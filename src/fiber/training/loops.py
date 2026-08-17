"""Training / evaluation loops shared by Phases 2 and 3.

Cross-fit discipline (PLAN.md §4) is enforced by the CALLER choosing the split:

    discovery       : split A   (fit Q; the extractor trained here is thrown away)
    evaluation      : split B   (a fresh extractor, identical init/capacity/schedule)
    reporting       : test / test_heldout_prompts

EVERY arm — including the random ones — trains its evaluation extractor on B
only, so the extractor-training budget is identical across arms.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..channels import ChannelBank
from ..diffusion.cache_dataset import FiberDataset
from ..metrics import coord_pearson, sign_ber
from ..models import build_extractor
from ..utils.logging import get_logger
from ..utils.seeding import derive_seed

log = get_logger("fiber.train")


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 8
    w_regression: float = 1.0
    w_sign: float = 1.0
    frame_lr_scale: float = 1.0
    extractor_arch: str = "resnet18"
    # P0-7 D3: a hard sign target severs the gradient to an in-subspace rotation just
    # as it does to a frame, so basis discovery uses a smooth surrogate tanh(W/tau).
    # tau is a hyperparameter and is selected on val like any other, never on test.
    target_transform: str = "identity"
    soft_sign_tau: float = 0.5

    @classmethod
    def from_config(cls, cfg) -> "TrainConfig":
        t = cfg["training"]
        heads = cfg["extractor"]["heads"]
        return cls(
            batch_size=t["batch_size"], lr=t["lr"], weight_decay=t["weight_decay"],
            epochs=t["epochs"], warmup_steps=t["warmup_steps"], grad_clip=t["grad_clip"],
            amp=t["amp"],
            w_regression=heads["regression"]["weight"] if heads["regression"]["enabled"] else 0.0,
            w_sign=heads["sign"]["weight"] if heads["sign"]["enabled"] else 0.0,
            extractor_arch=cfg["extractor"].get("arch", "resnet18"),
        )


def make_loader(root, split: str, bank: ChannelBank, *, attacks=None, mode="sampled",
                crossfit: str | None = None, crossfit_sub: str | None = None,
                batch_size: int = 16, workers: int = 8, shuffle: bool = True,
                epoch_salt: str = "", limit: int = 0,
                subset_size: int = 0, subset_seed: int = 0) -> DataLoader:
    ds = FiberDataset(root, split, bank, attacks=attacks, mode=mode, crossfit=crossfit,
                      crossfit_sub=crossfit_sub, epoch_salt=epoch_salt)
    if subset_size:
        # NESTED random subsets, for sample-size experiments. A seeded permutation
        # taken as a prefix means N=100 is a subset of N=200 is a subset of the whole
        # split, so a learning curve varies the SIZE and nothing else. `limit` below
        # slices the records in their stored order instead, which is fine for a debug
        # smoke test and wrong here: the order carries the sample index, and therefore
        # the latent seed and the prompt draw, so a prefix confounds sample size with
        # subset composition.
        # The permutation depends only on subset_seed, never on epoch_salt, so the
        # subset is identical across epochs while the attack draw still varies.
        import numpy as _np
        perm = _np.random.default_rng(derive_seed("subset", subset_seed)
                                      % (2**31)).permutation(len(ds.records))
        ds.records = [ds.records[i] for i in perm[:subset_size]]
    elif limit:
        ds.records = ds.records[:limit]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=False, persistent_workers=workers > 0)


def _lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    p = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def soft_sign(w: torch.Tensor, tau: float) -> torch.Tensor:
    """tanh(w/tau): large tau approaches a linear target, small tau approaches sign(w).
    So tau decides whether basis discovery optimises continuous recoverability or
    near-sign recoverability -- which is exactly why it cannot be tuned on test."""
    return torch.tanh(w / tau)


def head_losses(out: dict, w_true: torch.Tensor, cfg: TrainConfig):
    loss = w_true.new_zeros(())
    parts = {}
    if cfg.target_transform == "soft_sign":
        w_true = soft_sign(w_true, cfg.soft_sign_tau)
    elif cfg.target_transform != "identity":
        raise ValueError(f"unknown target_transform {cfg.target_transform!r}")
    if cfg.w_regression and "w_hat" in out:
        # MSE, never L1: the regression head is also the teacher objective and
        # must converge to the conditional MEAN (PLAN.md §3.3).
        parts["mse"] = F.mse_loss(out["w_hat"], w_true)
        loss = loss + cfg.w_regression * parts["mse"]
    if cfg.w_sign and "sign_logits" in out:
        bits = (w_true > 0).float()
        parts["bce"] = F.binary_cross_entropy_with_logits(out["sign_logits"], bits)
        loss = loss + cfg.w_sign * parts["bce"]
    return loss, parts


def train_extractor(frame, root, bank: ChannelBank, cfg: TrainConfig, *, split="train",
                    crossfit: str | None = "B", device="cuda:0", seed: int = 0,
                    learn_frame: bool = False, attacks=None, limit: int = 0,
                    subset_size: int = 0, subset_seed: int = 0, log_every: int = 50):
    """Returns (extractor, frame, history). `learn_frame=True` is the DISCOVERY
    stage of arm E only; evaluation always runs with the frame frozen."""
    device = torch.device(device)
    torch.manual_seed(derive_seed("extractor", seed) % (2**31))
    model = build_extractor(cfg.extractor_arch, k=frame.k).to(device)
    frame = frame.to(device)
    frame.requires_grad_(learn_frame)

    if learn_frame and cfg.w_sign:
        # A hard sign target has NO gradient path to the frame: (w > 0) returns a bool
        # tensor, so the autograd graph is severed and frame.grad is None -- not small,
        # absent. Discovery is therefore an MSE objective and is labelled as one, rather
        # than appearing to optimise sign BER while silently optimising MSE (P0-4).
        raise ValueError(
            "learn_frame=True with a non-zero sign weight: the hard sign target 1[W>0] "
            "contributes no gradient to the frame, so this would look like sign-BER "
            "optimisation while actually being MSE-only. Set w_sign=0 for discovery.")
    groups = [{"params": list(model.parameters()), "lr": cfg.lr, "lr_scale": 1.0}]
    frame_params = [p for p in frame.parameters() if p.requires_grad]
    if learn_frame and frame_params:
        # weight_decay=0 on the frame: H(v) depends only on v/||v||, so decay does not
        # regularise the transform at all -- it just shrinks ||v|| toward the numerically
        # unstable point where the reflection is undefined.
        groups.append({"params": frame_params, "lr": cfg.lr * cfg.frame_lr_scale,
                       "lr_scale": cfg.frame_lr_scale, "weight_decay": 0.0})
    elif learn_frame:
        raise ValueError("learn_frame=True but the frame has no trainable parameters")
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    history, step = [], 0
    for epoch in range(cfg.epochs):
        loader = make_loader(root, split, bank, attacks=attacks, crossfit=crossfit,
                             batch_size=cfg.batch_size, workers=cfg.num_workers,
                             epoch_salt=f"e{epoch}", limit=limit,
                             subset_size=subset_size, subset_seed=subset_seed)
        total_steps = cfg.epochs * max(len(loader), 1)
        model.train()
        t0, run = time.time(), []
        for batch in loader:
            lr = _lr_at(step, total_steps, cfg)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            x = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            with torch.no_grad() if not learn_frame else torch.enable_grad():
                w = frame.project(z)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(x)
            out = {kk: v.float() for kk, v in out.items()}
            loss, parts = head_losses(out, w, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if learn_frame:
                torch.nn.utils.clip_grad_norm_(frame.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            run.append(float(loss))
            step += 1
        entry = {"epoch": epoch, "loss": float(np.mean(run)), "lr": lr,
                 "seconds": round(time.time() - t0, 1)}
        if learn_frame:
            entry["orthonormality_error"] = frame.orthonormality_error()
        history.append(entry)
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            log.info("epoch %d/%d loss %.4f (%.1fs)%s", epoch + 1, cfg.epochs, entry["loss"],
                     entry["seconds"],
                     f" ortho {entry['orthonormality_error']:.2e}" if learn_frame else "")
    frame.requires_grad_(False)
    return model, frame, history


@torch.no_grad()
def evaluate(model, frame, root, bank: ChannelBank, split: str, attacks, *, device="cuda:0",
             batch_size: int = 32, workers: int = 8, limit: int = 0) -> dict:
    """Per-attack sign BER, kept PER SAMPLE so Gate 3A's paired bootstrap can
    difference arms on the same (z_i, prompt_i, attack_i)."""
    device = torch.device(device)
    model.eval().to(device)
    frame = frame.to(device)
    out: dict[str, dict] = {}
    for attack in attacks:
        loader = make_loader(root, split, bank, attacks=[attack], mode="fixed",
                             batch_size=batch_size, workers=workers, shuffle=False, limit=limit)
        per_sample, ids, w_all, w_hat_all = [], [], [], []
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            w = frame.project(z)
            pred = model(x)
            per_sample.append(sign_ber(pred["sign_logits"].float(), w, reduce="per_sample").cpu())
            if "w_hat" in pred:
                w_all.append(w.cpu())
                w_hat_all.append(pred["w_hat"].float().cpu())
            ids.extend(batch["sample_id"])
        per_sample = torch.cat(per_sample)
        rec = {"sign_ber": float(per_sample.mean()),
               "per_sample": per_sample.numpy(),
               "sample_ids": ids, "n": len(ids)}
        if w_all:
            w = torch.cat(w_all); w_hat = torch.cat(w_hat_all)
            rho = coord_pearson(w_hat, w)
            rec.update({"coord_mse": float((w_hat - w).pow(2).mean()),
                        "pearson_mean": float(rho.mean()),
                        "pearson_per_coord": rho.numpy(),
                        # same units as the observability spectrum: 1 - MMSE
                        "observed_capacity": float(1 - (w_hat - w).pow(2).mean())})
        out[attack] = rec
        log.info("%-8s %-20s sign_ber %.4f", split, attack, rec["sign_ber"])
    return out
