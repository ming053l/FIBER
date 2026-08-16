"""Teacher M_theta(Y) ~= E[Z | Y], the estimator behind C_obs.

MSE and only MSE (PLAN.md §3.3): L2 regression converges to the conditional
MEAN, L1 to the conditional MEDIAN, and lambda = 1 - MMSE is a law-of-total-
variance statement about the mean. The loss is not a parameter of this function.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..channels import ChannelBank
from ..models import build_teacher
from ..utils.logging import get_logger
from ..utils.seeding import derive_seed
from .loops import TrainConfig, _lr_at, make_loader

log = get_logger("fiber.teacher")


def train_teacher(root, bank: ChannelBank, cfg: TrainConfig, *, d: int = 16384,
                  arch: str = "resnet18", latent_shape=(4, 64, 64),
                  split: str = "train", crossfit: str | None = None,
                  crossfit_sub: str | None = "A_teacher", device="cuda:0",
                  seed: int = 0, attacks=None, limit: int = 0):
    """Trained on A_teacher only: the operator is estimated on A_operator, and a
    sample must never serve both roles (P0-1)."""
    device = torch.device(device)
    torch.manual_seed(derive_seed("teacher", seed) % (2**31))
    model = build_teacher(arch, d=d, latent_shape=latent_shape).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    history, step = [], 0
    for epoch in range(cfg.epochs):
        loader = make_loader(root, split, bank, attacks=attacks, crossfit=crossfit,
                             crossfit_sub=crossfit_sub, batch_size=cfg.batch_size,
                             workers=cfg.num_workers, epoch_salt=f"teacher-e{epoch}",
                             limit=limit)
        total = cfg.epochs * max(len(loader), 1)
        model.train()
        t0, run = time.time(), []
        for batch in loader:
            lr = _lr_at(step, total, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                m_hat = model(x)
            loss = torch.nn.functional.mse_loss(m_hat.float(), z)   # MSE and only MSE
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            run.append(float(loss))
            step += 1
        history.append({"epoch": epoch, "mse": float(np.mean(run)),
                        "seconds": round(time.time() - t0, 1)})
        if epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1:
            log.info("teacher epoch %d/%d mse %.4f (%.1fs)", epoch + 1, cfg.epochs,
                     history[-1]["mse"], history[-1]["seconds"])
    return model, history


@torch.no_grad()
def teacher_outputs(model, root, bank: ChannelBank, split: str, *, attacks=None,
                    mode: str = "sampled", crossfit: str | None = None,
                    crossfit_sub: str | None = None, device="cuda:0",
                    batch_size: int = 32, workers: int = 8, limit: int = 0,
                    epoch_salt: str = "spectrum") -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (F, Z): teacher outputs and the true latents, both on the CPU.

    BOTH are needed -- C_cert is built from the cross term E[z_c f_c'] as well as
    E[f_c f_c'], which is exactly what makes it a certified lower bound."""
    device = torch.device(device)
    model.eval().to(device)
    loader = make_loader(root, split, bank, attacks=attacks, mode=mode, crossfit=crossfit,
                         crossfit_sub=crossfit_sub, batch_size=batch_size, workers=workers,
                         shuffle=False, epoch_salt=epoch_salt, limit=limit)
    M, Z = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        M.append(model(x).float().cpu())
        Z.append(batch["z"])
    return torch.cat(M), torch.cat(Z)
