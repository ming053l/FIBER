#!/usr/bin/env python
"""Train + evaluate ONE arm under the cross-fit protocol (PLAN.md §4).

    Discovery : fit Q on split A                (arms D, E only)
    Freeze    : Q frozen, discovery extractor DISCARDED
    Re-fit    : fresh extractor on split B      (EVERY arm, identical budget)
    Evaluate  : test + test_heldout_prompts, all eval attacks, per sample

    python scripts/train_coordinates.py --arm C_hadamard --k 64 --seed 0 --tag pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.training import TrainConfig, evaluate, train_extractor
from fiber.transforms import build_frame
from fiber.transforms.spectral import SpectralFrame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

log = get_logger("arm")


def build_arm_frame(cfg, arm: str, k: int, seed: int, tag: str, d: int):
    spec = dict(cfg["fiber"]["arms"][arm])
    if spec["type"] == "spectral_topk":
        path = Path(cfg["paths"]["data_root"]) / "spectrum" / f"{tag}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run scripts/fit_observability_spectrum.py --tag {tag} "
                f"--seed {seed} first (arm D is the spectrum, not a baseline)")
        return SpectralFrame(d=d, k=k, path=path), {"spectrum_file": str(path)}
    return build_frame(spec, d=d, k=k, seed=seed), {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--discovery-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    # val is evaluated too, because ALL model selection happens there (P0-3) and the
    # test evaluation is not allowed to choose anything.
    ap.add_argument("--eval-splits", nargs="*",
                    default=["val", "test", "test_heldout_prompts"])
    # Overrides for smoke-testing the pipeline before the full cache exists.
    # Leave them alone for real runs: the cross-fit protocol is the experiment.
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--crossfit", default=None, help="'A', 'B' or 'none'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extractor-arch", default=None,
                    help="receiver architecture; `spatial` is the no-GAP control (P0-5)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / args.tag
    d = int(cfg["latent"]["dim"])
    k = args.k or int(cfg["fiber"]["robust_dims"])
    if args.arm not in cfg["fiber"]["arms"]:
        raise SystemExit(f"unknown arm {args.arm}; have {sorted(cfg['fiber']['arms'])}")

    xfit_eval = cfg["dataset"]["crossfit"]["extractor_split"]
    if args.crossfit is not None:
        xfit_eval = None if args.crossfit.lower() == "none" else args.crossfit
    tcfg = TrainConfig.from_config(cfg)
    if args.extractor_arch:
        tcfg.extractor_arch = args.extractor_arch
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.batch_size:
        tcfg.batch_size = args.batch_size

    frame, extra = build_arm_frame(cfg, args.arm, k, args.seed, args.tag, d)
    learnable = any(p.requires_grad for p in frame.parameters()) or \
        cfg["fiber"]["arms"][args.arm]["type"] == "householder"
    t0 = time.time()
    meta = {"discovery": None}

    # ---- discovery (arm E only): joint (Q, H) on split A, extractor discarded
    if learnable:
        dcfg = TrainConfig(**{**tcfg.__dict__})
        dcfg.epochs = args.discovery_epochs or tcfg.epochs
        # Discovery is MSE-only (P0-4). The sign head is trained afterwards, on the
        # frozen frame, and is what the communication metric reads.
        dcfg.w_sign = 0.0
        log.info("[%s] discovery on split A (%d epochs)", args.arm, dcfg.epochs)
        _, frame, dhist = train_extractor(
            frame, root, bank, dcfg, split=args.train_split,
            crossfit=cfg["dataset"]["crossfit"]["discovery_split"] if args.crossfit is None else xfit_eval,
            device=args.device, seed=args.seed, learn_frame=True, attacks=bank.train,
            limit=args.limit)
        meta["discovery"] = {"epochs": dcfg.epochs, "objective": "mse",
                             "history": dhist[-3:],
                             "orthonormality_error": frame.orthonormality_error()}
        # the discovery extractor is thrown away here, on purpose (PLAN.md §4)

    ortho = frame.orthonormality_error()
    tol = float(cfg["gate3a"]["orthogonality_tol"])
    if ortho > tol:
        raise SystemExit(f"frame is not orthonormal: ‖RRᵀ−I‖={ortho:.2e} > {tol:.0e}")

    # ---- evaluation extractor: split B, identical budget for every arm
    log.info("[%s] evaluation extractor on split B (%d epochs)", args.arm, tcfg.epochs)
    model, frame, hist = train_extractor(
        frame, root, bank, tcfg, split=args.train_split, crossfit=xfit_eval,
        device=args.device, seed=args.seed, learn_frame=False, attacks=bank.train,
        limit=args.limit)

    results, arrays = {}, {}
    for split in args.eval_splits:
        ev = evaluate(model, frame, root, bank, split, bank.eval, device=args.device,
                      limit=args.limit)
        results[split] = {a: {kk: v for kk, v in r.items()
                              if kk not in ("per_sample", "sample_ids", "pearson_per_coord")}
                          for a, r in ev.items()}
        for a, r in ev.items():
            arrays[f"{split}|{a}|per_sample"] = r["per_sample"]
            arrays[f"{split}|{a}|sample_ids"] = np.array(r["sample_ids"])
            if "pearson_per_coord" in r:
                arrays[f"{split}|{a}|pearson"] = r["pearson_per_coord"]

    out_dir = Path(cfg["paths"]["data_root"]) / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.arm}_k{k}_s{args.seed}"
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    rows = frame.rows().detach().cpu().contiguous()
    rows_digest = hashlib.blake2s(rows.numpy().tobytes(), digest_size=16).hexdigest()
    torch.save({"rows_digest": rows_digest, "rows": rows,
                "state_dict": frame.state_dict()}, out_dir / f"{stem}_frame.pt")
    # The arm's own hyperparameters travel WITH the run, so selection can tell two
    # runs of the same arm and k apart, and the lock can name exactly one of them.
    arm_spec = {kk: vv for kk, vv in cfg["fiber"]["arms"][args.arm].items() if kk != "seeds"}
    hp_fp = hashlib.blake2s(json.dumps(arm_spec, sort_keys=True, default=str).encode(),
                            digest_size=8).hexdigest()
    summary = {
        "arm": args.arm, "type": cfg["fiber"]["arms"][args.arm]["type"], "k": k,
        "arm_spec": arm_spec, "hyperparameters_fingerprint": hp_fp,
        "seed": args.seed, "tag": args.tag,
        "orthonormality_error": ortho, "rows_digest": rows_digest,
        "extractor_arch": tcfg.extractor_arch,
        "epochs": tcfg.epochs, "final_train_loss": hist[-1]["loss"],
        "train_split": args.train_split, "crossfit": xfit_eval,
        "seconds": round(time.time() - t0, 1),
        "results": results, **extra, **meta,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("[%s] done in %.1f s -> %s", args.arm, summary["seconds"], out_dir / f"{stem}.json")
    for split in args.eval_splits:
        mean = np.mean([r["sign_ber"] for r in results[split].values()])
        log.info("  %-24s mean sign BER %.4f", split, mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
