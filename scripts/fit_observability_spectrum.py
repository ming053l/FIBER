#!/usr/bin/env python
"""Fit the Generative Observability Spectrum  C_obs = Cov(E[Z|Y])  (PLAN.md §3).

    teacher M_theta(Y) ~= E[Z|Y]   trained with MSE on split A (discovery)
    C_hat = (1/N) sum m m^T        never formed: randomized SVD of M
    lambda_j = 1 - MMSE_j in [0,1] fraction of direction j that survives
    Tr(C_hat) = sum lambda_j       effective number of recoverable dimensions

Eigenvectors are fit on the discovery split; the REPORTED eigenvalues are
re-measured on a held-out split (cross-fit), because in-sample eigenvalues of a
covariance are upward biased and would inflate Tr.

    python scripts/fit_observability_spectrum.py --tag pilot --epochs 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fiber.channels import ChannelBank
from fiber.spectrum import fit_spectrum, subspace_alignment, trace_c_obs
from fiber.training import TrainConfig, teacher_outputs, train_teacher
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

log = get_logger("spectrum")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="how many directions to keep")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--holdout-split", default="val")
    ap.add_argument("--per-attack", action="store_true", help="also fit one spectrum per attack")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / args.tag
    scfg = cfg["spectrum"]
    d = int(cfg["latent"]["dim"])
    k = args.k or max([int(cfg["fiber"]["robust_dims"])] + [256])

    tcfg = TrainConfig.from_config(cfg)
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.batch_size:
        tcfg.batch_size = args.batch_size

    t0 = time.time()
    teacher, hist = train_teacher(root, bank, tcfg, d=d, crossfit=cfg["dataset"]["crossfit"]["discovery_split"],
                                  device=args.device, seed=args.seed, attacks=bank.train,
                                  limit=args.limit)
    log.info("teacher trained in %.1f s (final mse %.4f)", time.time() - t0, hist[-1]["mse"])

    # discovery outputs (directions) and held-out outputs (honest eigenvalues)
    M_fit, Z_fit = teacher_outputs(teacher, root, bank, "train", attacks=bank.train,
                                   crossfit=cfg["dataset"]["crossfit"]["discovery_split"],
                                   device=args.device, limit=args.limit)
    M_held, Z_held = teacher_outputs(teacher, root, bank, args.holdout_split, attacks=bank.train,
                                     device=args.device, limit=args.limit,
                                     epoch_salt="spectrum-holdout")

    spec = fit_spectrum(M_fit, k=k, cfg=scfg, seed=args.seed, strict=False)
    lam_held = spec.evaluate_on(M_held)
    # 1 - MMSE cross-check along the recovered directions, on held-out data
    proj_err = ((Z_held @ spec.eigenvectors.T) - (M_held @ spec.eigenvectors.T)).pow(2).mean(0)
    one_minus_mmse = (1.0 - proj_err)

    out_dir = Path(cfg["paths"]["data_root"]) / "spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_seed{args.seed}"
    torch.save({"eigenvectors": spec.eigenvectors, "eigenvalues_insample": spec.eigenvalues,
                "eigenvalues_heldout": lam_held, "trace_insample": spec.trace,
                "trace_heldout": trace_c_obs(M_held), "k": k, "d": d,
                "teacher_loss": "mse", "seed": args.seed},
               out_dir / f"{stem}.pt")

    summary = {
        "tag": args.tag, "seed": args.seed, "k_kept": int(spec.eigenvectors.shape[0]),
        "teacher_final_mse": hist[-1]["mse"],
        "teacher_epochs": tcfg.epochs,
        "n_discovery": int(M_fit.shape[0]), "n_holdout": int(M_held.shape[0]),
        # THE headline number: effective number of recoverable dimensions
        "trace_c_obs_heldout": trace_c_obs(M_held),
        "trace_c_obs_insample": spec.trace,
        "lambda_top1_heldout": float(lam_held[0]),
        "lambda_top64_mean_heldout": float(lam_held[:64].mean()),
        "lambda_flatness_top64": float(lam_held[:64].max() / lam_held[:64].min().clamp_min(1e-9)),
        "one_minus_mmse_top1": float(one_minus_mmse[0]),
        "one_minus_mmse_top64_mean": float(one_minus_mmse[:64].mean()),
        "effective_rank": spec.effective_rank(),
        "estimator": spec.method,
        "seconds": round(time.time() - t0, 1),
    }

    if args.per_attack:
        per = {}
        for attack in bank.eval:
            Ma, _ = teacher_outputs(teacher, root, bank, args.holdout_split, attacks=[attack],
                                    mode="fixed", device=args.device, limit=args.limit)
            sp_a = fit_spectrum(Ma, k=min(k, Ma.shape[0] - 1), cfg=scfg, seed=args.seed, strict=False)
            per[attack] = {
                "trace_c_obs": trace_c_obs(Ma),
                "lambda_top1": float(sp_a.eigenvalues[0]),
                "alignment_with_mixture": subspace_alignment(
                    sp_a.eigenvectors[:64], spec.eigenvectors[:64]),
            }
            log.info("%-10s Tr(C_obs) %.2f  align %.3f", attack, per[attack]["trace_c_obs"],
                     per[attack]["alignment_with_mixture"])
        summary["per_attack"] = per

    rep = Path(cfg["paths"]["reports_dir"]) / f"spectrum_{stem}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(summary, indent=2))
    log.info("Tr(C_obs) held-out = %.2f of d=%d  (lambda_1 = %.3f)",
             summary["trace_c_obs_heldout"], d, summary["lambda_top1_heldout"])
    log.info("wrote %s and %s", out_dir / f"{stem}.pt", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
