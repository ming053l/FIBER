#!/usr/bin/env python
"""Fit the DECODER-CERTIFIED observability operator (P0-1).

    C_cert(f) = E[ z_c f_c' + f_c z_c' - f_c f_c' ]  =  C_obs - E[(m-f)(m-f)']  <=  C_obs
    v' C_cert v = Var(v'Z) - E[(v'z_c - v'f_c)^2]    ( = 1 - MSE_v for Z ~ N(0,I) )

so a weak decoder UNDERSTATES observability and can never manufacture it. This
replaces Cov(f(Y)), which is not a lower bound on C_obs for an approximate teacher.

Every sample plays exactly one role:

    A_teacher  -> fit f
    A_operator -> estimate C_cert, take the top-k directions, FREEZE
    val        -> cross-fit the reported eigenvalues and the validity diagnostic

In-sample eigenvalues are inflated far past the theoretical bound of 1 (measured
lambda_max = 7.2 with an EXACT teacher at N=200, d=4096), so they only ever select
directions. Everything reported is the held-out lambda_skill.

    python scripts/fit_observability_spectrum.py --tag pilot --per-attack
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.spectrum import fit_certified, subspace_alignment, teacher_validity
from fiber.training import TrainConfig, teacher_outputs, train_teacher
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

log = get_logger("spectrum")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--cache-tag", default=None,
                    help="image cache namespace; defaults to --tag. A triage writes its "
                         "artifacts under a fresh --tag while reusing an existing cache.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None, help="directions to keep")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--report-split", default=None)
    ap.add_argument("--teacher", default=None,
                    choices=["resnet18", "spatial", "spatial_sharedtrunk"],
                    help="decoder class the operator is certified BY (P0-5)")
    ap.add_argument("--per-attack", action="store_true",
                    help="fixed-decoder operational spectrum per attack (NOT Cov(E[Z|Y,T=t]))")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / (args.cache_tag or args.tag)
    scfg = cfg["spectrum"]
    xfit = cfg["dataset"]["crossfit"]
    d = int(cfg["latent"]["dim"])
    k = args.k or max(int(cfg["fiber"]["robust_dims"]), 256)
    report_split = args.report_split or scfg.get("report_split", "val")

    if str(scfg.get("operator", "certified")) != "certified":
        raise SystemExit("spectrum.operator must be 'certified'; Cov(f) is a diagnostic (P0-1)")

    tcfg = TrainConfig.from_config(cfg)
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.batch_size:
        tcfg.batch_size = args.batch_size

    t0 = time.time()
    # ---- 1. teacher on A_teacher, MSE only ------------------------------
    arch = args.teacher or scfg.get("teacher_arch", "resnet18")
    teacher, hist = train_teacher(root, bank, tcfg, d=d, arch=arch,
                                  latent_shape=(cfg["latent"]["channels"],
                                                cfg["latent"]["height"], cfg["latent"]["width"]),
                                  crossfit_sub=xfit["teacher_subsplit"],
                                  device=args.device, seed=args.seed, attacks=bank.train,
                                  limit=args.limit)
    log.info("teacher trained on %s in %.1f s (final mse %.4f)",
             xfit["teacher_subsplit"], time.time() - t0, hist[-1]["mse"])

    # ---- 2. discovery on A_operator, disjoint from the teacher's samples --
    F_op, Z_op = teacher_outputs(teacher, root, bank, "train", attacks=bank.train,
                                 crossfit_sub=xfit["operator_subsplit"], device=args.device,
                                 limit=args.limit, epoch_salt="spectrum-operator")
    spec = fit_certified(Z_op, F_op, k=k, oversampling=int(scfg.get("oversampling", 32)),
                         seed=args.seed, center=bool(scfg.get("center", True)),
                         method="range_eigh" if scfg.get("eigensolver") == "range_eigh" else "eigsh")
    V = spec.eigenvectors[:k]
    log.info("C_cert on %s: n=%d, in-sample lambda_1=%.3f (in-sample only, never reported)",
             xfit["operator_subsplit"], spec.n_samples, float(spec.eigenvalues[0]))

    # ---- 3. cross-fit the reported numbers on a held-out split -----------
    F_rep, Z_rep = teacher_outputs(teacher, root, bank, report_split, attacks=bank.train,
                                   device=args.device, limit=args.limit,
                                   epoch_salt="spectrum-report")
    valid = teacher_validity(Z_rep, F_rep, V,
                             bootstrap=int(scfg.get("certification_bootstrap", 2000)))
    lam = valid["lambda_skill"]           # per-coordinate: basis dependent, diagnostic
    sub = valid["subspace"]               # basis invariant: THE headline (P0-1.1)
    tol = float(scfg.get("validity_tol", 0.10))

    summary = {
        "tag": args.tag, "cache_tag": args.cache_tag or args.tag,
        "seed": args.seed, "operator": "certified",
        "solver": spec.solver, "k_kept": int(V.shape[0]),
        "teacher_arch": arch, "teacher_loss": "mse",
        "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
        "teacher_trunk_parameters": sum(p.numel() for n, p in teacher.named_parameters()
                                        if n.startswith("trunk")),
        "teacher_final_mse": hist[-1]["mse"], "teacher_epochs": tcfg.epochs,
        "n_teacher_split": None, "n_operator": int(spec.n_samples),
        "n_report": int(valid["n_heldout"]), "report_split": report_split,

        # ---- SUBSPACE: basis-invariant, cross-fitted. THE headline. ----
        # D_cert_subspace = sum_j max(mu_j, 0) with mu = eig(V C_cert^held V^T).
        # Clipping the DIAGONAL of that matrix instead would not be a property of
        # the discovered subspace: C_V = [[-1,2],[2,-1]] clips to 0 while its
        # eigenvalues are (1,-3), i.e. one rotation from a certified direction.
        "D_cert_subspace": sub["D_cert_subspace"],
        # The bound is what the word "certified" has to rest on: tau only excludes
        # floating-point noise, a one-sided LCB above zero is a claim.
        "D_cert_LCB": sub.get("D_cert_LCB"),
        "D_cert_LCB_per_fold": sub.get("D_cert_LCB_per_fold"),
        # a count of positive quadratic forms, NOT a rank -- see the inertia field
        "certified_positive_direction_count": sub.get("certified_positive_direction_count"),
        "certified_positive_inertia": sub.get("certified_positive_inertia"),
        "weyl_radius": sub.get("weyl_radius"),
        "fold_masses": sub.get("fold_masses"),
        "D_cert_subspace_insample": sub["D_cert_subspace_insample"],
        "trace_C_V": sub["trace_C_V"],
        "inner_crossfit": sub["crossfit"], "rotation_split": sub["rotation_split"],
        # numerical, NOT certified: tau excludes floating-point noise only. The
        # certified quantity is the inertia below, which carries a confidence radius.
        "numerical_positive_rank": sub["numerical_positive_rank"],
        "requested_k": sub["requested_k"],
        "mu_max": sub["mu_max"], "mu_min": sub["mu_min"],
        "zero_tolerance": sub["zero_tolerance"],

        # ---- COORDINATES: basis dependent. Diagnostics, and the quantity that
        # sign coding actually depends on (P0-7). Never the observability headline.
        "coordinate_skill_top1": float(lam[0]),
        "coordinate_skill_top64_mean": float(lam[:64].mean()),
        "D_coordinate_clipped": sub["D_coordinate_clipped"],
        "n_negative_coordinates": int(valid["n_negative_directions"]),

        # ---- validity: lambda_var == lambda_skill only for an exact mean --
        "validity_mean_abs_gap": valid["mean_abs_gap"],
        "validity_max_abs_gap": valid["max_abs_gap"],
        "validity_tol": tol,
        "validity_pass": bool(valid["mean_abs_gap"] < tol),
        "teacher_output_mean_norm": valid["teacher_output_mean_norm"],

        # ---- in-sample, for contrast only -----------------------------
        "insample_lambda_top1": float(spec.eigenvalues[0]),
        "insample_trace_signed": spec.trace_signed,
        "insample_positive_mass": spec.total_positive_mass(),
        "insample_negative_mass": spec.negative_mass(),
        "insample_spectrum_complete": spec.spectrum_is_complete,

        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or "uncommitted",
        "seconds": round(time.time() - t0, 1),
    }

    if args.per_attack:
        per = {}
        for attack in bank.eval:
            Fa, Za = teacher_outputs(teacher, root, bank, report_split, attacks=[attack],
                                     mode="fixed", device=args.device, limit=args.limit)
            va = teacher_validity(Za, Fa, V)
            sp_a = fit_certified(Za, Fa, k=min(64, Za.shape[0] - 2), oversampling=8,
                                 seed=args.seed, method="range_eigh")
            per[attack] = {
                "D_cert_subspace": va["subspace"]["D_cert_subspace"],
                "numerical_positive_rank": va["subspace"]["numerical_positive_rank"],
                "coordinate_skill_top1": float(va["lambda_skill"][0]),
                "alignment_with_mixture": subspace_alignment(
                    torch.from_numpy(sp_a.eigenvectors[:64]).float(),
                    torch.from_numpy(V[:64]).float()),
            }
            log.info("%-10s D_cert_subspace=%7.3f (rank %d)  align=%.3f", attack,
                     per[attack]["D_cert_subspace"],
                     per[attack]["numerical_positive_rank"],
                     per[attack]["alignment_with_mixture"])
        summary["per_attack"] = per
        summary["per_attack_note"] = (
            "fixed-decoder operational spectrum under attack t: one attack-mixture "
            "teacher evaluated under each attack. NOT Cov(E[Z|Y,T=t]) -- that would "
            "need a decoder calibrated per attack.")

    out_dir = Path(cfg["paths"]["data_root"]) / "spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_seed{args.seed}" + ("" if arch == "resnet18" else f"_{arch}")
    torch.save({
        "operator": "certified",
        "eigenvectors": torch.from_numpy(V).float(),
        "coordinate_skill_heldout": torch.from_numpy(np.asarray(lam)).float(),
        "mu_subspace_heldout": torch.from_numpy(np.asarray(sub["mu"])).float(),
        "D_cert_subspace": sub["D_cert_subspace"],
        "lambda_var_heldout": torch.from_numpy(np.asarray(valid["lambda_var"])).float(),
        "eigenvalues_insample": torch.from_numpy(spec.eigenvalues[:k]).float(),
        "validity_pass": summary["validity_pass"], "k": k, "d": d,
        "teacher_loss": "mse", "teacher_arch": arch, "seed": args.seed, "k": k,
        "tag": args.tag, "cache_tag": args.cache_tag or args.tag,
        "commit": summary["commit"],
    }, out_dir / f"{stem}.pt")

    # Report-split outputs, so the cross-decoder certificate matrix can be computed
    # without re-running either teacher.
    torch.save({"F_rep": F_rep, "Z_rep": Z_rep, "teacher_arch": arch,
                "report_split": report_split},
               out_dir / f"{stem}_report_outputs.pt")

    rep = Path(cfg["paths"]["reports_dir"]) / f"spectrum_{stem}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(summary, indent=2, default=float))

    log.info("SUBSPACE  D_cert = %.3f (LCB %s) over %d numerically positive / %s "
             "certified positive inertia of %d  (tr(C_V) = %.3f)",
             summary["D_cert_subspace"],
             f"{summary['D_cert_LCB']:.3f}" if summary.get("D_cert_LCB") is not None else "n/a",
             summary["numerical_positive_rank"],
             summary.get("certified_positive_inertia", "n/a"),
             summary["requested_k"], summary["trace_C_V"])
    log.info("          without the inner cross-fit it would read %.3f -- the gap is "
             "selection bias, not observability", summary["D_cert_subspace_insample"])
    log.info("COORDS    diagonal-clipped %.3f, top-1 %.3f  (basis dependent; not the headline)",
             summary["D_coordinate_clipped"], summary["coordinate_skill_top1"])
    log.info("teacher validity: mean|lambda_var - lambda_skill| = %.4f (tol %.2f) -> %s",
             summary["validity_mean_abs_gap"], tol,
             "PASS" if summary["validity_pass"] else "FAIL")
    if not summary["validity_pass"]:
        log.warning("validity gate FAILED: the teacher is far from a conditional mean, so "
                    "these directions are certified by a poor decoder. Report as such.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
