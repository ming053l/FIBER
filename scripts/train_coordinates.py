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
from fiber.transforms.rotation import RotatedFrame
from fiber.transforms.spectral import SpectralFrame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

log = get_logger("arm")


BASIS_ONLY_TYPES = {"rotated_random", "rotated_learned"}


def run_stem(arm: str, k: int, seed: int, extractor_arch: str, receiver_seed: int,
             scope: str = "gate") -> str:
    """A run's identity is (arm, k, structure seed, receiver architecture, receiver seed,
    analysis scope). Anything less and runs overwrite each other on disk.

    Two collisions this has already had to fix:
      * the P0-5 receiver control reran C2/D/E at seed 0 with `--extractor-arch spatial`
        under the same stem as the primary receiver, silently replacing the Gate runs;
      * the P0-7 basis analysis reruns D_spectral at seed 0 with receiver seed 0 -- the
        same arm, k, seed, architecture and receiver as the Gate run. Without the scope
        in the stem it overwrote the Gate run AND flipped its analysis_scope to
        p0_7_basis, at which point the selector skips it and the Gate loses that seed
        entirely.
    """
    return f"{arm}_k{k}_s{seed}_rx{extractor_arch}_r{receiver_seed}_sc{scope}"


def build_arm_frame(cfg, arm: str, k: int, seed: int, tag: str, d: int):
    spec = dict(cfg["fiber"]["arms"][arm])
    if spec["type"] in ("rotated_random", "rotated_learned"):
        # P0-7: the ambient subspace comes from the certified spectral fit and is
        # FROZEN; only the in-subspace basis varies between D1, D2 and D3.
        #
        # The arm's own seed drives the ROTATION ONLY. `base_seed` is pinned, so every
        # D2 draw and every D3 seed sits in the SAME subspace as D1 at that base seed.
        # Letting the arm seed drive both would put different draws in different
        # subspaces, and the whole subspace-versus-basis decomposition would be
        # comparing two things at once.
        base_arm = spec.get("base_arm", "D_spectral")
        base_seed = int(spec.get("base_seed", 0))
        base, extra = build_arm_frame(cfg, base_arm, k, base_seed, tag, d)
        mode = "random" if spec["type"] == "rotated_random" else "learned"
        return RotatedFrame(base, k=k, mode=mode, seed=seed), {
            **extra, "rotation_mode": mode, "base_arm": base_arm, "base_seed": base_seed}
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
    ap.add_argument("--scope", default=None,
                    choices=["gate", "receiver_control", "p0_7_basis"],
                    help="which analysis this run belongs to; the Gate selector reads "
                         "ONLY gate-scope runs")
    ap.add_argument("--receiver-seed", type=int, default=None,
                    help="P0-7.1: separate the receiver's randomness from the BASIS "
                         "seed, so a D2 spread can be attributed to the basis")
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

    # args.seed selects the arm's structure (subspace, or basis for a rotation arm);
    # receiver_seed selects the extractor initialisation and training order. Sharing one
    # seed for both makes a D2 spread a mixture of basis variability and extractor
    # training noise, which cannot then be attributed to the coding basis.
    receiver_seed = args.seed if args.receiver_seed is None else args.receiver_seed
    arm_type = cfg["fiber"]["arms"][args.arm]["type"]
    scope = args.scope
    if scope is None:
        # A control run must never default into the Gate pool.
        if tcfg.extractor_arch != cfg["extractor"].get("arch", "resnet18"):
            scope = "receiver_control"
        elif arm_type in BASIS_ONLY_TYPES:
            scope = "p0_7_basis"
        else:
            scope = "gate"
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
        # P0-7 D3 discovers a BASIS, so its surrogate target is smooth-sign rather than
        # the raw coordinate; tau travels in the arm spec and is therefore part of the
        # hyperparameter fingerprint the val lock records.
        if cfg["fiber"]["arms"][args.arm]["type"] == "rotated_learned":
            dcfg.target_transform = "soft_sign"
            dcfg.soft_sign_tau = float(cfg["fiber"]["arms"][args.arm].get("tau", 0.5))
            meta["soft_sign_tau"] = dcfg.soft_sign_tau
        log.info("[%s] discovery on split A (%d epochs)", args.arm, dcfg.epochs)
        _, frame, dhist = train_extractor(
            frame, root, bank, dcfg, split=args.train_split,
            crossfit=cfg["dataset"]["crossfit"]["discovery_split"] if args.crossfit is None else xfit_eval,
            device=args.device, seed=args.seed, learn_frame=True, attacks=bank.train,
            limit=args.limit)   # discovery randomness follows the structure seed
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
        device=args.device, seed=receiver_seed, learn_frame=False, attacks=bank.train,
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
    stem = run_stem(args.arm, k, args.seed, tcfg.extractor_arch, receiver_seed, scope)
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
        # `seed` is the arm's structure seed; these name what it actually varies, and
        # `analysis_scope` keeps control runs out of the Gate candidate pool
        "structure_seed": args.seed, "basis_seed": args.seed,
        "receiver_seed": receiver_seed, "analysis_scope": scope,
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
