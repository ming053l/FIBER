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
from fiber.training import TrainConfig, evaluate, make_loader, train_extractor
from fiber.transforms import build_frame
from fiber.transforms.rotation import RotatedFrame
from fiber.transforms.spectral import SpectralFrame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean
from fiber.utils.seeding import set_determinism

# B1 physical test isolation: nothing produced before the val lock may touch the test
# splits. "The selection script does not read test" is weaker than "no test prediction,
# no test corruption and no test metric existed yet" -- only the second is checkable by
# someone auditing the artifacts.
FORBIDDEN_PRE_LOCK = ("test",)

log = get_logger("arm")


BASIS_ONLY_TYPES = {"rotated_random", "rotated_learned"}


def run_stem(arm: str, k: int, seed: int, extractor_arch: str, receiver_seed: int,
             scope: str = "gate", subset_size: int = 0, subset_seed: int = 0) -> str:
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
    stem = f"{arm}_k{k}_s{seed}_rx{extractor_arch}_r{receiver_seed}_sc{scope}"
    # A sample-size sweep reruns the same (arm, k, seed, receiver) at several training
    # set sizes. Without this in the stem every point of the curve overwrites the last
    # one and the surviving file is whichever happened to run last.
    if subset_size:
        stem += f"_n{subset_size}_ss{subset_seed}"
    return stem


GATE_SCOPE = "gate"


def _bind_spectrum(path: Path, blob: dict, tag: str, seed: int, scope: str, prov: dict):
    """What the Gate requires of a frozen spectral frame before it becomes a treatment.

    The producer stamps provenance; without this nothing consumed it. That left a legal
    path where `fit_observability_spectrum.py --allow-dirty` writes git_dirty=True, the
    tree is then committed, and a clean Gate run loads the dirty frame and records
    git_dirty=False -- its own. Downstream clean provenance masking upstream dirty
    evidence is the exact failure the provenance discipline exists to prevent, and the
    run summary kept only `spectrum_file`, a path, so it was undetectable afterwards.

    Exploratory scopes are exempt on purpose: the k diagnostic reuses a frame from an
    earlier commit deliberately, and that reuse is registered in reports/powercurve.md.
    """
    facts = {"spectrum_file": str(path),
             "spectrum_digest": hashlib.blake2s(path.read_bytes(), digest_size=16).hexdigest(),
             "spectrum_git_commit": blob.get("git_commit"),
             "spectrum_git_dirty": blob.get("git_dirty"),
             "spectrum_tag": blob.get("tag"), "spectrum_seed": blob.get("seed"),
             "spectrum_cache_tag": blob.get("cache_tag"),
             "spectrum_provenance_enforced": scope == GATE_SCOPE}
    if scope != GATE_SCOPE:
        return facts
    problems = []
    if blob.get("git_dirty") is not False:
        problems.append(f"git_dirty={blob.get('git_dirty')!r} (needs False; a frame fit "
                        "from an edited tree cannot name the code that produced it)")
    if prov.get("git_commit") and blob.get("git_commit") != prov["git_commit"]:
        problems.append(f"fit at {blob.get('git_commit')}, this run is at "
                        f"{prov['git_commit']}")
    if blob.get("tag") != tag:
        problems.append(f"tag {blob.get('tag')!r} != {tag!r}")
    if blob.get("seed") != seed:
        problems.append(f"seed {blob.get('seed')!r} != {seed!r}")
    if problems:
        raise SystemExit(
            f"refusing a Gate run on {path}:\n  - " + "\n  - ".join(problems)
            + "\nRefit the spectrum on the current committed tree, or run this as an "
              "exploratory scope (--scope powercurve), which is exempt and is recorded "
              "as such in the run summary.")
    return facts


def build_arm_frame(cfg, arm: str, k: int, seed: int, tag: str, d: int,
                    scope: str = GATE_SCOPE, prov: dict | None = None):
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
        base, extra = build_arm_frame(cfg, base_arm, k, base_seed, tag, d, scope, prov)
        mode = "random" if spec["type"] == "rotated_random" else "learned"
        return RotatedFrame(base, k=k, mode=mode, seed=seed), {
            **extra, "rotation_mode": mode, "base_arm": base_arm, "base_seed": base_seed}
    if spec["type"] == "spectral_topk":
        path = Path(cfg["paths"]["data_root"]) / "spectrum" / f"{tag}_seed{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run scripts/fit_observability_spectrum.py --tag {tag} "
                f"--seed {seed} first (arm D is the spectrum, not a baseline)")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        facts = _bind_spectrum(path, blob if isinstance(blob, dict) else {}, tag, seed,
                               scope, prov or {})
        return SpectralFrame(d=d, k=k, path=path), facts
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
    # val ONLY. Test is computed for the first time by scripts/evaluate_locked.py,
    # after selection is frozen (B1).
    ap.add_argument("--eval-splits", nargs="*", default=["val"])
    ap.add_argument("--allow-dirty", action="store_true",
                    help="throwaway run from an uncommitted tree; not for the record")
    # Overrides for smoke-testing the pipeline before the full cache exists.
    # Leave them alone for real runs: the cross-fit protocol is the experiment.
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--crossfit", default=None, help="'A', 'B' or 'none'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-tag", default=None,
                    help="image cache namespace; defaults to --tag. A triage writes its "
                         "artifacts under a fresh --tag while reusing an existing cache.")
    ap.add_argument("--eval-attacks", nargs="*", default=None,
                    help="restrict the evaluated attacks (triage uses one per family); "
                         "recorded in the run so the locked test evaluation matches")
    ap.add_argument("--scope", default=None,
                    choices=["gate", "receiver_control", "p0_7_basis", "powercurve"],
                    help="which analysis this run belongs to; the Gate selector reads "
                         "ONLY gate-scope runs")
    ap.add_argument("--receiver-seed", type=int, default=None,
                    help="P0-7.1: separate the receiver's randomness from the BASIS "
                         "seed, so a D2 spread can be attributed to the basis")
    ap.add_argument("--subset-size", type=int, default=0,
                    help="EXPLORATORY: train on a nested random subset of this many "
                         "samples from the training split. Evaluation splits are never "
                         "subset. Requires a non-gate --scope.")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="which nested subset; N=100 is a subset of N=200 at the same "
                         "seed, so the curve varies size and nothing else")
    ap.add_argument("--extractor-arch", default=None,
                    help="receiver architecture; `spatial` is the no-GAP control (P0-5)")
    args = ap.parse_args()

    bad = [s for s in args.eval_splits if s.startswith(FORBIDDEN_PRE_LOCK)]
    if bad:
        raise SystemExit(
            f"--eval-splits {bad} is a protocol violation (B1): training runs are "
            "pre-lock artifacts and may not produce test predictions, corruptions or "
            "metrics. scripts/evaluate_locked.py computes the test splits after the "
            "val selection is frozen.")

    cfg = load_config(args.config)
    prov = require_clean("a training run", allow_dirty=args.allow_dirty)
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    if args.eval_attacks:
        unknown = [a for a in args.eval_attacks if a not in bank.attacks]
        if unknown:
            raise SystemExit(f"unknown attacks {unknown}")
        bank.eval = list(args.eval_attacks)
    cache_tag = args.cache_tag or args.tag
    root = Path(cfg["paths"]["cache_dir"]) / cache_tag
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

    if args.subset_size and scope == "gate":
        raise SystemExit(
            "--subset-size is an exploratory diagnostic and must not produce a run the "
            "selector can see. Pass an explicit --scope (e.g. --scope powercurve); "
            "select_method.py only reads analysis_scope == 'gate'.")
    frame, extra = build_arm_frame(cfg, args.arm, k, args.seed, args.tag, d, scope, prov)
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
            limit=args.limit,
        subset_size=args.subset_size, subset_seed=args.subset_seed)   # discovery randomness follows the structure seed
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
        limit=args.limit,
        subset_size=args.subset_size, subset_seed=args.subset_seed)

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

    # What was actually trained on. Asked-for and delivered differ whenever
    # subset_size exceeds the split, and a learning curve plotted against the request
    # would show a phantom plateau at the top end.
    n_train_samples = len(make_loader(
        root, args.train_split, bank, crossfit=xfit_eval, workers=0,
        subset_size=args.subset_size, subset_seed=args.subset_seed).dataset)

    out_dir = Path(cfg["paths"]["data_root"]) / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_stem(args.arm, k, args.seed, tcfg.extractor_arch, receiver_seed, scope,
                    args.subset_size, args.subset_seed)
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    rows = frame.rows().detach().cpu().contiguous()
    rows_digest = hashlib.blake2s(rows.numpy().tobytes(), digest_size=16).hexdigest()
    torch.save({"rows_digest": rows_digest, "rows": rows,
                "state_dict": frame.state_dict()}, out_dir / f"{stem}_frame.pt")
    # The receiver is frozen HERE, before the lock. evaluate_locked.py loads this exact
    # checkpoint rather than retraining, so the test evaluation cannot benefit from a
    # receiver that was trained after anyone had seen a selection outcome.
    torch.save({"state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()},
                "arch": tcfg.extractor_arch, "k": k,
                "receiver_seed": receiver_seed, "epochs": tcfg.epochs},
               out_dir / f"{stem}_extractor.pt")
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
        # 0 = the whole split. Non-zero marks an exploratory sample-size point.
        "subset_size": args.subset_size, "subset_seed": args.subset_seed,
        "n_train_samples": n_train_samples,
        "orthonormality_error": ortho, "rows_digest": rows_digest,
        "extractor_arch": tcfg.extractor_arch,
        **prov,
        "train_config": {kk: vv for kk, vv in tcfg.__dict__.items()},
        "eval_splits": list(args.eval_splits), "eval_attacks": list(bank.eval),
        "cache_tag": cache_tag,
        "crossfit_eval_split": xfit_eval, "train_split": args.train_split,
        "limit": args.limit,
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
