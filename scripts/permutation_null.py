#!/usr/bin/env python
"""Empirical null for the linear explained variance, from frozen checkpoints.

    R2_lin(V) = (1/k) sum_j rho_j^2

is reported against 1/(n-1). That is the standard zero-correlation reference for a
Pearson r under the usual null model, but it is an ANALYTIC reference: it assumes a
distribution the receiver's outputs need not follow, and it says nothing about the
dependence between coordinates and between attacks. A negative result that rests on it
is attackable exactly where it should not be.

This recomputes W and W_hat from the saved frame and extractor -- no retraining, no new
experiment -- and builds the null by permuting the pairing between them. Under the
permutation the marginals of both are exactly the empirical ones, so the reference stops
depending on a distributional assumption.

    python scripts/permutation_null.py --tag triage1 --scope powercurve --perms 2000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.models import build_extractor
from fiber.training import make_loader
from fiber.transforms.spectral import SpectralFrame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger

log = get_logger("permnull")


def _r2(w_hat: np.ndarray, w: np.ndarray) -> float:
    """(1/k) sum_j rho_j^2, correlations taken over the sample axis."""
    a = w_hat - w_hat.mean(0, keepdims=True)
    b = w - w.mean(0, keepdims=True)
    denom = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    rho = (a * b).sum(0) / np.clip(denom, 1e-12, None)
    return float((rho ** 2).mean())


def permutation_null(per_attack: dict, perms: int, rng, shared: bool = True):
    """Null distribution of the pooled R2 under re-pairing of W_hat and W.

    ONE permutation per replicate, applied to every attack. The pairing is a property of
    the SAMPLE: sample i has one latent, seen through many attacks, and one prediction
    per attack that are strongly correlated with each other. Drawing an independent
    permutation per attack averages `A` independent estimates where the observed
    statistic averages `A` dependent ones -- the same mean, a far smaller variance, and
    therefore p-values that are much too small.

    Measured on this project's 36 runs: independent-per-attack reported 13 runs at
    p < 0.05 (including in the random-reference arm) and a null sd of 0.00022 at k=64;
    the shared permutation gives 3 runs at p < 0.05 against 1.8 expected by chance, and
    a null sd of 0.00065. `shared=False` exists only so a test can demonstrate this.
    """
    n = next(iter(per_attack.values()))[1].shape[0]
    out = np.empty(perms)
    for b in range(perms):
        pi = rng.permutation(n)
        out[b] = float(np.mean([_r2(wh, w[pi if shared else rng.permutation(n)])
                                for wh, w in per_attack.values()]))
    return out


def collect(run_json: Path, cfg, bank, split: str, device: str):
    """W and W_hat per attack, from the frozen artifacts of one run."""
    meta = json.loads(run_json.read_text())
    stem = run_json.with_suffix("").name
    d = run_json.parent
    fb = torch.load(d / f"{stem}_frame.pt", map_location="cpu", weights_only=True)
    ck = torch.load(d / f"{stem}_extractor.pt", map_location="cpu", weights_only=True)
    rows = fb["rows"].float()
    # carrying the frozen rows is exactly equivalent to rebuilding the arm class, and
    # cannot silently rebuild it differently (same construction as evaluate_locked.py)
    frame = SpectralFrame(d=rows.shape[1], k=rows.shape[0], rows=rows,
                          reorthonormalise=False)
    model = build_extractor(ck["arch"], k=int(ck["k"])).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    frame = frame.to(device)

    root = Path(cfg["paths"]["cache_dir"]) / meta.get("cache_tag", meta["tag"])
    out, order = {}, None
    for attack in bank.eval:
        loader = make_loader(root, split, bank, attacks=[attack], mode="fixed",
                             batch_size=32, workers=4, shuffle=False)
        W, WH, ids = [], [], []
        with torch.no_grad():
            for batch in loader:
                ids.extend(batch["sample_id"])
                z = batch["z"].to(device)
                pred = model(batch["image"].to(device))
                if "w_hat" not in pred:
                    return meta, None
                W.append(frame.project(z).float().cpu().numpy())
                WH.append(pred["w_hat"].float().cpu().numpy())
        # One permutation is shared across attacks below, so row i must be the same
        # sample in every attack. shuffle=False gives that; this refuses to rely on it.
        if order is None:
            order = ids
        elif ids != order:
            raise SystemExit(f"{attack}: sample order differs from {bank.eval[0]!r}; a "
                             "shared permutation would compare different samples")
        out[attack] = (np.concatenate(WH), np.concatenate(W))
    return meta, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="triage1")
    ap.add_argument("--scope", default="powercurve")
    ap.add_argument("--split", default="val")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bank = ChannelBank(cfg)
    rng = np.random.default_rng(0)
    runs = sorted(glob.glob(str(Path(cfg["paths"]["data_root"]) / "results" / args.tag
                                / f"*sc{args.scope}*.json")))
    rows = []
    for rj in runs:
        meta, per_attack = collect(Path(rj), cfg, bank, args.split, args.device)
        if per_attack is None:
            log.warning("%s has no regression head; skipped", Path(rj).name)
            continue
        # Pool the coordinates of every attack, exactly as the reported R2_lin does.
        obs = float(np.mean([_r2(wh, w) for wh, w in per_attack.values()]))
        null = permutation_null(per_attack, args.perms, rng)
        n = next(iter(per_attack.values()))[1].shape[0]
        rows.append({"stem": Path(rj).stem, "arm": meta["arm"], "k": meta["k"],
                     "receiver_seed": meta.get("receiver_seed"),
                     "subset_size": meta.get("subset_size", 0),
                     "n": int(n), "analytic_null": 1.0 / (n - 1),
                     "R2_observed": obs,
                     "perm_null_mean": float(null.mean()),
                     "perm_null_sd": float(null.std(ddof=1)),
                     "perm_null_q95": float(np.quantile(null, 0.95)),
                     "p_one_sided": float((null >= obs).mean()),
                     "perms": args.perms})
        log.info("%-46s obs %.5f  perm-null %.5f +- %.5f  p=%.3f  analytic %.5f",
                 rows[-1]["stem"][:46], obs, null.mean(), null.std(ddof=1),
                 rows[-1]["p_one_sided"], rows[-1]["analytic_null"])
    out = Path(args.out or (Path(cfg["paths"]["reports_dir"])
                            / f"permutation_null_{args.tag}_{args.scope}.json"))
    out.write_text(json.dumps({"split": args.split, "perms": args.perms, "runs": rows},
                              indent=2))
    log.info("wrote %s (%d runs)", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
