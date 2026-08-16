#!/usr/bin/env python
"""Frame geometry at the ACTUAL configured dimension, not extrapolated from a toy.

Cheap (CPU, seconds) and it settles one question a reviewer will ask: is the
random-Householder control anything like a uniform random subspace at d = 16384,
m = 128? It is not -- it is very nearly the identity.

    python scripts/diagnose_frames.py --seeds 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fiber.transforms import (HaarRandomFrame, HadamardFrame, RandomHouseholderFrame,
                              SignedPermutationFrame)
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger

log = get_logger("frames")


def row_stats(rows: torch.Tensor) -> dict:
    """rows: [n_seeds, d], one leading row per draw."""
    sq = rows.double() ** 2
    return {
        "mean_leading_coord_energy": float(sq[:, 0].mean()),
        "mean_max_coord_energy": float(sq.max(1).values.mean()),
        # 1 / sum(r_i^4): 1 means one coordinate carries everything, d/3 is uniform
        "participation_ratio": float((1.0 / (sq**2).sum(1)).mean()),
        "n_draws": int(rows.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default="reports/frame_geometry.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = int(cfg["latent"]["dim"])
    m = int(cfg["fiber"]["arms"]["C3_rand_hh"]["num_reflectors"])

    builders = {
        "haar": lambda s: HaarRandomFrame(d, 1, seed=s),
        f"random_householder_m{m}": lambda s: RandomHouseholderFrame(d, 1, num_reflectors=m, seed=s),
        "hadamard": lambda s: HadamardFrame(d, 1, seed=s),
        "signed_permutation": lambda s: SignedPermutationFrame(d, 1, seed=s),
    }
    out = {"d": d, "reflectors": m, "uniform_reference": {
        "mean_leading_coord_energy": 1.0 / d,
        "mean_max_coord_energy": 2 * float(np.log(d)) / d,
        "participation_ratio": d / 3.0}}
    for name, build in builders.items():
        rows = torch.stack([build(s).rows()[0] for s in range(args.seeds)])
        out[name] = row_stats(rows)
        log.info("%-26s E[r0^2] %.3e  E[max r^2] %.3e  participation %.1f",
                 name, out[name]["mean_leading_coord_energy"],
                 out[name]["mean_max_coord_energy"], out[name]["participation_ratio"])

    rhh = out[f"random_householder_m{m}"]
    out["conclusion"] = {
        "random_householder_is_near_identity": rhh["participation_ratio"] < 2.0,
        "note": (f"With m={m} reflections in d={d}, each reflection moves e_0 by about "
                 f"2/sqrt(d) = {2/d**0.5:.4f}, so a random walk of m steps displaces it by "
                 f"~sqrt(m)*2/sqrt(d) = {(m**0.5)*2/d**0.5:.3f}. The product is therefore "
                 "essentially the identity at initialisation; reaching a Haar-like draw "
                 f"would need m ~ d/4 = {d//4} reflections. This is a statement about the "
                 "INITIALISATION, not about representational capacity: m >= k Householder "
                 "reflections can represent any k-frame exactly."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
