#!/usr/bin/env python
"""B3 — does arm E's reflector count suffice at each k?

Arm E fixes m = 128 reflectors while k sweeps to 256. If the reachable family at some k
is strictly smaller than the set of k-frames, then `BER_learned > BER_spectral` there
would reflect PARAMETERISATION CAPACITY rather than channel geometry, and the
comparison would not mean what Gate 3B says it means.

A necessary condition comes from dimension counting. The Grassmannian Gr(k, d) has
dimension k(d-k); m reflector directions carry at most m(d-1) degrees of freedom, so
generically covering it needs

    m >= k(d-k)/(d-1)

which at d = 16384 is about 16 / 64 / 127 / 252 for k = 16 / 64 / 128 / 256. Necessary,
not sufficient -- hence the empirical audit.

Two target families, so a failure can be attributed:

  generic  T ~ Haar on Gr(k, d)        the real coverage stress test
  reachable T = a frame built by the SAME parameterisation, random parameters
             -> reachable fails too  => optimiser or initialisation
             -> reachable succeeds, generic fails => capacity / coverage

Metrics. For orthonormal k-frames R, T,

    ||R'R - T'T||_F^2 = 2k - 2||R T'||_F^2      and     ||T'T||_F = sqrt(k)

so with A_sub = ||R T'||_F^2 / k = (1/k) sum_i cos^2(theta_i),

    E_sub = ||R'R - T'T||_F / ||T'T||_F = sqrt(2(1 - A_sub))

The projector error and the mean squared principal cosine are therefore the same
quantity, and only the k x k matrix R T' is ever formed -- never the d x d projector.

    python scripts/audit_reflector_capacity.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fiber.transforms import HaarRandomFrame, HouseholderFrame
from fiber.utils.logging import get_logger

log = get_logger("capacity")


def alignment(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """A_sub = ||R T'||_F^2 / k, in [0, 1]. Only k x k is formed."""
    return (R @ T.T).pow(2).sum() / R.shape[0]


def projector_error(a_sub: float) -> float:
    return float((2.0 * max(0.0, 1.0 - a_sub)) ** 0.5)


def necessary_m(k: int, d: int) -> float:
    return k * (d - k) / (d - 1)


def make_target(kind: str, d: int, k: int, m: int, seed: int, device) -> torch.Tensor:
    if kind == "generic":
        return HaarRandomFrame(d, k, seed=1000 + seed).rows().to(device)
    if kind == "reachable":
        # same parameterisation, random (unpaired) parameters: a frame this family
        # provably contains, so failing it is an optimisation result, not a capacity one
        f = HouseholderFrame(d, k, num_reflectors=m, seed=2000 + seed,
                             base="haar", paired_init=False).to(device)
        return f.rows().detach()
    raise ValueError(kind)


def fit(d: int, k: int, m: int, kind: str, seed: int, steps: int, lr: float,
        device) -> dict:
    torch.manual_seed(seed)
    frame = HouseholderFrame(d, k, num_reflectors=m, seed=seed,
                             base="haar", paired_init=True).to(device)
    target = make_target(kind, d, k, m, seed, device)
    opt = torch.optim.Adam(frame.parameters(), lr=lr)
    start = float(alignment(frame.rows().detach(), target))
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = 1.0 - alignment(frame.rows(), target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = float(alignment(frame.rows(), target))
        ortho = frame.orthonormality_error()
    return {"k": k, "m": m, "target": kind, "seed": seed,
            "alignment_start": start, "alignment_final": a,
            "projector_error": projector_error(a),
            "orthonormality_error": ortho,
            "necessary_m": round(necessary_m(k, d), 1),
            "m_meets_necessary_condition": m >= necessary_m(k, d),
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16384)
    ap.add_argument("--ks", nargs="*", type=int, default=[16, 64, 128, 256])
    ap.add_argument("--ms", nargs="*", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--targets", nargs="*", default=["generic", "reachable"])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="reports/reflector_capacity.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = []
    for kind in args.targets:
        for k in args.ks:
            for m in args.ms:
                if m < 2:
                    continue
                res = [fit(args.d, k, m, kind, s, args.steps, args.lr, device)
                       for s in args.seeds]
                a = sum(r["alignment_final"] for r in res) / len(res)
                e = sum(r["projector_error"] for r in res) / len(res)
                row = {**res[0], "seed": args.seeds, "alignment_final": a,
                       "projector_error": e,
                       "alignment_spread": max(r["alignment_final"] for r in res)
                       - min(r["alignment_final"] for r in res)}
                rows.append(row)
                log.info("%-9s k=%-4d m=%-4d  A_sub %.4f  E_sub %.4f  "
                         "(need m>=%.0f: %s)", kind, k, m, a, e,
                         row["necessary_m"], "yes" if row["m_meets_necessary_condition"] else "NO")

    out = {"d": args.d, "steps": args.steps, "lr": args.lr, "rows": rows,
           "metric_note": ("E_sub = ||R'R - T'T||_F / ||T'T||_F = sqrt(2(1 - A_sub)); the "
                           "projector error and the mean squared principal cosine are the "
                           "same quantity, and only R T' (k x k) is ever formed"),
           "necessary_condition": "m >= k(d-k)/(d-1), necessary not sufficient"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
