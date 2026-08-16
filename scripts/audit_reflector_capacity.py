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


# Pre-registered decision rule, fixed BEFORE the sweep finished so that m(k) is chosen
# by a rule rather than by eye after seeing the table. The requirement is the MINIMUM
# over seeds, not the mean: an m that works on one draw and not another is not a
# capacity a locked experiment can rely on.
CAPACITY_SUFFICIENT = 0.99
CAPACITY_MARGINAL = 0.95


def classify(a_sub: float) -> str:
    if a_sub >= CAPACITY_SUFFICIENT:
        return "sufficient"
    if a_sub >= CAPACITY_MARGINAL:
        return "marginal"
    return "insufficient"


def recommend_m(rows: list[dict]) -> dict:
    """Smallest m whose GENERIC alignment clears the threshold at every seed.

    Generic, not reachable: reachable only says the optimiser can hit targets the
    family provably contains, which is the diagnostic for attributing a failure. What
    arm E needs is coverage of subspaces it was not built from.
    """
    out = {}
    for row in sorted(rows, key=lambda r: (r["k"], r["m"])):
        if row["target"] != "generic":
            continue
        k = row["k"]
        worst = row.get("alignment_min", row.get("alignment_final"))
        if worst is None:
            raise KeyError(f"row for k={row['k']} m={row['m']} has no alignment")
        entry = out.setdefault(k, {"k": k, "recommended_m": None, "by_m": {}})
        entry["by_m"][row["m"]] = {"alignment_min_over_seeds": worst,
                                   "class": classify(worst)}
        if entry["recommended_m"] is None and classify(worst) == "sufficient":
            entry["recommended_m"] = row["m"]
    return out


def alignment(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """A_sub = ||R T'||_F^2 / k, in [0, 1]. Only k x k is formed."""
    return (R @ T.T).pow(2).sum() / R.shape[0]


def projector_error(a_sub: float) -> float:
    return float((2.0 * max(0.0, 1.0 - a_sub)) ** 0.5)


def necessary_m(k: int, d: int) -> float:
    return k * (d - k) / (d - 1)


def make_target(kind: str, fitter: HouseholderFrame, d: int, k: int, m: int,
                seed: int, device) -> torch.Tensor:
    """`fitter` is passed in on purpose: the reachable target must be built on the
    fitter's OWN frozen Haar base.

    `HouseholderFrame(seed=s)` uses s for the reflectors AND for the base, so a target
    at seed 2000+s sits on a DIFFERENT base B'. Then T = B' Q_* need not lie in the
    fitter's family {B Q : Q a product of m reflections} at all, and a failure could not
    be attributed to optimisation -- which is the whole point of having this control.
    """
    if kind == "generic":
        return HaarRandomFrame(d, k, seed=1000 + seed).rows().to(device)
    if kind == "reachable":
        f = HouseholderFrame(d, k, num_reflectors=m, seed=2000 + seed,
                             base="haar", paired_init=False).to(device)
        with torch.no_grad():
            f.B.copy_(fitter.B)            # same base => provably inside the family
        return f.rows().detach()
    raise ValueError(kind)


def fit(d: int, k: int, m: int, kind: str, seed: int, steps: int, lr: float,
        device) -> dict:
    torch.manual_seed(seed)
    frame = HouseholderFrame(d, k, num_reflectors=m, seed=seed,
                             base="haar", paired_init=True).to(device)
    target = make_target(kind, frame, d, k, m, seed, device)
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
    ap.add_argument("--classify-only", action="store_true",
                    help="apply the pre-registered rule to an existing result file")
    args = ap.parse_args()

    if args.classify_only:
        blob = json.loads(Path(args.out).read_text())
        rows = blob["rows"]
        for row in rows:
            # two seeds: mean and spread determine the minimum exactly
            if "alignment_min" not in row:
                row["alignment_min"] = row["alignment_final"] - row.get("alignment_spread", 0) / 2
            row["capacity_class"] = classify(row["alignment_min"])
        blob["recommended_m_by_k"] = recommend_m(rows)
        blob["decision_rule"] = {"sufficient": CAPACITY_SUFFICIENT,
                                 "marginal": CAPACITY_MARGINAL,
                                 "statistic": "minimum generic alignment over seeds"}
        Path(args.out).write_text(json.dumps(blob, indent=2))
        for k, e in sorted(blob["recommended_m_by_k"].items(), key=lambda x: int(x[0])):
            log.info("k=%-4s recommended m=%s  %s", k, e["recommended_m"],
                     {m: v["class"] for m, v in sorted(e["by_m"].items())})
        return 0

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
                worst = min(r["alignment_final"] for r in res)
                row = {**res[0], "seed": args.seeds, "alignment_final": a,
                       "projector_error": e,
                       "alignment_min": worst, "capacity_class": classify(worst),
                       "alignment_spread": max(r["alignment_final"] for r in res) - worst}
                rows.append(row)
                log.info("%-9s k=%-4d m=%-4d  A_sub %.4f  E_sub %.4f  "
                         "(need m>=%.0f: %s)", kind, k, m, a, e,
                         row["necessary_m"], "yes" if row["m_meets_necessary_condition"] else "NO")

    out = {"d": args.d, "steps": args.steps, "lr": args.lr, "rows": rows,
           "decision_rule": {"sufficient": CAPACITY_SUFFICIENT,
                             "marginal": CAPACITY_MARGINAL,
                             "statistic": "minimum generic alignment over seeds",
                             "registered": "before the sweep completed"},
           "recommended_m_by_k": recommend_m(rows),
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
