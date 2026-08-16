#!/usr/bin/env python
"""Cost of the exact range-restricted certified solver at production dimension.

`range_eigh` is exact and avoids ARPACK's stall on the degenerate null space, but
it is O(d p^2) with p = 2 N_operator, so its cost grows quadratically in the number
of discovery samples. The full run has ~5x the pilot's A_operator split, and this
benchmark decides whether that needs a top-k solver instead.

Each N runs in its own process so peak RSS is not contaminated by the previous one.

    python scripts/benchmark_solver.py                  # all sizes, writes a report
    python scripts/benchmark_solver.py --n 500          # one size, prints JSON
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

D = 16384


def bench(n: int, k: int = 64, oversampling: int = 32, try_eigsh: bool = True) -> dict:
    from scipy.sparse.linalg import eigsh

    from fiber.spectrum.certified import CertifiedObservabilityOperator

    rng = np.random.default_rng(0)
    n_obs = 96
    A = rng.standard_normal((n_obs, D)) / D**0.5 * 3
    Z = rng.standard_normal((n, D))
    Y = Z @ A.T + 0.6 * rng.standard_normal((n, n_obs))
    Sy = A @ A.T + 0.36 * np.eye(n_obs)
    F = np.linalg.solve(Sy, Y.T).T @ A
    op = CertifiedObservabilityOperator(Z, F)

    t0 = time.time()
    G = np.concatenate([op.Zc.T, op.Fc.T], axis=1)
    t_concat = time.time() - t0
    t0 = time.time()
    Q, R = np.linalg.qr(G)
    t_qr = time.time() - t0
    t0 = time.time()
    R1, R2 = R[:, :n], R[:, n:]
    B = (R1 @ R2.T + R2 @ R1.T - R2 @ R2.T) / n
    B = (B + B.T) / 2
    t_form = time.time() - t0
    t0 = time.time()
    w, P = np.linalg.eigh(B)
    t_eigh = time.time() - t0
    t0 = time.time()
    _ = (Q @ P[:, -(k + oversampling):]).T
    t_back = time.time() - t0

    out = {
        "n_operator": n, "d": D, "p": int(B.shape[0]),
        "seconds": {"concat": round(t_concat, 2), "qr": round(t_qr, 2),
                    "form_B": round(t_form, 2), "eigh": round(t_eigh, 2),
                    "backproject": round(t_back, 2)},
        "seconds_total": round(t_concat + t_qr + t_form + t_eigh + t_back, 2),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2),
        "lambda_max": float(w[-1]),
    }

    if try_eigsh:
        t0 = time.time()
        try:
            v0 = rng.standard_normal(D)
            vals, _ = eigsh(op.as_linear_operator(), k=k + oversampling, which="LA", v0=v0)
            out["eigsh"] = {"seconds": round(time.time() - t0, 2),
                            "lambda_max": float(vals.max()),
                            "agrees_with_range_eigh": bool(abs(vals.max() - w[-1]) < 1e-6)}
        except Exception as exc:
            out["eigsh"] = {"seconds": round(time.time() - t0, 2),
                            "error": f"{type(exc).__name__}: {str(exc)[:90]}"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="child mode: one size")
    ap.add_argument("--sizes", nargs="*", type=int, default=[500, 1000, 2500])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", default="reports/solver_benchmark.json")
    args = ap.parse_args()

    if args.n is not None:
        print(json.dumps(bench(args.n, k=args.k)))
        return 0

    rows = []
    for n in args.sizes:
        proc = subprocess.run([sys.executable, __file__, "--n", str(n), "--k", str(args.k)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"benchmark failed at n={n}")
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        e = row.get("eigsh", {})
        print(f"  N={n:5d}  p={row['p']:5d}  total {row['seconds_total']:7.2f}s "
              f"(qr {row['seconds']['qr']:.2f}, eigh {row['seconds']['eigh']:.2f})  "
              f"peak {row['peak_rss_gb']:.2f} GB  |  eigsh "
              f"{e.get('seconds', float('nan')):.2f}s {e.get('error', 'ok')}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"d": D, "k": args.k, "rows": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
