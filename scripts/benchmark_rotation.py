#!/usr/bin/env python
"""Cost of the SO(k) rotation parameterisation (P0-7).

D3 evaluates `matrix_exp(S - S^T)` and backpropagates through it on every step. At the
main k = 64 that is irrelevant; the sweep goes to k = 256, where it might not be. If it
is, the alternatives are Cayley, Givens or a Householder product in k-space -- but the
matrix exponential stays the default while it is affordable, because it is the cleanest
statement of "an element of SO(k)".

    python scripts/benchmark_rotation.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def bench(k: int, batch: int = 16, d: int = 16384, steps: int = 20, device: str = "cpu") -> dict:
    dev = torch.device(device)
    S = torch.zeros(k, k, device=dev, requires_grad=True)
    V = torch.randn(k, d, device=dev)
    z = torch.randn(batch, d, device=dev)
    target = torch.randn(batch, k, device=dev)

    def step():
        A = torch.matrix_exp(S - S.T)
        loss = ((z @ V.T) @ A.T - target).pow(2).mean()
        loss.backward()
        S.grad = None

    step()                                  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / steps
    return {"k": k, "device": device, "seconds_per_step": round(per_step, 5),
            "ms_per_step": round(per_step * 1000, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/rotation_benchmark.json")
    args = ap.parse_args()

    rows = [bench(k, device=args.device) for k in args.ks]
    base = rows[0]["seconds_per_step"]
    for r in rows:
        r["relative_to_k%d" % rows[0]["k"]] = round(r["seconds_per_step"] / base, 2)
        print(f"  k={r['k']:4d}  {r['ms_per_step']:8.2f} ms/step  "
              f"({r['relative_to_k%d' % rows[0]['k']]:.2f}x)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
