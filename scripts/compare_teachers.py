#!/usr/bin/env python
"""P0-5: is the observability geometry a property of the channel or of the decoder?

The default teacher is ResNet18 -> global average pool -> Linear(512, d), so its output
covariance has rank at most 512 whatever the channel carries, and GAP is biased toward
global content. A spectrum measured only through it is the geometry OF THAT DECODER.

This runs the certified fit with both teacher architectures and reports, for the
directions each one discovers:

    D_cert (subspace, cross-fitted)          how much each decoder can certify
    principal-angle alignment between them   do they find the SAME subspace?

Interpretation, fixed in advance:
  * high alignment  -> evidence the geometry is a property of the channel
  * low alignment   -> the geometry is architecture-dependent, and the report says so
                       rather than quoting whichever number is larger

Until they agree, the wording stays "decoder-certified observability geometry", never
"intrinsic".

    python scripts/compare_teachers.py --tag pilot --k 64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from fiber.spectrum import subspace_alignment
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger

log = get_logger("teachers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-fit", action="store_true", help="reuse existing spectrum files")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec_dir = Path(cfg["paths"]["data_root"]) / "spectrum"
    archs = [cfg["spectrum"].get("teacher_arch", "resnet18"),
             cfg["spectrum"].get("teacher_arch_control", "spatial")]

    paths = {}
    for arch in archs:
        stem = f"{args.tag}_seed{args.seed}" + ("" if arch == "resnet18" else f"_{arch}")
        paths[arch] = spec_dir / f"{stem}.pt"
        if args.skip_fit and paths[arch].exists():
            log.info("%s: reusing %s", arch, paths[arch].name)
            continue
        cmd = [sys.executable, "scripts/fit_observability_spectrum.py",
               "--config", args.config, "--tag", args.tag, "--seed", str(args.seed),
               "--k", str(args.k), "--teacher", arch, "--device", args.device]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        log.info("fitting the certified operator with the %s teacher", arch)
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit(f"spectrum fit failed for teacher {arch}")

    blobs, summaries = {}, {}
    for arch, path in paths.items():
        blobs[arch] = torch.load(path, map_location="cpu", weights_only=True)
        rep = Path(cfg["paths"]["reports_dir"]) / f"spectrum_{path.stem}.json"
        summaries[arch] = json.loads(rep.read_text()) if rep.exists() else {}

    a, b = archs
    k = min(args.k, blobs[a]["eigenvectors"].shape[0], blobs[b]["eigenvectors"].shape[0])
    align = subspace_alignment(blobs[a]["eigenvectors"][:k], blobs[b]["eigenvectors"][:k])
    d = int(cfg["latent"]["dim"])
    chance = k / d          # mean squared cosine between unrelated k-subspaces

    out = {
        "tag": args.tag, "seed": args.seed, "k": k,
        "teachers": {arch: {
            "D_cert_subspace": summaries[arch].get("D_cert_subspace"),
            "numerical_positive_rank": summaries[arch].get("numerical_positive_rank",
                                                           summaries[arch].get("certified_positive_rank")),
            "validity_mean_abs_gap": summaries[arch].get("validity_mean_abs_gap"),
            "validity_pass": summaries[arch].get("validity_pass"),
        } for arch in archs},
        "subspace_alignment": align,
        "chance_alignment": chance,
        "alignment_over_chance": align / chance if chance else float("nan"),
        "verdict": ("consistent across decoder architectures" if align > 0.5 else
                    "ARCHITECTURE-DEPENDENT: the two decoders do not agree on the "
                    "subspace, so the geometry cannot be attributed to the channel"),
        "wording": "decoder-certified observability geometry (not intrinsic)",
    }
    rep = Path(cfg["paths"]["reports_dir"]) / f"teacher_comparison_{args.tag}_seed{args.seed}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=2, default=float))

    for arch in archs:
        t = out["teachers"][arch]
        log.info("%-9s D_cert %-8s rank %-5s validity %s", arch,
                 f"{t['D_cert_subspace']:.3f}" if t["D_cert_subspace"] is not None else "n/a",
                 t["numerical_positive_rank"], t["validity_pass"])
    log.info("top-%d subspace alignment %.4f (chance %.4f, %.1fx) -> %s",
             k, align, chance, out["alignment_over_chance"], out["verdict"])
    log.info("wrote %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
