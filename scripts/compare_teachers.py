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

import torch as _torch

from fiber.spectrum import principal_cosines, subspace_alignment
from fiber.spectrum.certified import subspace_certificate
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean

log = get_logger("teachers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-fit", action="store_true",
                    help="reuse existing spectrum files, after verifying their provenance")
    ap.add_argument("--cache-tag", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prov = require_clean("a teacher comparison", allow_dirty=args.allow_dirty)
    spec_dir = Path(cfg["paths"]["data_root"]) / "spectrum"
    archs = [cfg["spectrum"].get("teacher_arch", "resnet18"),
             cfg["spectrum"].get("teacher_arch_control", "spatial_sharedtrunk")]

    paths = {}
    for arch in archs:
        stem = f"{args.tag}_seed{args.seed}" + ("" if arch == "resnet18" else f"_{arch}")
        paths[arch] = spec_dir / f"{stem}.pt"
        if args.skip_fit and paths[arch].exists():
            # Reuse only what this experiment actually produced. "The file exists" is not
            # provenance: a spectrum from another commit, k, seed or teacher would be
            # silently folded into this comparison.
            blob = torch.load(paths[arch], map_location="cpu", weights_only=True)
            # same code, same experiment, same image cache
            want = {"commit": prov["git_commit_short"], "tag": args.tag,
                    "cache_tag": args.cache_tag or args.tag, "k": args.k,
                    "seed": args.seed, "teacher_arch": arch}
            got = {kk: blob.get(kk) for kk in want}
            if got != want:
                raise SystemExit(
                    f"{paths[arch].name} does not match this run: {got} != {want}. "
                    "Drop --skip-fit to refit, or delete the stale artifact.")
            log.info("%s: reusing %s (verified)", arch, paths[arch].name)
            continue
        cmd = [sys.executable, "scripts/fit_observability_spectrum.py",
               "--config", args.config, "--tag", args.tag, "--seed", str(args.seed),
               "--k", str(args.k), "--teacher", arch, "--device", args.device]
        if args.cache_tag:
            cmd += ["--cache-tag", args.cache_tag]
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
    Va, Vb = blobs[a]["eigenvectors"][:k], blobs[b]["eigenvectors"][:k]
    align = subspace_alignment(Va, Vb)
    cos2 = principal_cosines(Va, Vb)
    d = int(cfg["latent"]["dim"])
    chance = k / d          # mean squared cosine between unrelated k-subspaces

    validity = {arch: bool(summaries[arch].get("validity_pass", False)) for arch in archs}
    both_valid = all(validity.values())
    tau = float(cfg["spectrum"].get("alignment_tol", 0.5))

    # A teacher that failed its own validity check has an eigenspace that may be
    # nothing but noise, so a low alignment cannot distinguish "the geometry is
    # architecture-dependent" from "one decoder was not fitted". Validity gates the
    # comparison; alignment is only read once both decoders are admissible.
    if not both_valid:
        verdict = ("INCONCLUSIVE: " + ", ".join(f"{arch} validity {'PASS' if v else 'FAIL'}"
                                                for arch, v in validity.items())
                   + " -- a decoder that fails its validity check has an eigenspace that "
                     "may be noise, so alignment says nothing about the channel")
    elif align >= tau:
        verdict = "ARCHITECTURE-STABLE: both decoders admissible and they agree"
    else:
        verdict = ("ARCHITECTURE-DEPENDENT: both decoders admissible and they do NOT "
                   "agree, so the geometry cannot be attributed to the channel")

    out = {
        "tag": args.tag, "seed": args.seed, "k": k,
        "teachers": {arch: {
            "D_cert_subspace": summaries[arch].get("D_cert_subspace"),
            "numerical_positive_rank": summaries[arch].get("numerical_positive_rank"),
            "validity_mean_abs_gap": summaries[arch].get("validity_mean_abs_gap"),
            "validity_pass": validity[arch],
            # capacity is a confound if left unreported; the two heads cannot be
            # parameter-matched by construction (dense GAP+FC vs weight-shared conv)
            "parameters": summaries[arch].get("teacher_parameters"),
            "trunk_parameters": summaries[arch].get("teacher_trunk_parameters"),
        } for arch in archs},
        "both_valid": both_valid,
        "alignment_threshold": tau,
        "subspace_alignment": align,
        "chance_alignment": chance,
        "alignment_over_chance": align / chance if chance else float("nan"),
        # a mean can hide "a few directions agree strongly" vs "all agree weakly"
        "principal_cos2": {
            "top1": float(cos2[0]), "top5_mean": float(cos2[:5].mean()),
            "median": float(cos2.median()), "min": float(cos2[-1]),
            "spectrum": [float(v) for v in cos2],
        },
        "verdict": verdict,
        "wording": "decoder-certified observability geometry (not intrinsic)",
    }

    # ---- 2x2 cross-decoder restricted certificate ------------------------
    # D(V_i ; f_j): the subspace one decoder discovered, certified by the other. High
    # off-diagonals are stronger evidence of shared channel structure than eigenspace
    # alignment alone; a high-diagonal / low-off-diagonal pattern is decoder
    # specialisation.
    outputs = {}
    for arch, path in paths.items():
        op = path.with_name(path.stem + "_report_outputs.pt")
        if op.exists():
            outputs[arch] = torch.load(op, map_location="cpu", weights_only=True)
    if len(outputs) == len(archs):
        V = {a: Va, b: Vb}
        matrix = {}
        for vi in archs:
            for fj in archs:
                cert = subspace_certificate(outputs[fj]["Z_rep"].numpy(),
                                            outputs[fj]["F_rep"].numpy(),
                                            V[vi].numpy())
                matrix[f"V[{vi}]|f[{fj}]"] = cert["D_cert_subspace"]
        out["cross_decoder_certificate"] = matrix
        diag = [matrix[f"V[{x}]|f[{x}]"] for x in archs]
        off = [matrix[f"V[{x}]|f[{y}]"] for x in archs for y in archs if x != y]
        out["cross_decoder_ratio"] = (float(_torch.tensor(off).mean() /
                                            max(float(_torch.tensor(diag).mean()), 1e-9)))
    else:
        out["cross_decoder_certificate"] = None
    rep = Path(cfg["paths"]["reports_dir"]) / f"teacher_comparison_{args.tag}_seed{args.seed}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=2, default=float))

    for arch in archs:
        t = out["teachers"][arch]
        log.info("%-20s D_cert %-8s rank %-5s validity %-5s params %.2fM", arch,
                 f"{t['D_cert_subspace']:.3f}" if t["D_cert_subspace"] is not None else "n/a",
                 t["numerical_positive_rank"], t["validity_pass"],
                 (t["parameters"] or 0) / 1e6)
    pc = out["principal_cos2"]
    log.info("top-%d alignment %.4f (chance %.4f, %.1fx) | cos^2 top1 %.3f top5 %.3f "
             "median %.3f", k, align, chance, out["alignment_over_chance"],
             pc["top1"], pc["top5_mean"], pc["median"])
    if out.get("cross_decoder_certificate"):
        for kk, vv in out["cross_decoder_certificate"].items():
            log.info("  %-34s D_cert %.3f", kk, vv)
    log.info("VERDICT: %s", out["verdict"])
    log.info("wrote %s", rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
