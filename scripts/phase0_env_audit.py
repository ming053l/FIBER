#!/usr/bin/env python
"""Gate 0 — environment audit (PLAN.md §7).

| #   | Criterion                                                             |
|-----|-----------------------------------------------------------------------|
| 0.1 | PSNR(G(z)^1, G(z)^2) measured, reported as a REPRODUCIBILITY AUDIT     |
| 0.2 | init_noise_sigma == 1.0; z round-trips through prepare_latents        |
| 0.3 | z persisted and reloadable bit-exactly                                |
| 0.4 | [4,64,64], d = 16384; VAE 0.18215 documented as NOT applying to z      |
| 0.5 | generation is no_grad; N samples, zero NaN                            |
| 0.6 | DDIM, eta = 0, nothing sets sde-*                                     |
| 0.7 | Batch-composition sensitivity of generation, measured and reported     |

R5: 0.1 is an audit, NOT a BER noise floor. X is generated once and cached and
every extractor reads the same cached X, so generation nondeterminism never
enters the extractor's error.

0.7 qualifies 0.1: reproducibility holds at FIXED batch composition. fp16 batched
matmuls reduce in a different order at a different batch size, and 25 diffusion
steps amplify it, so the batch size is part of a cache's identity rather than a
speed knob. Recorded in every shard marker.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from fiber.diffusion import FrozenGenerator, GeneratorSpec, load_captions, sample_prompts
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

log = get_logger("phase0")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--repro-pairs", type=int, default=8)
    ap.add_argument("--out", default="reports/phase0.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    results: dict[str, dict] = {}

    # ---- 0.6 / 0.2 : constructing the generator runs assert_channel_is_native
    t0 = time.time()
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])
    sched = gen.pipe.scheduler
    results["0.6_native_channel"] = {
        "pass": True,
        "scheduler": type(sched).__name__,
        "eta": float(gen.spec.eta),
        "prediction_type": sched.config.prediction_type,
        "clip_sample": bool(sched.config.clip_sample),
        "steps": gen.spec.num_inference_steps,
        "load_seconds": round(time.time() - t0, 1),
    }

    # ---- 0.4 latent geometry
    z = gen.sample_latent("gate0", "audit", batch=args.batch)
    d = int(np.prod(z.shape[1:]))
    results["0.4_latent_geometry"] = {
        "pass": tuple(z.shape[1:]) == (4, 64, 64) and d == 16384,
        "shape": list(z.shape[1:]), "d": d, "is_power_of_two": d & (d - 1) == 0,
        "vae_scaling_factor": cfg["vae"]["scaling_factor"],
        "vae_scaling_applied_to_z": False,
        "note": "0.18215 belongs to the VAE latent, never to z_T (PLAN.md §1.3)",
        "z_mean": float(z.mean()), "z_std": float(z.std()),
        "z_dtype_rounded_to": gen.spec.dtype,
    }

    # ---- 0.2 round-trip through prepare_latents
    err = gen.latent_roundtrip_error(z)
    results["0.2_latent_roundtrip"] = {
        "pass": err == 0.0 and abs(float(sched.init_noise_sigma) - 1.0) < 1e-6,
        "init_noise_sigma": float(sched.init_noise_sigma),
        "max_abs_error": err,
    }

    # ---- 0.3 persistence
    root = Path(cfg["paths"]["data_root"]) / "gate0"
    root.mkdir(parents=True, exist_ok=True)
    zp = root / "z_roundtrip.pt"
    torch.save(z, zp)
    z2 = torch.load(zp, map_location="cpu", weights_only=True)
    bitexact = bool(torch.equal(z, z2))
    results["0.3_persistence"] = {
        "pass": bitexact, "path": str(zp), "bit_exact": bitexact,
        "bytes": zp.stat().st_size,
    }

    # ---- 0.5 N samples, zero NaN, no_grad
    prompts_pool = load_captions(cfg["dataset"]["prompts"]["source"])
    prompts = sample_prompts(prompts_pool, args.num_samples, "gate0", "prompts")
    n_nan, n_done, grad_leak = 0, 0, False
    t0 = time.time()
    for lo in range(0, args.num_samples, args.batch):
        bs = min(args.batch, args.num_samples - lo)
        zb = gen.sample_latent("gate0", "sample", lo, batch=bs)
        imgs = gen.generate(zb, prompts[lo:lo + bs], return_float=True)
        grad_leak |= isinstance(imgs, torch.Tensor) and imgs.requires_grad
        n_nan += int((~np.isfinite(imgs)).sum())
        n_done += bs
        if lo % (args.batch * 5) == 0:
            log.info("0.5 generated %d/%d (%.1fs)", n_done, args.num_samples, time.time() - t0)
    gen_seconds = time.time() - t0
    results["0.5_generation_health"] = {
        "pass": n_nan == 0 and not grad_leak,
        "samples": n_done, "non_finite_values": n_nan,
        "grad_enabled_in_output": grad_leak,
        "seconds_total": round(gen_seconds, 1),
        "seconds_per_image": round(gen_seconds / max(n_done, 1), 2),
        "guidance_scale": gen.spec.guidance_scale,
    }

    # ---- 0.1 reproducibility audit (NOT a BER floor -- R5)
    psnrs, exact = [], 0
    zr = gen.sample_latent("gate0", "repro", batch=args.repro_pairs)
    pr = sample_prompts(prompts_pool, args.repro_pairs, "gate0", "repro_prompts")
    a = gen.generate(zr, pr)
    b = gen.generate(zr, pr)
    for i in range(args.repro_pairs):
        psnrs.append(psnr(a[i], b[i]))
        exact += int(np.array_equal(a[i], b[i]))
    finite = [p for p in psnrs if np.isfinite(p)]
    results["0.1_reproducibility_audit"] = {
        "pass": True,   # measured and reported; never a pass/fail threshold (R5)
        "pairs": args.repro_pairs,
        "bit_identical_pairs": exact,
        "psnr_min": float(np.min(psnrs)) if psnrs and np.isfinite(np.min(psnrs)) else None,
        "psnr_mean_finite": float(np.mean(finite)) if finite else None,
        "psnr_all_infinite": len(finite) == 0,
        "role": "generation reproducibility audit, NOT a BER noise floor (PLAN.md R5)",
        "scope": "same process, same batch composition; see 0.7 for what breaks it",
    }

    # ---- 0.7 batch-composition sensitivity (qualifies 0.1)
    nb = min(8, args.repro_pairs * 2) if args.repro_pairs >= 4 else 4
    zb = gen.sample_latent("gate0", "batchdep", batch=nb)
    pb = sample_prompts(prompts_pool, nb, "gate0", "batch_prompts")
    big = gen.generate(zb, pb)                                  # one batch of nb
    half = np.concatenate([gen.generate(zb[: nb // 2], pb[: nb // 2]),
                           gen.generate(zb[nb // 2:], pb[nb // 2:])])   # two of nb/2
    d = np.abs(big.astype(int) - half.astype(int))
    identical = int(sum(d[i].max() == 0 for i in range(nb)))
    results["0.7_batch_composition"] = {
        "pass": True,   # measured and reported, never a threshold
        "batch": nb, "split_batch": nb // 2,
        "bit_identical_images": identical,
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": float(d.mean()),
        "fraction_pixels_changed": float((d > 0).mean()),
        "psnr": psnr(big, half),
        "note": ("generation is bit-reproducible only at identical batch composition; "
                 "the batch size is part of a cache's identity and is recorded in each "
                 "shard marker"),
    }

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "checkpoint": gen.spec.checkpoint,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or "uncommitted",
    }
    try:
        import diffusers
        env["diffusers"] = diffusers.__version__
    except Exception:
        pass

    passed = all(v["pass"] for v in results.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# Phase 0 — Environment audit (Gate 0)\n\n")
        fh.write(f"**Verdict: {'PASS' if passed else 'FAIL'}**\n\n")
        fh.write("| env | |\n|---|---|\n")
        for k, v in env.items():
            fh.write(f"| {k} | `{v}` |\n")
        fh.write("\n| criterion | result |\n|---|---|\n")
        for k, v in sorted(results.items()):
            fh.write(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} |\n")
        fh.write("\n## Detail\n\n```json\n")
        fh.write(json.dumps(results, indent=2))
        fh.write("\n```\n\n")
        fh.write("## Notes\n\n"
                 "- **0.7 qualifies 0.1.** Reproducibility holds at fixed batch\n"
                 "  composition. Regenerating the same `z` and prompt at a different batch\n"
                 "  size changes the image materially, so the batch size is part of a\n"
                 "  cache's identity and is recorded in every shard marker.\n"
                 "- **0.1 is an audit, not a noise floor (R5).** `X` is generated once and\n"
                 "  cached; every extractor reads the same cached `X`, so generation\n"
                 "  nondeterminism never enters the extractor's error. It would only become\n"
                 "  channel noise in a variant where the receiver regenerates the carrier —\n"
                 "  which is InvCISD's situation and not FIBER's.\n"
                 "- `z` is drawn in float32 and rounded to the generator's dtype before being\n"
                 "  stored, so the persisted `z` is bit-exactly what the UNet consumed and the\n"
                 "  extractor's target `W = Qz` refers to that same value.\n")
    def _jsonable(o):
        if isinstance(o, float) and not np.isfinite(o):
            return "inf" if o > 0 else "-inf"   # keep the file strict-JSON parseable
        raise TypeError(o)

    with open(out.with_suffix(".json"), "w") as fh:
        json.dump({"env": env, "results": results, "pass": passed}, fh, indent=2,
                  allow_nan=False, default=_jsonable)
    log.info("wrote %s (verdict %s)", out, "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
