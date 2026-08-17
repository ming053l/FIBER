#!/usr/bin/env python
"""alpha-sweep on the REAL frozen diffusion channel, with the receivers already trained.

For a cached latent z and its own prompt, step along one frame direction,

    z_perp  = (I - v v') z,        z_alpha = z_perp + alpha * v,

regenerate with the frozen generator, and read the trained receiver's prediction of that
coordinate. Nothing is retrained; the frames and receivers are the frozen artifacts from
the earlier runs.

The component along v is REMOVED before the step is added, so alpha is the coordinate's
value rather than an offset on top of whatever it already was. Writing z + alpha*v gives
v'z ~ N(alpha, 1) instead of N(0, 1), i.e. a shifted marginal; the correct statement even
for z_perp + alpha*v is a CONTROLLED TRAVERSAL WITHIN A TYPICAL COORDINATE RANGE -- fixing
alpha is a conditional slice, not a draw from the prior, and it must not be described as
in-distribution sampling.

WHAT THIS CAN AND CANNOT SAY. No direction of the estimated spectrum is certified:
certified positive inertia is 0/64 on both teachers and both fail the validity gate. The
rows are therefore labelled by their rank in the IN-SAMPLE spectrum, never as
"high-certified". The registered protocol from the synthetic section applies: a direction
may not be called more observable because perturbing it changes the image more.

alpha = 0 is REGENERATED rather than taken from the cache. The cache was produced in
batches of 12 and fp16 reductions depend on batch composition, so reusing the cached
image would put the alpha = 0 point on a different generation path from the rest of its
own sweep.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/ssd1/ming/FIBER/src")
from fiber.diffusion import FrozenGenerator, GeneratorSpec          # noqa: E402
from fiber.models import build_extractor                            # noqa: E402
from fiber.transforms.spectral import SpectralFrame                 # noqa: E402
from fiber.utils.config import load_config                          # noqa: E402

RES = "/ssd2/ming/FIBER/results/triage1"
ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
N_BASE = 6
BATCH = 12                                    # same as the cache


def load_run(stem):
    fb = torch.load(f"{RES}/{stem}_frame.pt", map_location="cpu", weights_only=True)
    ck = torch.load(f"{RES}/{stem}_extractor.pt", map_location="cpu", weights_only=True)
    rows = fb["rows"].float()
    frame = SpectralFrame(d=rows.shape[1], k=rows.shape[0], rows=rows,
                          reorthonormalise=False)
    model = build_extractor(ck["arch"], k=int(ck["k"]))
    model.load_state_dict(ck["state_dict"])
    return frame, model.eval()


def main():
    cfg = load_config("/ssd1/ming/FIBER/configs/linear_fiber.yaml")
    dev = "cuda:0"
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])
    idx = [json.loads(l) for l in open("/ssd2/ming/FIBER/cache/pilot/index.jsonl")]
    val = sorted([r for r in idx if r["split"] == "val"], key=lambda r: r["index"])[:N_BASE]

    # C2_haar only: Phase A is locked to this frame, so an appendix diagnostic that used
    # a different one would not be comparable to the primary experiment. No direction of
    # the spectral fit is certified (0/64 positive inertia), so nothing here may be
    # labelled a high-observability direction.
    rows = [(f"Haar coord {c}", "C2_haar_k64_s0_rxresnet18_r0_scgate", c)
            for c in (0, 31, 63)]

    out = {}
    for label, stem, coord in rows:
        frame, model = load_run(stem)
        frame, model = frame.to(dev), model.to(dev)
        v = frame.rows()[coord].detach().to(dev)
        rec = np.zeros((N_BASE, len(ALPHAS)))
        dpix = np.zeros((N_BASE, len(ALPHAS)))
        for bi, r in enumerate(val):
            z0 = gen.sample_latent(*[r["latent_seed_parts"][0], *r["latent_seed_parts"][1:]])[0] \
                 if isinstance(r["latent_seed_parts"], list) else None
            z0 = gen.sample_latent(*r["latent_seed_parts"])[0]
            zf = z0.to(dev).flatten()
            z_perp = zf - (v @ zf) * v                 # remove the existing component
            zs = torch.stack([(z_perp + a * v).reshape(z0.shape).cpu() for a in ALPHAS])
            imgs = gen.generate(zs, [r["prompt"]] * len(ALPHAS))
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(dev) / 255.0
            x = (x - 0.5) / 0.5
            with torch.no_grad():
                w = model(x)["w_hat"][:, coord].float().cpu().numpy()
            base = imgs[ALPHAS.index(0.0)].astype(np.float32)
            rec[bi] = w - w[ALPHAS.index(0.0)]
            dpix[bi] = [np.sqrt(((im.astype(np.float32) - base) ** 2).mean())
                        for im in imgs]
        slope = np.polyfit(np.tile(ALPHAS, N_BASE), rec.ravel(), 1)[0]
        out[label] = {"recovered": rec.tolist(), "rms_pixel_change": dpix.tolist(),
                      "slope": float(slope), "coord": coord, "stem": stem}
        print(f"{label:22s} coord {coord:2d}  slope {slope:+.4f}  "
              f"RMS pixel change at |a|=2: {dpix[:, 0].mean():.2f}/{dpix[:, -1].mean():.2f}")
    json.dump({"alphas": list(ALPHAS), "n_base": N_BASE, "rows": out},
              open("figures/alpha_sweep_real.json", "w"), indent=2)
    print("wrote figures/alpha_sweep_real.json")


if __name__ == "__main__":
    main()
