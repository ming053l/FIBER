#!/usr/bin/env python
"""Verify a cached shard against the frozen generator, and (re)stamp its marker.

Two jobs, both about not throwing away GPU-hours on a guess:

1. **Integrity.** Regenerate a spot sample of images from the index and compare them
   BIT-EXACTLY with what is stored.

   Generation is bit-reproducible ONLY at identical batch composition. fp16 batched
   matmuls reduce in a different order at a different batch size, and 25 diffusion
   steps amplify that: measured, the same z and prompt generated at batch 8 instead of
   batch 12 differs in 26% of pixels with max |diff| = 115 grey levels, while at the
   original batch 12 it is bit-identical (12/12, max |diff| = 0).

   So the spot check must replay the ORIGINAL batching: whole aligned groups of
   `--batch` consecutive records, with `--batch` equal to the value the cache was
   written with (recorded in the shard marker as `batch`). A mismatch here means the
   cache and the config really have diverged; it is not a tolerance question.

2. **Marker migration.** `.done` markers record the config fingerprint that produced
   them. If the fingerprint FUNCTION changes -- as it did in P0-1, when it was
   narrowed to the fields that actually determine pixels -- every marker looks stale
   and a full regeneration would be triggered for no reason. Re-stamping is only
   allowed here after the spot check passes, and the old value is kept for audit.

    python scripts/verify_cache.py --tag pilot --spot 8 --restamp
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fiber.diffusion import FrozenGenerator, GeneratorSpec
from fiber.diffusion.cache_dataset import build_index
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import derive_seed, set_determinism

from cache_native_dataset import config_fingerprint  # noqa: E402

log = get_logger("verify-cache")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--spot", type=int, default=12, help="images re-generated per shard")
    ap.add_argument("--batch", type=int, default=None,
                    help="batch size the cache was WRITTEN with; read from the marker "
                         "when omitted. Must match, or nothing will be bit-identical.")
    ap.add_argument("--restamp", action="store_true",
                    help="rewrite .done markers with the current fingerprint if verified")
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / args.tag
    records = build_index(cfg, pilot=(args.tag == "pilot"))
    fingerprint = config_fingerprint(cfg)
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])

    by_shard: dict[tuple, list] = {}
    for r in records:
        if args.splits and r.split not in args.splits:
            continue
        by_shard.setdefault((r.split, r.shard), []).append(r)

    report, ok_all = {}, True
    for (split, shard), items in sorted(by_shard.items()):
        sdir = root / split
        img_path = sdir / f"images_{shard:05d}.npy"
        if not img_path.exists():
            log.warning("%s shard %d: no images, skipping", split, shard)
            continue
        images = np.load(img_path, mmap_mode="r")
        latents = np.load(sdir / f"latents_{shard:05d}.npy", mmap_mode="r")
        marker = sdir / f"shard_{shard:05d}.done"
        prev = json.loads(marker.read_text()) if marker.exists() else {}
        batch = args.batch or prev.get("batch")
        if batch is None:
            raise SystemExit(
                f"{marker} does not record the batch size the cache was written with, "
                "and --batch was not given. Comparing at the wrong batch size makes "
                "every image differ (26% of pixels, measured), so guessing is worse "
                "than failing here.")
        items = sorted(items, key=lambda r: r.offset)

        # replay ORIGINAL batches: whole aligned groups, never an arbitrary subset
        n_groups = max(1, -(-args.spot // batch))
        rng = np.random.default_rng(derive_seed("verify", args.tag, split, shard))
        total_groups = -(-len(items) // batch)
        gsel = sorted(rng.choice(total_groups, size=min(n_groups, total_groups), replace=False))
        groups = [items[g * batch:(g + 1) * batch] for g in gsel]
        chosen = [r for g in groups for r in g]

        t0, img_bad, lat_bad = time.time(), 0, 0
        for chunk in groups:
            z = torch.stack([gen.sample_latent(*r.latent_seed_parts)[0] for r in chunk])
            x = gen.generate(z, [r.prompt for r in chunk])
            for j, r in enumerate(chunk):
                img_bad += int(not np.array_equal(x[j], np.asarray(images[r.offset])))
                lat_bad += int(not np.array_equal(z[j].numpy().astype(np.float16),
                                                 np.asarray(latents[r.offset])))
        ok = img_bad == 0 and lat_bad == 0
        ok_all &= ok
        report[f"{split}/{shard}"] = {
            "checked": len(chosen), "image_mismatches": img_bad,
            "latent_mismatches": lat_bad, "pass": ok, "seconds": round(time.time() - t0, 1),
        }
        report[f"{split}/{shard}"]["batch"] = batch
        report[f"{split}/{shard}"]["groups"] = [int(g) for g in gsel]
        log.info("%s shard %d: %d/%d bit-identical (batch %d)  %s", split, shard,
                 len(chosen) - img_bad, len(chosen), batch, "OK" if ok else "MISMATCH")

        if args.restamp and ok:
            marker.write_text(json.dumps({
                "n": len(items), "shard_size": max(len(items), prev.get("shard_size", len(items))),
                "fingerprint": fingerprint, "batch": batch,
                "seconds": prev.get("seconds"),
                "migrated_from_fingerprint": prev.get("fingerprint"),
                "verified_spot": len(chosen),
            }))

    out = Path(cfg["paths"]["reports_dir"]) / f"cache_verify_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fingerprint": fingerprint, "spot": args.spot,
                               "restamped": bool(args.restamp), "pass": ok_all,
                               "shards": report}, indent=2))
    log.info("%s -> %s", "PASS" if ok_all else "FAIL", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
