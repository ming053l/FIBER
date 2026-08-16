#!/usr/bin/env python
"""Phase 1 — cache the native channel dataset  (z, prompt, seed, clean X).

Attacks are NOT cached: they are applied in the dataloader from a derived seed
(PLAN.md §2a). Caching them would cost ~36 GB and freeze the attack set, and X
is a cached constant either way.

    python scripts/cache_native_dataset.py --pilot
    python scripts/cache_native_dataset.py                 # full 10k/1k/1k/1k

Re-running resumes: a shard with a `.done` marker is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from fiber.diffusion import FrozenGenerator, GeneratorSpec
from fiber.diffusion.cache_dataset import (build_index, split_sizes, verify_index,
                                           write_index)
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.seeding import derive_seed, set_determinism

log = get_logger("phase1")


def config_fingerprint(cfg) -> str:
    keep = {k: cfg[k] for k in ("model", "latent", "vae", "dataset") if k in cfg}
    blob = json.dumps(keep, sort_keys=True, default=str).encode()
    return hashlib.blake2s(blob, digest_size=8).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images/split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    tag = "pilot" if args.pilot else "full"
    root = Path(cfg["paths"]["cache_dir"]) / tag
    root.mkdir(parents=True, exist_ok=True)

    records = build_index(cfg, pilot=args.pilot, shard_size=args.shard_size)
    report = verify_index(records)
    if not report["pass"]:
        log.error("Gate 1 split discipline FAILED: %s", json.dumps(report, indent=2))
        return 1
    write_index(records, root)
    log.info("index ok: %s", report["counts"])

    fingerprint = config_fingerprint(cfg)
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])
    splits = args.splits or sorted({r.split for r in records})
    res = cfg["model"]["resolution"]
    lshape = tuple(gen.latent_shape)
    timings: dict[str, float] = {}

    for split in splits:
        rs = [r for r in records if r.split == split]
        if args.limit:
            rs = rs[: args.limit]
        sdir = root / split
        sdir.mkdir(parents=True, exist_ok=True)
        shards = sorted({r.shard for r in rs})
        t_split = time.time()
        for shard in shards:
            items = [r for r in rs if r.shard == shard]
            done = sdir / f"shard_{shard:05d}.done"
            n = len(items)
            if done.exists():
                # A marker alone is not enough: a rerun with a different shard size
                # or a different generator config would silently reuse stale images.
                try:
                    prev = json.loads(done.read_text())
                except json.JSONDecodeError:
                    prev = {}
                if prev.get("n") == n and prev.get("fingerprint") == fingerprint:
                    log.info("%s shard %d: already cached", split, shard)
                    continue
                log.warning("%s shard %d: stale cache (%s), regenerating", split, shard, prev)
                done.unlink()
            img_path = sdir / f"images_{shard:05d}.npy"
            lat_path = sdir / f"latents_{shard:05d}.npy"
            images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                               shape=(n, res, res, 3))
            latents = np.lib.format.open_memmap(lat_path, mode="w+", dtype=np.float16,
                                                shape=(n, *lshape))
            t0 = time.time()
            for lo in range(0, n, args.batch):
                chunk = items[lo:lo + args.batch]
                z = torch.stack([gen.sample_latent(*r.latent_seed_parts)[0] for r in chunk])
                x = gen.generate(z, [r.prompt for r in chunk])
                for j, r in enumerate(chunk):
                    images[r.offset] = x[j]
                    latents[r.offset] = z[j].numpy().astype(np.float16)
                if lo % (args.batch * 10) == 0:
                    rate = (lo + len(chunk)) / max(time.time() - t0, 1e-6)
                    log.info("%s shard %d: %d/%d  %.2f img/s", split, shard,
                             lo + len(chunk), n, rate)
            images.flush(); latents.flush()
            del images, latents
            done.write_text(json.dumps({"n": n, "shard_size": args.shard_size,
                                        "fingerprint": fingerprint,
                                        "seconds": round(time.time() - t0, 1)}))
        timings[split] = round(time.time() - t_split, 1)
        log.info("%s done in %.1f s", split, timings[split])

    manifest = {
        "tag": tag,
        "config": str(cfg["_config_path"]),
        "config_fingerprint": fingerprint,
        "counts": report["counts"],
        "sizes": split_sizes(cfg, args.pilot),
        "shard_size": args.shard_size,
        "resolution": res,
        "latent_shape": list(lshape),
        "guidance_scale": cfg["model"]["guidance_scale"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "attacks_are_cached": False,
        "attack_seed_rule": "blake2s(sample_id|attack|severity|split_salt)",
        "split_seconds": timings,
        "gate1": report,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
