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
import os
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


# Only what determines the PIXELS. The fingerprint answers exactly one question --
# "would regenerating produce different images?" -- so downstream protocol choices
# (cross-fit roles, attack lists, training budget) must not invalidate a cache.
# Getting this wrong costs a full regeneration: 2768 images is ~1 GPU-hour, 12768 is ~4.4.
IMAGE_DETERMINING = {
    "model": None,                                   # checkpoint, scheduler, steps, CFG, dtype
    "latent": None,                                  # shape of z
    "vae": None,
    "dataset": ["splits", "pilot", "prompts"],       # which z and which prompt per sample
}


def config_fingerprint(cfg) -> str:
    keep = {}
    for block, fields in IMAGE_DETERMINING.items():
        if block not in cfg:
            continue
        keep[block] = (cfg[block] if fields is None
                       else {f: cfg[block][f] for f in fields if f in cfg[block]})
    blob = json.dumps(keep, sort_keys=True, default=str).encode()
    return hashlib.blake2s(blob, digest_size=8).hexdigest()


class SyntheticGenerator:
    """A stand-in for FrozenGenerator that makes pixels out of z arithmetically.

    It exists so the whole protocol -- caching, the --post-lock gate, training,
    selection, the lock, the test materialisation, the locked evaluation and the gate --
    can be exercised end to end on a CPU in under a minute. NOTHING it produces is
    evidence about the diffusion channel, and every artifact it touches is stamped
    `synthetic_pixels: true` so a dry-run result cannot be mistaken for one later.

    z is drawn exactly as the real generator draws it (same derive_seed, same CPU
    randn, same dtype rounding), so split disjointness and the W = Rz target are real.
    Only G is fake, and it is deliberately a LOSSY function of z -- upsampled, offset by
    a prompt-dependent constant, clipped and quantised to uint8 -- so the recoverable
    information is partial rather than trivially perfect.
    """

    def __init__(self, cfg):
        lc = cfg["latent"]
        self.latent_shape = (int(lc["channels"]), int(lc["height"]), int(lc["width"]))
        self.res = int(cfg["model"]["resolution"])
        self.dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32

    def sample_latent(self, *seed_parts, batch: int = 1) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(derive_seed(*seed_parts) % (2**63 - 1))
        z = torch.randn(batch, *self.latent_shape, generator=g, dtype=torch.float32)
        return z.to(self.dtype).float()

    def generate(self, z: torch.Tensor, prompts) -> np.ndarray:
        if z.dim() == 3:
            z = z.unsqueeze(0)
        img = torch.nn.functional.interpolate(z[:, :3], size=(self.res, self.res),
                                              mode="bilinear", align_corners=False)
        off = torch.tensor([[(int(hashlib.blake2s(p.encode(), digest_size=2).hexdigest(), 16)
                              % 64) - 32] for p in prompts], dtype=torch.float32)
        img = 128.0 + 48.0 * img + off[:, :, None, None]
        return img.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--splits", nargs="*", default=None)
    # B1, strongest form: test PIXELS are generated only after the method is locked, so
    # "no test sample existed before the lock" is a statement about the filesystem
    # rather than about which script read what.
    ap.add_argument("--post-lock", metavar="SELECTION",
                    help="path to reports/selection_<tag>.json; required to cache any "
                         "test split")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N images/split")
    ap.add_argument("--synthetic", action="store_true",
                    help="PLUMBING ONLY: make pixels arithmetically instead of loading "
                         "the diffusion model. Every artifact is stamped "
                         "synthetic_pixels so it cannot be read as evidence.")
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
    gen = (SyntheticGenerator(cfg) if args.synthetic
           else FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"]))
    if args.synthetic:
        log.warning("SYNTHETIC pixels: this cache is a plumbing fixture, not evidence")
    splits = args.splits or sorted({r.split for r in records})
    test_splits = [s for s in splits if s.startswith("test")]
    if test_splits and not args.post_lock:
        raise SystemExit(
            f"refusing to cache {test_splits} before a method lock (B1). Test pixels "
            "generated pre-lock only support 'the test set was never accessed'; "
            "generating them after selection supports the stronger 'no test sample "
            "existed'. Pass --post-lock reports/selection_<tag>.json, or --splits "
            "train val for the pre-lock stage.")
    lock_sha = None
    if args.post_lock:
        sel_path = Path(args.post_lock)
        if not sel_path.exists():
            raise SystemExit(f"{sel_path} not found: --post-lock names the selection "
                             "artifact the test cache is bound to")
        lock_sha = hashlib.blake2s(sel_path.read_bytes(), digest_size=16).hexdigest()
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
                fresh = prev.get("n") == n and prev.get("fingerprint") == fingerprint
                if fresh and split.startswith("test") and lock_sha:
                    # A pre-existing test shard may NOT be skipped and then covered by a
                    # freshly written manifest: that would prove the manifest was created
                    # after the lock, not that the pixels were. The binding lives in the
                    # shard's own marker and is written only when the shard is generated.
                    if prev.get("selection_sha") != lock_sha:
                        log.warning("%s shard %d predates this lock (%s); regenerating so "
                                    "the pixels are demonstrably post-lock", split, shard,
                                    prev.get("selection_sha"))
                        fresh = False
                if fresh:
                    log.info("%s shard %d: already cached", split, shard)
                    continue
                log.warning("%s shard %d: stale cache (%s), regenerating. If only the "
                            "PROTOCOL changed, stop and run scripts/verify_cache.py "
                            "--restamp instead of paying for a regeneration.",
                            split, shard, prev)
                done.unlink()
            # Write to temporary files and rename only on completion. open_memmap(w+)
            # creates a ZERO-FILLED file, so writing in place means an interrupted or
            # failed run destroys good data before it has regenerated it.
            img_path = sdir / f"images_{shard:05d}.npy"
            lat_path = sdir / f"latents_{shard:05d}.npy"
            img_tmp = img_path.with_suffix(".npy.tmp")
            lat_tmp = lat_path.with_suffix(".npy.tmp")
            images = np.lib.format.open_memmap(img_tmp, mode="w+", dtype=np.uint8,
                                               shape=(n, res, res, 3))
            latents = np.lib.format.open_memmap(lat_tmp, mode="w+", dtype=np.float16,
                                                shape=(n, *lshape))
            t0 = time.time()
            for lo in range(0, n, args.batch):
                chunk = items[lo:lo + args.batch]
                z = torch.stack([gen.sample_latent(*r.latent_seed_parts)[0] for r in chunk])
                x = gen.generate(z, [r.prompt for r in chunk])
                for j, r in enumerate(chunk):
                    images[r.offset] = x[j]
                    latents[r.offset] = z[j].numpy().astype(np.float16)
                if (lo // args.batch) % 10 == 0 or lo + len(chunk) >= n:
                    rate = (lo + len(chunk)) / max(time.time() - t0, 1e-6)
                    log.info("%s shard %d: %d/%d  %.2f img/s", split, shard,
                             lo + len(chunk), n, rate)
            images.flush(); latents.flush()
            del images, latents
            os.replace(img_tmp, img_path)      # atomic
            os.replace(lat_tmp, lat_path)
            # `batch` is part of the cache's identity, not just a speed knob: fp16
            # batched matmuls reduce in a different order at a different batch size,
            # so re-generating at another batch does NOT reproduce these pixels.
            done.write_text(json.dumps({"n": n, "shard_size": args.shard_size,
                                        "synthetic_pixels": bool(args.synthetic),
                                        "batch": args.batch, "fingerprint": fingerprint,
                                        # written only here, when the pixels were made
                                        "selection_sha": lock_sha if split.startswith("test") else None,
                                        "generated_under_lock": bool(lock_sha) and split.startswith("test"),
                                        "seconds": round(time.time() - t0, 1)}))
        timings[split] = round(time.time() - t_split, 1)
        log.info("%s done in %.1f s", split, timings[split])

    manifest = {
        "tag": tag,
        # Read by the locked evaluator, which refuses to produce a gate artifact from
        # synthetic pixels unless the run is explicitly declared a dry run.
        "synthetic_pixels": bool(args.synthetic),
        "config": str(cfg["_config_path"]),
        "config_fingerprint": fingerprint,
        "counts": report["counts"],
        "sizes": split_sizes(cfg, args.pilot),
        "shard_size": args.shard_size,
        "batch": args.batch,
        "generation_is_batch_dependent": True,
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
    if lock_sha:
        # The manifest only SUMMARISES what the shard markers say; it never asserts more.
        bound, unbound = [], []
        for split in test_splits:
            for marker in sorted((root / split).glob("*.done")):
                prev = json.loads(marker.read_text())
                (bound if prev.get("selection_sha") == lock_sha else unbound).append(
                    f"{split}/{marker.stem}")
        (root / "test_cache_manifest.json").write_text(json.dumps({
            "selection_sha": lock_sha, "selection": str(Path(args.post_lock).resolve()),
            "splits": test_splits, "config_fingerprint": fingerprint,
            "shards_generated_under_this_lock": bound,
            "shards_not_bound_to_this_lock": unbound,
            "generated_after_lock": not unbound}, indent=2))
        log.info("test cache: %d shards generated under this lock, %d not",
                 len(bound), len(unbound))
    log.info("wrote %s", root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
