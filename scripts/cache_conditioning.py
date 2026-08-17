#!/usr/bin/env python
"""Cache S = C(P), the conditioning tensor the generator actually consumed.

    python scripts/cache_conditioning.py --pilot
    python scripts/cache_conditioning.py --pilot --verify 4     # regenerate and compare

`S` is defined in reports/blind_vs_sideinfo.md as the conditioning ACTUALLY SUPPLIED to
the generator, not "the prompt re-encoded". Those differ: the text encoder is not
batch-composition invariant in fp16 -- re-encoding the same prompt alone rather than in
its original chunk moves the embedding by up to 0.024 -- exactly the reduction-order
effect that already made batch size part of the image cache's identity.

So this REPLAYS the caching loop: same index, same shard order, same chunking, same
`batch`, and the same `encode_prompt` call with classifier-free guidance and the same
negative prompt. The batch size is read from the image cache manifest and a mismatch is
refused, because a different one silently produces a different S.

What is stored is therefore, precisely:

    S_i = the SAMPLE-SPECIFIC POSITIVE CFG CONDITIONING EMBEDDING used during generation

Not "the UNet's conditioning tensor". The negative branch is deliberately excluded, and
calling it "a constant" would have been wrong: measured here, the unconditional embedding
is identical for every row WITHIN a batch (max|diff| 0.000000) but differs by 0.015625
between batch sizes 12 and 8/4/1. Its text content is fixed; its value is not. It is a
fixed part of the CHANNEL PROTOCOL, not receiver side information, so it is not something
the side receiver is given.
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
from fiber.diffusion.cache_dataset import build_index
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import provenance
from fiber.utils.seeding import set_determinism

log = get_logger("conditioning")

# Bump when anything about how S is produced changes: the encoder call, which half is
# stored, the negative-prompt protocol, the dtype. A cached shard whose marker carries a
# different version is regenerated rather than reused.
CONDITIONING_PROTOCOL = 1


def shard_identity(n: int, batch: int, shape, cfg_fingerprint: str, negative: str) -> dict:
    """Everything S depends on. `n` alone is not freshness: the text encoder is not
    batch-composition invariant, so a shard with the right sample count can still hold a
    conditioning produced under a different chunking, a different negative-prompt
    protocol or a different image-cache identity -- and be silently reused. This is the
    same stale-marker failure the image cache already had to fix."""
    return {"n": int(n), "batch": int(batch),
            "shape": list(shape) if shape else None,
            "image_config_fingerprint": cfg_fingerprint,
            "negative_prompt": negative,
            "protocol": CONDITIONING_PROTOCOL}


def encode_chunk(gen, prompts: list[str]) -> torch.Tensor:
    """The conditional embedding, from the pipeline's own code path."""
    with torch.no_grad():
        cond, _ = gen.pipe.encode_prompt(
            prompt=list(prompts), device=gen.pipe.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[gen.spec.negative_prompt] * len(prompts))
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=None,
                    help="defaults to the image cache's batch; a mismatch is refused")
    ap.add_argument("--shard-size", type=int, default=None)
    ap.add_argument("--post-lock", metavar="SELECTION",
                    help="path to the side-info selection artifact; required to cache "
                         "conditioning for any test split")
    ap.add_argument("--verify", action="store_true",
                    help="replay representative chunks -- first full, partial tail, a "
                         "later shard -- from the stored S and require the cached pixels "
                         "back bit-identically")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    tag = "pilot" if args.pilot else "full"
    root = Path(cfg["paths"]["cache_dir"]) / tag
    man_path = root / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{man_path} not found: cache the images first — S has to be "
                         "replayed against the chunking that produced them")
    image_manifest = json.loads(man_path.read_text())
    batch = args.batch or int(image_manifest["batch"])
    shard_size = args.shard_size or int(image_manifest["shard_size"])
    if batch != int(image_manifest["batch"]):
        raise SystemExit(
            f"--batch {batch} != the image cache's {image_manifest['batch']}. The text "
            "encoder is not batch-composition invariant, so this would store a "
            "conditioning the generator never saw.")

    records = build_index(cfg, pilot=args.pilot, shard_size=shard_size)
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])
    splits = args.splits or sorted({r.split for r in records})

    # B1, mirrored for the side-information branch. S is not harmless metadata here: it
    # is a RECEIVER INPUT, so test conditioning materialised before the side-info method
    # is locked has the same status as test pixels generated pre-lock.
    test_splits = [x for x in splits if x.startswith("test")]
    lock_sha = None
    if test_splits and not args.post_lock:
        raise SystemExit(
            f"refusing to cache conditioning for {test_splits} before a side-info method "
            "lock. S is a receiver input in this branch, not metadata. Pass --post-lock "
            "<selection artifact>, or --splits train val for the pre-lock stage.")
    if args.post_lock:
        sel = Path(args.post_lock)
        if not sel.exists():
            raise SystemExit(f"{sel} not found: --post-lock names the selection artifact "
                             "the test conditioning is bound to")
        lock_sha = hashlib.blake2s(sel.read_bytes(), digest_size=16).hexdigest()
    out_root = root / "conditioning"
    out_root.mkdir(parents=True, exist_ok=True)
    shape = expected_shape = None
    timings = {}

    for split in splits:
        rs = [r for r in records if r.split == split]
        sdir = out_root / split
        sdir.mkdir(parents=True, exist_ok=True)
        t_split = time.time()
        for shard in sorted({r.shard for r in rs}):
            items = [r for r in rs if r.shard == shard]
            n = len(items)
            done = sdir / f"shard_{shard:05d}.done"
            want = shard_identity(n, batch, shape or expected_shape,
                                  image_manifest.get("config_fingerprint"),
                                  gen.spec.negative_prompt)
            if done.exists():
                try:
                    prev = json.loads(done.read_text())
                except json.JSONDecodeError:
                    prev = {}
                # shape is unknown until the first shard is encoded; do not let that
                # single unknown field wave a stale shard through
                cmp_keys = [k for k in want if not (k == "shape" and want[k] is None)]
                if all(prev.get(k) == want[k] for k in cmp_keys):
                    log.info("%s shard %d: already cached", split, shard)
                    if expected_shape is None and prev.get("shape"):
                        expected_shape = tuple(prev["shape"])
                    continue
                log.warning("%s shard %d: stale conditioning (%s), regenerating",
                            split, shard, {k: prev.get(k) for k in cmp_keys})
                done.unlink()
            path = sdir / f"cond_{shard:05d}.npy"
            tmp = path.with_suffix(".npy.tmp")
            arr = None
            t0 = time.time()
            # EXACTLY the caching loop's chunking: same order, same size
            for lo in range(0, n, batch):
                chunk = items[lo:lo + batch]
                cond = encode_chunk(gen, [r.prompt for r in chunk]).cpu()
                if arr is None:
                    shape = tuple(cond.shape[1:])
                    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                                    shape=(n, *shape))
                for j, r in enumerate(chunk):
                    arr[r.offset] = cond[j].numpy().astype(np.float16)
            arr.flush(); del arr
            os.replace(tmp, path)
            done.write_text(json.dumps({
                **shard_identity(n, batch, shape,
                                 image_manifest.get("config_fingerprint"),
                                 gen.spec.negative_prompt),
                "selection_sha": lock_sha if split.startswith("test") else None,
                "generated_under_lock": bool(lock_sha) and split.startswith("test"),
                "seconds": round(time.time() - t0, 1)}))
            log.info("%s shard %d: %d encoded in %.1fs", split, shard, n,
                     time.time() - t0)
        timings[split] = round(time.time() - t_split, 1)

    manifest = {
        "tag": tag, "splits": splits, "batch": batch, "shard_size": shard_size,
        "shape": list(shape) if shape else None,
        "stores": ("sample-specific positive CFG conditioning embedding only; the "
                   "negative branch is part of the channel protocol, not side "
                   "information, and is itself batch-composition dependent "
                   "(0.015625 between B=12 and B=8/4/1)"),
        "negative_prompt": gen.spec.negative_prompt,
        "image_cache_batch": int(image_manifest["batch"]),
        "image_config_fingerprint": image_manifest.get("config_fingerprint"),
        "replayed_generation_chunking": True,
        "text_encoder_is_batch_composition_invariant": False,
        "split_seconds": timings,
        "protocol": CONDITIONING_PROTOCOL,
        "test_splits_bound_to": lock_sha,
        "test_conditioning_is_post_lock": bool(lock_sha),
        **provenance(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", out_root / "manifest.json")

    if args.verify:
        rc = verify(gen, cfg, root, out_root, records, splits, batch)
        if rc:
            return rc
    return 0


def _chunks(records, split: str, batch: int):
    """The generation loop's chunking, reconstructed: shard by shard, batch by batch."""
    rs = [r for r in records if r.split == split]
    for shard in sorted({r.shard for r in rs}):
        items = sorted([r for r in rs if r.shard == shard], key=lambda r: r.offset)
        for lo in range(0, len(items), batch):
            yield shard, lo, items[lo:lo + batch]


def representative_chunks(records, splits, batch: int):
    """Cover the regimes where a replay could differ, not just the easy one.

    A full leading chunk proves nothing about the PARTIAL TAIL, and the tails are where
    the batch composition actually changes: 500 % 12 = 8 for a train shard, 256 % 12 = 4
    for val. A single shard proves nothing about shard addressing either.
    """
    out = []
    for split in splits:
        chunks = list(_chunks(records, split, batch))
        if not chunks:
            continue
        full = [c for c in chunks if len(c[2]) == batch]
        tail = [c for c in chunks if len(c[2]) != batch]
        picked = []
        if full:
            picked.append(("first full chunk", full[0]))
        if tail:
            picked.append((f"partial tail ({len(tail[-1][2])})", tail[-1]))
        later = [c for c in full if c[0] > 0]
        if later:
            picked.append((f"shard {later[len(later)//2][0]} addressing",
                           later[len(later)//2]))
        for label, c in picked:
            out.append((split, label, c))
    return out


def verify(gen, cfg, root: Path, out_root: Path, records, splits, batch: int) -> int:
    """Feed the stored S back through the UNet and require the cached pixels back.

    Every check replays the COMPLETE original chunk and then compares the samples in it.
    Verifying a single image by re-encoding it at batch 1 would test the opposite of the
    property in question -- the whole point is that the chunk composition is part of S.
    """
    checks = representative_chunks(records, splits, batch)
    if not checks:
        log.warning("nothing to verify"); return 0
    bad = 0
    for split, label, (shard, lo, items) in checks:
        imgs = np.load(root / split / f"images_{shard:05d}.npy", mmap_mode="r")
        cond = np.load(out_root / split / f"cond_{shard:05d}.npy", mmap_mode="r")
        z = torch.stack([gen.sample_latent(*r.latent_seed_parts)[0] for r in items])
        pe = torch.from_numpy(np.asarray(cond[[r.offset for r in items]])).to(
            gen.pipe.device, gen.dtype)
        with torch.no_grad():
            neg, _ = gen.pipe.encode_prompt(
                prompt=[gen.spec.negative_prompt] * len(items), device=gen.pipe.device,
                num_images_per_prompt=1, do_classifier_free_guidance=False)
            out = gen.pipe(prompt_embeds=pe, negative_prompt_embeds=neg,
                           latents=z.to(gen.pipe.device, gen.dtype),
                           num_inference_steps=gen.spec.num_inference_steps,
                           guidance_scale=gen.spec.guidance_scale, eta=gen.spec.eta,
                           height=gen.spec.resolution, width=gen.spec.resolution,
                           output_type="np").images
        regen = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        same = sum(np.array_equal(regen[j], np.asarray(imgs[r.offset]))
                   for j, r in enumerate(items))
        worst = max(int(np.abs(regen[j].astype(int)
                               - np.asarray(imgs[r.offset]).astype(int)).max())
                    for j, r in enumerate(items))
        log.info("%-22s %-28s chunk of %2d at shard %d offset %3d: %d/%d bit-identical, "
                 "max|diff| %d", split, label, len(items), shard, lo, same, len(items),
                 worst)
        bad += len(items) - same
    if bad:
        log.error("%d images differ: the stored S is NOT the conditioning that produced "
                  "the cache", bad)
        return 1
    log.info("all %d chunks bit-identical across full, partial-tail and later-shard "
             "regimes: S is the conditioning the generator consumed", len(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
