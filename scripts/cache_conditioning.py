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

Only the CONDITIONAL half is stored. The unconditional half is a constant (the negative
prompt is "" for every sample) and carries no sample information.
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
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="regenerate N cached images from the stored S and require them "
                         "to be bit-identical")
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
    out_root = root / "conditioning"
    out_root.mkdir(parents=True, exist_ok=True)
    shape = None
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
            if done.exists() and json.loads(done.read_text()).get("n") == n:
                log.info("%s shard %d: already cached", split, shard)
                continue
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
            done.write_text(json.dumps({"n": n, "batch": batch, "shape": list(shape),
                                        "seconds": round(time.time() - t0, 1)}))
            log.info("%s shard %d: %d encoded in %.1fs", split, shard, n,
                     time.time() - t0)
        timings[split] = round(time.time() - t_split, 1)

    manifest = {
        "tag": tag, "splits": splits, "batch": batch, "shard_size": shard_size,
        "shape": list(shape) if shape else None,
        "stores": "conditional embedding only; the unconditional half is a constant",
        "negative_prompt": gen.spec.negative_prompt,
        "image_cache_batch": int(image_manifest["batch"]),
        "image_config_fingerprint": image_manifest.get("config_fingerprint"),
        "replayed_generation_chunking": True,
        "text_encoder_is_batch_composition_invariant": False,
        "split_seconds": timings, **provenance(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("wrote %s", out_root / "manifest.json")

    if args.verify:
        rc = verify(gen, cfg, root, out_root, records, args.verify, batch)
        if rc:
            return rc
    return 0


def verify(gen, cfg, root: Path, out_root: Path, records, n_check: int, batch: int) -> int:
    """Feed the stored S back through the UNet and require the cached pixels back.

    This is what makes "the conditioning the generator actually consumed" checkable
    rather than asserted: if S were re-encoded at the wrong batch composition, the
    pixels would not come back bit-identical.
    """
    split = "train"
    rs = sorted([r for r in records if r.split == split], key=lambda r: r.index)[:batch]
    if not rs:
        log.warning("nothing to verify"); return 0
    imgs = np.load(root / split / f"images_{rs[0].shard:05d}.npy", mmap_mode="r")
    cond = np.load(out_root / split / f"cond_{rs[0].shard:05d}.npy", mmap_mode="r")
    chunk = rs[:n_check] if n_check < len(rs) else rs
    z = torch.stack([gen.sample_latent(*r.latent_seed_parts)[0] for r in rs])
    pe = torch.from_numpy(np.asarray(cond[[r.offset for r in rs]])).to(
        gen.pipe.device, gen.dtype)
    with torch.no_grad():
        neg, _ = gen.pipe.encode_prompt(
            prompt=[gen.spec.negative_prompt] * len(rs), device=gen.pipe.device,
            num_images_per_prompt=1, do_classifier_free_guidance=False)
    out = gen.pipe(prompt_embeds=pe, negative_prompt_embeds=neg,
                   latents=z.to(gen.pipe.device, gen.dtype),
                   num_inference_steps=gen.spec.num_inference_steps,
                   guidance_scale=gen.spec.guidance_scale, eta=gen.spec.eta,
                   height=gen.spec.resolution, width=gen.spec.resolution,
                   output_type="np").images
    regen = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bad = 0
    for j, r in enumerate(chunk):
        same = np.array_equal(regen[j], np.asarray(imgs[r.offset]))
        diff = int(np.abs(regen[j].astype(int) - np.asarray(imgs[r.offset]).astype(int)).max())
        log.info("verify %s: bit-identical=%s max|diff|=%d", r.sample_id, same, diff)
        bad += not same
    if bad:
        log.error("%d/%d regenerated images differ: the stored S is NOT the conditioning "
                  "that produced the cache", bad, len(chunk))
        return 1
    log.info("all %d regenerated images bit-identical: S is the conditioning the "
             "generator consumed", len(chunk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
