"""Cached native-channel dataset (Phase 1).

We cache only (z, prompt, seed, CLEAN image) and apply the channel in the
dataloader (PLAN.md §2a): caching 9 attack variants would cost ~36 GB and would
freeze the attack set, and X is a cached constant either way so nothing needs to
be differentiable.

Layout under `paths.cache_dir`:

    <split>/images_00000.npy    uint8  [n, 512, 512, 3]   lossless, memmapped
    <split>/latents_00000.npy   float16[n, 4, 64, 64]     exactly what the UNet consumed
    index.jsonl                 one record per sample
    manifest.json               config fingerprint + counts

Split discipline (Gate 1):
  * latent seeds are derived from (salt, split, i) so they are disjoint by
    construction across splits;
  * prompt sets are drawn disjointly across splits, and the held-out-prompt test
    split draws from COCO val2017 instead of train2017;
  * train is partitioned into cross-fit halves A (discovery) and B (evaluation
    extractor), disjoint by construction (PLAN.md §4);
  * A is further split into A_teacher / A_operator (P0-1). Fitting the teacher and
    estimating the certified observability operator on the SAME samples makes the
    operator read back the teacher's own fitting noise, so no sample may play both
    roles. The full chain, with every sample in exactly one role:

        A_teacher  -> teacher fitting
        A_operator -> coordinate discovery, then frozen
        B          -> fresh evaluation extractor
        val        -> all model / hyperparameter selection
        test       -> locked evaluation only
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..channels import ChannelBank
from ..utils.seeding import derive_seed
from .prompts import load_captions, sample_prompts

SALT = "fiber-v1"


@dataclass
class Sample:
    sample_id: str
    split: str
    crossfit: str          # "A", "B" or "-"
    crossfit_sub: str      # "A_teacher", "A_operator", "B" or "-"   (P0-1)
    index: int
    prompt: str
    latent_seed_parts: tuple
    shard: int
    offset: int


def split_sizes(cfg, pilot: bool = False) -> dict[str, int]:
    ds = cfg["dataset"]
    sizes = dict(ds["pilot"] if pilot else ds["splits"])
    # Held-out PROMPT domain test set (Gate 3A: 'held-out latents AND prompts').
    sizes.setdefault("test_heldout_prompts", sizes.get("test", 0))
    return sizes


def build_index(cfg, pilot: bool = False, shard_size: int = 500) -> list[Sample]:
    """Pure function of the config: the same config always yields the same index,
    including prompt assignment, so a rebuild is verifiable."""
    ds = cfg["dataset"]
    pcfg = ds["prompts"]
    sizes = split_sizes(cfg, pilot)
    tag = "pilot" if pilot else "full"

    train_pool = load_captions(pcfg["source"])
    heldout_pool = load_captions(pcfg["heldout_source"])

    in_domain = [s for s in sizes if s != "test_heldout_prompts"]
    used: set[str] = set()
    prompts: dict[str, list[str]] = {}
    for split in sorted(in_domain):                      # fixed order => stable draws
        prompts[split] = sample_prompts(train_pool, sizes[split], SALT, tag, "prompts", split,
                                        exclude=used)
        used |= set(prompts[split])
    if sizes.get("test_heldout_prompts"):
        prompts["test_heldout_prompts"] = sample_prompts(
            heldout_pool, sizes["test_heldout_prompts"], SALT, tag, "prompts", "heldout")

    records: list[Sample] = []
    for split in sorted(sizes):
        n = sizes[split]
        for i in range(n):
            if split == "train":
                # cross-fit halves, deterministic and balanced
                crossfit = "A" if derive_seed(SALT, tag, "crossfit", i) % 2 == 0 else "B"
                if crossfit == "A":
                    # A_teacher / A_operator, derived from i alone so that adding this
                    # sub-split cannot perturb sample_id, shard, offset, latent seed or
                    # prompt -- i.e. an existing image cache stays addressable.
                    sub = ("A_teacher" if derive_seed(SALT, tag, "crossfit_sub", i) % 2 == 0
                           else "A_operator")
                else:
                    sub = "B"
            else:
                crossfit, sub = "-", "-"
            records.append(Sample(
                sample_id=f"{tag}/{split}/{i:06d}", split=split, crossfit=crossfit,
                crossfit_sub=sub, index=i,
                prompt=prompts[split][i],
                latent_seed_parts=(SALT, tag, "latent", split, i),
                shard=i // shard_size, offset=i % shard_size,
            ))
    return records


def verify_index(records: list[Sample]) -> dict:
    """Gate 1 assertions, returned as a dict so the report can quote them."""
    by_split: dict[str, list[Sample]] = {}
    for r in records:
        by_split.setdefault(r.split, []).append(r)

    prompts = {s: {r.prompt for r in rs} for s, rs in by_split.items()}
    seeds = {s: {derive_seed(*r.latent_seed_parts) for r in rs} for s, rs in by_split.items()}

    prompt_overlaps, seed_overlaps = {}, {}
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            po, so = prompts[a] & prompts[b], seeds[a] & seeds[b]
            if po:
                prompt_overlaps[f"{a}|{b}"] = len(po)
            if so:
                seed_overlaps[f"{a}|{b}"] = len(so)

    train = by_split.get("train", [])
    a_ids = {r.sample_id for r in train if r.crossfit == "A"}
    b_ids = {r.sample_id for r in train if r.crossfit == "B"}
    at_ids = {r.sample_id for r in train if r.crossfit_sub == "A_teacher"}
    ao_ids = {r.sample_id for r in train if r.crossfit_sub == "A_operator"}
    all_seeds = [derive_seed(*r.latent_seed_parts) for r in records]

    return {
        "counts": {s: len(rs) for s, rs in by_split.items()},
        "prompt_overlaps": prompt_overlaps,
        "latent_seed_overlaps": seed_overlaps,
        "duplicate_seeds_global": len(all_seeds) - len(set(all_seeds)),
        "crossfit": {"A": len(a_ids), "B": len(b_ids), "overlap": len(a_ids & b_ids)},
        "crossfit_sub": {
            "A_teacher": len(at_ids), "A_operator": len(ao_ids),
            "teacher_operator_overlap": len(at_ids & ao_ids),
            "sub_covers_A": len(at_ids | ao_ids) == len(a_ids),
            "sub_leaks_into_B": len((at_ids | ao_ids) & b_ids),
        },
        "unique_prompts_per_split": {s: len(p) for s, p in prompts.items()},
        "pass": (not prompt_overlaps and not seed_overlaps and not (a_ids & b_ids)
                 and not (at_ids & ao_ids) and (at_ids | ao_ids) == a_ids
                 and len(all_seeds) == len(set(all_seeds))),
    }


# ---------------------------------------------------------------- runtime


class FiberDataset(torch.utils.data.Dataset):
    """Reads cached (clean X, z) and applies the channel on the fly.

    mode="fixed"   : one named attack for every item (evaluation)
    mode="sampled" : attack drawn per item from `attacks`, deterministically from
                     (sample_id, epoch_salt) so an epoch is reproducible
    """

    def __init__(self, root, split: str, bank: ChannelBank, attacks=None,
                 mode: str = "sampled", crossfit: str | None = None,
                 crossfit_sub: str | None = None,
                 epoch_salt: str = "", normalise: bool = True):
        self.root = Path(root)
        self.split = split
        self.bank = bank
        self.attacks = list(attacks or bank.train)
        self.mode = mode
        self.epoch_salt = epoch_salt
        self.normalise = normalise
        if mode == "fixed" and len(self.attacks) != 1:
            raise ValueError("mode='fixed' takes exactly one attack")

        with open(self.root / "index.jsonl") as fh:
            recs = [json.loads(line) for line in fh]
        recs = [r for r in recs if r["split"] == split]
        if crossfit:
            recs = [r for r in recs if r["crossfit"] == crossfit]
        if crossfit_sub:
            if any("crossfit_sub" not in r for r in recs):
                raise KeyError("index.jsonl predates the A_teacher/A_operator split "
                               "(P0-1): rebuild it with scripts/cache_native_dataset.py "
                               "(images are not regenerated)")
            recs = [r for r in recs if r["crossfit_sub"] == crossfit_sub]
        self.records = sorted(recs, key=lambda r: r["index"])
        self._shards: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _shard(self, shard: int):
        if shard not in self._shards:
            d = self.root / self.split
            self._shards[shard] = (
                np.load(d / f"images_{shard:05d}.npy", mmap_mode="r"),
                np.load(d / f"latents_{shard:05d}.npy", mmap_mode="r"),
            )
        return self._shards[shard]

    def attack_for(self, rec: dict) -> str:
        if self.mode == "fixed":
            return self.attacks[0]
        i = derive_seed(rec["sample_id"], self.epoch_salt) % len(self.attacks)
        return self.attacks[i]

    def __getitem__(self, i: int):
        rec = self.records[i]
        images, latents = self._shard(rec["shard"])
        img = np.asarray(images[rec["offset"]])
        z = np.asarray(latents[rec["offset"]]).astype(np.float32)
        name = self.attack_for(rec)
        y = self.bank.apply(img, name, sample_id=rec["sample_id"], split_salt=self.split)
        x = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1).float() / 255.0
        if self.normalise:
            x = (x - 0.5) / 0.5
        return {
            "image": x,
            "z": torch.from_numpy(z).reshape(-1),
            "attack": name,
            "sample_id": rec["sample_id"],
            "index": rec["index"],
        }


def write_index(records: list[Sample], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "index.jsonl", "w") as fh:
        for r in records:
            d = asdict(r)
            d["latent_seed_parts"] = list(d["latent_seed_parts"])
            d["latent_seed"] = derive_seed(*r.latent_seed_parts)
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
