"""Prompt pool.

PLAN.md R6: 100 prompts (InvCISD's text_prompt.yaml) would let the extractor
memorise prompt->image structure and infer z indirectly. FIBER draws from COCO
captions (~590 k) and keeps the prompt sets DISJOINT across splits, so a
held-out-prompt number can be reported as first-class.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..utils.seeding import derive_seed


def load_captions(path: str | Path) -> list[str]:
    """Deterministic, deduplicated, sorted — the order must not depend on json
    iteration order or filesystem state."""
    with open(path) as fh:
        blob = json.load(fh)
    caps = {a["caption"].strip().replace("\n", " ") for a in blob["annotations"]}
    caps = {c for c in caps if 8 <= len(c) <= 250}
    return sorted(caps)


def sample_prompts(pool: list[str], n: int, *seed_parts, exclude: set[str] | None = None) -> list[str]:
    import numpy as np
    exclude = exclude or set()
    candidates = [c for c in pool if c not in exclude]
    if n > len(candidates):
        raise ValueError(f"asked for {n} prompts, pool has {len(candidates)}")
    rng = np.random.default_rng(derive_seed(*seed_parts))
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(idx.tolist())]


def split_prompts(pool: list[str], counts: dict[str, int], salt: str = "fiber-v1") -> dict[str, list[str]]:
    """Disjoint prompt sets, allocated in a fixed order so adding a split later
    cannot reshuffle the existing ones."""
    used: set[str] = set()
    out: dict[str, list[str]] = {}
    for name in sorted(counts):
        out[name] = sample_prompts(pool, counts[name], salt, name, exclude=used)
        used |= set(out[name])
    return out
