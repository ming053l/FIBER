"""S = C(P): the conditioning tensor, and the two objects the placebo arms need.

Registered in reports/blind_vs_sideinfo.md:

  * `S_null` is the TRAINING-SPLIT MEAN conditioning tensor, taken per token position and
    per feature so the shape is preserved, computed once on train and reused unchanged.
  * `S_shuffled` comes from a DIFFERENT-CONDITIONING derangement: the constraint is on
    the value, `S_pi(i) != S_i`, not merely on the index, `pi(i) != i`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class ConditioningStore:
    """Read-only access to the sharded conditioning cache, addressed like the images."""

    def __init__(self, cache_root: str | Path, split: str):
        self.root = Path(cache_root) / "conditioning" / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"{self.root} missing: run scripts/cache_conditioning.py. S must be "
                "replayed against the chunking that produced the images, so it cannot "
                "be re-encoded on the fly.")
        self._shards: dict[int, np.ndarray] = {}

    def get(self, shard: int, offset: int) -> np.ndarray:
        if shard not in self._shards:
            self._shards[shard] = np.load(self.root / f"cond_{shard:05d}.npy",
                                          mmap_mode="r")
        return np.asarray(self._shards[shard][offset])


def train_mean_conditioning(cache_root: str | Path, split: str = "train") -> np.ndarray:
    """S_null. Per token position and per feature -- never pooled first, so it has the
    exact shape of a real S and the side encoder cannot tell them apart structurally.

    Accumulated in float64 and returned in float32: a mean over thousands of float16
    tensors loses precision fast if summed in float16, and S_null is a fixed constant
    reused for every sample, so any error in it is systematic rather than averaging out.
    """
    root = Path(cache_root) / "conditioning" / split
    shards = sorted(root.glob("cond_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no conditioning shards under {root}")
    total, n = None, 0
    for sh in shards:
        arr = np.load(sh, mmap_mode="r")
        block = np.asarray(arr, dtype=np.float64).sum(axis=0)
        total = block if total is None else total + block
        n += arr.shape[0]
    return (total / n).astype(np.float32)


def _digests(prompts: list[str]) -> list[str]:
    """Conditioning identity is the PROMPT, not the bytes of S.

    The registered rule is `S_pi(i) != S_i`. Taken literally on the stored tensors it
    would be the weaker test: the text encoder is not batch-composition invariant, so two
    samples with the SAME prompt in different chunks hold numerically different S -- and
    swapping them would pass a byte comparison while handing the placebo semantically
    correct side information, which is exactly what the guard exists to prevent. The
    prompt is the identity that decides information content, so it is what is digested.
    Documented rather than silent (PLAN.md discipline).
    """
    return [hashlib.blake2s(p.encode(), digest_size=16).hexdigest() for p in prompts]


def different_conditioning_derangement(prompts: list[str], seed: int = 0) -> np.ndarray:
    """pi with prompt(pi(i)) != prompt(i) for every i.

    Construction: order the indices grouped by conditioning, shuffle within the ordering
    with a fixed seed, then rotate by the size of the largest group. If positions p and
    p + m (mod n) shared a group, that group would occupy m + 1 consecutive positions,
    contradicting m being its maximum size -- so the rotation is a valid derangement by
    construction rather than by rejection sampling, which would not terminate when
    collisions are dense.

    Raises when no valid derangement exists (one conditioning holding more than half the
    split). It must never quietly fall back to an ordinary shuffle: that is precisely the
    case where the placebo would leak most.
    """
    n = len(prompts)
    if n < 2:
        raise ValueError("a derangement needs at least two samples")
    digs = _digests(prompts)
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(digs):
        groups.setdefault(d, []).append(i)
    m = max(len(g) for g in groups.values())
    if 2 * m > n:
        raise ValueError(
            f"no different-conditioning derangement exists: one conditioning covers "
            f"{m} of {n} samples. Refusing to fall back to an ordinary shuffle -- the "
            "placebo would then receive correct side information for those samples.")
    rng = np.random.default_rng(seed)
    order: list[int] = []
    for d in sorted(groups):                       # sorted: independent of dict order
        g = list(groups[d])
        rng.shuffle(g)
        order.extend(g)
    order = np.asarray(order)
    rotated = np.roll(order, m)
    pi = np.empty(n, dtype=np.int64)
    pi[order] = rotated                            # sample at position p receives S from p+m
    assert not any(digs[pi[i]] == digs[i] for i in range(n))
    return pi


def derangement_manifest(prompts: list[str], pi: np.ndarray, seed: int) -> dict:
    """What has to travel with the placebo arm so it is auditable, not just reproducible
    in principle."""
    digs = _digests(prompts)
    return {
        "n": len(prompts), "seed": int(seed),
        "distinct_conditionings": len(set(digs)),
        "largest_group": max(sum(1 for d in digs if d == x) for x in set(digs)),
        "fixed_points": int((pi == np.arange(len(pi))).sum()),
        "same_conditioning_pairs": int(sum(digs[pi[i]] == digs[i] for i in range(len(pi)))),
        "permutation_digest": hashlib.blake2s(np.ascontiguousarray(pi).tobytes(),
                                              digest_size=16).hexdigest(),
        "identity_rule": "blake2s(prompt); NOT the bytes of S -- see conditioning.py",
    }
