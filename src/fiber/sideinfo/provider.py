"""What each Phase A arm feeds the receiver as S.

Four arms, one architecture. The only thing that varies is this object.

    blind      no S at all (the legacy Extractor, not a side model)
    null       S_null, the training-split mean, identical for every sample
    shuffled   S from a DIFFERENT sample, within the same cross-fit role
    correct    the sample's own S

Two invariants live here rather than in the training script, because a training script
is where they get lost:

  * the derangement is FIXED ACROSS RECEIVER SEEDS. Receiver seeds replicate optimisation
    randomness; if the placebo pairing moved with them, the between-seed spread would mix
    in "which wrong prompt happened to be attached" and the equivalence margin -- derived
    from between-seed spread -- would be measuring the wrong thing.
  * the derangement is WITHIN CROSS-FIT ROLE. A_teacher donates only to A_teacher,
    A_operator to A_operator, B to B, val to val. Otherwise a teacher would read
    conditioning belonging to samples reserved for the operator or the receiver, which
    breaks the one-sample-one-role discipline the whole protocol rests on.

Every mode returns float32 [77, 768]. The cache is fp16 and S_null is float32, so
without this the null arm would differ from the other two in numeric representation as
well as in content.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .conditioning import (ConditioningStore, different_conditioning_derangement,
                           train_mean_conditioning)

MODES = ("blind", "null", "shuffled", "correct")
DTYPE = np.float32


def _role(rec: dict) -> str:
    return rec.get("crossfit_sub") or rec.get("crossfit") or "-"


class SideConditioning:
    """Mode-dependent S, addressed by sample_id.

    `index_records` must be EVERY record of the split, not the filtered view a particular
    dataset happens to hold: the donor of a sample must not change because a loader was
    constructed with a different attack list or a different cross-fit filter.
    """

    def __init__(self, cache_root: str | Path, split: str, index_records: list[dict],
                 mode: str, s_null: np.ndarray | None = None,
                 derangement_seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"unknown side mode {mode!r}, expected one of {MODES}")
        self.mode, self.split = mode, split
        self.cache_root = Path(cache_root)
        self.derangement_seed = int(derangement_seed)
        recs = [r for r in index_records if r["split"] == split]
        if not recs:
            raise ValueError(f"no index records for split {split!r}")
        self._by_id = {r["sample_id"]: r for r in recs}

        if mode == "blind":
            self.store = None
            return
        self.store = ConditioningStore(cache_root, split)
        if mode == "null":
            self._null = np.asarray(
                s_null if s_null is not None else train_mean_conditioning(cache_root),
                dtype=DTYPE)
            return
        self._donor: dict[str, dict] = {}
        if mode == "shuffled":
            groups: dict[str, list[dict]] = {}
            for r in recs:
                groups.setdefault(_role(r), []).append(r)
            for role in sorted(groups):
                g = sorted(groups[role], key=lambda r: r["index"])
                pi = different_conditioning_derangement(
                    [r["prompt"] for r in g],
                    # split and role, never the receiver seed
                    seed=self.derangement_seed
                    + int(hashlib.blake2s(f"{split}|{role}".encode(),
                                          digest_size=4).hexdigest(), 16) % 10_000)
                for i, r in enumerate(g):
                    self._donor[r["sample_id"]] = g[pi[i]]

    def get(self, sample_id: str) -> np.ndarray | None:
        if self.mode == "blind":
            return None
        if self.mode == "null":
            return self._null
        rec = self._donor[sample_id] if self.mode == "shuffled" else self._by_id[sample_id]
        return np.asarray(self.store.get(rec["shard"], rec["offset"]), dtype=DTYPE)

    def manifest(self) -> dict:
        """Everything an auditor needs to reconstruct which S each sample received."""
        out = {"side_mode": self.mode, "split": self.split, "dtype": str(np.dtype(DTYPE)),
               "derangement_seed": self.derangement_seed,
               "derangement_scope": "within cross-fit role",
               "derangement_depends_on_receiver_seed": False}
        if self.mode == "shuffled":
            pairs = sorted(self._donor.items())
            out["donor_digest"] = hashlib.blake2s(
                "".join(f"{a}->{b['sample_id']}" for a, b in pairs).encode(),
                digest_size=16).hexdigest()
            out["roles"] = sorted({_role(self._by_id[a]) for a, _ in pairs})
            out["same_role_pairs"] = sum(
                _role(self._by_id[a]) == _role(b) for a, b in pairs)
            out["n_pairs"] = len(pairs)
            out["fixed_points"] = sum(a == b["sample_id"] for a, b in pairs)
            out["same_prompt_pairs"] = sum(
                self._by_id[a]["prompt"] == b["prompt"] for a, b in pairs)
        if self.mode == "null":
            out["s_null_digest"] = hashlib.blake2s(
                np.ascontiguousarray(self._null).tobytes(), digest_size=16).hexdigest()
            out["s_null_shape"] = list(self._null.shape)
        return out
