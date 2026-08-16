"""Attack registry + the reproducibility contract.

Every stochastic attack draws from a generator seeded with

    seed = blake2s(f"{sample_id}|{name}|{severity}|{split_salt}")

so a given (sample, attack) is byte-identical across runs, machines and
dataloader orderings (configs/channels.yaml). `severity` is the attack's own
parameter dict rendered canonically, so bumping jpeg90 -> jpeg85 changes the
noise draw rather than silently reusing the old one.
"""
from __future__ import annotations

import json

import numpy as np

from ..utils.seeding import derive_seed
from .ops import OPS, STOCHASTIC


def severity_key(spec: dict) -> str:
    return json.dumps({k: v for k, v in sorted(spec.items()) if k != "type"}, sort_keys=True)


def attack_seed(sample_id, name: str, spec: dict, split_salt: str = "") -> int:
    return derive_seed(sample_id, name, severity_key(spec), split_salt)


class ChannelBank:
    """Holds the attack table from configs/channels.yaml and applies it."""

    def __init__(self, cfg):
        self.attacks = dict(cfg["attacks"])
        self.train = list(cfg.get("train_attacks", []))
        self.eval = list(cfg.get("eval_attacks", []))
        self.holdout = list(cfg.get("holdout_attacks", []))
        self._validate()

    def _validate(self) -> None:
        for group in (self.train, self.eval, self.holdout):
            for name in group:
                if name not in self.attacks:
                    raise KeyError(f"attack {name!r} listed but not defined")
        for name, spec in self.attacks.items():
            if spec["type"] not in OPS:
                raise KeyError(f"attack {name!r} has unknown type {spec['type']!r}")
        leaked = set(self.holdout) & (set(self.train) | set(self.eval))
        if leaked:
            raise ValueError(f"held-out attacks leaked into train/eval: {sorted(leaked)}")

    def apply(self, img: np.ndarray, name: str, sample_id=0, split_salt: str = "") -> np.ndarray:
        spec = self.attacks[name]
        kind = spec["type"]
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        if kind in STOCHASTIC:
            kwargs["rng"] = np.random.default_rng(attack_seed(sample_id, name, spec, split_salt))
        out = OPS[kind](img, **kwargs)
        if out.shape != img.shape or out.dtype != np.uint8:
            raise RuntimeError(f"attack {name!r} changed shape/dtype: {out.shape} {out.dtype}")
        return out

    def group_of(self, name: str) -> str:
        """The 5 channel groups Gate 3A counts over (>=3 of 5 must improve)."""
        return self.attacks[name]["type"].replace("gaussian_", "")
