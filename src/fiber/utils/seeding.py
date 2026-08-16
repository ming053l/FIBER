"""Deterministic seed derivation and global determinism control.

Every stochastic decision in FIBER (latent draw, prompt draw, attack noise) is
derived from a *string* via blake2s rather than from a mutable global RNG, so a
given (sample, attack) is byte-identical across runs, machines and orderings.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

SEED_BITS = 63
_MASK = (1 << SEED_BITS) - 1


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from any tuple of parts. Order matters; types do not."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & _MASK


def numpy_rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


def torch_generator(*parts: object, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(derive_seed(*parts) % (2**63 - 1))
    return g


def set_determinism(cfg=None, seed: int = 0) -> None:
    """Apply the determinism block of a config (or sane defaults)."""
    det = (cfg or {}).get("determinism", {}) if cfg is not None else {}
    seed = int(det.get("torch_seed", seed))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(det.get("cudnn_benchmark", False))
    torch.backends.cudnn.deterministic = bool(det.get("cudnn_deterministic", True))
    if det.get("use_deterministic_algorithms", False):
        torch.use_deterministic_algorithms(True)
