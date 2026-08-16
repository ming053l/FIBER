from __future__ import annotations

from .base import Frame
from .hadamard import HadamardFrame
from .householder import HouseholderFrame
from .identity import IdentityFrame
from .signperm import SignedPermutationFrame
from .spectral import SpectralFrame

FRAMES = {
    "identity": IdentityFrame,
    "signed_permutation": SignedPermutationFrame,
    "hadamard": HadamardFrame,
    "spectral_topk": SpectralFrame,
    "householder": HouseholderFrame,
}


def build_frame(spec: dict, d: int, k: int, seed: int = 0, **overrides) -> Frame:
    """spec is one entry of `fiber.arms` in linear_fiber.yaml."""
    spec = dict(spec)
    kind = spec.pop("type")
    spec.pop("seeds", None)
    if kind not in FRAMES:
        raise KeyError(f"unknown transform type {kind!r}; have {sorted(FRAMES)}")
    spec.update(overrides)
    return FRAMES[kind](d=d, k=k, seed=seed, **spec)
