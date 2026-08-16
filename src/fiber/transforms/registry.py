from __future__ import annotations

from .base import Frame
from .haar import HaarRandomFrame
from .hadamard import HadamardFrame
from .householder import FrozenHouseholderOnHaarFrame, HouseholderFrame
from .identity import IdentityFrame
from .rotation import RotatedFrame
from .signperm import SignedPermutationFrame
from .spectral import SpectralFrame

FRAMES = {
    "identity": IdentityFrame,
    "signed_permutation": SignedPermutationFrame,
    "hadamard": HadamardFrame,
    "haar": HaarRandomFrame,
    "frozen_householder_on_haar": FrozenHouseholderOnHaarFrame,
    "random_householder": FrozenHouseholderOnHaarFrame,   # deprecated alias
    "spectral_topk": SpectralFrame,
    "rotated_random": RotatedFrame,
    "rotated_learned": RotatedFrame,
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
