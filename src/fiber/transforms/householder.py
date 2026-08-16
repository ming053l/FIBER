"""Arm E — learned Q as a product of Householder reflections.

    Q = H_m ... H_1 ,   H_i = I - 2 v_i v_iᵀ / ‖v_i‖²

Orthogonality is exact for ANY value of the parameters (no penalty, no
re-projection, no KL term), which is the whole point: measure preservation is
structural, not approximate. Cost is O(m d) per sample, and only the first k
rows are ever read.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.seeding import derive_seed
from .base import Frame


class HouseholderFrame(Frame):
    def __init__(self, d: int, k: int, num_reflectors: int = 128, seed: int = 0,
                 init_scale: float = 1.0, **_):
        super().__init__(d, k)
        g = torch.Generator().manual_seed(derive_seed("householder", d, k, seed) % (2**63 - 1))
        v = torch.randn(num_reflectors, d, generator=g) * init_scale
        self.V = nn.Parameter(v)
        self.num_reflectors = int(num_reflectors)
        self.eps = 1e-12

    def _apply_reflections(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        V = self.V.to(x.dtype)
        order = range(self.num_reflectors - 1, -1, -1) if reverse else range(self.num_reflectors)
        for i in order:
            v = V[i]
            denom = v.dot(v) + self.eps
            coef = (x @ v) * (2.0 / denom)
            x = x - coef.unsqueeze(-1) * v
        return x

    def project(self, z: torch.Tensor) -> torch.Tensor:
        # Q z = H_m ( ... (H_1 z) ), then keep the first k coordinates.
        return self._apply_reflections(z, reverse=False)[..., : self.k]

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        # Qᵀ = H_1 ... H_m  (each H_i is symmetric), applied to [a; 0].
        out = torch.zeros(*a.shape[:-1], self.d, device=a.device, dtype=a.dtype)
        out[..., : self.k] = a
        return self._apply_reflections(out, reverse=True)
