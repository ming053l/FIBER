from __future__ import annotations

import torch

from .base import Frame


class IdentityFrame(Frame):
    """Q = I: the robust coordinates are the first k raw latent entries.

    Sanity arm only. Identity loses to any global transform for a trivial
    locality reason (PLAN.md R1), so no gate ever compares against it.
    """

    def __init__(self, d: int, k: int, **_):
        super().__init__(d, k)
        self.register_buffer("_anchor", torch.zeros(1))

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.k]

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*a.shape[:-1], self.d, device=a.device, dtype=a.dtype)
        out[..., : self.k] = a
        return out
