from __future__ import annotations

import torch

from ..utils.seeding import derive_seed
from .base import Frame


class SignedPermutationFrame(Frame):
    """Random signed permutation: W_j = s_j z_{pi(j)}.

    Exactly orthogonal by construction, and a *local* random baseline: each
    robust coordinate still reads a single latent pixel. Contrast with the
    Hadamard arm, which is global. The gap between the two is itself the
    measurement of R1's locality effect.
    """

    def __init__(self, d: int, k: int, seed: int = 0, **_):
        super().__init__(d, k)
        g = torch.Generator().manual_seed(derive_seed("signperm", d, k, seed) % (2**63 - 1))
        idx = torch.randperm(d, generator=g)[:k]
        signs = (torch.randint(0, 2, (k,), generator=g).float() * 2 - 1)
        self.register_buffer("idx", idx)
        self.register_buffer("signs", signs)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., self.idx] * self.signs.to(z.dtype)

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*a.shape[:-1], self.d, device=a.device, dtype=a.dtype)
        out[..., self.idx] = a * self.signs.to(a.dtype)
        return out
