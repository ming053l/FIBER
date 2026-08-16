"""Arm C2 — a genuinely uniform random k-dimensional subspace (P0-2).

The existing random arms are not this. A signed permutation reads one latent pixel
per coordinate (local); a Walsh-Hadamard row is `±1/sqrt(d)` in every entry (global
but structured). Beating either shows only that the derived directions beat a
particular family. The claim FIBER wants -- that the frozen generative channel has
ANISOTROPIC observability -- is a claim against a uniformly random subspace, so Haar
is the primary reference and the structured families are controls.

Sampling: A ~ N(0,1)^{d x k}, thin QR, then multiply by sign(diag(R)). LAPACK's
Householder QR does not promise a sign convention, and Haar-ness must not depend on
one -- the correction makes it exact by construction rather than by luck.
"""
from __future__ import annotations

import torch

from ..utils.seeding import derive_seed
from .base import Frame


class HaarRandomFrame(Frame):
    def __init__(self, d: int, k: int, seed: int = 0, **_):
        super().__init__(d, k)
        g = torch.Generator().manual_seed(derive_seed("haar", d, k, seed) % (2**63 - 1))
        A = torch.randn(d, k, generator=g, dtype=torch.float64)
        Q, R = torch.linalg.qr(A)
        sign = torch.sign(torch.diagonal(R))
        sign[sign == 0] = 1.0                      # a zero pivot would erase a column
        Q = Q * sign
        self.register_buffer("R", Q.T.float().contiguous())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.R.to(z.dtype).T

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.R.to(a.dtype)
