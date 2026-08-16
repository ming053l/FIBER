"""P0-7 — separating WHERE information survives from HOW it should be coded.

Rayleigh-Ritz identifies a top-k SUBSPACE. Sign coding depends on a particular BASIS
inside it. For any A with AᵀA = I,

    span(A V) = span(V),      so      D_cert(A V) = D_cert(V)

exactly, while the sign bits `1[(A V z)_j > 0]` change. So `BER_learned < BER_spectral`
does not by itself mean a better subspace was found -- it may be a better coding basis
in the same subspace. Three arms hold the subspace fixed and vary only the basis:

    D1  V                    the certified eigenbasis
    D2  A_rand V             random basis, A_rand ~ Haar on O(k), MANY draws
    D3  A_phi V              learned basis, A_phi = expm(S - Sᵀ) in SO(k)

`expm` of a skew-symmetric matrix has determinant +1, so D3 explores **SO(k)**, not all
of O(k). The reflection component is not included: for sign coding a reflection largely
amounts to flipping which bits are complemented, and naming the parameterisation
honestly is worth more than the extra generality. D2 does cover full O(k), since a QR
of a Gaussian matrix has determinant of either sign.

The ambient frame V is a BUFFER here, never a parameter. If gradients could reach it,
D3 would quietly stop being a within-subspace rotation and become subspace discovery
again, and the whole decomposition would collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.seeding import derive_seed
from .base import Frame


def haar_orthogonal(k: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """A ~ Haar on O(k). QR of a Gaussian with the sign correction; det is either sign,
    so this covers reflections as well as rotations."""
    g = torch.Generator().manual_seed(derive_seed("rotation", k, seed) % (2**63 - 1))
    q, r = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=dtype))
    sign = torch.sign(torch.diagonal(r))
    sign[sign == 0] = 1.0
    return q * sign


class RotatedFrame(Frame):
    """R = A V with V FROZEN and A an in-subspace orthogonal map."""

    def __init__(self, base: Frame | torch.Tensor, k: int | None = None,
                 mode: str = "random", seed: int = 0, **_):
        rows = base.rows().detach() if isinstance(base, Frame) else base
        k = int(k or rows.shape[0])
        rows = rows[:k].contiguous()
        super().__init__(d=rows.shape[1], k=k)
        if mode not in ("identity", "random", "learned"):
            raise ValueError(f"unknown rotation mode {mode!r}")
        self.mode = mode
        # buffer, not parameter: gradients must never reach the ambient subspace
        self.register_buffer("V", rows.float())
        if mode == "learned":
            g = torch.Generator().manual_seed(derive_seed("so_k", k, seed) % (2**63 - 1))
            # S starts at zero => A(0) = I => D3 starts at D1, so any gain is the
            # rotation's doing and not a different starting basis
            self.S = nn.Parameter(torch.zeros(k, k) + 0.0 * torch.randn(k, k, generator=g))
        elif mode == "random":
            self.register_buffer("A_fixed", haar_orthogonal(k, seed).float())
        else:
            self.register_buffer("A_fixed", torch.eye(k))

    def rotation(self) -> torch.Tensor:
        if self.mode == "learned":
            S = self.S
            return torch.matrix_exp(S - S.T)      # skew-symmetric => SO(k)
        return self.A_fixed

    def project(self, z: torch.Tensor) -> torch.Tensor:
        A = self.rotation().to(z.dtype)
        return (z @ self.V.to(z.dtype).T) @ A.T   # A (V z)

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        A = self.rotation().to(a.dtype)
        return (a @ A) @ self.V.to(a.dtype)       # Vᵀ Aᵀ a
