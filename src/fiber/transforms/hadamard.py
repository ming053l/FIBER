"""Structured orthogonal transform  Q = P2 H D P1  (Walsh-Hadamard).

d = 16384 = 2^14, so H is applied by the fast Walsh-Hadamard transform in
O(d log d) with no dense matrix. Each output coordinate is a GLOBAL functional
of z (a sign pattern over all 16384 entries), which is precisely why this arm is
expected to beat identity for the trivial reason recorded in PLAN.md R1.
"""
from __future__ import annotations

import torch

from ..utils.seeding import derive_seed
from .base import Frame


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised fast Walsh-Hadamard transform along the last axis.

    H is symmetric and H H = d I, so the orthogonal transform is fwht(x)/sqrt(d)
    and it is its own inverse (up to that scale).
    """
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(f"fwht needs a power-of-two length, got {d}")
    orig_shape = x.shape
    x = x.reshape(-1, d).clone()
    h = 1
    while h < d:
        x = x.view(-1, d // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, d)
        h *= 2
    return x.reshape(orig_shape)


class HadamardFrame(Frame):
    def __init__(self, d: int, k: int, seed: int = 0, structure: str = "P2_H_D_P1", **_):
        super().__init__(d, k)
        if structure != "P2_H_D_P1":
            raise ValueError(f"unsupported structure {structure!r}")
        g = torch.Generator().manual_seed(derive_seed("hadamard", d, k, seed) % (2**63 - 1))
        perm1 = torch.randperm(d, generator=g)              # P1 (gather)
        diag = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1   # D
        sel = torch.randperm(d, generator=g)[:k]            # the k rows P2 keeps
        self.register_buffer("perm1", perm1)
        self.register_buffer("diag", diag)
        self.register_buffer("sel", sel)
        self.register_buffer("scale", torch.tensor(d, dtype=torch.float64).sqrt().float())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        v = z[..., self.perm1] * self.diag.to(z.dtype)
        v = fwht(v) / self.scale.to(z.dtype)
        return v[..., self.sel]

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        # (P2 H D P1)^T = P1^T D H P2^T   (H symmetric, D diagonal)
        v = torch.zeros(*a.shape[:-1], self.d, device=a.device, dtype=a.dtype)
        v[..., self.sel] = a
        v = fwht(v) / self.scale.to(a.dtype)
        v = v * self.diag.to(a.dtype)
        out = torch.zeros_like(v)
        out[..., self.perm1] = v
        return out
