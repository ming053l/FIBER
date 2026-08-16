"""Measure-preserving coordinate frames.

Level 1 uses W = QZ with QᵀQ = I. Only the k *robust* coordinates W_R ever
matter: the remaining d-k coordinates are never read, and measure preservation
holds for ANY orthogonal completion of the first k rows. So the object we
actually implement is not a d x d matrix (d = 16384 would be a 1 GB dense
matrix) but a **k-frame**

    R in R^{k x d},  R Rᵀ = I_k

with two operations:

    project(z) = R z          (transmit side: which coordinates we use)
    expand(a)  = Rᵀ a         (adjoint; used for orthonormality checks and for
                               materialising the frame rows)

Everything downstream (extractor targets, sign BER, the observability spectrum)
is expressed in terms of these two.
"""
from __future__ import annotations

import abc

import torch


def flatten_latent(z: torch.Tensor) -> torch.Tensor:
    """[B,4,64,64] (or [4,64,64]) -> [B,16384] in C order."""
    if z.dim() == 3:
        z = z.unsqueeze(0)
    return z.reshape(z.shape[0], -1)


def unflatten_latent(z: torch.Tensor, shape=(4, 64, 64)) -> torch.Tensor:
    return z.reshape(z.shape[0], *shape)


class Frame(torch.nn.Module, abc.ABC):
    """k orthonormal rows of a measure-preserving transform on R^d."""

    def __init__(self, d: int, k: int):
        super().__init__()
        if k > d:
            raise ValueError(f"k={k} > d={d}")
        self.d = int(d)
        self.k = int(k)

    @abc.abstractmethod
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """[..., d] -> [..., k]"""

    @abc.abstractmethod
    def expand(self, a: torch.Tensor) -> torch.Tensor:
        """[..., k] -> [..., d]   (the adjoint of project)"""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.project(z)

    # -- diagnostics -----------------------------------------------------
    def rows(self) -> torch.Tensor:
        """Materialise R as [k, d]. k*d is ~1M floats at k=64, fine."""
        eye = torch.eye(self.k, device=self.device, dtype=self.dtype)
        return self.expand(eye)

    def gram(self) -> torch.Tensor:
        """R Rᵀ, computed without ever forming a d x d matrix."""
        eye = torch.eye(self.k, device=self.device, dtype=self.dtype)
        return self.project(self.expand(eye))

    def orthonormality_error(self) -> float:
        """‖R Rᵀ − I_k‖_inf. Gate 3A requires < 1e-5."""
        g = self.gram()
        eye = torch.eye(self.k, device=g.device, dtype=g.dtype)
        return float((g - eye).abs().max())

    @property
    def device(self) -> torch.device:
        for p in self.buffers():
            return p.device
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        for p in self.buffers():
            if p.is_floating_point():
                return p.dtype
        for p in self.parameters():
            return p.dtype
        return torch.float32
