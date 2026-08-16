"""Arm D — the Generative Observability Spectrum.

R = top-k eigenvectors of C_obs = Cov(E[Z|Y]).  By Rayleigh-Ritz this is the
Bayes-optimal k-dimensional linear readout under MMSE (PLAN.md §3.1). The rows
come from `fiber.spectrum.observability`; this class only carries them, checks
orthonormality and re-orthonormalises if the estimator drifted.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .base import Frame


class SpectralFrame(Frame):
    def __init__(self, d: int, k: int, rows: torch.Tensor | None = None,
                 path: str | Path | None = None, reorthonormalise: bool = True, **_):
        super().__init__(d, k)
        if rows is None:
            if path is None:
                raise ValueError("SpectralFrame needs either rows= or path=")
            blob = torch.load(path, map_location="cpu", weights_only=True)
            rows = blob["eigenvectors"] if isinstance(blob, dict) else blob
        rows = rows[:k].float()
        if rows.shape != (k, d):
            raise ValueError(f"expected rows {(k, d)}, got {tuple(rows.shape)}")
        if reorthonormalise:
            # QR on the transpose: numerically exact orthonormal rows spanning
            # the same subspace. The subspace is the scientific object; the
            # in-subspace basis is not.
            q, _ = torch.linalg.qr(rows.T.double())
            rows = q.T.float()
        self.register_buffer("R", rows.contiguous())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.R.to(z.dtype).T

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.R.to(a.dtype)
