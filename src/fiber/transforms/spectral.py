"""Arm D — the decoder-certified observability directions.

R = top-k ALGEBRAICALLY largest eigenvectors of

    C_cert(f) = C_obs - E[(m-f)(m-f)']  <=  C_obs

produced by `fiber.spectrum.certified`. By Rayleigh-Ritz these maximise the
variance the decoder can certify it recovers; they are the Bayes-optimal MMSE
readout only in the limit f -> E[Z|Y] (PLAN.md §3, P0-1). This class carries the
rows, refuses a file produced by the demoted Cov(f) estimator, checks
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
            if isinstance(blob, dict):
                op = blob.get("operator")
                if op != "certified":
                    raise ValueError(
                        f"{path} was produced by operator={op!r}. Arm D must use the "
                        "decoder-certified operator: Cov(f) is not a lower bound on "
                        "C_obs for an approximate teacher (P0-1).")
                if blob.get("validity_pass") is False:
                    print(f"WARNING: {path} failed the teacher-validity gate; these "
                          "directions are certified by a poor decoder.")
                rows = blob["eigenvectors"]
            else:
                rows = blob
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
