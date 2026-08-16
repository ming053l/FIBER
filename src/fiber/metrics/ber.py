"""Communication metrics on the robust coordinates W_R.

Primary metric is SIGN BER: b_j = 1[W_j > 0]. Under W ~ N(0, I_k) the bits are
uniform and independent, so chance is exactly 0.5 and BER needs no calibration.
Regression MSE and Pearson rho are diagnostics only (PLAN.md §5.2).
"""
from __future__ import annotations

import torch


def sign_bits(w: torch.Tensor) -> torch.Tensor:
    return (w > 0).to(torch.uint8)


def sign_ber(logits: torch.Tensor, w_true: torch.Tensor, reduce: str = "mean"):
    """logits: [N, k] (positive => predict bit 1). Returns scalar or per-sample."""
    pred = (logits > 0).to(torch.uint8)
    err = (pred != sign_bits(w_true)).float()
    if reduce == "mean":
        return float(err.mean())
    if reduce == "per_sample":
        return err.mean(dim=-1)          # the unit a paired bootstrap resamples
    if reduce == "per_coord":
        return err.mean(dim=0)
    raise ValueError(reduce)


def coord_mse(w_hat: torch.Tensor, w_true: torch.Tensor, per_coord: bool = False):
    se = (w_hat - w_true).pow(2)
    return se.mean(0) if per_coord else float(se.mean())


def coord_pearson(w_hat: torch.Tensor, w_true: torch.Tensor) -> torch.Tensor:
    """Per-coordinate rho over the batch dimension: [k]."""
    a = w_hat - w_hat.mean(0, keepdim=True)
    b = w_true - w_true.mean(0, keepdim=True)
    denom = a.norm(dim=0) * b.norm(dim=0)
    return (a * b).sum(0) / denom.clamp_min(1e-12)


def observed_capacity(w_hat: torch.Tensor, w_true: torch.Tensor) -> float:
    """1 - MMSE averaged over the k robust coordinates: the same units as the
    observability spectrum, so the extractor can be compared against Tr(C_obs)/k
    directly. (Var(W_j) = 1 by construction, so this needs no normalisation.)"""
    return float(1.0 - (w_hat - w_true).pow(2).mean())
