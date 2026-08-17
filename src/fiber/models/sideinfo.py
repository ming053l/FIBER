"""Side-information receiver and teacher: f(Y, S), late fusion.

Preregistered in reports/blind_vs_sideinfo.md and frozen:

    S in R^{77 x 768}
      -> shared Linear(768 -> 64) applied per token        T in R^{77 x 64}
      -> single learned-query attention pooling            h_S in R^{64}
    [h_Y (512) ; h_S (64)] -> the same prediction heads

Deliberately small, and deliberately not a Transformer branch. Phase A asks whether
side information does anything at all, not which prompt encoder is best, and a large
side branch would reopen the capacity/overfitting question on a 2000-sample pilot that
the four-arm ladder exists to close.

LATE fusion: S never touches the image trunk. Early fusion would make any improvement
inseparable from "conditioning rewrote the image feature extractor".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .extractor import Extractor, Teacher, _resnet18_trunk

S_TOKENS, S_FEATURES, S_DIM = 77, 768, 64


class SideEncoder(nn.Module):
    """S -> h_S in R^64.

    The pooling query is ZERO-INITIALISED, so at init the softmax is exactly uniform and
    the encoder starts as a mean-pooled linear projection. That matters for the same
    reason arm C3 (frozen Householder, paired init) matters: the parameterisation
    contributes nothing at step zero, so anything it gains is learning rather than a
    different starting point.
    """

    def __init__(self, tokens: int = S_TOKENS, features: int = S_FEATURES,
                 dim: int = S_DIM):
        super().__init__()
        self.proj = nn.Linear(features, dim)          # shared across token positions
        self.query = nn.Parameter(torch.zeros(dim))   # uniform attention at init
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        t = self.proj(s)                              # [B, tokens, dim]
        logits = (t @ self.query) / math.sqrt(self.dim)
        alpha = torch.softmax(logits, dim=1)
        return (alpha.unsqueeze(-1) * t).sum(1)       # [B, dim]

    def attention(self, s: torch.Tensor) -> torch.Tensor:
        t = self.proj(s)
        return torch.softmax((t @ self.query) / math.sqrt(self.dim), dim=1)


class SideExtractor(nn.Module):
    """The receiver, with side information concatenated after the image trunk."""

    def __init__(self, k: int, arch: str = "resnet18", pretrained: bool = False,
                 regression: bool = True, sign: bool = True, side_dim: int = S_DIM,
                 s_tokens: int = S_TOKENS, s_features: int = S_FEATURES):
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"unsupported arch {arch!r}")
        if pretrained:
            raise ValueError("pretrained=True is forbidden for the extractor")
        self.trunk, feat = _resnet18_trunk(False)
        self.side = SideEncoder(s_tokens, s_features, side_dim)
        self.k = int(k)
        fused = feat + side_dim
        self.head_reg = nn.Linear(fused, k) if regression else None
        self.head_sign = nn.Linear(fused, k) if sign else None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> dict[str, torch.Tensor]:
        if s is None:
            raise ValueError("a side receiver requires S; the blind arm is Extractor")
        h = torch.cat([self.trunk(x), self.side(s)], dim=1)
        out = {}
        if self.head_reg is not None:
            out["w_hat"] = self.head_reg(h)
        if self.head_sign is not None:
            out["sign_logits"] = self.head_sign(h)
        return out


class SideTeacher(nn.Module):
    """f_s(Y, S) ~= E[Z | Y, S], MSE only, same rule as Teacher.

    Note the capacity accounting differs by role: the teacher's head is Linear(., d) with
    d = 16384, so concatenating 64 dimensions adds 64 * 16384 = 1,048,576 weights, while
    the receiver's head is Linear(., k) and adds only 64 * k per head. Report the total
    per arm, never "the side branch is 49K parameters".
    """

    def __init__(self, d: int = 16384, arch: str = "resnet18", pretrained: bool = False,
                 side_dim: int = S_DIM, s_tokens: int = S_TOKENS,
                 s_features: int = S_FEATURES):
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"unsupported arch {arch!r}")
        self.trunk, feat = _resnet18_trunk(pretrained)
        self.side = SideEncoder(s_tokens, s_features, side_dim)
        self.head = nn.Linear(feat + side_dim, d)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.trunk(x), self.side(s)], dim=1))


def parameter_report(model: nn.Module) -> dict:
    """Per-arm totals, so a preregistration cannot quote the side branch alone."""
    total = sum(p.numel() for p in model.parameters())
    side = sum(p.numel() for n, p in model.named_parameters() if n.startswith("side"))
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head"))
    return {"total": total, "side_branch": side, "heads": head,
            "trunk": total - side - head}
