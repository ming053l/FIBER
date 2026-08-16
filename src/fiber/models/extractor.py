"""Receiver H(Y).

Threat model (linear_fiber.yaml): the receiver observes the received image and
nothing else — not z, not the clean image, not the reference, not the prompt.

Heads are GAUSSIAN, not circular (PLAN.md §5.2): in Level 1, W = QZ ~ N(0,I) is
not a torus variable, and (cos 2piW, sin 2piW) would alias W and W+1 onto the
same target. Circular heads belong to Phase 4/5, after U = Phi(Z).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _resnet18_trunk(pretrained: bool = False) -> tuple[nn.Module, int]:
    from torchvision.models import resnet18
    net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    feat = net.fc.in_features
    net.fc = nn.Identity()
    return net, feat


class Extractor(nn.Module):
    def __init__(self, k: int, arch: str = "resnet18", pretrained: bool = False,
                 regression: bool = True, sign: bool = True):
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"unsupported arch {arch!r}")
        if pretrained:
            # An ImageNet trunk is a semantic shortcut: it would let the extractor
            # read content rather than coordinates (linear_fiber.yaml).
            raise ValueError("pretrained=True is forbidden for the extractor")
        self.trunk, feat = _resnet18_trunk(pretrained)
        self.k = int(k)
        self.head_reg = nn.Linear(feat, k) if regression else None
        self.head_sign = nn.Linear(feat, k) if sign else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        out = {}
        if self.head_reg is not None:
            out["w_hat"] = self.head_reg(h)
        if self.head_sign is not None:
            out["sign_logits"] = self.head_sign(h)
        return out


class Teacher(nn.Module):
    """M_theta(Y) ~= E[Z | Y], trained with MSE and ONLY MSE.

    L2 regression converges to the conditional mean; L1 to the conditional
    median, which would silently corrupt C_obs (PLAN.md §3.3). The loss lives
    here, next to the model, so no training script can quietly swap it.
    """

    def __init__(self, d: int = 16384, arch: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"unsupported arch {arch!r}")
        self.trunk, feat = _resnet18_trunk(pretrained)
        self.head = nn.Linear(feat, d)
        self.d = int(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    @staticmethod
    def loss(m_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(m_hat, z)
