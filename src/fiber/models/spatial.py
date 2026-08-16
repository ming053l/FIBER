"""Architecture controls for P0-5.

The default teacher is ResNet18 -> global average pool -> Linear(512, 16384), so its
output covariance has rank at most 512 no matter what the channel carries, and GAP is
biased toward global/semantic content. Measuring "the observability geometry of the
generative channel" with only that decoder measures the decoder as much as the channel.

Two controls, both without global pooling:

* `SpatialTeacher`  — a convolutional pyramid to 64x64 with a 3x3 head, so the output
  covariance is not rank-limited by a bottleneck and local structure has a path to the
  prediction.
* `SpatialExtractor` — the same critique applies to the RECEIVER: if GAP cannot read
  local structure, a global frame can win the BER comparison for a reason that is about
  the extractor rather than the channel. This one keeps the ResNet trunk but replaces
  GAP with a 1x1 channel reduction and a flatten, so spatial position survives to the
  head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class SpatialTeacher(nn.Module):
    """M(Y) ~= E[Z|Y] with NO global pooling.

    Input 512x512x3 -> /8 pyramid -> residual blocks at 64x64 -> 3x3 head -> [4,64,64].
    The output is flattened to d so it is a drop-in for the global teacher, but its
    covariance is not confined to a 512-dimensional affine subspace.

    Trained with MSE and only MSE, for the same reason as the global teacher: the
    certified operator is defined through the conditional MEAN.
    """

    def __init__(self, latent_shape=(4, 64, 64), width: int = 64, blocks: int = 4,
                 in_channels: int = 3):
        super().__init__()
        c1, c2 = width, width * 2
        self.latent_shape = tuple(latent_shape)
        self.d = int(torch.tensor(self.latent_shape).prod())
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[_ResBlock(c2) for _ in range(blocks)])
        self.head = nn.Conv2d(c2, self.latent_shape[0], 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.body(self.stem(x)))
        return h.reshape(h.shape[0], -1)

    @staticmethod
    def loss(m_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(m_hat, z)


class SpatialExtractor(nn.Module):
    """Receiver H(Y) without global average pooling.

    GAP discards where a feature occurred, which is exactly the information a LOCAL
    frame needs. Reading only through GAP could therefore favour global frames for a
    receiver-side reason, so the locked method and the Haar reference are additionally
    evaluated with this extractor (P0-5).
    """

    def __init__(self, k: int, reduce_channels: int = 32, spatial: int = 16,
                 pretrained: bool = False, regression: bool = True, sign: bool = True):
        super().__init__()
        if pretrained:
            raise ValueError("pretrained=True is forbidden for the extractor")
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                   net.layer1, net.layer2, net.layer3, net.layer4)
        self.reduce = nn.Conv2d(512, reduce_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d(spatial)   # fixes the head size, keeps position
        feat = reduce_channels * spatial * spatial
        self.k = int(k)
        self.head_reg = nn.Linear(feat, k) if regression else None
        self.head_sign = nn.Linear(feat, k) if sign else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.pool(self.reduce(self.trunk(x))).flatten(1)
        out = {}
        if self.head_reg is not None:
            out["w_hat"] = self.head_reg(h)
        if self.head_sign is not None:
            out["sign_logits"] = self.head_sign(h)
        return out


class SharedTrunkSpatialTeacher(nn.Module):
    """The GAP-vs-spatial comparison with the TRUNK held fixed.

    `SpatialTeacher` is a different network end to end, so a disagreement with the
    global teacher shows the geometry is architecture-dependent but cannot say the
    global pooling caused it. This one keeps the identical ResNet18 trunk and changes
    only the head:

        global :  trunk -> global average pool -> Linear(512, d)
        this   :  trunk -> 1x1 conv -> upsample to the latent grid -> 3x3 conv -> d

    so the isolated variable is whether spatial position survives to the output.

    The two heads are NOT parameter-matched and cannot be: a dense GAP+FC head needs
    512*d weights to address d outputs while a convolutional head shares weights across
    positions. Parameter counts are reported with every comparison rather than implied
    to be equal.
    """

    def __init__(self, latent_shape=(4, 64, 64), width: int = 64, pretrained: bool = False):
        super().__init__()
        if pretrained:
            raise ValueError("pretrained=True is forbidden for the teacher")
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                   net.layer1, net.layer2, net.layer3, net.layer4)
        self.latent_shape = tuple(latent_shape)
        self.d = int(torch.tensor(self.latent_shape).prod())
        self.reduce = nn.Sequential(nn.Conv2d(512, width, 1, bias=False),
                                    nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(width, self.latent_shape[0], 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(self.trunk(x))
        h = nn.functional.interpolate(h, size=self.latent_shape[1:], mode="bilinear",
                                      align_corners=False)
        return self.head(h).reshape(x.shape[0], -1)
