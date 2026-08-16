"""Arm E — learned coordinates as an orthogonal transform of a frozen Haar frame.

    R_E = Q_phi H,      Q_phi in O(d) a Householder product,   H a FROZEN Haar frame

Orthogonality is exact for ANY value of the parameters (no penalty, no reprojection,
no KL term), which is the whole point: measure preservation is structural, not
approximate. Only the first k rows are ever read.

**Why a Haar base (P0-4).** Measured at the configured size (d = 16384, m = 128,
`scripts/diagnose_frames.py`), a Householder product of randomly drawn reflectors is
nearly the IDENTITY: E[r_0^2] = 0.969, participation ratio 1.1, against Haar's 5.8e-5
and 5470. Each reflection displaces e_0 by about 2/sqrt(d); a random walk of m steps
covers only sqrt(m)*2/sqrt(d) = 0.18, and a heuristic mixing-scale argument
(sqrt(m)*2/sqrt(d) ~ 1) puts the crossover near m ~ d/4 -- an order-of-magnitude
estimate, not a theorem about Householder mixing times.

So without a base, arm E would start at identity, the one point R1 says loses for a
trivial locality reason, and a failure would be indistinguishable from an optimiser
that never escaped a bad start. With the base, arm E starts exactly AT the gate
denominator and any gain over Haar is attributable to learning.

**Exactly, not nearly.** The reflectors are initialised in identical adjacent PAIRS.
Since H(v)H(v) = I, the product is the identity at initialisation up to floating-point
rounding -- not merely close to it. Both members of each pair are independent trainable
parameters that happen to start equal, so the first gradient step separates them and
the frame leaves the Haar point.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.seeding import derive_seed
from .base import Frame
from .haar import HaarRandomFrame


class HouseholderFrame(Frame):
    def __init__(self, d: int, k: int, num_reflectors: int = 128, seed: int = 0,
                 init_scale: float = 1.0, trainable: bool = True,
                 base: str | None = "haar", paired_init: bool = True, **_):
        super().__init__(d, k)
        if paired_init and num_reflectors % 2:
            raise ValueError(f"paired_init needs an even reflector count, got {num_reflectors}")
        if base not in (None, "none", "haar"):
            raise ValueError(f"unsupported base {base!r}")
        self.has_base = base == "haar"
        if self.has_base:
            # frozen: the learned part is Q_phi, never H
            self.register_buffer("B", HaarRandomFrame(d, k, seed=seed).R.clone())
        # The seed string is deliberately shared with the frozen control arm, so
        # RandomHouseholderFrame(seed=s) IS this frame at initialisation. That makes
        # the control architecture-matched in the strict sense: same parameterisation,
        # same reflector count, same init draw -- only the optimisation differs.
        g = torch.Generator().manual_seed(derive_seed("householder", d, k, seed) % (2**63 - 1))
        if paired_init:
            # identical adjacent pairs => H(v)H(v) = I => Q_phi(0) = I exactly
            half = torch.randn(num_reflectors // 2, d, generator=g) * init_scale
            v = half.repeat_interleave(2, dim=0)
        else:
            v = torch.randn(num_reflectors, d, generator=g) * init_scale
        if trainable:
            self.V = nn.Parameter(v)
        else:
            self.register_buffer("V", v)
        self.num_reflectors = int(num_reflectors)
        self.eps = 1e-12

    def _apply_reflections(self, x: torch.Tensor, reverse: bool) -> torch.Tensor:
        V = self.V.to(x.dtype)
        order = range(self.num_reflectors - 1, -1, -1) if reverse else range(self.num_reflectors)
        for i in order:
            v = V[i]
            denom = v.dot(v) + self.eps
            coef = (x @ v) * (2.0 / denom)
            x = x - coef.unsqueeze(-1) * v
        return x

    def project(self, z: torch.Tensor) -> torch.Tensor:
        q = self._apply_reflections(z, reverse=False)          # Q z
        if self.has_base:
            return q @ self.B.to(q.dtype).T                    # H Q z
        return q[..., : self.k]

    def expand(self, a: torch.Tensor) -> torch.Tensor:
        # Qᵀ = H_1 ... H_m (each H_i is symmetric), applied to Hᵀa (or to [a; 0]).
        if self.has_base:
            x = a @ self.B.to(a.dtype)
        else:
            x = torch.zeros(*a.shape[:-1], self.d, device=a.device, dtype=a.dtype)
            x[..., : self.k] = a
        return self._apply_reflections(x, reverse=True)


class FrozenHouseholderOnHaarFrame(HouseholderFrame):
    """Arm C3 — the learned arm's architecture and starting point, frozen.

    With the Haar base and the paired initialisation, this is bit-identical to a Haar
    draw at the same seed. That is the honest statement rather than a defect: at
    initialisation the parameterisation contributes nothing, so any arm-E gain is
    attributable to learning alone. It is retained as an executable check of exactly
    that -- if it ever diverges from arm C2 at the same seed, the base or the paired
    init is broken.

    The name is deliberate: `random_householder` invited reading it as a
    random-subspace baseline, which it never was.

    The name is deliberate: `random_householder` invited reading it as a
    random-subspace baseline, which it never was.

    MEASURED at the configured size (d = 16384, m = 128, 100 draws), this frame is
    not merely "not Haar" -- it is very nearly the IDENTITY:

        E[r_0^2]            0.969        (Haar: 5.8e-5, uniform ideal 1/d = 6.1e-5)
        participation ratio 1.1          (Haar: 5470, uniform ideal d/3 = 5461)

    Each reflection displaces e_0 by about 2/sqrt(d) = 0.016, so m random reflections
    move it by only ~sqrt(m)*2/sqrt(d) = 0.18; reaching a Haar-like draw would need
    m ~ d/4 = 4096. So this control behaves like arm A, and PLAN.md R1 applies to it.

    This concerns the INITIALISATION, not representational capacity: the reflector
    count controls the expressivity of the learned frame, and `scripts/audit_reflector_
    capacity.py` (B3) audits empirically whether the configured count suffices at each
    k rather than asserting a theorem without a citation. What it does mean is that the
    LEARNED arm starts essentially at identity -- the one starting point R1 says is bad
    for a trivial locality reason -- so an arm-E failure would be confounded with a poor
    init. See PLAN.md and the P0-4 entry in reports/p0_fix_plan.md.
    """

    def __init__(self, d: int, k: int, num_reflectors: int = 128, seed: int = 0, **kw):
        kw.pop("trainable", None)
        super().__init__(d, k, num_reflectors=num_reflectors, seed=seed, trainable=False, **kw)
