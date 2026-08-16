"""Paired bootstrap (PLAN.md §7, Gate 3A).

Every arm is evaluated on the SAME (z_i, prompt_i, attack_i), so the correct
statistic is the paired difference, not two independent CIs. 'Do the CIs
overlap' is both wrong and weaker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class PairedResult:
    mean_delta: float          # BER_reference - BER_derived  (positive = derived wins)
    ci_low: float
    ci_high: float
    p_one_sided: float         # P(delta <= 0) under the bootstrap distribution
    relative_reduction: float  # mean_delta / BER_reference
    baseline: float
    treatment: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def paired_bootstrap(baseline_per_sample, treatment_per_sample, resamples: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> PairedResult:
    """Both inputs are per-sample BER over the SAME samples, in the same order."""
    b, t = _to_np(baseline_per_sample), _to_np(treatment_per_sample)
    if b.shape != t.shape:
        raise ValueError(f"paired bootstrap needs aligned samples: {b.shape} vs {t.shape}")
    n = b.shape[0]
    delta = b - t
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    base = float(b.mean())
    return PairedResult(
        mean_delta=float(delta.mean()),
        ci_low=float(lo), ci_high=float(hi),
        p_one_sided=float((boot <= 0).mean()),
        relative_reduction=float(delta.mean() / base) if base > 0 else 0.0,
        baseline=base, treatment=float(t.mean()), n=n,
    )


def seed_average(per_seed) -> np.ndarray:
    """Seeds are REPLICATIONS, never a selection axis (P0-3). Averaging them before the
    sample bootstrap is the minimum; `hierarchical_paired_bootstrap` additionally
    propagates the draw-to-draw uncertainty."""
    arr = np.stack([_to_np(x) for x in per_seed])
    return arr.mean(axis=0)


def hierarchical_paired_bootstrap(reference_by_draw, treatment_by_seed,
                                  resamples: int = 10000, seed: int = 0,
                                  alpha: float = 0.05) -> PairedResult:
    """Resample BOTH the random draws and the test examples.

    The paper's claim is about E_{Q ~ Haar}[BER(Q)], so the particular 8 Haar draws are
    themselves a sample. A bootstrap over test examples alone gives a CI conditional on
    those draws, which is a narrower and different statement. This resamples draws (and
    training seeds) with replacement as well, so the interval covers both sources.

    Report it alongside the conditional-on-draws interval, not instead of it: with few
    draws the hierarchical interval is wide and honest, and the conditional one is
    exactly what it says on the tin.
    """
    R = np.stack([_to_np(x) for x in reference_by_draw])      # [S, N]
    T = np.stack([_to_np(x) for x in treatment_by_seed])      # [M, N]
    if R.shape[1] != T.shape[1]:
        raise ValueError(f"unaligned samples: {R.shape} vs {T.shape}")
    S, N = R.shape
    M = T.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for b in range(resamples):
        ridx = rng.integers(0, S, S)
        tidx = rng.integers(0, M, M)
        sidx = rng.integers(0, N, N)
        boot[b] = float((R[np.ix_(ridx, sidx)].mean(0) - T[np.ix_(tidx, sidx)].mean(0)).mean())
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    base, treat = float(R.mean()), float(T.mean())
    delta = base - treat
    return PairedResult(
        mean_delta=delta, ci_low=float(lo), ci_high=float(hi),
        p_one_sided=float((boot <= 0).mean()),
        relative_reduction=float(delta / base) if base > 0 else 0.0,
        baseline=base, treatment=treat, n=N,
    )


def gate3a_condition(result: PairedResult, cfg: dict) -> dict:
    """Apply the Gate 3A thresholds to one channel group.

    All three must hold: CI95(delta) > 0, relative reduction >= 20%, absolute
    reduction >= 0.02. The absolute floor exists so that 0.005 -> 0.004 cannot be
    sold as '20% better'.
    """
    ci_ok = result.ci_low > float(cfg.get("paired_ci_lower_bound", 0.0))
    rel_ok = result.relative_reduction >= float(cfg.get("target_relative_reduction", 0.20))
    abs_ok = result.mean_delta >= float(cfg.get("min_absolute_reduction", 0.02))
    return {
        "ci_ok": bool(ci_ok), "relative_ok": bool(rel_ok), "absolute_ok": bool(abs_ok),
        "passed": bool(ci_ok and rel_ok and abs_ok), **result.as_dict(),
    }


def gate3a_verdict(conditions: dict, cfg: dict, provisional: bool = False) -> dict:
    """conditions: {channel_group: gate3a_condition(...)}.

    PASS needs >= min_conditions_improved groups. The ONLY kill condition is
    'data-derived is indistinguishable from random' — everything between is
    'inconclusive, keep going', never a silent pass.
    """
    passed = [g for g, c in conditions.items() if c["passed"]]
    need = int(cfg.get("min_conditions_improved", 3))
    kill_thresh = float(cfg.get("kill_if_relative_reduction_below", 0.01))
    best_rel = max((c["relative_reduction"] for c in conditions.values()), default=0.0)
    verdict = "PASS" if len(passed) >= need else ("KILL" if best_rel < kill_thresh else "INCONCLUSIVE")
    if provisional:
        # A pilot is a rehearsal. It may say "worth continuing" or "no sign of an
        # effect at this scale"; it may not close the scientific question.
        verdict = {"PASS": "PROVISIONAL_PASS", "KILL": "PROVISIONAL_FAIL"}.get(verdict, verdict)
    return {
        "verdict": verdict,
        "provisional": bool(provisional),
        "groups_passed": sorted(passed),
        "n_passed": len(passed),
        "n_required": need,
        "best_relative_reduction": best_rel,
    }
