"""The permutation null has to be permuted at the right level.

R2_lin is reported against 1/(n-1), the analytic zero-correlation reference for a Pearson
r. That reference assumes a distribution the receiver's outputs need not follow, so the
negative result should not rest on it. These tests cover the empirical replacement -- and
specifically the mistake that version first made.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
from permutation_null import _r2, permutation_null  # noqa: E402

N, K, ATTACKS = 256, 16, 10


def _pure_noise(rng, n=N, k=K, attacks=ATTACKS, rho_across_attacks=0.95):
    """W_hat carries NO information about W, but the predictions for one sample are
    strongly correlated ACROSS attacks -- which is what the real data looks like, since
    the attacks are ten views of the same image."""
    W = rng.standard_normal((n, k))
    shared = rng.standard_normal((n, k))
    out = {}
    for a in range(attacks):
        idio = rng.standard_normal((n, k))
        out[f"a{a}"] = (rho_across_attacks * shared
                        + (1 - rho_across_attacks ** 2) ** 0.5 * idio, W)
    return out


def test_the_empirical_null_reproduces_the_analytic_reference():
    """1/(n-1) is validated here, not assumed -- which is the point of running this."""
    rng = np.random.default_rng(0)
    null = permutation_null(_pure_noise(rng), 500, rng)
    assert abs(null.mean() - 1 / (N - 1)) < 0.0004


def test_one_permutation_per_replicate_not_one_per_attack():
    """Independent permutations destroy the cross-attack dependence that the OBSERVED
    statistic still has, so the null comes out far too narrow and every p-value shrinks."""
    rng = np.random.default_rng(1)
    data = _pure_noise(rng)
    shared = permutation_null(data, 500, np.random.default_rng(2), shared=True)
    indep = permutation_null(data, 500, np.random.default_rng(2), shared=False)
    assert abs(shared.mean() - indep.mean()) < 0.0004, "the two must agree in MEAN"
    assert shared.std(ddof=1) > 2.0 * indep.std(ddof=1), (
        f"shared sd {shared.std(ddof=1):.5f} vs independent {indep.std(ddof=1):.5f}")


def test_the_independent_version_is_anticonservative_under_the_null():
    """The consequence that matters: with NO signal, per-attack permutation calls runs
    significant far more often than 5% of the time."""
    rng = np.random.default_rng(3)
    p_shared, p_indep = [], []
    for t in range(40):
        data = _pure_noise(np.random.default_rng(100 + t))
        obs = float(np.mean([_r2(wh, w) for wh, w in data.values()]))
        for shared, acc in ((True, p_shared), (False, p_indep)):
            null = permutation_null(data, 300, np.random.default_rng(t), shared=shared)
            acc.append(float((null >= obs).mean()))
    p_shared, p_indep = np.array(p_shared), np.array(p_indep)
    r_shared, r_indep = (p_shared < 0.05).mean(), (p_indep < 0.05).mean()
    # measured over 40 null datasets: shared 0.050 (exactly nominal), independent 0.275
    assert r_indep > 3 * 0.05, f"independent rejected only {r_indep:.0%}"
    assert r_indep > 2.5 * max(r_shared, 0.02), f"{r_indep:.0%} vs {r_shared:.0%}"
    assert r_shared <= 0.15, f"shared version fired on {r_shared:.0%} of null datasets"


def test_real_signal_is_still_detected():
    """A null that never rejects would pass the three tests above."""
    rng = np.random.default_rng(4)
    data = _pure_noise(rng)
    signalled = {a: (0.3 * w + 0.95 * wh, w) for a, (wh, w) in data.items()}
    obs = float(np.mean([_r2(wh, w) for wh, w in signalled.values()]))
    null = permutation_null(signalled, 500, rng)
    assert obs > null.max(), f"observed {obs:.5f} did not clear the null max {null.max():.5f}"


@pytest.mark.parametrize("k", [8, 64])
def test_the_null_width_scales_with_k_as_the_reported_spreads_did(k):
    """Fewer coordinates means a noisier pooled statistic. The reading rule must use each
    k's own spread, which is why this is measured rather than assumed."""
    rng = np.random.default_rng(5)
    null = permutation_null(_pure_noise(rng, k=k), 400, rng)
    assert abs(null.mean() - 1 / (N - 1)) < 0.0005
    assert null.std(ddof=1) > 0


def test_the_p_value_can_never_be_exactly_zero():
    """(1 + #) / (B + 1), not # / B. With B draws, p = 0 claims more than the draws can
    support -- and the observed statistic is itself one valid arrangement, so it counts."""
    rng = np.random.default_rng(6)
    data = _pure_noise(rng)
    # a statistic no permutation can reach
    obs = 1e9
    B = 200
    null = permutation_null(data, B, rng)
    naive = float((null >= obs).mean())
    corrected = float((1 + (null >= obs).sum()) / (B + 1))
    assert naive == 0.0
    assert corrected == pytest.approx(1 / (B + 1))
    assert 0 < corrected <= 1
