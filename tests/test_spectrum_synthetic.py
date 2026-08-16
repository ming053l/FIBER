"""PLAN.md §3.5 — validate the Cov(f) estimator with zero GPU time.

P0-1 demoted this estimator to a DIAGNOSTIC: Cov(f) equals C_obs only for the exact
conditional mean, which is what this file feeds it. The certified operator that
discovery actually uses is tested in tests/test_certified_operator.py.

Toy channel  Y = A Z + sigma * e,  Z ~ N(0, I_d),  e ~ N(0, I_n).  Then

    E[Z|Y] = Aᵀ (A Aᵀ + sigma² I)^{-1} Y
    C_obs  = Aᵀ (A Aᵀ + sigma² I)^{-1} A = V diag(s²/(s²+sigma²)) Vᵀ

with A = U S Vᵀ. So both the eigenvalues AND the eigenvectors are known in
closed form and the estimator can be checked against ground truth.
"""
import functools

import numpy as np
import pytest
import torch

from fiber.spectrum import (fit_gram, fit_randomized, fit_spectrum,
                            subspace_alignment, trace_teacher_covariance)

D, N_OBS, N, SIGMA = 256, 48, 6000, 0.5


def synthetic(seed=0, decay=0.85):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(N_OBS, N_OBS, generator=g).double())
    V, _ = torch.linalg.qr(torch.randn(D, N_OBS, generator=g).double())
    s = torch.tensor([decay**i for i in range(N_OBS)], dtype=torch.float64) * 3.0
    A = U @ torch.diag(s) @ V.T                      # n x d
    lam = (s**2 / (s**2 + SIGMA**2))                 # analytic eigenvalues
    Z = torch.randn(N, D, generator=g, dtype=torch.float64)
    Y = Z @ A.T + SIGMA * torch.randn(N, N_OBS, generator=g, dtype=torch.float64)
    Sy = A @ A.T + SIGMA**2 * torch.eye(N_OBS, dtype=torch.float64)
    M = torch.linalg.solve(Sy, Y.T).T @ A            # exact E[Z|Y], N x d
    return dict(A=A, Z=Z, Y=Y, M=M.float(), lam=lam.float(), V=V.T.float())


DATA = synthetic()


def test_gram_recovers_analytic_eigenvalues():
    """Cross-fit: directions from the first half, eigenvalues measured on the
    second. In-sample eigenvalues are upward biased at this N (that bias is what
    _finite_sample_ceiling budgets for); the cross-fit ones are the honest number."""
    M = DATA["M"]
    sp = fit_gram(M[: N // 2])
    lam = sp.evaluate_on(M[N // 2:])
    analytic = DATA["lam"]
    # Per-direction: only the leading eigenvalue is well separated here; the rest
    # are near-degenerate, so individual eigenvectors rotate within the cluster and
    # per-index comparison is not the identified quantity.
    assert abs(float(lam[0] - analytic[0])) < 0.02
    # Rotation-invariant, and exactly what Rayleigh-Ritz optimises: the energy
    # captured by the top-k SUBSPACE, Tr(V C Vᵀ).
    assert abs(float(lam[:16].sum() - analytic[:16].sum())) < 0.3


def test_gram_recovers_the_analytic_eigenvectors():
    sp = fit_gram(DATA["M"])
    assert subspace_alignment(sp.eigenvectors[:8], DATA["V"][:8]) > 0.95


def test_randomized_matches_gram():
    """The config's estimator is randomized_svd; it must agree with the exact one."""
    exact = fit_gram(DATA["M"])
    approx = fit_randomized(DATA["M"], k=16, oversampling=32, n_iter=4, seed=0)
    assert (approx.eigenvalues[:16] - exact.eigenvalues[:16]).abs().max() < 1e-3
    assert subspace_alignment(approx.eigenvectors[:16], exact.eigenvectors[:16]) > 0.999


def test_trace_is_the_sum_of_the_analytic_spectrum():
    """Tr(C_obs) is the headline number and is computed exactly (no SVD)."""
    assert abs(trace_teacher_covariance(DATA["M"]) - float(DATA["lam"].sum())) < 0.5


def test_crossfit_eigenvalues_lie_in_the_unit_interval():
    """C_obs ⪯ I is a theorem (PLAN.md §3.2), about the POPULATION spectrum.
    Cross-fit eigenvalues are the estimate that must respect it."""
    M = DATA["M"]
    sp = fit_gram(M[: N // 2])
    lam = sp.evaluate_on(M[N // 2:])
    assert float(lam.max()) <= 1.0 + 1e-3
    assert float(lam.min()) >= -1e-6


def test_lambda_is_one_minus_mmse():
    """The whole interpretation rests on this identity. Check it directly against
    the empirical MMSE along the recovered directions, on held-out samples."""
    Z, M = DATA["Z"].float(), DATA["M"]
    sp = fit_gram(M[: N // 2])
    lam = sp.evaluate_on(M[N // 2:])
    for j in [0, 3, 10]:
        v = sp.eigenvectors[j]
        mmse = ((Z[N // 2:] @ v) - (M[N // 2:] @ v)).pow(2).mean()
        assert abs(float(1 - mmse) - float(lam[j])) < 0.05, j


def test_estimator_flags_an_impossible_spectrum():
    bogus = DATA["M"] * 3.0     # a "teacher" that overshoots the conditional mean
    with pytest.raises(ValueError, match="must be"):
        fit_gram(bogus, strict=True)


def test_fit_spectrum_rejects_a_non_mse_teacher():
    with pytest.raises(ValueError, match="conditional MEAN"):
        fit_spectrum(DATA["M"], k=8, cfg={"teacher_loss": "l1"})


def test_flat_spectrum_is_detected():
    """If the channel is isotropic there is no subspace to find and FIBER is
    dead (PLAN.md §3.2). The estimator must not manufacture anisotropy."""
    flat = synthetic(seed=1, decay=1.0)
    sp = fit_gram(flat["M"][: N // 2])
    lam = sp.evaluate_on(flat["M"][N // 2:])[:N_OBS]
    assert float(lam.max() / lam.min()) < 1.3
    assert sp.effective_rank() > 0.9 * N_OBS


# --------------------------------------------------------------------------
# The MSE mandate (PLAN.md §3.3). Under a SKEWED channel the conditional mean
# and the conditional median differ, and only the mean satisfies
#     E[(Z - hat)²] = Var(Z) - Var(hat).
# A median (L1) teacher therefore reports a spectrum that is not 1 - MMSE.
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _skewed_posterior_stats(n=20000, seed=0, b=2.0):
    """Z ~ N(0,1), Y = Z + e with e ~ Exp(b) - b: zero-mean but strongly skewed,
    so the posterior of Z given Y is skewed and mean != median."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    e = rng.exponential(b, n) - b
    y = z + e
    grid = np.linspace(-10, 10, 2401)
    means, medians = np.empty(n), np.empty(n)
    for lo in range(0, n, 2000):
        yy = y[lo:lo + 2000, None]
        logp = -0.5 * grid[None, :] ** 2 - (yy - grid[None, :] + b) / b
        logp = np.where(yy - grid[None, :] >= -b, logp, -np.inf)
        p = np.exp(logp - logp.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        means[lo:lo + 2000] = (p * grid[None, :]).sum(1)
        cdf = np.cumsum(p, 1)
        medians[lo:lo + 2000] = grid[np.argmax(cdf >= 0.5, axis=1)]
    return z, means, medians


def test_mse_teacher_satisfies_the_variance_identity_and_l1_does_not():
    z, mean_hat, median_hat = _skewed_posterior_stats()
    for name, hat in (("mean", mean_hat), ("median", median_hat)):
        mmse = float(((z - hat) ** 2).mean())
        lam = float(hat.var())
        gap = abs((1.0 - mmse) - lam)
        if name == "mean":
            assert gap < 0.015, f"law of total variance broken for the mean: {gap:.4f}"
        else:
            assert gap > 0.02, (
                "the median teacher happened to satisfy the identity; this test "
                "no longer demonstrates the L1 hazard")


def test_median_teacher_reports_a_different_lambda():
    _, mean_hat, median_hat = _skewed_posterior_stats()
    lam_mean, lam_median = float(mean_hat.var()), float(median_hat.var())
    assert abs(lam_median - lam_mean) / lam_mean > 0.05
