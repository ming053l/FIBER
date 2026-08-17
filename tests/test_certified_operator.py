"""P0-1: the decoder-certified observability operator.

Toy channel Y = A Z + sigma e with Z ~ N(0,I_d), so both E[Z|Y] and C_obs are known
in closed form and every claim below is checked against ground truth:

    m(Y)  = A^T (A A^T + sigma^2 I)^{-1} Y
    C_obs = A^T (A A^T + sigma^2 I)^{-1} A
"""
import numpy as np
import pytest
import torch

from fiber.spectrum.certified import (CertifiedObservabilityOperator, fit_certified,
                                      quadratic_form, teacher_validity, variance_form)

D, N_OBS, N, SIGMA = 64, 24, 30000, 0.6


def toy(seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N_OBS, N_OBS)))
    V, _ = np.linalg.qr(rng.standard_normal((D, N_OBS)))
    s = 3.0 * 0.88 ** np.arange(N_OBS)
    A = U @ np.diag(s) @ V.T
    Z = rng.standard_normal((N, D))
    Y = Z @ A.T + SIGMA * rng.standard_normal((N, N_OBS))
    Sy = A @ A.T + SIGMA**2 * np.eye(N_OBS)
    m = np.linalg.solve(Sy, Y.T).T @ A                  # exact E[Z|Y]
    C_obs = A.T @ np.linalg.solve(Sy, A)
    lam = s**2 / (s**2 + SIGMA**2)
    return dict(A=A, Z=Z, m=m, C_obs=C_obs, V=V.T, lam=lam)


T = toy()


def cov(F, ddof=1):
    """ddof=1 to match the operator: the data are sample-centered."""
    Fc = F - F.mean(0)
    return Fc.T @ Fc / max(len(Fc) - ddof, 1)


def max_eig(M):
    return float(np.linalg.eigvalsh((M + M.T) / 2).max())


def cert_dense(Z, F, center=True):
    return CertifiedObservabilityOperator(Z, F, center=center).dense()


# ---------------------------------------------------------------- identities
def test_exact_teacher_reproduces_c_obs():
    C = cert_dense(T["Z"], T["m"])
    assert np.abs(C - T["C_obs"]).max() < 0.02


def test_zero_teacher_gives_zero_operator():
    C = cert_dense(T["Z"], np.zeros_like(T["m"]))
    assert np.abs(C).max() < 1e-12


def test_quadratic_form_is_one_minus_mmse():
    """v^T C_cert v = Var(v^T Z) − E[(v^T Z − v^T f)^2], which for Z ~ N(0,I) and
    the exact conditional mean is 1 − MMSE_v."""
    V = T["V"][:8]
    lam = quadratic_form(T["Z"], T["m"], V)
    mmse = ((T["Z"] @ V.T) - (T["m"] @ V.T)).var(0)
    assert np.abs(lam - (1 - mmse)).max() < 0.02
    assert np.abs(lam - T["lam"][:8]).max() < 0.03


# ---------------------------------------------------------------- the bound
def test_covariance_of_a_bad_teacher_is_not_a_lower_bound():
    """The reason C_cert exists. An over-scaled teacher inflates Cov(f) without
    bound, so Cov(f) can NOT be presented as a lower bound on C_obs."""
    bad = 3 * T["m"]
    assert max_eig(cov(bad) - T["C_obs"]) > 1.0


def test_certified_operator_stays_below_c_obs_for_the_same_bad_teacher():
    bad = 3 * T["m"]
    assert max_eig(cert_dense(T["Z"], bad) - T["C_obs"]) < 0.05


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_certified_operator_is_a_lower_bound_at_every_teacher_scale(scale):
    """The population bound C_cert <= C_obs is exact; the slack here is finite-sample
    noise, measured at most +0.066 (at scale 1, where the bound is tight) for
    N = 30000, d = 64, rank 24."""
    assert max_eig(cert_dense(T["Z"], scale * T["m"]) - T["C_obs"]) < 0.10


def test_only_the_covariance_estimator_diverges_with_teacher_scale():
    """The discriminating fact, stated as a test: as the teacher is scaled away from
    the conditional mean, Cov(f) violates the bound without limit while C_cert does
    not move. Measured: Cov violation 0.03 -> 23.3 over scale 1 -> 5."""
    viol_cov, viol_cert = [], []
    for scale in (1.0, 2.0, 3.0, 5.0):
        f = scale * T["m"]
        viol_cov.append(max_eig(cov(f) - T["C_obs"]))
        viol_cert.append(max_eig(cert_dense(T["Z"], f) - T["C_obs"]))
    assert viol_cov[-1] > 10 * viol_cov[0]
    assert max(viol_cert) < 0.10


def test_certified_operator_equals_c_obs_minus_teacher_error():
    """C_cert(f) = C_obs − E[(m−f)(m−f)^T] exactly."""
    f = 0.7 * T["m"] + 0.3 * np.roll(T["m"], 3, axis=1)
    err = T["m"] - f
    err = err - err.mean(0)
    predicted = T["C_obs"] - err.T @ err / len(err)
    assert np.abs(cert_dense(T["Z"], f) - predicted).max() < 0.03


# ---------------------------------------------------------------- centering
def test_centering_removes_a_constant_teacher_bias():
    bias = np.full(D, 0.4)
    biased = T["m"] + bias
    assert np.abs(cert_dense(T["Z"], biased) - cert_dense(T["Z"], T["m"])).max() < 1e-9
    uncentered = CertifiedObservabilityOperator(T["Z"], biased, center=False).dense()
    assert np.abs(uncentered - cert_dense(T["Z"], T["m"])).max() > 0.1


def test_uncentered_bias_injects_a_rank_one_direction():
    """The concrete failure centering prevents: a constant offset shows up as a
    spurious leading direction pointing along the bias."""
    bias = np.zeros(D); bias[0] = 1.5
    F = T["m"] + bias
    C = CertifiedObservabilityOperator(T["Z"], F, center=False).dense()
    w, v = np.linalg.eigh(C)
    worst = v[:, np.argmin(w)]
    assert abs(worst[0]) > 0.9        # the spurious direction is the bias direction


# ------------------------------------------------- matrix-free == dense
def test_matrix_free_matches_dense():
    op = CertifiedObservabilityOperator(T["Z"], T["m"])
    V = np.random.default_rng(0).standard_normal((D, 5))
    assert np.abs(op.matmat(V) - op.dense() @ V).max() < 1e-9
    assert np.abs(op.matvec(V[:, 0]) - op.dense() @ V[:, 0]).max() < 1e-9


def test_trace_is_exact_and_signed():
    op = CertifiedObservabilityOperator(T["Z"], 3 * T["m"])
    assert abs(op.trace() - np.trace(op.dense())) < 1e-6
    assert op.trace() < 0, "an over-scaled teacher must give a NEGATIVE signed trace"


def test_dense_is_refused_at_production_dimension():
    op = CertifiedObservabilityOperator(np.zeros((4, 16384)), np.zeros((4, 16384)))
    with pytest.raises(MemoryError):
        op.dense()


# ------------------------------------------- ALGEBRAICALLY largest (the trap)
def _indefinite_case():
    """Teacher scaled per-eigendirection: c=1 on the top directions (good) and c=4
    on the next ones (badly over-scaled). Since
        v_j^T C_cert v_j = lambda_j (2 c_j − c_j^2),
    c=1 gives +lambda_j and c=4 gives −8 lambda_j, so the largest-MAGNITUDE
    eigenvalue is negative while the largest ALGEBRAIC one is positive."""
    V, m, lam = T["V"], T["m"], T["lam"]
    coef = np.where(np.arange(len(lam)) < 4, 1.0, 4.0)
    proj = m @ V.T
    F = (proj * coef) @ V
    return F, V, lam, coef


def test_largest_magnitude_eigenvalue_is_negative_in_this_case():
    F, V, lam, _ = _indefinite_case()
    w = np.linalg.eigvalsh(cert_dense(T["Z"], F))
    assert w.min() < 0 < w.max()
    assert abs(w.min()) > w.max(), "test case is not indefinite enough to be a trap"


def test_fit_certified_returns_the_algebraically_largest_direction():
    F, V, lam, coef = _indefinite_case()
    sp = fit_certified(T["Z"], F, k=3, oversampling=8, seed=0)
    assert sp.eigenvalues[0] > 0
    # The four c=1 eigenvalues are near-degenerate, so the recovered vector is a
    # mixture within that block; what must hold is that its ENERGY sits in the
    # well-estimated subspace, not that it matches one eigenvector.
    energy = (np.abs(sp.eigenvectors[0] @ V.T)) ** 2
    assert energy[:4].sum() > 0.95, "top direction leaks into the over-scaled block"
    assert energy[4:].sum() < 0.05


def test_magnitude_based_selection_would_pick_the_wrong_direction():
    """Guards against ever reverting to an SVD/`which='LM'` solver."""
    F, V, _, _ = _indefinite_case()
    C = cert_dense(T["Z"], F)
    w, vecs = np.linalg.eigh(C)
    by_magnitude = vecs[:, np.argmax(np.abs(w))]
    energy = (np.abs(by_magnitude @ V.T)) ** 2
    assert energy[4:].sum() > 0.95, "magnitude selection did not pick a bad direction"
    assert w[np.argmax(np.abs(w))] < 0


def test_eigsh_matches_dense_eigendecomposition():
    sp = fit_certified(T["Z"], T["m"], k=8, oversampling=8, seed=0)
    dense_vals = np.linalg.eigvalsh(cert_dense(T["Z"], T["m"]))[::-1]
    assert np.abs(sp.eigenvalues[:8] - dense_vals[:8]).max() < 1e-6


def test_fit_is_deterministic_given_a_seed():
    a = fit_certified(T["Z"], T["m"], k=6, oversampling=8, seed=3)
    b = fit_certified(T["Z"], T["m"], k=6, oversampling=8, seed=3)
    assert np.array_equal(a.eigenvalues, b.eigenvalues)


def test_in_sample_eigenvalues_can_exceed_the_theoretical_bound():
    """Why every REPORTED number must be cross-fitted. C_cert <= C_obs <= I holds in
    the population, but estimating a rank-2N operator in d dimensions from N samples
    inflates the in-sample spectrum badly: measured lambda_max = 7.2 at N=200, d=4096
    with an EXACT teacher. In-sample spectra select directions; they never get quoted."""
    rng = np.random.default_rng(0)
    d, n, n_obs = 512, 60, 24
    A = rng.standard_normal((n_obs, d)) / d**0.5 * 3
    Z = rng.standard_normal((n, d))
    Y = Z @ A.T + 0.6 * rng.standard_normal((n, n_obs))
    Sy = A @ A.T + 0.36 * np.eye(n_obs)
    m = np.linalg.solve(Sy, Y.T).T @ A                 # EXACT conditional mean
    sp = fit_certified(Z, m, k=4, oversampling=4, seed=0)
    assert sp.eigenvalues[0] > 1.5
    # ... and the cross-fit evaluation on fresh samples does not
    Z2 = rng.standard_normal((4000, d))
    Y2 = Z2 @ A.T + 0.6 * rng.standard_normal((4000, n_obs))
    m2 = np.linalg.solve(Sy, Y2.T).T @ A
    assert quadratic_form(Z2, m2, sp.eigenvectors[:4]).max() < 1.05


# ---------------------------------------------------------------- reporting
def test_d_cert_is_the_positive_mass_not_the_trace():
    F, _, _, _ = _indefinite_case()
    sp = fit_certified(T["Z"], F, k=8, oversampling=16, seed=0)
    assert sp.d_cert() >= 0
    assert sp.negative_mass() > 0
    assert sp.trace_signed < sp.d_cert(), "signed trace must not be sold as a mass"


def test_d_cert_of_an_exact_teacher_approaches_the_true_spectrum():
    sp = fit_certified(T["Z"], T["m"], k=8, oversampling=16, seed=0)
    assert abs(sp.d_cert(8) - T["lam"][:8].sum()) < 0.1


def test_range_eigh_returns_the_complete_non_zero_spectrum():
    """range(C_cert) is spanned by the rows of Zc and Fc, so the p = 2N eigenvalues
    it returns account for the entire operator: their sum must equal the exact trace,
    which makes the positive and negative masses exact rather than tail estimates."""
    sp = fit_certified(T["Z"], T["m"], k=8, oversampling=16, seed=0)
    assert sp.spectrum_is_complete and sp.positive_mass_is_complete()
    assert abs(sp.eigenvalues_full.sum() - sp.trace_signed) < 1e-6
    assert abs((sp.total_positive_mass() - sp.negative_mass()) - sp.trace_signed) < 1e-6


def test_range_eigh_agrees_with_the_lanczos_reference():
    """scipy eigsh(which="LA") is the independent reference implementation. It is
    NOT the default because in production d = 16384 >> 2N, so the zero eigenvalue is
    massively degenerate and ARPACK stalls inside that cluster."""
    exact = fit_certified(T["Z"], T["m"], k=6, oversampling=8, seed=0, method="range_eigh")
    lanczos = fit_certified(T["Z"], T["m"], k=6, oversampling=8, seed=0, method="eigsh")
    assert np.abs(exact.eigenvalues[:6] - lanczos.eigenvalues[:6]).max() < 1e-6
    align = np.abs(np.diag(exact.eigenvectors[:6] @ lanczos.eigenvectors[:6].T))
    assert align.min() > 0.999


def test_recovered_directions_are_orthonormal():
    sp = fit_certified(T["Z"], T["m"], k=8, oversampling=8, seed=0)
    V = sp.eigenvectors
    assert np.abs(V @ V.T - np.eye(len(V))).max() < 1e-8


def test_eigenvalues_never_exceed_one_for_a_valid_teacher():
    """C_cert <= C_obs <= I, so lambda <= 1 up to sampling noise. A value well above
    1 means the teacher saw z, or the target was mis-scaled."""
    sp = fit_certified(T["Z"], T["m"], k=8, oversampling=16, seed=0)
    assert sp.eigenvalues.max() < 1.02


# ---------------------------------------------------------------- validity
def test_validity_gap_is_zero_for_an_exact_teacher():
    V = T["V"][:16]
    rep = teacher_validity(T["Z"], T["m"], V)
    assert rep["mean_abs_gap"] < 0.02


def test_validity_gap_exposes_a_scaled_teacher():
    """lambda_var (what Cov(f) reports) and lambda_skill (what C_cert certifies)
    coincide only for the exact conditional mean."""
    V = T["V"][:16]
    rep = teacher_validity(T["Z"], 3 * T["m"], V)
    assert rep["mean_abs_gap"] > 1.0
    assert rep["n_negative_directions"] > 0
    assert (rep["lambda_var"] > rep["lambda_skill"]).all()


def test_variance_form_is_the_old_estimator():
    V = T["V"][:8]
    assert np.abs(variance_form(T["m"], V) - np.diag(V @ cov(T["m"]) @ V.T)).max() < 1e-9


def test_cannot_ask_for_more_directions_than_the_operator_has_rank():
    Z = np.random.default_rng(0).standard_normal((8, D))
    with pytest.raises(ValueError, match="rank"):
        fit_certified(Z, Z, k=32, oversampling=8)


# ==========================================================================
# P0-1.1: the reported observability must be a property of the SUBSPACE, not of
# the basis the discovery step happened to hand back.
# ==========================================================================
from fiber.spectrum.certified import (project_operator,  # noqa: E402
                                      subspace_certificate, zero_tolerance)


def _rotation(k, seed=0):
    g = np.random.default_rng(seed)
    q, r = np.linalg.qr(g.standard_normal((k, k)))
    return q * np.sign(np.diag(r))          # exactly orthogonal, sign-corrected


def test_subspace_score_is_invariant_under_within_subspace_rotation():
    """D_cert_subspace(V) == D_cert_subspace(R V) for R in O(k), because
    C_{RV} = R C_V R^T has the same eigenvalues. This is what makes it a property
    of the discovered subspace and lets P0-7 attribute a BER change to the BASIS."""
    V = T["V"][:8]
    base = subspace_certificate(T["Z"], T["m"], V)
    for seed in (0, 1, 2):
        rot = subspace_certificate(T["Z"], T["m"], _rotation(8, seed) @ V)
        assert abs(rot["D_cert_subspace"] - base["D_cert_subspace"]) < 1e-9
        assert rot["numerical_positive_rank"] == base["numerical_positive_rank"]
        assert np.abs(np.sort(rot["mu"]) - np.sort(base["mu"])).max() < 1e-9


def test_coordinate_skill_is_NOT_invariant():
    """The contrast that motivates the fix: the per-coordinate diagonal moves under
    rotation, so clipping it cannot be the subspace headline."""
    V = T["V"][:8]
    base = subspace_certificate(T["Z"], T["m"], V)
    rot = subspace_certificate(T["Z"], T["m"], _rotation(8, 5) @ V)
    assert np.abs(rot["coordinate_skill"] - base["coordinate_skill"]).max() > 1e-3


def _counterexample_data(n=40000, d=16, seed=0):
    """Realise the reviewer's C_V = [[-1, 2], [2, -1]] as actual (Z, F) samples.

    With B = A M and A whitened, C_V = M + M^T - M^T M = 2M - M^2, so M = [[2,-1],[-1,2]]
    gives eigenvalues (1, -3) with diagonal (-1, -1).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, 2))
    cov = A.T @ A / n
    w, U = np.linalg.eigh(cov)
    A = A @ (U / np.sqrt(w)) @ U.T                     # exactly whitened
    M = np.array([[2.0, -1.0], [-1.0, 2.0]])
    B = A @ M
    Z = np.concatenate([A, rng.standard_normal((n, d - 2))], axis=1)
    F = np.concatenate([B, rng.standard_normal((n, d - 2))], axis=1)
    V = np.eye(d)[:2]
    return Z, F, V


def test_diagonal_clipping_reports_zero_where_the_subspace_certifies_one():
    """The concrete failure of the previous metric, on data rather than on paper:
    every coordinate looks worse than the prior mean, yet the subspace contains a
    direction certified at mu = 1."""
    Z, F, V = _counterexample_data()
    C_V = project_operator(Z, F, V)
    assert np.abs(C_V - np.array([[-1.0, 2.0], [2.0, -1.0]])).max() < 0.05

    cert = subspace_certificate(Z, F, V)
    assert (cert["coordinate_skill"] < 0).all()
    assert cert["D_coordinate_clipped"] == 0.0          # what the old metric said
    assert abs(cert["D_cert_subspace"] - 1.0) < 0.05    # what the subspace certifies
    assert cert["numerical_positive_rank"] == 1


def test_subspace_score_comes_from_eigenvalues_not_from_the_diagonal():
    Z, F, V = _counterexample_data()
    cert = subspace_certificate(Z, F, V)
    assert cert["D_cert_subspace"] != cert["D_coordinate_clipped"]
    # `mu` is fold 0's; the headline averages the symmetric folds, so the identity is
    # checked against the fold it belongs to
    assert abs(cert["D_cert_subspace_fold0"] - np.clip(cert["mu"], 0, None).sum()) < 1e-12
    assert abs(cert["D_cert_subspace"] - float(np.mean(cert["fold_masses"]))) < 1e-12


def test_projected_operator_matches_the_analytic_restriction():
    """For the EXACT conditional mean, V C_cert V^T must equal V C_obs V^T."""
    V = T["V"][:10]
    C_V = project_operator(T["Z"], T["m"], V)
    assert np.abs(C_V - V @ T["C_obs"] @ V.T).max() < 0.03


def test_numerical_zero_does_not_become_certified_rank():
    """A decoder that recovers nothing must report rank 0, not k directions of
    1e-16 'observability'."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((5000, 16))
    F = np.zeros_like(Z)
    cert = subspace_certificate(Z, F, np.eye(16)[:6])
    assert cert["numerical_positive_rank"] == 0
    assert cert["D_cert_subspace"] == 0.0
    assert cert["zero_tolerance"] > 0


def test_zero_tolerance_is_a_floor_then_scales_with_the_spectrum():
    eps = np.finfo(np.float64).eps
    small = zero_tolerance(np.array([1e-3, -1e-3]))
    assert small == 1e-9, "tiny spectra must fall back to the absolute floor"
    huge = np.array([1e9, -1e9])
    assert zero_tolerance(huge) == pytest.approx(len(huge) * eps * 1e9)
    assert zero_tolerance(huge) > small


def test_rank_is_reported_against_the_requested_k():
    """If only some of the k requested directions are positively certified, that is
    a result to report, not a number to quietly round up."""
    Z, F, V = _counterexample_data()
    cert = subspace_certificate(Z, F, V)
    assert cert["requested_k"] == 2 and cert["numerical_positive_rank"] == 1


# --------------------------------------------------------------------------
# P0-1.1 continued: the eigen-step is itself a selection, so it needs its own
# cross-fit. Choosing the within-subspace rotation and measuring it on the same
# held-out samples rectifies noise into certified mass.
# --------------------------------------------------------------------------
def _null_case(n, k, d=512, seed=0):
    """A decoder with ZERO true skill: f is independent of Z, so C_cert = 0 in the
    population and every certified mass must go to zero."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d))
    F = rng.standard_normal((n, d)) * 0.5
    V = np.linalg.qr(rng.standard_normal((d, k)))[0].T
    return Z, F, V


def test_inner_crossfit_removes_the_selection_bias_under_the_null():
    Z, F, V = _null_case(256, 32)
    c = subspace_certificate(Z, F, V)
    assert c["crossfit"] and c["D_cert_subspace_insample"] > 0.3
    assert c["D_cert_subspace"] < 0.05, "inner cross-fit did not remove the bias"


def test_selection_bias_is_worse_at_small_n_and_large_k():
    """Documents the regime where the in-sample version is untrustworthy."""
    tiny = subspace_certificate(*_null_case(64, 32))["D_cert_subspace_insample"]
    big = subspace_certificate(*_null_case(1000, 32))["D_cert_subspace_insample"]
    assert tiny > 1.0 and big < 0.05


def test_trace_companion_is_unbiased_for_the_true_restricted_trace():
    """sum(mu) = tr(C_V) is basis-invariant and unclipped, so it estimates the TRUE
    restricted trace rather than being rectified upward.

    Note what the null actually is: f is independent of Z but not zero, so
    C_cert = -Cov(f) = -0.25 I and the restriction to k=32 has trace exactly -8.0.
    "Zero skill" means the certified MASS is zero, not that the operator vanishes."""
    traces = [subspace_certificate(*_null_case(256, 32, seed=s))["trace_C_V"]
              for s in range(8)]
    assert abs(float(np.mean(traces)) + 8.0) < 0.4
    # and the clipped headline correctly reports no certified mass
    masses = [subspace_certificate(*_null_case(256, 32, seed=s))["D_cert_subspace"]
              for s in range(8)]
    assert max(masses) < 0.05


def test_crossfit_score_is_still_rotation_invariant():
    """The whole point of the subspace score survives the inner cross-fit: the chosen
    rotation absorbs R, so the measured eigenvalues are unchanged."""
    V = T["V"][:16]
    a = subspace_certificate(T["Z"], T["m"], V)
    b = subspace_certificate(T["Z"], T["m"], _rotation(16, 7) @ V)
    assert abs(a["D_cert_subspace"] - b["D_cert_subspace"]) < 1e-9
    assert abs(a["trace_C_V"] - b["trace_C_V"]) < 1e-9


def test_crossfit_score_recovers_real_signal():
    """Removing the bias must not remove the signal: with the exact conditional mean
    the cross-fit score still matches the analytic restricted mass."""
    V = T["V"][:8]
    c = subspace_certificate(T["Z"], T["m"], V)
    assert abs(c["D_cert_subspace"] - T["lam"][:8].sum()) < 0.15


def test_reported_mu_range_is_consistent_with_the_certified_rank():
    """After the inner cross-fit mu is ordered by the rotation from half 1, not
    sorted, so mu[0]/mu[-1] are not the extremes. Reporting a max below the zero
    tolerance while also reporting a positive rank is self-contradictory."""
    for case in (_null_case(256, 32), (T["Z"], T["m"], T["V"][:16])):
        c = subspace_certificate(*case)
        assert c["mu_max"] == pytest.approx(float(np.max(c["mu"])))
        assert c["mu_min"] == pytest.approx(float(np.min(c["mu"])))
        if c["numerical_positive_rank"] > 0:
            assert c["mu_max"] > c["zero_tolerance"]
        else:
            assert c["mu_max"] <= c["zero_tolerance"]


def test_operator_uses_the_unbiased_denominator():
    """The data are sample-centered, so N-1 is the unbiased denominator. At small N
    the difference is visible: with ddof=0 the estimate is shrunk by (N-1)/N."""
    n = 12
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n, 8))
    F = 0.5 * Z + 0.1 * rng.standard_normal((n, 8))
    V = np.eye(8)[:4]
    c1 = project_operator(Z, F, V, ddof=1)
    c0 = project_operator(Z, F, V, ddof=0)
    assert np.allclose(c1 * (n - 1) / n, c0)


def test_unbiased_denominator_recovers_the_population_operator():
    """Averaged over many small samples, ddof=1 converges to the truth and ddof=0
    does not."""
    rng = np.random.default_rng(0)
    d, n, reps = 6, 10, 4000
    V = np.eye(d)[:3]
    # f = alpha * Z is a legitimate Y-measurable decoder for Y = Z
    alpha = 0.6
    truth = (2 * alpha - alpha**2)          # v'C_cert v for every direction
    acc1, acc0 = [], []
    for _ in range(reps):
        Z = rng.standard_normal((n, d))
        F = alpha * Z
        acc1.append(np.diag(project_operator(Z, F, V, ddof=1)).mean())
        acc0.append(np.diag(project_operator(Z, F, V, ddof=0)).mean())
    assert abs(np.mean(acc1) - truth) < 0.01
    assert abs(np.mean(acc0) - truth) > 0.03


def test_positive_rank_is_named_as_numerical_not_statistical():
    """`tau` only excludes floating-point noise; it is not a significance test."""
    c = subspace_certificate(*_null_case(256, 32))
    assert "numerical_positive_rank" in c
    # the alias is gone: it let a caller read a numerical count under a name that
    # promised certification, and hid a field mismatch until runtime
    assert "certified_positive_rank" not in c


def test_principal_cosines_expose_structure_a_mean_would_hide():
    """Two subspaces sharing one direction strongly and none otherwise have the same
    MEAN alignment as two sharing every direction weakly. The spectrum separates them."""
    from fiber.spectrum import principal_cosines, subspace_alignment

    rng = np.random.default_rng(0)
    d, k = 64, 4
    Q = np.linalg.qr(rng.standard_normal((d, 3 * k)))[0]
    A = Q[:, :k].T
    shared_one = np.concatenate([Q[:, :1], Q[:, k:k + k - 1]], axis=1).T
    a = principal_cosines(torch.from_numpy(A), torch.from_numpy(shared_one))
    assert a[0] > 0.99 and a[-1] < 0.01
    assert abs(subspace_alignment(torch.from_numpy(A),
                                  torch.from_numpy(shared_one)) - float(a.mean())) < 1e-9


# --------------------------------------------------------------------------
# Statistical certification: tau excludes floating-point noise, a one-sided
# lower bound above zero is what makes "certified" a claim.
# --------------------------------------------------------------------------
def test_null_certifies_nothing_statistically():
    Z, F, V = _null_case(2000, 16)
    c = subspace_certificate(Z, F, V, bootstrap=400)
    assert c["certified_positive_direction_count_per_fold"] == [0, 0]
    assert c["D_cert_LCB"] == 0.0


def test_real_signal_survives_the_lower_bound():
    c = subspace_certificate(T["Z"], T["m"], T["V"][:8], bootstrap=400)
    assert c["certified_positive_direction_count_per_fold"] == [8, 8]
    assert 0 < c["D_cert_LCB"] <= c["D_cert_subspace"] + 1e-9


def test_the_bound_is_never_above_the_point_estimate():
    for case in (_null_case(600, 8), (T["Z"], T["m"], T["V"][:8])):
        c = subspace_certificate(*case, bootstrap=300)
        assert c["D_cert_LCB"] <= c["D_cert_subspace"] + 1e-9


def test_symmetric_folds_use_every_sample_and_agree():
    """Both halves take a turn as the measurement half, so the score stops depending on
    which one the split seed happened to pick."""
    c = subspace_certificate(T["Z"], T["m"], T["V"][:8])
    assert c["folds"] == 2 and len(c["fold_masses"]) == 2
    assert abs(c["D_cert_subspace"] - float(np.mean(c["fold_masses"]))) < 1e-12
    assert abs(c["fold_masses"][0] - c["fold_masses"][1]) < 0.5


def test_symmetric_score_is_less_split_dependent_than_one_fold():
    spreads = {}
    for name, sym in (("one_fold", False), ("symmetric", True)):
        vals = [subspace_certificate(T["Z"], T["m"], T["V"][:8], symmetric=sym, seed=s
                                     )["D_cert_subspace"] for s in range(6)]
        spreads[name] = float(np.std(vals))
    assert spreads["symmetric"] <= spreads["one_fold"] + 1e-9


def test_certification_stays_rotation_invariant():
    a = subspace_certificate(T["Z"], T["m"], T["V"][:8], bootstrap=300)
    b = subspace_certificate(T["Z"], T["m"], _rotation(8, 4) @ T["V"][:8], bootstrap=300)
    assert abs(a["D_cert_subspace"] - b["D_cert_subspace"]) < 1e-9
    # per fold, because index j means "fold f's own j-th fitted direction" -- but
    # still rotation invariant: rotating V by A rotates each fold's operator by A and
    # that fold's chosen rotation absorbs it
    assert (a["certified_positive_direction_count_per_fold"]
            == b["certified_positive_direction_count_per_fold"])
    assert a["certified_positive_inertia"] == b["certified_positive_inertia"]


def test_per_direction_bound_is_multiplicity_corrected():
    """Both levels count: k directions inspected inside each of the two folds, so the
    union bound is alpha/(2k) per direction per fold -- which needs no independence
    between the folds."""
    c = subspace_certificate(T["Z"], T["m"], T["V"][:8], bootstrap=300)
    assert c["per_direction_correction"] == "bonferroni over k and folds"


# --------------------------------------------------------------------------
# Counting positive quadratic forms is NOT counting positive eigenvalues.
# --------------------------------------------------------------------------
def _positive_diagonal_negative_eigenvalue_data(n=60000, d=8, seed=0):
    """Realise a C_V with both diagonals positive but one NEGATIVE eigenvalue.

    With B = A M and A whitened, C_V = 2M - M^2, so eigenvalues of M at (1, 2.0954)
    give C_V eigenvalues (1, -0.2). Rotating M by 45 degrees puts C_V's diagonal at
    (0.4, 0.4): two positive quadratic forms, positive inertia one.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, 2))
    w, U = np.linalg.eigh(A.T @ A / n)
    A = A @ (U / np.sqrt(w)) @ U.T
    R = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
    M = R @ np.diag([1.0, 1.0 + np.sqrt(1.2)]) @ R.T
    B = A @ M
    Z = np.concatenate([A, rng.standard_normal((n, d - 2))], axis=1)
    F = np.concatenate([B, rng.standard_normal((n, d - 2))], axis=1)
    return Z, F, np.eye(d)[:2]


def test_positive_direction_count_is_not_a_rank():
    """The name was the claim: `statistically_certified_rank` counted directions whose
    lower bound cleared zero, which is not the number of positive eigenvalues."""
    Z, F, V = _positive_diagonal_negative_eigenvalue_data()
    C_V = project_operator(Z, F, V)
    diag, eig = np.diag(C_V), np.linalg.eigvalsh(C_V)
    assert (diag > 0).all(), diag                  # two positive quadratic forms
    assert (eig < 0).sum() == 1, eig               # one negative eigenvalue
    assert abs(eig.max() - 1.0) < 0.05 and abs(eig.min() + 0.2) < 0.05

    c = subspace_certificate(Z, F, V, bootstrap=300, crossfit=False)
    assert "statistically_certified_rank" not in c, "the misleading name is still exported"
    assert c["certified_positive_inertia"] <= 1


def test_inertia_certificate_never_exceeds_the_true_positive_count():
    for case in (_null_case(4000, 8), (T["Z"], T["m"], T["V"][:8]),
                 _positive_diagonal_negative_eigenvalue_data()):
        c = subspace_certificate(*case, bootstrap=200)
        truth = int((np.linalg.eigvalsh(project_operator(*case)) > 0).sum())
        assert c["certified_positive_inertia"] <= truth
        assert c["weyl_radius"] > 0


def test_inertia_is_zero_under_the_null_and_full_under_clean_signal():
    assert subspace_certificate(*_null_case(4000, 8),
                                bootstrap=200)["certified_positive_inertia"] == 0
    c = subspace_certificate(T["Z"], T["m"], T["V"][:8], bootstrap=200)
    assert c["certified_positive_inertia"] == 8


def test_per_sample_terms_average_to_the_operator_diagonal_exactly():
    """The docstring says "exactly", so it has to be exact: the operator uses N-1 and a
    plain mean uses N."""
    from fiber.spectrum.certified import _per_sample_contributions

    V = T["V"][:8]
    got = _per_sample_contributions(T["Z"], T["m"], V).mean(0)
    want = np.diag(project_operator(T["Z"], T["m"], V))
    assert np.abs(got - want).max() < 1e-12


def test_multiplicity_spans_directions_and_folds():
    c = subspace_certificate(T["Z"], T["m"], T["V"][:8], bootstrap=200)
    assert c["per_direction_correction"] == "bonferroni over k and folds"


# --- the certified mass bound must actually be a bound ----------------------------

def _near_null(n, k, eps, d=512, seed=0):
    """f independent of Z at scale eps, so C_cert = -eps^2 I: every true eigenvalue is
    negative and the true certified mass sum_j max(lambda_j, 0) is exactly 0. Small eps
    is the interesting regime -- it puts the per-direction sampling noise ACROSS zero,
    where a rectified statistic has nowhere to go but up."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)), rng.standard_normal((n, d)) * eps,
            np.linalg.qr(rng.standard_normal((d, k)))[0].T)


def test_certified_mass_bound_is_zero_under_a_near_null():
    """Bootstrapping sum_j max(b_j, 0) directly cannot produce this. That statistic is
    non-negative, so its whole distribution -- and therefore every quantile of it --
    sits above zero even when the truth is exactly zero."""
    v = [subspace_certificate(*_near_null(256, 128, 0.1, seed=s), bootstrap=400)
         for s in range(4)]
    assert float(np.mean([x["D_cert_LCB"] for x in v])) < 0.05, "positive mass under H0"


def test_a_lower_bound_never_exceeds_its_own_point_estimate():
    """Self-evident without knowing the truth, and the sharpest statement of the bug:
    the old estimator reported LCB 0.55 against a point estimate of 0.42."""
    for s in range(4):
        for eps, k in ((0.1, 128), (0.03, 128), (0.5, 32)):
            c = subspace_certificate(*_near_null(256, k, eps, seed=s), bootstrap=400)
            assert c["D_cert_LCB"] <= c["D_cert_subspace"] + 1e-9, (
                f"LCB {c['D_cert_LCB']:.4f} > point estimate "
                f"{c['D_cert_subspace']:.4f} at eps={eps}, k={k}")


def test_the_corrected_bound_still_detects_real_skill():
    """A bound that is always zero would satisfy the two tests above and be useless."""
    rng = np.random.default_rng(0)
    d, k, a, sig = 256, 32, 0.5, 0.3
    Z = rng.standard_normal((256, d))
    F = a * Z + sig * rng.standard_normal((256, d))
    V = np.linalg.qr(rng.standard_normal((d, k)))[0].T
    c = subspace_certificate(Z, F, V, bootstrap=400)
    per_direction = 2 * a - a * a - sig * sig          # 0.66
    assert 0 < c["D_cert_LCB"] < c["D_cert_subspace"]
    assert c["D_cert_subspace"] == pytest.approx(k * per_direction, rel=0.10)


def test_mass_bound_is_derived_from_the_directional_bounds():
    """It must be sum_j max(L_j, 0) of the reported per-direction bounds, not a
    separately bootstrapped statistic -- otherwise the two can disagree and the alpha
    budget is spent twice."""
    c = subspace_certificate(*_near_null(256, 64, 0.2, seed=3), bootstrap=400)
    assert max(c["D_cert_LCB_per_fold"]) == pytest.approx(c["D_cert_LCB"])
    for L, m in zip(c["lcb_per_direction_per_fold"], c["D_cert_LCB_per_fold"]):
        assert float(np.clip(L, 0, None).sum()) == pytest.approx(m)


def test_trace_uses_every_held_out_sample_not_the_measurement_half():
    """The trace needs no inner cross-fit. It is linear in the per-sample terms, so it
    is unbiased, and invariant to the within-subspace rotation, so a rotation chosen on
    the same samples cannot select anything favourable out of it. Splitting it only
    halved the sample."""
    Z, F, V = _null_case(256, 32, seed=0)
    c = subspace_certificate(Z, F, V)
    assert c["n_trace"] == Z.shape[0]
    assert c["trace_C_V"] == pytest.approx(float(np.trace(project_operator(Z, F, V))),
                                           abs=1e-9)


def test_full_heldout_trace_has_the_variance_of_twice_the_sample():
    """sqrt(2) tighter than the fold-0 version, which is the whole point of the change.
    Both remain unbiased -- this buys precision, not a different estimand."""
    full, fold0 = [], []
    for s in range(100):
        c = subspace_certificate(*_null_case(256, 32, seed=s))
        full.append(c["trace_C_V"])
        fold0.append(c["trace_C_V_fold0"])
    full, fold0 = np.array(full), np.array(fold0)
    assert abs(full.mean() + 8.0) < 0.15 and abs(fold0.mean() + 8.0) < 0.20
    ratio = fold0.std(ddof=1) / full.std(ddof=1)
    assert 1.2 < ratio < 1.7, f"expected ~sqrt(2), got {ratio:.3f}"
