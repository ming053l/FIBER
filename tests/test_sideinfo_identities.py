"""The identities reports/blind_vs_sideinfo.md preregisters, checked numerically.

Nothing here is implemented machinery -- that experiment has not been written. These pin
the algebra the design document relies on, so a claim in a preregistration is not just
prose. Everything is a closed-form Gaussian construction where the Bayes estimators are
computable, which is the only setting where the population objects are available at all.
"""
import numpy as np
import pytest

D, P, Q, N = 4, 3, 6, 400_000


def _world(seed=0):
    """Jointly Gaussian (Z, S, Y) with Y = A Z + B S + noise, so E[Z|Y] and E[Z|Y,S] are
    exact linear maps."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Q, D)) * 0.6
    B = rng.standard_normal((Q, P)) * 0.8
    Z = rng.standard_normal((N, D))
    S = rng.standard_normal((N, P))
    Y = Z @ A.T + S @ B.T + 0.5 * rng.standard_normal((N, Q))
    return rng, Z, S, Y


def _cond_mean(target, given):
    C = np.cov(np.hstack([target, given]).T)
    t = target.shape[1]
    return given @ np.linalg.solve(C[t:, t:], C[:t, t:].T)


def _pieces(seed=0):
    rng, Z, S, Y = _world(seed)
    m_b = _cond_mean(Z, Y)
    m_s = _cond_mean(Z, np.hstack([Y, S]))
    return rng, Z, Y, m_b, m_s, m_s - m_b


def _C_gain(Z, h):
    return (Z.T @ h + h.T @ Z - h.T @ h) / Z.shape[0]


def test_side_information_cannot_reduce_bayes_observability():
    """Delta C_side = Cov(m_s) - Cov(m_b) >= 0, by the law of total covariance."""
    _, _, _, m_b, m_s, _ = _pieces()
    dC = np.cov(m_s.T) - np.cov(m_b.T)
    assert np.linalg.eigvalsh(dC).min() > -1e-6


def test_the_gain_is_the_covariance_of_the_residual():
    """Delta C_side = E[Cov(m_s|Y)] = Cov(r) with r = m_s - m_b, and E[r|Y] = 0."""
    _, _, Y, m_b, m_s, r = _pieces()
    dC = np.cov(m_s.T) - np.cov(m_b.T)
    assert np.abs(dC - np.cov(r.T)).max() < 1e-8
    assert np.abs(_cond_mean(r, Y)).mean() < 1e-8


@pytest.mark.parametrize("scale,noise", [(1.0, 0.0), (0.5, 0.0), (1.0, 0.3), (0.2, 0.6)])
def test_any_residualised_predictor_certifies_a_lower_bound(scale, noise):
    """For h with E[h|Y] = 0,  C_gain(h) = Cov(r) - Cov(r - h) <= Delta C_side.

    This is what makes a DIRECT certificate possible, rather than subtracting two
    certificates whose approximation errors need not cancel."""
    rng, Z, _, m_b, m_s, r = _pieces()
    h = scale * r + noise * rng.standard_normal(r.shape)   # still E[h|Y] = 0
    dC = np.cov(m_s.T) - np.cov(m_b.T)
    assert np.linalg.eigvalsh(dC - _C_gain(Z, h)).min() > -1e-3


def test_the_bound_fails_when_the_residualisation_is_not_exact():
    """The documented failure mode, and the reason the whole difficulty moves into
    E[h|Y] = 0: a leftover Y-measurable component adds E[m_b b^T + b m_b^T], which has no
    sign, so the certificate BREAKS rather than merely loosening."""
    _, Z, _, m_b, m_s, r = _pieces()
    dC = np.cov(m_s.T) - np.cov(m_b.T)
    bad = r + 0.8 * m_b                     # E[h|Y] = 0.8 m_b != 0
    assert np.linalg.eigvalsh(dC - _C_gain(Z, bad)).min() < -0.1


def test_C_gain_has_the_same_algebraic_form_as_C_cert():
    """C_gain(h) = E[Z h' + h Z' - h h'] is C_cert(f) with f -> h, so the existing
    certified-operator machinery applies unchanged; only the constraint is new."""
    from fiber.spectrum.certified import CertifiedObservabilityOperator

    _, Z, _, _, _, r = _pieces()
    op = CertifiedObservabilityOperator(Z, r, center=False)
    v = np.zeros(D)
    v[0] = 1.0
    # up to the denominator: the operator uses N-1 so that the per-sample contributions
    # average exactly to v' C v, while the plain moment above divides by N
    n = Z.shape[0]
    assert float(v @ op.matvec(v)) == pytest.approx(
        float(_C_gain(Z, r)[0, 0]) * n / (n - 1), rel=1e-9)
