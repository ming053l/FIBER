"""Gate 3A requires ‖R Rᵀ − I‖_inf < 1e-5 for every arm, and the frames must be
orthonormal *by construction* — never by penalty. These tests are the guard."""
import pytest
import torch

from fiber.transforms import (HadamardFrame, HouseholderFrame, IdentityFrame,
                              SignedPermutationFrame, SpectralFrame, build_frame, fwht)

D = 1024
K = 32
TOL = 1e-5


def frames():
    yield "identity", IdentityFrame(D, K)
    yield "signperm", SignedPermutationFrame(D, K, seed=3)
    yield "hadamard", HadamardFrame(D, K, seed=3)
    yield "householder", HouseholderFrame(D, K, num_reflectors=16, seed=3)
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(D, K, generator=g).double())
    yield "spectral", SpectralFrame(D, K, rows=q.T.float())


@pytest.mark.parametrize("name,frame", list(frames()), ids=lambda x: x if isinstance(x, str) else "")
def test_rows_are_orthonormal(name, frame):
    assert frame.orthonormality_error() < TOL, name


@pytest.mark.parametrize("name,frame", list(frames()), ids=lambda x: x if isinstance(x, str) else "")
def test_project_is_adjoint_of_expand(name, frame):
    """<project(z), a> == <z, expand(a)> — otherwise expand() is not the adjoint
    and every orthonormality check above is vacuous."""
    z = torch.randn(8, D)
    a = torch.randn(8, K)
    lhs = (frame.project(z) * a).sum()
    rhs = (z * frame.expand(a)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4), name


@pytest.mark.parametrize("name,frame", list(frames()), ids=lambda x: x if isinstance(x, str) else "")
def test_projection_preserves_the_gaussian_measure(name, frame):
    """W_R = R Z with R Rᵀ = I  =>  W_R ~ N(0, I_k) exactly. This is a bug
    detector, not the security argument (README: 'empirical distribution tests
    are bug detection, never the security mechanism')."""
    torch.manual_seed(0)
    z = torch.randn(20000, D)
    w = frame.project(z)
    cov = w.T @ w / w.shape[0]
    assert w.mean().abs() < 0.05, name
    assert (cov - torch.eye(K)).abs().max() < 0.12, name


def test_fwht_matches_dense_hadamard():
    d = 256
    h = torch.tensor([[1.0]])
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    x = torch.randn(4, d)
    assert torch.allclose(fwht(x), x @ h.T, atol=1e-4)


def test_hadamard_rows_are_global():
    """R1: the point of the Hadamard arm is that each robust coordinate reads
    the whole latent, not one pixel. If rows were sparse the arm would be a
    signed permutation in disguise."""
    frame = HadamardFrame(D, K, seed=0)
    rows = frame.rows()
    assert (rows.abs() > 1e-8).float().mean() > 0.99
    assert torch.allclose(rows.abs(), torch.full_like(rows, D**-0.5), atol=1e-6)


def test_householder_stays_orthonormal_after_a_gradient_step():
    frame = HouseholderFrame(D, K, num_reflectors=16, seed=1)
    opt = torch.optim.SGD(frame.parameters(), lr=10.0)
    z = torch.randn(16, D)
    for _ in range(5):
        opt.zero_grad()
        loss = frame.project(z).pow(2).mean()
        loss.backward()
        opt.step()
    assert frame.orthonormality_error() < TOL


def test_registry_builds_every_arm_type():
    for spec in [{"type": "identity"}, {"type": "signed_permutation", "seeds": [0]},
                 {"type": "hadamard", "structure": "P2_H_D_P1"},
                 {"type": "householder", "num_reflectors": 8}]:
        f = build_frame(spec, d=D, k=K, seed=1)
        assert f.orthonormality_error() < TOL


# ==========================================================================
# P0-2: a genuinely uniform random subspace, and an architecture-matched
# random control for the learned arm.
# ==========================================================================
import numpy as np  # noqa: E402

from fiber.transforms import (FrozenHouseholderOnHaarFrame,  # noqa: E402
                              HaarRandomFrame)


def test_haar_and_random_householder_are_orthonormal():
    for f in (HaarRandomFrame(D, K, seed=1),
              FrozenHouseholderOnHaarFrame(D, K, num_reflectors=16, seed=1)):
        assert f.orthonormality_error() < TOL


def test_haar_preserves_the_gaussian_measure():
    torch.manual_seed(0)
    w = HaarRandomFrame(D, K, seed=2).project(torch.randn(20000, D))
    cov = w.T @ w / w.shape[0]
    assert w.mean().abs() < 0.05
    assert (cov - torch.eye(K)).abs().max() < 0.12


def _haar_rows(n_draws, d, k=1):
    return torch.stack([HaarRandomFrame(d, k, seed=s).rows()[0] for s in range(n_draws)])


def test_haar_sampling_is_isotropic():
    """E[r_i r_j] = delta_ij / d for a uniform direction on the sphere."""
    d, n = 64, 3000
    rows = _haar_rows(n, d).double()
    second = rows.T @ rows / n
    assert (second.diag() - 1.0 / d).abs().max() < 0.004
    off = second - torch.diag(second.diag())
    assert off.abs().max() < 0.006


def test_haar_is_random_where_hadamard_is_structured():
    """Isotropy alone does NOT separate Haar from Hadamard: a Hadamard row also has
    E[r_i r_j] = delta_ij/d, because every entry is exactly +-1/sqrt(d). The
    distinguishing statistic is the SPREAD of a squared entry, which is
    Beta(1/2, (d-1)/2) for Haar and identically zero for Hadamard."""
    d, n = 64, 3000
    haar_sq = (_haar_rows(n, d)[:, 0] ** 2).double()
    beta_var = 2 * (d - 1) / (d**2 * (d + 2))
    assert abs(float(haar_sq.mean()) - 1.0 / d) < 0.002
    assert abs(float(haar_sq.var()) - beta_var) < 0.5 * beta_var

    had = torch.stack([HadamardFrame(d, 1, seed=s).rows()[0] for s in range(200)])
    assert float((had[:, 0] ** 2).var()) < 1e-12
    assert torch.allclose(had.abs(), torch.full_like(had, d**-0.5), atol=1e-6)


def test_haar_entry_signs_are_unbiased():
    """The sign(diag(R)) correction exists so that Haar-ness cannot depend on an
    undocumented LAPACK convention. A convention leak would show up as a non-zero
    mean on the leading entries."""
    rows = _haar_rows(3000, 64).double()
    means = rows[:, :4].mean(0)
    assert means.abs().max() < 3.5 * (1.0 / 64) ** 0.5 / 3000**0.5 * 3


def test_frozen_control_is_the_learned_arm_at_initialisation():
    """Architecture-matched control: same parameterisation, same reflector count,
    same initial draw, same Haar base. Only the optimisation differs."""
    learned = HouseholderFrame(D, K, num_reflectors=16, seed=3)
    frozen = FrozenHouseholderOnHaarFrame(D, K, num_reflectors=16, seed=3)
    assert torch.equal(learned.V.detach(), frozen.V)
    assert sum(p.numel() for p in frozen.parameters()) == 0
    z = torch.randn(8, D)
    assert torch.allclose(learned.project(z), frozen.project(z), atol=1e-6)


def test_unbased_householder_product_is_not_a_haar_sample():
    """A product of m reflections is only close to uniform once m is comparable to d,
    so arm C3 must never stand in for the primary random reference (arm C2).

    For a Haar direction E[r_0^2] = 1/d. Measured at d=64 the Householder product
    approaches that from far above, because a handful of reflections barely moves
    e_1: m=4 -> 0.78, m=16 -> 0.38, m=64 -> 0.035, m=128 -> 0.014, against 1/64 =
    0.0156. The configured arm uses m=128 in d=16384, i.e. m/d = 1/128 -- deep in the
    regime where it is nothing like a uniform subspace, which is exactly why it is a
    control for arm E rather than a random baseline.
    """
    d, n = 64, 400
    means = {}
    for m in (4, 16, 64):
        rows = torch.stack([HouseholderFrame(d, 1, num_reflectors=m, seed=s,
                                             base=None, paired_init=False).rows()[0]
                            for s in range(n)]).double()
        means[m] = float((rows[:, 0] ** 2).mean())
    uniform = 1.0 / d
    assert means[4] > 20 * uniform, "m << d should be far from uniform"
    assert means[4] > means[16] > means[64], "more reflections must move toward uniform"
    assert means[64] < 4 * uniform


def test_registry_exposes_both_new_arms():
    for spec in ({"type": "haar"},
                 {"type": "frozen_householder_on_haar", "num_reflectors": 8}):
        f = build_frame(spec, d=D, k=K, seed=2)
        assert f.orthonormality_error() < TOL


# ==========================================================================
# P0-4: arm E starts EXACTLY at the gate denominator, not near it.
# ==========================================================================
def test_learned_frame_initialises_exactly_at_the_haar_base():
    """R_E = Q_phi H with Q_phi(0) = I exactly, because the reflectors start as
    identical adjacent pairs and H(v)H(v) = I. 'Near identity' would leave arm E
    starting somewhere the gate denominator is not."""
    for seed in (0, 1, 2):
        e = HouseholderFrame(D, K, num_reflectors=16, seed=seed)
        h = HaarRandomFrame(D, K, seed=seed)
        assert (e.rows() - h.rows()).abs().max() < 1e-6
        assert e.orthonormality_error() < TOL


def test_paired_initialisation_requires_an_even_reflector_count():
    with pytest.raises(ValueError, match="even reflector count"):
        HouseholderFrame(D, K, num_reflectors=15, paired_init=True)


def test_one_gradient_step_leaves_the_haar_point_and_reduces_the_loss():
    """An exact Haar start must not be a fixed point of the optimiser."""
    torch.manual_seed(0)
    frame = HouseholderFrame(D, K, num_reflectors=16, seed=4)
    haar = HaarRandomFrame(D, K, seed=4)
    z = torch.randn(64, D)
    target = torch.randn(64, K)
    opt = torch.optim.AdamW(frame.parameters(), lr=1e-2, weight_decay=0.0)

    before = float(((frame.project(z) - target) ** 2).mean())
    opt.zero_grad()
    torch.nn.functional.mse_loss(frame.project(z), target).backward()
    assert frame.V.grad is not None and float(frame.V.grad.abs().max()) > 0
    opt.step()

    after = float(((frame.project(z) - target) ** 2).mean())
    assert after < before, "the Haar initialisation is a fixed point"
    assert (frame.rows() - haar.rows()).abs().max() > 1e-4, "the frame did not move"
    assert frame.orthonormality_error() < TOL, "orthogonality lost after a step"


def test_frozen_control_coincides_with_haar_by_construction():
    """C3 == C2 at the same seed. Not a defect: at initialisation the parameterisation
    contributes nothing, so any arm-E gain is attributable to learning."""
    frozen = FrozenHouseholderOnHaarFrame(D, K, num_reflectors=16, seed=5)
    haar = HaarRandomFrame(D, K, seed=5)
    assert (frozen.rows() - haar.rows()).abs().max() < 1e-6


def test_base_can_be_disabled_for_the_diagnostic():
    """base=None reproduces the pre-P0-4 arm: first k rows of the raw product."""
    f = HouseholderFrame(D, K, num_reflectors=16, seed=0, base=None, paired_init=False)
    assert not f.has_base and f.orthonormality_error() < TOL


def test_reachable_target_shares_the_fitters_frozen_base():
    """B3's attribution rests on the reachable target really being reachable.
    `HouseholderFrame(seed=s)` seeds the reflectors AND the Haar base, so a target
    built at another seed sits on a different base and need not lie in the fitter's
    family at all -- at which point a failure says nothing about the optimiser."""
    import sys
    sys.path.insert(0, "scripts")
    from audit_reflector_capacity import alignment, make_target

    d, k, m = 512, 8, 16
    fitter = HouseholderFrame(d, k, num_reflectors=m, seed=0, base="haar", paired_init=True)
    naive = HouseholderFrame(d, k, num_reflectors=m, seed=2000, base="haar",
                             paired_init=False)
    assert not torch.allclose(fitter.B, naive.B), "the bases were already identical"

    T = make_target("reachable", fitter, d, k, m, 0, torch.device("cpu"))
    # the target is reachable: copying its parameters into the fitter reproduces it
    src = HouseholderFrame(d, k, num_reflectors=m, seed=2000, base="haar",
                           paired_init=False)
    with torch.no_grad():
        fitter.V.copy_(src.V)
    assert float(alignment(fitter.rows(), T)) > 1 - 1e-5


def test_generic_target_is_independent_of_the_fitter():
    import sys
    sys.path.insert(0, "scripts")
    from audit_reflector_capacity import alignment, make_target

    d, k, m = 512, 8, 16
    fitter = HouseholderFrame(d, k, num_reflectors=m, seed=0, base="haar", paired_init=True)
    T = make_target("generic", fitter, d, k, m, 0, torch.device("cpu"))
    assert float(alignment(fitter.rows().detach(), T)) < 0.5
