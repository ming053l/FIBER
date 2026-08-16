"""P0-7: the observability score belongs to the SUBSPACE, the sign BER to the BASIS.

For any A with AᵀA = I, span(AV) = span(V), so every certified quantity must be
unchanged while the sign bits move. If that invariance does not hold numerically, a BER
gap from D3 could be mistaken for finding more observable information, which is the one
confusion P0-7 exists to prevent.
"""
import numpy as np
import pytest
import torch

from fiber.spectrum.certified import project_operator, subspace_certificate
from fiber.transforms import HaarRandomFrame, RotatedFrame, haar_orthogonal
from fiber.utils.config import load_config

CFG = load_config("configs/linear_fiber.yaml")
D, K, N = 256, 16, 4000


def _channel(seed=0):
    """Linear-Gaussian toy with a genuinely anisotropic operator."""
    rng = np.random.default_rng(seed)
    n_obs = 32
    A = rng.standard_normal((n_obs, D)) / D**0.5 * 3
    Z = rng.standard_normal((N, D))
    Y = Z @ A.T + 0.6 * rng.standard_normal((N, n_obs))
    Sy = A @ A.T + 0.36 * np.eye(n_obs)
    F = np.linalg.solve(Sy, Y.T).T @ A
    V = np.linalg.svd(A, full_matrices=False)[2][:K]
    return Z, F, V


ZC, FC, VC = _channel()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_certified_score_is_invariant_under_any_in_subspace_rotation(seed):
    """The central invariant. Computed in float64 throughout."""
    A = haar_orthogonal(K, seed=seed).numpy()
    base = subspace_certificate(ZC, FC, VC, crossfit=False)
    rot = subspace_certificate(ZC, FC, A @ VC, crossfit=False)
    assert abs(base["D_cert_subspace"] - rot["D_cert_subspace"]) < 1e-9
    assert abs(base["trace_C_V"] - rot["trace_C_V"]) < 1e-9
    assert base["numerical_positive_rank"] == rot["numerical_positive_rank"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_restricted_operator_eigenvalues_are_invariant(seed):
    """C_{AV} = A C_V Aᵀ, so the eigenvalues must agree to numerical precision."""
    A = haar_orthogonal(K, seed=seed).numpy()
    e0 = np.sort(np.linalg.eigvalsh(project_operator(ZC, FC, VC)))
    e1 = np.sort(np.linalg.eigvalsh(project_operator(ZC, FC, A @ VC)))
    assert np.abs(e0 - e1).max() < 1e-10


def test_rotation_preserves_the_subspace_exactly():
    from fiber.spectrum import subspace_alignment

    A = haar_orthogonal(K, seed=7)
    V = torch.from_numpy(VC)
    assert abs(subspace_alignment(V, A.double() @ V) - 1.0) < 1e-12


def test_the_control_is_not_vacuous_because_sign_bits_do_move():
    """If a rotation could not change the coding, D2 and D3 would test nothing."""
    base = HaarRandomFrame(D, K, seed=0)
    z = torch.randn(512, D)
    b0 = (RotatedFrame(base, mode="identity").project(z) > 0)
    b1 = (RotatedFrame(base, mode="random", seed=1).project(z) > 0)
    assert 0.2 < float((b0 != b1).float().mean()) < 0.8


def test_random_rotations_cover_both_components_of_O_k():
    """D2 is a random BASIS, so it should sample reflections as well as rotations;
    measured over 400 draws the determinant is positive about half the time."""
    dets = [float(torch.det(haar_orthogonal(8, seed=s))) for s in range(400)]
    frac_pos = sum(d > 0 for d in dets) / len(dets)
    assert 0.4 < frac_pos < 0.6
    assert max(abs(abs(d) - 1) for d in dets) < 1e-10


def test_learned_rotation_is_in_SO_k_and_starts_at_the_identity():
    """expm of a skew-symmetric matrix has determinant +1: D3 explores SO(k), not all
    of O(k), and the code says SO(k). Starting at A = I means D3 starts at D1, so a
    gain is the rotation's doing rather than a different starting basis."""
    f = RotatedFrame(HaarRandomFrame(D, K, seed=0), mode="learned", seed=3)
    A = f.rotation()
    assert torch.allclose(A, torch.eye(K), atol=1e-6)
    with torch.no_grad():
        f.S.add_(torch.randn(K, K) * 0.3)
    A = f.rotation()
    assert abs(float(torch.det(A)) - 1.0) < 1e-5
    assert torch.allclose(A @ A.T, torch.eye(K), atol=1e-5)
    assert f.orthonormality_error() < 1e-5


def test_gradients_reach_the_rotation_and_never_the_subspace():
    """If gradients could reach V, D3 would stop being a within-subspace rotation and
    become subspace discovery again, and the whole decomposition would collapse."""
    f = RotatedFrame(HaarRandomFrame(D, K, seed=0), mode="learned", seed=1)
    assert not f.V.requires_grad
    loss = (f.project(torch.randn(32, D)) - torch.randn(32, K)).pow(2).mean()
    loss.backward()
    assert f.S.grad is not None and float(f.S.grad.abs().max()) > 0
    assert f.V.grad is None


def test_soft_sign_target_carries_gradient_where_the_hard_one_does_not():
    from fiber.training.loops import soft_sign

    for mode, target_fn in (("hard", lambda w: (w > 0).float()),
                            ("soft", lambda w: soft_sign(w, 0.5))):
        f = RotatedFrame(HaarRandomFrame(D, K, seed=0), mode="learned", seed=2)
        pred = torch.randn(32, K, requires_grad=True)
        torch.nn.functional.mse_loss(pred, target_fn(f.project(torch.randn(32, D)))).backward()
        if mode == "hard":
            assert f.S.grad is None
        else:
            assert f.S.grad is not None and float(f.S.grad.abs().max()) > 0


def test_tau_is_registered_as_a_hyperparameter_of_the_arm():
    """tau moves D3 between continuous and near-sign recoverability, so it belongs in
    the arm spec, where the hyperparameter fingerprint makes it a val-locked choice."""
    arm = CFG["fiber"]["arms"]["D3_rot_learn"]
    assert arm["type"] == "rotated_learned" and "tau" in arm
    assert CFG["fiber"]["arms"]["D2_rot_rand"]["type"] == "rotated_random"
    assert len(CFG["fiber"]["arms"]["D2_rot_rand"]["seeds"]) >= 5


def test_all_three_basis_arms_share_one_subspace():
    from fiber.spectrum import subspace_alignment

    base = HaarRandomFrame(D, K, seed=0)
    frames = {m: RotatedFrame(base, mode=m, seed=1) for m in ("identity", "random", "learned")}
    rows = {m: f.rows().double() for m, f in frames.items()}
    for m in ("random", "learned"):
        assert abs(subspace_alignment(rows["identity"], rows[m]) - 1.0) < 1e-9


def test_rotation_seed_must_not_move_the_subspace():
    """The arm seed varies the BASIS; the subspace is pinned by base_seed. If the arm
    seed drove both, different D2 draws would sit in different subspaces and the
    comparison would vary two things at once -- which is exactly what P0-7 forbids."""
    from fiber.spectrum import subspace_alignment

    base = HaarRandomFrame(D, K, seed=0)
    rows = [RotatedFrame(base, mode="random", seed=s).rows().double() for s in range(4)]
    for r in rows[1:]:
        assert abs(subspace_alignment(rows[0], r) - 1.0) < 1e-9
    # ... and the bases really do differ
    assert (rows[0] - rows[1]).abs().max() > 1e-3


def test_config_pins_the_base_seed_for_both_rotation_arms():
    for arm in ("D2_rot_rand", "D3_rot_learn"):
        assert "base_seed" in CFG["fiber"]["arms"][arm], arm
