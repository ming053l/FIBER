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
