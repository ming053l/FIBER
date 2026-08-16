"""Teacher-output covariance  Cov(f(Y))  —  A DIAGNOSTIC, NOT C_obs.

This module used to be presented as an estimator of the Generative Observability
Spectrum. It is not one, and P0-1 demoted it:

    Cov(f) == C_obs  ONLY when f is the exact conditional mean E[Z|Y].

For any approximate teacher there is no PSD ordering `Cov(f) <= C_obs` — an
over-scaled teacher inflates it without bound (measured: f = 3m gives
max eig(Cov(f) - C_obs) = +7.97). The valid, certified lower-bound operator lives
in `fiber.spectrum.certified` and is what discovery and reporting now use.

What survives here is the quantity `lambda_var = Var(vᵀf(Y))`, whose GAP against
the certified `lambda_skill` is the teacher-validity diagnostic
(`certified.teacher_validity`). Nothing in this module may be reported as C_obs,
as an observability mass, or as a channel capacity.

With a teacher M_theta(Y) and m_i = M_theta(y_i),

    Cov(f) = (1/N) Σ_i (m_i - m̄)(m_i - m̄)ᵀ   (d x d, NEVER formed: d = 16384)

Two estimators, both reading the eigenvectors of Ĉ_obs off M = [m_1 … m_N]ᵀ
without materialising Ĉ_obs:

  * "randomized"  — randomized range finder + power iterations (config default)
  * "gram"        — exact when N <= d: eig of the N x N Gram matrix M Mᵀ / N.
                    Used by tests/test_spectrum_synthetic.py to check the
                    randomized path against ground truth.

MSE IS MANDATORY for the teacher. L2 regression converges to the conditional
MEAN; L1 converges to the conditional MEDIAN, and the identity λ = 1 - MMSE
above is a law-of-total-variance statement about the MEAN. A median teacher
silently corrupts C_obs — see tests/test_spectrum_synthetic.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

LAMBDA_TOL = 1e-3  # numerical slack on top of the finite-sample allowance below


@dataclass
class Spectrum:
    """Spectrum of Cov(f(Y)) — lambda_var. Never report this as C_obs."""

    eigenvalues: torch.Tensor           # [r], descending, PSD
    eigenvectors: torch.Tensor          # [r, d], orthonormal rows
    trace: float                        # Tr(Ĉ_obs), computed exactly (not from the top-r)
    n_samples: int
    method: str
    meta: dict = field(default_factory=dict)

    def topk(self, k: int) -> torch.Tensor:
        if k > self.eigenvectors.shape[0]:
            raise ValueError(f"asked for {k} directions, estimator kept "
                             f"{self.eigenvectors.shape[0]}")
        return self.eigenvectors[:k].contiguous()

    def evaluate_on(self, M_held: torch.Tensor) -> torch.Tensor:
        """Cross-fit eigenvalues: directions from THIS fit, variance measured on a
        held-out set of teacher outputs. Removes the in-sample upward bias, so
        these are the numbers a report should quote."""
        proj = M_held.double() @ self.eigenvectors.double().T
        return (proj.pow(2).mean(0)).float()

    def effective_rank(self) -> float:
        """exp(entropy of the normalised spectrum) — how spread the survival is."""
        p = self.eigenvalues / self.eigenvalues.sum().clamp_min(1e-12)
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()))

    def summary(self) -> dict:
        lam = self.eigenvalues
        return {
            "trace_c_obs": self.trace,
            "lambda_max": float(lam[0]),
            "lambda_top64_mean": float(lam[:64].mean()),
            "lambda_median_of_kept": float(lam.median()),
            "effective_rank": self.effective_rank(),
            "n_samples": self.n_samples,
            "method": self.method,
            **self.meta,
        }


def trace_teacher_covariance(M: torch.Tensor, center: bool = True) -> float:
    """Tr(Cov(f)) = ‖M − m̄‖_F² / N, exact and O(Nd).

    A DIAGNOSTIC. Not Tr(C_obs), not an observability mass, not capacity. Without
    centering, a teacher output bias m̄ contributes ‖m̄‖² of pure artefact.
    """
    Md = M.double()
    if center:
        Md = Md - Md.mean(0, keepdim=True)
    return float(Md.pow(2).sum() / M.shape[0])


def _finite_sample_ceiling(trace: float, lam_max: float, N: int) -> float:
    """In-sample eigenvalues of a covariance are upward biased: with effective
    rank r = Tr(C)/lambda_max, the Marchenko-Pastur edge sits at
    (1 + sqrt(r/N))² times the population value. So `lambda > 1` in-sample is NOT
    by itself a bug; only `lambda` above this ceiling is.

    The honest fix is cross-fitting, not a bigger tolerance: fit the directions on
    one split and re-measure the eigenvalues on another (`Spectrum.evaluate_on`),
    which removes the bias instead of budgeting for it.
    """
    r = max(1.0, trace / max(lam_max, 1e-12))
    return (1.0 + (r / max(N, 1)) ** 0.5) ** 2 + LAMBDA_TOL


def _check_spectrum(lam: torch.Tensor, trace: float, N: int, strict: bool) -> None:
    """C_obs ⪯ I  =>  population λ ≤ 1. A λ far above 1 means the teacher is not a
    conditional mean (wrong loss, leaked target, or z-scaling bug)."""
    if not lam.numel():
        return
    lam_max = float(lam.max())
    ceiling = _finite_sample_ceiling(trace, lam_max, N)
    if lam_max > ceiling:
        msg = (f"lambda_max = {lam_max:.4f} exceeds the finite-sample ceiling "
               f"{ceiling:.4f} (N={N}): C_obs must be ⪯ I. Check the teacher loss "
               "(must be MSE), target scaling, and that the teacher never sees z.")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)


def fit_gram(M: torch.Tensor, k: int | None = None, strict: bool = True,
             center: bool = True) -> Spectrum:
    """Exact top-r eigenpairs of Cov(f) via the N x N Gram matrix. Requires N <= d.

    DIAGNOSTIC ONLY (see the module docstring): these are lambda_var, not C_obs.
    """
    N, d = M.shape
    Md = M.double()
    if center:
        Md = Md - Md.mean(0, keepdim=True)
    G = (Md @ Md.T) / N
    evals, evecs = torch.linalg.eigh(G)              # ascending
    evals = evals.flip(0).clamp_min(0.0)
    evecs = evecs.flip(1)
    r = min(k or N, N)
    evals, evecs = evals[:r], evecs[:, :r]
    # V = Mᵀ U / sqrt(N λ)  are the unit eigenvectors of Ĉ_obs = MᵀM/N
    V = (Md.T @ evecs) / (N * evals).clamp_min(1e-30).sqrt()
    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-30)
    lam = evals.float()
    tr = trace_teacher_covariance(M, center=center)
    _check_spectrum(lam, tr, N, strict)
    return Spectrum(lam, V.T.float().contiguous(), tr, N, "gram")


def fit_randomized(M: torch.Tensor, k: int, oversampling: int = 32, n_iter: int = 4,
                   seed: int = 0, strict: bool = True, center: bool = True) -> Spectrum:
    """Randomized SVD of the centered M (N x d); right singular vectors are the
    eigenvectors of Cov(f) and λ_j = s_j²/N.

    DIAGNOSTIC ONLY. An SVD is legitimate here because Cov(f) is PSD, so largest
    singular value == largest algebraic eigenvalue. That equivalence FAILS for the
    certified operator, which is indefinite — see `fiber.spectrum.certified`.
    """
    N, d = M.shape
    r = min(k + oversampling, min(N, d))
    Md = M.double()
    if center:
        Md = Md - Md.mean(0, keepdim=True)
    g = torch.Generator(device=M.device).manual_seed(int(seed) % (2**63 - 1))
    Omega = torch.randn(d, r, generator=g, device=M.device, dtype=torch.float64)
    Y = Md @ Omega
    Q, _ = torch.linalg.qr(Y)
    for _ in range(n_iter):                          # power iterations
        Q, _ = torch.linalg.qr(Md.T @ Q)
        Q, _ = torch.linalg.qr(Md @ Q)
    B = Q.T @ Md                                     # r x d
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    lam = (S.pow(2) / N)[:r].float()
    V = Vh[:r]
    tr = trace_teacher_covariance(M, center=center)
    _check_spectrum(lam, tr, N, strict)
    return Spectrum(lam, V.float().contiguous(), tr, N, "randomized",
                    meta={"oversampling": oversampling, "n_iter": n_iter, "seed": seed})


def fit_spectrum(M: torch.Tensor, k: int, cfg: dict | None = None, seed: int = 0,
                 strict: bool = True) -> Spectrum:
    """Diagnostic entry point for Cov(f). Discovery uses `certified.fit_certified`."""
    cfg = dict(cfg or {})
    loss = str(cfg.get("teacher_loss", "mse")).lower()
    if loss != "mse":
        raise ValueError(
            f"teacher_loss={loss!r}: the observability spectrum is defined through "
            "the conditional MEAN. L1/Huber converge to the conditional median and "
            "silently corrupt C_obs (PLAN.md §3.3)."
        )
    method = str(cfg.get("estimator", "randomized_svd")).lower()
    if method in ("gram", "exact"):
        return fit_gram(M, k=cfg.get("gram_rank"), strict=strict)
    return fit_randomized(M, k=k, oversampling=int(cfg.get("oversampling", 32)),
                          n_iter=int(cfg.get("num_power_iters", 4)), seed=seed, strict=strict)


def principal_cosines(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """cos^2 of the principal angles between the row-spaces of A and B, descending.

    The MEAN of these is `subspace_alignment`, and a mean can hide structure: a few
    directions agreeing strongly looks the same as every direction agreeing weakly, and
    those are different scientific statements. Report the spectrum.
    """
    Qa, _ = torch.linalg.qr(A.double().T)
    Qb, _ = torch.linalg.qr(B.double().T)
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return s.pow(2)      # svdvals is already descending


def subspace_alignment(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between row-spaces of A and B.
    1.0 = same subspace, ~k/d = unrelated. Used to compare per-attack spectra."""
    Qa, _ = torch.linalg.qr(A.double().T)
    Qb, _ = torch.linalg.qr(B.double().T)
    s = torch.linalg.svdvals(Qa.T @ Qb)
    return float(s.pow(2).mean())
