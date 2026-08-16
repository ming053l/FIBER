"""Decoder-certified observability operator (P0-1).

The quantity FIBER wants is

    C_obs = Cov(E[Z|Y]).

It is not available: E[Z|Y] is unknown and only an approximate teacher f(Y) exists.
Estimating C_obs by Cov(f) is INVALID, because for f != E[Z|Y] there is no PSD
ordering Cov(f) <= C_obs — an over-scaled teacher inflates it without bound
(measured: f = 3m gives max eig(Cov(f) − C_obs) = +7.97).

The certified operator instead is

    C_cert(f) = E[ z_c f_c^T + f_c z_c^T − f_c f_c^T ],       z_c = Z − E[Z],  f_c = f − E[f]

with two properties that make it the right object:

    v^T C_cert v = Var(v^T Z) − E[(v^T z_c − v^T f_c)^2]      ( = 1 − MSE_v for Z ~ N(0,I) )
    C_cert(f)    = C_obs − Cov(m − f)  <=  C_obs                (PSD order)

The subtracted term is a COVARIANCE, not a second moment: the operator centers
Z and f, so a constant decoder bias is calibrated away rather than charged
against the decoder. `E[(m−f)(m−f)^T]` is the UNCENTERED statement and the two
agree only when E[f] = E[m].

So a weak decoder UNDERSTATES observability; it can never manufacture it. What the
top-k eigenvectors give is therefore the best k-dimensional readout *certified by
this decoder class* — never "the true C_obs".

Two consequences that shape the implementation:

1. **C_cert is indefinite.** A badly scaled teacher produces large NEGATIVE
   eigenvalues (measured range [−3.00, +0.014] for f = 3m). Any solver that ranks
   by |eigenvalue| — every SVD-based routine — then returns the directions the
   teacher is WORST on as the "top-k", silently inverting the experiment. The top-k
   must come from an ALGEBRAICALLY-largest symmetric eigensolver.
2. **Tr(C_cert) is not an observability mass.** It can be negative. The reported
   scalar is the positive part

       D_cert^(k) = sum_{j<=k} max(lambda_j, 0)

   with the negative mass  D^- = sum_j max(−lambda_j, 0)  reported separately as a
   decoder-misspecification diagnostic. The raw signed spectrum is always preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# scipy's Lanczos is the reference implementation: `which="LA"` is algebraically
# largest, which is exactly the guarantee we need and exactly what SVD lacks.
from scipy.sparse.linalg import LinearOperator, eigsh

DENSE_LIMIT = 4096          # dense() is a test/debug path, never a production one


def _as_2d(x, dtype=np.float64) -> np.ndarray:
    a = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=dtype)
    if a.ndim != 2:
        raise ValueError(f"expected [N, d], got {a.shape}")
    return a


class CertifiedObservabilityOperator:
    """Matrix-free symmetric operator C_cert(f). Never forms d x d (d = 16384).

        C_cert V = ( Zc^T (Fc V) + Fc^T (Zc V) − Fc^T (Fc V) ) / N
    """

    def __init__(self, Z, F, center: bool = True, dtype=np.float64, ddof: int = 1):
        Z, F = _as_2d(Z, dtype), _as_2d(F, dtype)
        if Z.shape != F.shape:
            raise ValueError(f"Z {Z.shape} and f(Y) {F.shape} must align sample-wise")
        self.N, self.d = Z.shape
        self.center = bool(center)
        self.z_mean = Z.mean(0) if center else np.zeros(self.d, dtype=dtype)
        self.f_mean = F.mean(0) if center else np.zeros(self.d, dtype=dtype)
        self.Zc = Z - self.z_mean
        self.Fc = F - self.f_mean
        self.dtype = dtype
        # Z and f are SAMPLE-centered, so the unbiased denominator is N-1. With N the
        # estimator is shrunk by (N-1)/N -- immaterial at N >= 256, but then the word
        # "unbiased" would not be literally true.
        self.ddof = int(ddof)
        self.denom = max(self.N - self.ddof, 1)

    @property
    def shape(self):
        return (self.d, self.d)

    def matmat(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=self.dtype)
        return (self.Zc.T @ (self.Fc @ V)
                + self.Fc.T @ (self.Zc @ V)
                - self.Fc.T @ (self.Fc @ V)) / self.denom

    def matvec(self, v: np.ndarray) -> np.ndarray:
        return self.matmat(np.asarray(v, dtype=self.dtype).reshape(self.d, 1)).ravel()

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(self.shape, matvec=self.matvec, matmat=self.matmat,
                              rmatvec=self.matvec, dtype=self.dtype)

    def trace(self) -> float:
        """Tr(C_cert), exact in O(Nd) and SIGNED. Reported as a consistency check
        against the recovered spectrum, never as the observability mass."""
        return float((2 * (self.Zc * self.Fc).sum() - (self.Fc * self.Fc).sum()) / self.denom)

    def dense(self) -> np.ndarray:
        if self.d > DENSE_LIMIT:
            raise MemoryError(f"dense() refused for d={self.d}: use the matrix-free path")
        return (self.Zc.T @ self.Fc + self.Fc.T @ self.Zc - self.Fc.T @ self.Fc) / self.denom

    @property
    def max_rank(self) -> int:
        """range(C_cert) is spanned by the rows of Zc and Fc, so rank <= 2N."""
        return min(2 * self.N, self.d)


def quadratic_form(Z, F, V, center: bool = True, ddof: int = 1) -> np.ndarray:
    """lambda_skill_j = v_j^T C_cert v_j evaluated on the given samples.

    This is the CONSERVATIVE per-direction quantity: Var(v^T Z) minus the decoder's
    squared error along v. Cross-fitting it on a held-out split is what makes the
    reported eigenvalues honest.
    """
    Z, F, V = _as_2d(Z), _as_2d(F), _as_2d(V)
    a, b = Z @ V.T, F @ V.T
    if center:
        a, b = a - a.mean(0), b - b.mean(0)
    denom = max(a.shape[0] - ddof, 1)
    return (2 * (a * b).sum(0) - (b * b).sum(0)) / denom


# Numerical-rank tolerance for calling a projected eigenvalue positive. Standard
# relative rule: anything within k * eps of the largest magnitude is numerical zero,
# NOT certified observability. tau is reported alongside every rank.
def zero_tolerance(mu: np.ndarray, floor: float = 1e-9) -> float:
    k = max(len(mu), 1)
    return max(floor, k * np.finfo(np.float64).eps * float(np.abs(mu).max(initial=0.0)))


def project_operator(Z, F, V, center: bool = True, ddof: int = 1) -> np.ndarray:
    """C_V = V C_cert V^T, the certified operator RESTRICTED to span(V).

        A = Zc V^T,  B = Fc V^T,  C_V = (A^T B + B^T A - B^T B) / N

    Never touches d x d. Under any within-subspace rotation V' = R V (R in O(k)),
    C_V' = R C_V R^T, so its EIGENVALUES are a property of the subspace alone.
    """
    Z, F, V = _as_2d(Z), _as_2d(F), _as_2d(V)
    A, B = Z @ V.T, F @ V.T
    if center:
        A, B = A - A.mean(0), B - B.mean(0)
    C = (A.T @ B + B.T @ A - B.T @ B) / max(A.shape[0] - ddof, 1)
    return (C + C.T) / 2


def subspace_certificate(Z, F, V, center: bool = True, tol: float | None = None,
                         crossfit: bool = True, seed: int = 0) -> dict:
    """Cross-fitted, BASIS-INVARIANT score of a frozen discovered subspace.

        D_cert_subspace(V) = sum_j max(mu_j, 0),   mu = eig(V C_cert^held V^T)

    The discovery frame V diagonalises the DISCOVERY operator, not the held-out one,
    so `sum_j max(v_j^T C_held v_j, 0)` -- clipping the DIAGONAL -- is not a property
    of the subspace. Concretely, C_V = [[-1, 2], [2, -1]] has diagonal (-1, -1),
    which clips to 0, while its eigenvalues are (1, -3): the subspace does contain a
    positively certified direction, one rotation away.

    The per-coordinate diagonal is still returned as `coordinate_skill` -- it is the
    right diagnostic for sign coding, which IS basis dependent (P0-7) -- but it is
    never the observability headline.

    **The eigen-step needs its own cross-fit.** Summing positive eigenvalues is a
    max-type functional, so computing the eigenvectors and the eigenvalues on the same
    held-out samples rectifies noise into positive mass: measured on a decoder with
    ZERO true skill (f independent of Z), the in-sample version reports D = 3.35 at
    N=64, k=32 and 0.59 at N=256, k=32, where the truth is 0. That is the same
    selection error as trusting the discovery basis, moved one level down into the
    subspace. So by default the within-subspace rotation is chosen on one half of the
    held-out samples and measured on the other. Invariance survives: replacing V by RV
    rotates both halves' operators by R, the chosen rotation absorbs it, and the
    measured mu are unchanged.
    """
    Z, F, V = _as_2d(Z), _as_2d(F), _as_2d(V)
    C_V = project_operator(Z, F, V, center=center)
    # ascontiguousarray: the [::-1] view has a negative stride, which torch refuses
    mu_in = np.ascontiguousarray(np.linalg.eigvalsh(C_V)[::-1])

    mu, rotation_split = mu_in, None
    if crossfit and np.shape(Z)[0] >= 8:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(np.shape(Z)[0])
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        C1 = project_operator(Z[h1], F[h1], V, center=center)
        W = np.ascontiguousarray(np.linalg.eigh(C1)[1][:, ::-1].T)   # rotation from half 1
        C2 = project_operator(Z[h2], F[h2], V, center=center)
        mu = np.ascontiguousarray(np.diag(W @ C2 @ W.T))             # measured on half 2
        rotation_split = [int(h1.size), int(h2.size)]

    tau = zero_tolerance(mu) if tol is None else float(tol)
    positive = mu[mu > tau]
    return {
        # ---- subspace (basis-invariant) --------------------------------
        "mu": mu,
        "crossfit": bool(rotation_split is not None),
        "rotation_split": rotation_split,
        # same score without the inner cross-fit: optimistic, kept for contrast
        "D_cert_subspace_insample": float(np.clip(mu_in, 0, None).sum()),
        "mu_insample": mu_in,
        "D_cert_subspace": float(positive.sum()),
        # Unbiased companion: sum(mu) = tr(C_V), basis-invariant and NOT clipped, so
        # it estimates the TRUE restricted trace instead of being rectified upward.
        # Clipping is what leaves the headline with a residual positive bias at small
        # N (measured 0.44 at N=64, k=32 where the certified mass is 0; 0.00 by
        # N=256). Report both: the mass answers "how much is certified", the trace
        # answers "is this decoder net-positive at all" and can be negative.
        "trace_C_V": float(mu.sum()),
        # NOT a significance statement: tau only rules out floating-point noise. A
        # statistically certified rank needs a one-sided LCB > 0 on the measurement
        # half (recorded as pre-Gate-3 work in reports/p0_fix_plan.md).
        "numerical_positive_rank": int(positive.size),
        "certified_positive_rank": int(positive.size),   # deprecated alias
        "requested_k": int(V.shape[0]),
        # mu.max()/min(), NOT mu[0]/mu[-1]: after the inner cross-fit mu is ordered by
        # the rotation chosen on half 1, so it is no longer sorted descending.
        "mu_max": float(mu.max()) if mu.size else float("nan"),
        "mu_min": float(mu.min()) if mu.size else float("nan"),
        "zero_tolerance": tau,
        # ---- basis / coordinates (NOT invariant) -----------------------
        "coordinate_skill": np.diag(C_V).copy(),
        "D_coordinate_clipped": float(np.clip(np.diag(C_V), 0, None).sum()),
    }


def variance_form(F, V, center: bool = True, ddof: int = 1) -> np.ndarray:
    """lambda_var_j = Var(v_j^T f(Y)). Equals lambda_skill ONLY when f is the exact
    conditional mean, so the gap between them is the teacher-validity diagnostic."""
    F, V = _as_2d(F), _as_2d(V)
    b = F @ V.T
    if center:
        b = b - b.mean(0)
    return (b * b).sum(0) / max(b.shape[0] - ddof, 1)


@dataclass
class CertifiedSpectrum:
    eigenvalues: np.ndarray          # [r] SIGNED, algebraically descending (the kept ones)
    eigenvectors: np.ndarray         # [r, d] orthonormal rows
    trace_signed: float              # Tr(C_cert), exact, O(Nd)
    n_samples: int
    eigenvalues_full: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spectrum_is_complete: bool = False   # True when every non-zero eigenvalue is known
    solver: str = "range_eigh"
    meta: dict = field(default_factory=dict)

    # ---- the reported scalars ------------------------------------------
    def d_cert(self, k: int | None = None) -> float:
        """D_cert^(k) = sum_{j<=k} max(lambda_j, 0): the variance the decoder can
        CERTIFY it recovers in its best k directions. Not Shannon capacity."""
        lam = self.eigenvalues if k is None else self.eigenvalues[:k]
        return float(np.clip(lam, 0, None).sum())

    def positive_mass_is_complete(self) -> bool:
        """True when no positive eigenvalue can have been missed: either the whole
        non-zero spectrum is known (range_eigh), or the smallest computed algebraic
        eigenvalue is already <= 0 so everything below it is too."""
        return bool(self.spectrum_is_complete
                    or (self.eigenvalues.size and self.eigenvalues[-1] <= 0))

    def total_positive_mass(self) -> float:
        lam = self.eigenvalues_full if self.eigenvalues_full.size else self.eigenvalues
        return float(np.clip(lam, 0, None).sum())

    def negative_mass(self) -> float:
        """D^- = sum_j max(-lambda_j, 0): how much the decoder is actively WORSE
        than predicting the prior mean. Large values mean teacher misspecification
        (or, in-sample, plain overfitting), NOT negative observability."""
        lam = self.eigenvalues_full if self.eigenvalues_full.size else self.eigenvalues
        return float(np.clip(-lam, 0, None).sum())

    def summary(self, k: int | None = None) -> dict:
        lam = self.eigenvalues
        return {
            "d_cert_k": self.d_cert(k),
            "d_cert_all_computed": self.total_positive_mass(),
            "positive_mass_is_complete": self.positive_mass_is_complete(),
            "negative_mass": self.negative_mass(),
            "trace_signed": self.trace_signed,
            "lambda_max": float(lam[0]) if lam.size else float("nan"),
            "lambda_min_computed": float(lam[-1]) if lam.size else float("nan"),
            "n_positive": int((lam > 0).sum()),
            "n_computed": int(lam.size),
            "n_samples": self.n_samples,
            "solver": self.solver,
            **self.meta,
        }


def fit_certified(Z, F, k: int, oversampling: int = 32, seed: int = 0, center: bool = True,
                  method: str = "range_eigh", tol: float = 0.0,
                  maxiter: int | None = None, n_negative: int = 16) -> CertifiedSpectrum:
    """Top-k of C_cert by ALGEBRAICALLY largest eigenvalue.

    Never SVD and never `which="LM"`: for an indefinite operator those return the
    most negative directions, i.e. exactly the coordinates the decoder fails on.

    method="range_eigh" (default, exact)
        range(C_cert) is spanned by the rows of Zc and Fc, so with G = [Zc^T, Fc^T]
        (d x 2N) and a thin QR G = QR,

            C_cert = Q (R M R^T) Q^T,
            R M R^T = (R1 R2^T + R2 R1^T - R2 R2^T)/N,   R = [R1 R2]

        which is a 2N x 2N symmetric problem solved exactly by dense eigh. This is
        preferred over Lanczos because in production d = 16384 >> 2N, so the zero
        eigenvalue has multiplicity d - 2N and ARPACK stalls inside that degenerate
        cluster (observed: "No convergence, 8/11 eigenvectors converged"). It also
        returns the COMPLETE non-zero spectrum, making the positive and negative
        masses exact instead of tail estimates.

    method="eigsh" (reference)
        scipy Lanczos with which="LA". Kept as an independent cross-check of the
        exact route, and used as such in tests.
    """
    op = CertifiedObservabilityOperator(Z, F, center=center)
    r = int(min(k + oversampling, op.max_rank - 1, op.d - 2))
    if r < k:
        raise ValueError(f"cannot resolve k={k} directions from N={op.N} samples "
                         f"(rank(C_cert) <= 2N = {op.max_rank})")
    meta = {"oversampling": oversampling, "seed": seed, "centered": center,
            "max_rank": op.max_rank, "d": op.d, "k_requested": k}

    if method == "range_eigh":
        N = op.N
        G = np.concatenate([op.Zc.T, op.Fc.T], axis=1)      # d x 2N
        Q, R = np.linalg.qr(G)                              # Q d x p, R p x 2N
        R1, R2 = R[:, :N], R[:, N:]
        B = (R1 @ R2.T + R2 @ R1.T - R2 @ R2.T) / op.denom  # p x p, symmetric
        B = (B + B.T) / 2
        w, P = np.linalg.eigh(B)                            # ascending, exact
        order = np.argsort(w)[::-1]
        w, P = w[order], P[:, order]
        vecs = (Q @ P[:, :r]).T
        return CertifiedSpectrum(
            eigenvalues=w[:r], eigenvectors=np.ascontiguousarray(vecs),
            trace_signed=op.trace(), n_samples=op.N, eigenvalues_full=w,
            spectrum_is_complete=True, solver="range_eigh", meta=meta)

    if method == "eigsh":
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(op.d)          # seeded: Lanczos start is reproducible
        lin = op.as_linear_operator()
        vals, vecs = eigsh(lin, k=r, which="LA", v0=v0, tol=tol, maxiter=maxiter)
        order = np.argsort(vals)[::-1]                      # eigsh returns ascending
        vals, vecs = vals[order], vecs[:, order]
        full = vals
        n_negative = int(min(n_negative, op.max_rank - r - 1, op.d - r - 2))
        if n_negative > 0:
            neg, _ = eigsh(lin, k=n_negative, which="SA", v0=v0, tol=tol, maxiter=maxiter)
            full = np.concatenate([vals, neg])
        return CertifiedSpectrum(
            eigenvalues=vals, eigenvectors=np.ascontiguousarray(vecs.T),
            trace_signed=op.trace(), n_samples=op.N, eigenvalues_full=full,
            spectrum_is_complete=False, solver="eigsh_la", meta=meta)

    raise ValueError(f"unknown method {method!r}; use 'range_eigh' or 'eigsh'")


def teacher_validity(Z_held, F_held, V, center: bool = True) -> dict:
    """Cross-fit validity report for the recovered directions.

    lambda_var  = Var(v^T f(Y))                       (what Cov(f) would have reported)
    lambda_skill= Var(v^T Z) − E[(v^T z_c − v^T f_c)^2]  (what C_cert certifies)

    They coincide only for an exact conditional mean, so their gap measures how far
    the teacher is from E[Z|Y]. lambda_skill is always the conservative one.

    Both are PER-COORDINATE and therefore basis dependent. The reported observability
    headline is the basis-invariant `subspace` block (P0-1.1).
    """
    lam_skill = quadratic_form(Z_held, F_held, V, center=center)
    lam_var = variance_form(F_held, V, center=center)
    gap = np.abs(lam_var - lam_skill)
    F = _as_2d(F_held)
    return {
        # per-coordinate: basis dependent, diagnostics only (P0-1.1)
        "lambda_skill": lam_skill,
        "lambda_var": lam_var,
        "mean_abs_gap": float(gap.mean()),
        "max_abs_gap": float(gap.max()) if gap.size else 0.0,
        "n_negative_directions": int((lam_skill < 0).sum()),
        # subspace: basis invariant, THE headline
        "subspace": subspace_certificate(Z_held, F_held, V, center=center),
        # a teacher with a non-zero output mean injects a rank-1 direction into any
        # UNcentered estimate; report it so centering cannot be quietly dropped
        "teacher_output_mean_norm": float(np.linalg.norm(F.mean(0))),
        "n_heldout": int(F.shape[0]),
    }
