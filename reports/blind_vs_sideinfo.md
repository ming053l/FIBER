# Blind vs side-information observability — design note

**Status: design only. Not registered, not run, nothing implemented.** Written down now
because the k diagnostic closed the two hypotheses that pointed elsewhere, and because
the confounds below have to be fixed before a single number is produced.

## Why this, and not 10k

| hypothesis | status | evidence |
|---|---|---|
| sample count is the limiter | out, at fixed compute | `reports/powercurve.md`: 100 -> 928 gives no reproducible improvement, monotone in 0/3 and 1/3 seeds |
| 64 simultaneous targets is too hard | out | same report: k=8 on the top-8 directions sits at 0.825x the null |
| the receiver's information set is the limiter | **untested** | prompt-assisted DDIM inversion reached sign BER 0.119 on the same images where the prompt-free extractor sits at 0.50 |

That last row has been carried since Phase 0 as a "prompt-assisted reference that violates
the receiver protocol" — a footnote to a number that failed. If the two hypotheses above
are out, it stops being a footnote and becomes the obvious question.

## The quantity

Let `P` be the conditioning the generator actually received, and `C(P)` its encoding —
**not** raw prompt text. Defining side information as the generator's own conditioning
variable asks "what if the receiver knows what the generator knew", not "which NLP
encoder is better", and removes the CLIP-vs-one-hot-vs-text confound from the primary
comparison. A compression ladder `Y ⊂ (Y, q(P)) ⊂ (Y, P)` is a separate follow-up.

    m_b(Y)    = E[Z | Y]
    m_s(Y, P) = E[Z | Y, P]
    C_obs^blind = Cov(m_b),   C_obs^side = Cov(m_s)

Since `E[m_s | Y] = m_b` by the tower property, the law of total covariance gives

    Cov(m_s) = Cov(E[m_s | Y]) + E[Cov(m_s | Y)] = Cov(m_b) + E[Cov(m_s | Y)]

so the **side-information observability gain**

    Delta C_side = C_obs^side - C_obs^blind = E[Cov(m_s | Y)] >= 0

is positive semidefinite as a theorem, not as a hypothesis. It has a direct reading: how
much the posterior mean still moves once Y is known.

## Registered caveat — the estimator does not inherit the ordering

FIBER never estimates `C_obs`. It estimates `C_cert(f) = C_obs - Cov(m - f) <= C_obs` for
an approximate decoder. With two different decoders,

    C_cert^side(f_s)  = C_obs^side  - E_s,      E_s = Cov(m_s - f_s) >= 0
    C_cert^blind(f_b) = C_obs^blind - E_b,      E_b = Cov(m_b - f_b) >= 0
    =>  C_cert^side - C_cert^blind = Delta C_side - (E_s - E_b)

so the estimated difference **can** have negative eigenvalues even though the population
difference cannot. To be registered before running:

> Since Bayes observability is monotone under additional side information, statistically
> credible negative directions in the estimated side-minus-blind certified operator
> invalidate a direct Bayes-level interpretation of the decoder-certified difference.

The causes are not separable a priori — side-teacher approximation error, asymmetry
between the two teachers' errors, finite-sample estimation, or all three — so a negative
result may **not** be reported as "the side teacher underfit", and equally may not be
reported as "side information does not help".

## Design: fix the geometry, then the 2x2

Letting each information set discover its own eigenvectors changes the information set
and the subspace at once. Fix `V` first and ask whether the *same* directions become more
observable. But "which V" is itself a choice that favours one side, so report the full
matrix — the same shape `scripts/compare_teachers.py` already produces for teacher
architectures, where it separated a real effect (`V[resnet18] | f[spatial] = 1.445` vs
`V[resnet18] | f[resnet18] = 0.047`):

|  | `f_blind` | `f_side` |
|---|---|---|
| `V_blind` | `D_bb` | `D_bs` |
| `V_side` | `D_sb` | `D_ss` |

plus the principal angles between `span(V_blind)` and `span(V_side)`. This separates:

* `D_bs - D_bb` — do blind-discovered directions become more observable with conditioning
* `D_ss - D_sb` — how much the side-discovered directions depend on it
* the angles — whether the geometry itself rotates when the information set changes

## Design: the difference must be estimated paired, not by subtraction

Every held-out sample carries `(Z_i, Y_i, C(P_i))` at once. On a frozen `V`:

    A_i = Z_i V^T,   B_i^b = f_b(Y_i) V^T,   B_i^s = f_s(Y_i, C(P_i)) V^T

Estimate `Delta C_cert,V = V (C_cert^side - C_cert^blind) V^T` as **one object**,
bootstrapping the joint tuples `(Z_i, F_i^b, F_i^s)` and re-centring inside each
replicate. Shared image and latent variation then cancels within the pair; two
independent bootstraps leave it in both tables and inflate the interval. The
negative-direction check is a bound on `lambda_min(Delta C_cert,V)` or its spectral-norm
radius — not a comparison of two error-carrying tables.

**This needs the bootstrap machinery extended, not merely called twice.** The current
path (`certified.py`) centres once with the full measurement-half means and resamples the
already-centred terms; paired-difference inference specifically requires re-centring
inside the replicate.

## Not yet decided

* Whether the side teacher sees `C(P)` concatenated at the trunk output or fused earlier.
  It changes capacity as well as information, which is a confound of the same kind the
  P0-5 receiver control was built for.
* Whether `E_s` and `E_b` can be equalised by construction (identical architecture and
  budget, differing only in the conditioning input) well enough for the difference to be
  interpretable. If not, the honest object is the pair of certificates, never their
  difference.
* Sample size. `A_operator` was 537 at pilot scale; a higher-dimensional teacher input
  makes that harder, not easier, and the learning curve says nothing about this
  experiment.
