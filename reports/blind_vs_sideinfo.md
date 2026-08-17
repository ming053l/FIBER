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

Fixed notation, so that "conditioning" never has to be disambiguated later:

    P = the raw prompt (text)
    S = C(P) = the conditioning TENSOR actually supplied to the generator

The side information in this experiment is **S, never P**. That choice asks "what if the
receiver knows what the generator knew", not "which text encoder is better", and takes
the CLIP-vs-one-hot-vs-raw-text question out of the primary comparison entirely. A
compression ladder `Y ⊂ (Y, q(S)) ⊂ (Y, S)` is a separate follow-up.

    m_b(Y)    = E[Z | Y]
    m_s(Y, S) = E[Z | Y, S]
    C_obs^blind = Cov(m_b),   C_obs^side = Cov(m_s)

Since `E[m_s(Y, S) | Y] = m_b(Y)` by the tower property, the law of total covariance gives

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

Every held-out sample carries `(Z_i, Y_i, S_i)` at once. On a frozen `V`:

    A_i = Z_i V^T,   B_i^b = f_b(Y_i) V^T,   B_i^s = f_s(Y_i, S_i) V^T

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

## The control that decides whether any of this means anything

A side teacher takes an extra input, so it differs from the blind teacher in **two** ways
at once: it knows more, and it is a bigger model with a wider effective representation.
If the experiment returns `D_bs >> D_bb`, the first question asked will be whether the
conditioning helped or whether the larger model simply learned more easily. Without an
answer prepared in advance, the number is not interpretable and should not be produced.

The control that isolates information from capacity is a **shuffled-conditioning
placebo**: the same architecture, fed `S` from a DIFFERENT sample.

    blind                 f(Y)
    capacity-matched      f(Y, S_shuffled)     S permuted across the split
    true side information  f(Y, S_correct)

Across the second and third arms the architecture, the input dimensionality, the
parameter count, the optimiser and the budget are identical. The single difference is
whether `S` is correctly paired with `(Y, Z)`. The hierarchy to look for is

    (Y, S_shuffled) ~ Y     and     (Y, S_correct) >> (Y, S_shuffled)

which cannot be explained by capacity, because the placebo has all of it and none of the
information. The converse pattern — the placebo already gaining over blind — would say
the extra width alone is doing work, and the side arm's number could not then be read as
an information effect at all.

Note this is the same logical shape as the P0-5 receiver control (which isolated the
pooling architecture from the frame) and the C3 frozen-Householder arm (which isolated
the parameterisation from the learning). It is not a new kind of control for this project,
which is a good sign.

The permutation must be **within the split and applied once**, frozen like any other
protocol choice, so a "placebo" is not silently re-randomised each epoch into an easier
noise-averaging task.

## Not yet decided

* Whether the side teacher sees `S` concatenated at the trunk output or fused earlier.
  The placebo above controls for capacity at whichever choice is made, but the choice
  still has to be registered rather than tuned.
* Whether `E_s` and `E_b` can be equalised by construction (identical architecture and
  budget, differing only in the conditioning input) well enough for the difference to be
  interpretable. If not, the honest object is the pair of certificates, never their
  difference.
* Sample size. `A_operator` was 537 at pilot scale; a higher-dimensional teacher input
  makes that harder, not easier, and the learning curve says nothing about this
  experiment.
