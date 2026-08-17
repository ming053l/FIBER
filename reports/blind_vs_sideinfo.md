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

The control that isolates information from capacity is a shuffled-conditioning placebo.
But `f(Y)` and `f(Y, S)` still differ in architecture, so three arms leave the
capacity step and the distribution step tangled. **Four arms**, each isolating one thing:

| arm | input | isolates |
|---|---|---|
| `f(Y)` | image only | the legacy blind reference |
| `f(Y, S_null)` | side architecture, a **fixed** conditioning (null token / global mean) | architecture and capacity alone — no sample-specific information of any kind |
| `f(Y, S_shuffled)` | side architecture, the empirical `S` marginal, wrong pairing | having the real conditioning *distribution* without the pairing |
| `f(Y, S_correct)` | side architecture, correct `S` | the paired side-information effect |

so each step answers one question:

    f(Y)            -> f(Y, S_null)        architecture / capacity effect
    f(Y, S_null)    -> f(Y, S_shuffled)    real conditioning distribution, no pairing
    f(Y, S_shuffled)-> f(Y, S_correct)     PAIRED side information -- the quantity of interest

`S_null` must be registered in advance (a null token, or the split's mean conditioning),
not chosen once the numbers are in. Architecture, input dimensionality, parameter count,
optimiser and budget are identical across the last three.

The pattern that would support a side-information effect is a flat first two steps and a
large third. The converse — a gain already at `f(Y, S_null)` — would say the extra width
alone is doing work, and no arm's number could then be read as an information effect.

Note this is the same logical shape as the P0-5 receiver control (which isolated the
pooling architecture from the frame) and the C3 frozen-Householder arm (which isolated
the parameterisation from the learning). It is not a new kind of control for this project,
which is a good sign.

### The shuffle must be a different-conditioning derangement

`pi(i) != i` is **not** enough. If two samples share a conditioning, `S_pi(i)` can equal
`S_i` and the "placebo" is then fed correct side information for those samples — the
control silently leaks the thing it exists to withhold. The requirement is on the value,
not the index:

    S_pi(i) != S_i      for every i,   not merely   pi(i) != i

Implementation: digest each conditioning tensor, then draw a derangement constrained on
digests — within the split, fixed seed, and the realised permutation written into the run
manifest so the placebo is auditable rather than re-derivable-in-principle. If no valid
derangement exists (a split dominated by one conditioning), that must fail loudly, not
fall back to an ordinary shuffle.

Measured on the current pilot index: 2000 train samples, 2000 distinct prompts, maximum
repeat count 1 — so today the guard degenerates to an ordinary derangement. That is a
property of the config (`sample_prompts` draws without replacement, one prompt per
sample) and not a law; a smaller `num_train_prompts` would reintroduce collisions, which
is exactly when a silent fallback would be most damaging.

The permutation must also be **applied once and frozen**, like any other protocol choice.
Re-randomising each epoch turns the placebo into an easier noise-averaging task, and it
could then beat the blind arm for a reason that has nothing to do with information.

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
