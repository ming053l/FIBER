# Blind vs side-information observability — design note

**Status: Phase A preregistered and frozen; conditioning foundation and side encoder
implemented; no Phase A outcome observed.** Scope boundary: `reports/scope_v1.md`.

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
    S = C(P) = the SAMPLE-SPECIFIC POSITIVE CFG CONDITIONING EMBEDDING
               actually used during generation

`S` is deliberately not "the UNet's conditioning tensor". Under classifier-free guidance
the UNet also consumes an unconditional branch, and that branch is a fixed part of the
**channel protocol**, not something the receiver could know about a particular sample. It
is also not a constant in the naive sense: measured, it is identical across rows within a
batch (max|diff| 0.000000) yet differs by 0.015625 between batch sizes 12 and 8/4/1, so
its text being fixed does not make its value fixed. Excluding it keeps the question
clean — *what happens when the receiver knows the generator's sample-specific textual
conditioning* — with no semantic gap to argue about.

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

## Settled, and what genuinely remains open

Fusion point, `S_null`, the margins, the replication and the side encoder are all fixed
below. Nothing about the design is open; what remains is running it.

Still genuinely open, and out of scope for v1 per `reports/scope_v1.md`:

* Sample size. `A_operator` was 537 at pilot scale; a higher-dimensional teacher input
  makes that harder, not easier, and the learning curve says nothing about this
  experiment.


---

# Interpretation gate — three quantities that must not be conflated

The four-arm ladder drives the architecture/capacity confound down a long way. It does
**not** make `E_s - E_b` vanish, and that is not a gap to be engineered away. So the
preregistration limits the claim instead of pretending to close it.

## Why "make E_s = E_b" must NOT be registered as a control

`E_s = Cov(m_s - f_s)` and `E_b = Cov(m_b - f_b)` are not observable: `m_s` and `m_b` are
the Bayes estimators, which is exactly what is unknown. Matching architecture, parameter
count, sample size, optimiser and steps does not imply `E_s = E_b`, and neither does
matching any observable loss:

    MSE(f) = MMSE + approximation error

and side information changes the irreducible `MMSE` itself, so equal training loss, equal
validation MSE or matched early stopping all fail to pin the approximation error. A
procedure that looks fair here would have no mathematical guarantee behind it.

Worse, correct `S` plausibly makes the regression *easier*, i.e. `E_s < E_b`, in which
case

    Delta C_dec = C_cert^side - C_cert^blind = Delta C_side + (E_b - E_s)

**over**states the Bayes gain. So, written down explicitly:

> The difference of two certificates is **not** a lower bound on `Delta C_side`, in
> either direction.

## The three quantities

| | object | status |
|---|---|---|
| Bayes | `Delta C_side = C_obs^side - C_obs^blind = E[Cov(m_s\|Y)] >= 0` | theorem, unobservable |
| decoder-certified | `Delta C_dec = C_cert^correct - C_cert^placebo` | estimable, **not** a bound on the above |
| operational | held-out BER / R2_lin / MSE, correct vs placebo | estimable, no covariance claim |

`Delta C_dec` is named the **decoder-certified side-information contrast**, never the
"certified gain". It answers: at a fixed decoder family, capacity and training protocol,
does correct conditioning change what can be certified? Not: has `Delta C_side` been
lower-bounded.

## Primary contrast

    f(Y, S_correct)   vs   f(Y, S_shuffled)

Identical architecture, input shape, capacity, `S` marginal and training budget; the sole
intervention is the **pairing**. This is the comparison closest to an information
intervention, and it is the primary one. `f(Y)` and `f(Y, S_null)` are diagnostic
controls, not the treatment comparison.

## Reading rule

**Level 1 — operational side-information effect.** Correct conditioning must improve
held-out recoverability relative to the capacity-matched shuffled placebo under paired
evaluation. Supports: *receiver side information has operational value.*

**Level 2 — decoder-certified observability effect.** The same ordering must appear in
decoder-certified observability on frozen subspaces, stable across the registered
receiver controls. Supports: *side information changes decoder-certified observability.*

**Level 3 — Bayes-level interpretation.** Differences between certificates are **not**
interpreted as certified lower bounds on `Delta C_side`, because the approximation errors
need not cancel. A statistically credible `lambda_min(Delta C_dec) < 0` fails the
Bayes-level interpretation; it does **not** license "side information hurts".

Permitted on Levels 1+2: *correct receiver conditioning increases operational and
decoder-certified latent observability relative to a capacity- and distribution-matched
placebo.* Not permitted without Level 3 machinery: *we certify that
`C_obs^side - C_obs^blind > 0`.*

## The placebo checks need an equivalence margin, not a failed test

`S_null ~ S_shuffled` and `f(Y) ~ f(Y, S_null)` are the controls that make the primary
contrast interpretable, and `p > 0.05` does not establish either — it is equally
consistent with insufficient power, which is exactly the objection a reviewer would
raise. Register margins in advance, `|Delta R2_lin| < eps_R` and
`|Delta D_cert| < eps_D`, and read these two comparisons as equivalence tests. The
margins must be set from the between-seed spreads already measured, before any side-info
number exists.

## Future extension: a direct certificate of the gain

Not for the pilot. It becomes worth building only if the four-arm experiment shows a
large correct-vs-shuffled effect, because it addresses `E_s - E_b` head on rather than
limiting the claim around it.

With `r(Y, S) = m_s(Y, S) - m_b(Y)`, the tower property gives `E[r | Y] = 0` and

    Delta C_side = E[Cov(m_s | Y)] = E[r r^T] = Cov(r)

Now take any `h(Y, S)` satisfying `E[h | Y] = 0` and define

    C_gain(h) = E[ Z h^T + h Z^T - h h^T ]

Since `h` is a function of `(Y, S)`, `E[Z h^T] = E[m_s h^T] = E[m_b h^T] + E[r h^T]`, and
the first term vanishes because `m_b` is `Y`-measurable and `E[h | Y] = 0`. Hence

    C_gain(h) = E[r h^T + h r^T - h h^T] = Cov(r) - Cov(r - h)  <=  Delta C_side

a genuine one-sided certificate of the side-information gain, rather than a difference of
two certificates.

**The operator machinery carries over — but only once `h` is validly constructed.**
`C_gain(h)` is algebraically identical to `C_cert(f) = E[z f^T + f z^T - f f^T]` with `f`
replaced by `h`, so `CertifiedObservabilityOperator`, `fit_certified`, the inner cross-fit
and the bootstrap apply to it unchanged. That is a statement about the estimator, not
about validity: cross-fitting prevents a residual from absorbing its own fitting noise,
and it does **not** establish `E[h | Y] = 0`. Misspecification leaves `b(Y)` behind just
the same. So this stays a Phase B extension rather than riding along with the pilot —
otherwise it opens a second certification-validity problem while the first is unresolved.

**The whole difficulty moves into `E[h | Y] = 0`.** The natural construction is
residualisation, `h = g(Y, S) - g_hat(Y)` with `g_hat` cross-fitted — and it must be
cross-fitted, or the residual absorbs its own fitting noise exactly as the discovery
basis did. The cost of getting it wrong is explicit: if `E[h | Y] = b(Y) != 0` then

    C_gain(h) = Cov(r) - Cov(r - h) + E[m_b b^T + b m_b^T]

and that last term is **sign-indefinite**, so the bound fails rather than loosening. This
route therefore does not eliminate the difficulty; it exchanges an unidentifiable
quantity (`E_s - E_b`) for one that cross-fitting can at least attack, and it makes the
failure mode checkable.


---

# Phase A — preregistration

**Locked before implementation.** Every choice below is fixed; none may be revised once
numbers exist.

## The three engineering choices

**Fusion: late.** `S` never touches the image trunk. The existing receiver is
`Y -> ResNet18 trunk -> h_Y in R^512 -> linear heads`. The side branch is
`S -> side encoder -> h_S`, and `[h_Y, h_S]` feeds the *same* prediction heads. Early
fusion or cross-attention is rejected for the pilot: if the result improved, it would not
be separable from "conditioning rewrote the image feature extractor". All three side arms
use one identical side architecture.

**`S_null` = the training-split mean conditioning tensor.**

    S_null = (1/N_train) sum_{i in train} S_i

computed once on train and reused unchanged for val and test. Not a learned null token,
not a zero tensor: this is deterministic, carries no sample-specific information, has
exactly the shape of a real `S`, adds no parameters, and is far less likely to act as an
out-of-distribution shortcut. If `S` is a token sequence the mean is taken per token
position and per feature, preserving the tensor shape — never pooled first.

**Margins.** `eps_BER = 0.007`, grounded in the largest measured between-receiver-seed
sd, 0.00681 (D_spectral, k=16). **No `eps_R`, no `eps_D`.**

An `eps_R = 0.0015` derived from D_spectral's own spread (max 0.00134) would ignore the
largest measured spread across arms, C2_haar at k=8 with 0.00315 — twice the margin. A
margin below the noise floor makes the equivalence test unpassable by construction, which
is the mirror image of the `p > 0.05` fallacy and just as attackable. Widening it to
~0.005 instead would exceed the zero-correlation reference `1/(n-1) = 0.00392` itself, so
"equivalent" would permit a difference the size of the entire signal scale. Neither is
acceptable, so `R2_lin` carries no equivalence gate. Same reasoning as for `D_cert`:
equivalence margins are set only on quantities whose scale is already stable.

## Roles of the readouts

| readout | positive contrast | placebo equivalence |
|---|---|---|
| sign BER | yes | **yes**, `eps_BER = 0.007` |
| `R2_lin` | yes (secondary, paired) | no |
| `Delta D_dec` | yes (secondary, paired) | no |
| `Delta C_side` | **not claimed by Phase A at all** | — |

`R2_lin` is a secondary *inferential* contrast, not a descriptive aside: correct-vs-
shuffled is reported with paired CIs and the empirical permutation null. If side
information works, `BER` falling and `R2_lin` rising are two independent operational views
and agreement between them is worth much more than either alone.

## Primary treatment

    f(Y, S_correct)   vs   f(Y, S_shuffled)

paired across receiver seeds. Identical architecture, input shape, capacity, `S` marginal
and training budget; the only intervention is the pairing.

## Placebo chain — two separate equivalence tests, not one

    f(Y)           <-> f(Y, S_null)        architecture / capacity effect
    f(Y, S_null)   <-> f(Y, S_shuffled)    empirical S distribution, no pairing

Each BER contrast must fall inside `[-0.007, +0.007]` **on its own**. Failure of either
does not stop the experiment; it narrows what correct-vs-shuffled can mean:

* first fails -> the side architecture has an operational effect by itself;
* second fails -> the real `S` distribution changes receiver behaviour even mispaired.

## Replication: 6 receiver seeds

At the previously measured worst-case sd of 0.00681, the TOST 90% CI half-width is

    3 seeds  0.01148   (exceeds the margin — no resolving power)
    5 seeds  0.00649   (+0.00051 headroom — minimum viable)
    6 seeds  0.00560   (+0.00140 headroom)

6 seeds is chosen: 2.7x the headroom of 5 for four extra runs. Stated precisely, since
this is a design property and not a prediction: *if the true placebo difference is near
zero and the variance is comparable to the previously measured worst case, six receiver
seeds give the equivalence test enough precision to pass.* It does not guarantee passing.

4 arms x 6 seeds = 24 runs.

## Phase gate

Phase B — the direct `C_gain(h)` certificate and the `E[h|Y] = 0` residualisation problem
— is opened **only** if Phase A shows a strong `S_correct >> S_shuffled` effect. Building
inference machinery for an effect that may not exist is the wrong order.


## Side encoder — frozen

    S in R^{77 x 768}
      -> shared Linear(768 -> 64), one projection for all token positions
      -> single learned-query attention pooling, query ZERO-INITIALISED
      -> h_S in R^{64}
    [h_Y (512) ; h_S (64)] -> the same prediction heads

The zero-init query makes the softmax exactly uniform at step zero (measured:
`max|alpha - 1/77| = 0`), so the encoder begins as a mean-pooled linear projection: the
pooling is **non-selective at initialisation**, and any token preference is acquired
during training.

That claim is narrower than arm C3's paired initialisation and must not be stated as
"the parameterisation contributes nothing at step zero". It does contribute: the random
`768 -> 64` projection, the concatenated branch and the widened head all exist and all
affect the output from the first forward pass — which is exactly what
`test_S_actually_changes_the_output` asserts. Only the attention selectivity starts at
zero. No Transformer, no second MLP layer, no pooling ablation.

### Capacity, per arm and not per branch

The side branch is 49,280 parameters in both roles, but the concatenation also widens the
head — and the head is `Linear(., k)` for the receiver and `Linear(., d)` for the teacher:

| model | total | vs blind | side branch | head increase |
|---|---|---|---|---|
| Extractor, k=64 | 11,299,648 | **+57,472** | 49,280 | 64 x 64 x 2 = 8,192 |
| Teacher, d=16384 | 20,679,360 | **+1,097,856** | 49,280 | 64 x 16384 = 1,048,576 |

Quoting "the side branch is 49K parameters" would understate the teacher by a factor of
20. Report the total per arm.

This does not threaten the primary contrast: `S_correct` and `S_shuffled` run the *same*
576-d architecture, and the `S_null` arm exists precisely to measure whatever the extra
capacity buys on its own. Fusion is not redesigned to save the 1M — concatenation is what
was registered, and the effect it introduces is already captured by a control.
