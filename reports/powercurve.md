# Exploratory power diagnostic — learning curve

**Status: pre-registered, not yet run.** Written before the first run so the reading
rule cannot be revised after the numbers arrive.

**This is not a gate, not a selection, and not evidence for or against any claim about
the diffusion channel.** Every run is `--scope powercurve`; `select_method.py` reads
`analysis_scope == "gate"` only, and `tests/test_locked_selection.py::
test_powercurve_runs_never_enter_gate_selection` asserts it. The `triage1` lock is
already written and is write-once.

## Why this rather than caching 10k

The triage put all five arms at BER 0.5 with train loss falling from the chance value
1.693 to 0.53–0.69 — the extractors fit the training data and generalise at exactly
chance. Two explanations remained: sample-limited, or nothing to recover without the
prompt. Caching 10k images costs ~4.4 GPU-hours plus a full re-run, to bet on the first.

The per-coordinate correlations already argue against the sample-limited reading:

| statistic | observed | pure noise |
|---|---|---|
| variance ratio of per-coordinate rho | 0.968 | 1.000 |
| max abs rho over 2880 coordinates | 0.2234 | 0.2249 |
| pooled mean rho | +0.00370 | 0 |

There is a **weak positive aggregate correlation shift, and no coordinate-level
departure from the empirical noise pattern.** Not one coordinate stands out; the
distribution's width and its maximum are what pure noise produces.

That is deliberately weaker than "the channel is not empty". The runs share validation
and test images, architecture and attack set, and the 2880 per-coordinate correlations
are heavily dependent, so the 8/9 sign test (p=0.039) and the run-level t are not
independent replication inference. The honest reading is a small aggregate shift of
unknown origin.

So the prior going in is that the curve will be flat. It is worth 40 minutes to find out,
and it is not worth 4.4 hours to assume otherwise.

## Design

* **Nested random subsets.** N=100 is a subset of N=300 is a subset of split B (928), at
  each subset seed, so the curve varies size and nothing else. The existing `--limit`
  takes a prefix, and the record order carries the sample index and hence the latent seed
  and prompt draw, so a prefix would confound size with composition.
* **Constant optimisation steps, not constant epochs.** Target 2320 = 40 x ceil(928/16),
  the gate budget. At fixed epochs, small N gets proportionally fewer gradient steps and
  a flat curve could mean "not enough steps". Epochs: 331 / 122 / 40.
* **Two arms.** `C2_haar` (the null the claim is against) and `D_spectral` (derived).
* **3 subset seeds**, paired with the receiver seed, so the three replicates differ in
  both the subset and the extractor initialisation. Total 18 runs, ~40 min.
* Validation split is **never** subset: fixed, complete, 256 samples.
* Same cache, same frozen spectrum, same k=64 as the triage.

## Reading rule, fixed in advance

The decision is about a **trend across N**, consistent across subset seeds — never a
single point estimate.

Supports caching 10k if, monotonically in N and in at least 2 of 3 subset seeds:

    BER decreases, AND/OR mean rho increases, AND/OR coord_mse decreases toward 1.0

with the N=928 to N=100 difference in mean rho exceeding the between-seed spread.

Does **not** support caching 10k if BER stays at 0.5, mean rho stays within its
between-seed spread of 0, and coord_mse stays above 1.0 at every N. In that case the
next suspects are the prompt-free receiver bottleneck or the target/frame, not data
volume, and the follow-up is the cheap k in {8, 16, 64} diagnostic — if k=8 generalises
where k=64 does not, the problem is task dimensionality rather than an empty channel.

`coord_mse > 1.0` throughout would be its own statement: the trivial predictor `w_hat=0`
scores exactly 1.0, so the extractor is adding variance rather than information.

## Result

**Verdict: does not support caching 10k.** 18 runs, 2026-08-17, tag `triage1`,
scope `powercurve`, commit `24b56f2`.

> The raw artifacts behind this table were deleted by operator error — an
> `rm '*scpowercurve*'` meant to clear an aborted k-diagnostic also matched these runs,
> whose stems are `..._scpowercurve_n<N>_ss<seed>`. **Re-run and restored** on commit
> `5896363`: all 18 cells reproduced to the precision of the recorded table
> (max |ΔBER| = 5e-5, which is the rounding bound of a 4-decimal table; mean rho matched
> to all 5 recorded decimals), and every preregistered monotonicity count is unchanged.
>
> This is **deterministic reproducibility**, not a second stochastic replication: same
> receiver seeds, same subset seeds, same data, seeded training. It confirms the
> artifacts can be regenerated from the experimental identity; it provides no estimate of
> `Var(training | same receiver seed)`, because that stream is locked. The
> between-receiver-seed spread remains the variability quantity the reading rule uses,
> since the receiver seed is the stochastic axis the design varies on purpose.

Trajectories are reported per subset seed and are never averaged before being read —
the rule is about consistency across seeds, and a mean would hide exactly that.

### C2_haar

| seed | N | sign BER | coord MSE | mean rho | mean rho^2 | train loss |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.4936 | 1.1104 | +0.00417 | 0.00337 | 0.1222 |
| 0 | 300 | 0.4950 | 1.1126 | +0.01188 | 0.00345 | 0.2403 |
| 0 | 928 | 0.4966 | 1.1502 | +0.01129 | 0.00354 | 0.5687 |
| 1 | 100 | 0.4991 | 1.0983 | +0.01193 | 0.00460 | 0.1216 |
| 1 | 300 | 0.4994 | 1.1164 | −0.00306 | 0.00455 | 0.2323 |
| 1 | 928 | 0.4969 | 1.1512 | +0.00238 | 0.00318 | 0.5186 |
| 2 | 100 | 0.5000 | 1.1326 | −0.00393 | 0.00375 | 0.1328 |
| 2 | 300 | 0.4996 | 1.1247 | −0.00708 | 0.00404 | 0.2305 |
| 2 | 928 | 0.5031 | 1.1583 | −0.00270 | 0.00486 | 0.6172 |

### D_spectral

| seed | N | sign BER | coord MSE | mean rho | mean rho^2 | train loss |
|---|---|---|---|---|---|---|
| 0 | 100 | 0.5007 | 1.1041 | +0.00579 | 0.00377 | 0.1327 |
| 0 | 300 | 0.4984 | 1.1076 | +0.00680 | 0.00548 | 0.2493 |
| 0 | 928 | 0.4889 | 1.1394 | +0.01308 | 0.00488 | 0.5712 |
| 1 | 100 | 0.4981 | 1.0898 | +0.01190 | 0.00323 | 0.1189 |
| 1 | 300 | 0.5020 | 1.1004 | +0.00261 | 0.00466 | 0.2326 |
| 1 | 928 | 0.5006 | 1.1432 | +0.00201 | 0.00404 | 0.5160 |
| 2 | 100 | 0.4969 | 1.1152 | +0.00352 | 0.00483 | 0.1214 |
| 2 | 300 | 0.4958 | 1.1024 | +0.00152 | 0.00276 | 0.2311 |
| 2 | 928 | 0.4969 | 1.1444 | +0.00122 | 0.00420 | 0.6806 |

### Against the rule

| readout | C2_haar | D_spectral | required |
|---|---|---|---|
| sign BER monotone down | 0/3 | 1/3 | >= 2/3 |
| mean rho monotone up | 0/3 | 1/3 | >= 2/3 |
| coord MSE monotone down | 0/3 | 0/3 | >= 2/3 |
| mean rho, N=928 minus N=100 | −0.00040 +- 0.00846 | −0.00163 +- 0.00861 | > between-seed spread |
| between-seed sd at N=928 | 0.00709 | 0.00663 | — |

No readout improves monotonically in two of three seeds. The change in mean rho from
N=100 to N=928 is **negative** for both arms and an order of magnitude below the
between-seed spread it would have to exceed.

The single cleanest number is the linear explained variance of the current receiver
output, in the form that does not cancel signs:

    (1/k) sum_j rho_j^2  =  0.00407      pooled over all 18 runs (sd 0.00074)
    null E[rho^2] = 1/(n-1) = 0.00392    for n = 256 validation samples
    ratio = 1.037

It sits at the noise value **at every N**. Note what this quantity is and is not: it is
the R^2 of an optimally rescaled version of this receiver's output, since
min_a E[(W - a W_hat)^2] = 1 - rho^2 under the standardised conditions here. It is not a
Shannon capacity, not a channel capacity, not a bound on what a nonlinear decoder could
do, and not the C_obs ceiling.

`coord_mse` is the one consistent effect and it runs the wrong way: 1.10 at N=100 to
1.15 at N=928, monotone increasing in 6 of 6 trajectories, moving **away** from the 1.0
that the trivial predictor `w_hat = 0` scores. More data makes the receiver's output
variance hurt more, not less.

Train loss rises from 0.12 at N=100 (331 epochs over 100 samples) to 0.52–0.68 at
N=928. That is exactly what fixed-step training predicts — larger N means each image is
revisited fewer times — so it shows small-N runs memorise easily. It does **not** show
that N=928 reached an optimisation plateau, and this report does not claim it does.

### Consequence, per the rule as written

Do **not** cache 10k.

### What this does and does not establish

The curve holds **optimisation compute** fixed at 2320 steps, which is what makes sample
count the isolated variable — and is also the limit of the claim. The supported statement
is:

> At fixed optimisation compute, increasing the number of unique training samples from
> 100 to 928 produces no reproducible improvement in held-out recoverability. This
> diagnostic therefore does not justify the substantially more expensive 10k cache and
> run.

It is **not** "more data can never help", and this report does not say so. A 10k run under
the full protocol's 40 epochs would carry more optimisation compute, not the same:

    extractor trains on split B only, measured at 46.4% of train (2000 -> 928)
    full: train = 10000  ->  B ~ 4640  ->  40 * ceil(4640/16)  = 11600 steps
    here: N = 928                      ->  40 * ceil(928/16)   =  2320 steps
    ratio = 5.0x

so a 10k run would differ from this curve's top point in **two** ways at once. The
diagnostic answers the sample-count question at fixed compute; it does not answer the
joint one, and it was never registered to.

### Next

The next suspects are the prompt-free receiver bottleneck or the target/frame, not
sample count at this compute. Pre-registered follow-up: the k in {8, 16, 64} diagnostic.
If k=8 generalises where k=64 does not, the problem is task dimensionality; if k=8 is
also flat, it is not.

---

# Follow-up: the k in {8, 16, 64} diagnostic

**Status: pre-registered, not yet run.** Same discipline: reading rule fixed before the
numbers arrive, `--scope powercurve`, invisible to the selector.

## Why

The curve says the limiter is not data volume. The next cheapest hypothesis is **task
dimensionality**: 64 simultaneous targets from one 512-dimensional pooled feature vector,
trained on 928 images. If k=8 generalises where k=64 does not, the receiver is being
asked for too many coordinates at once. If k=8 is also flat, dimensionality is not the
limiter either, and what remains is the prompt-free receiver protocol or the target.

The three subspaces are **nested**: `SpectralFrame` takes `rows[:k]` of the same
`triage1_seed0` fit, so span(k=8) is a subspace of span(k=16) is a subspace of span(k=64).
The k=8 frame is the 8 most observable directions that fit found — the most favourable
case available, which is what a diagnostic wants.

Design: k in {8, 16, 64} x {C2_haar, D_spectral} x 3 receiver seeds = 18 runs. Full split
B (928) throughout, 40 epochs, so the optimisation budget is constant and k is the only
variable. The frame comes from commit `48e8351`, before the provenance fix; it is reused
deliberately and none of the closed fixes touch its eigenvectors.

## Reading rule, fixed in advance

`(1/k) sum_j rho_j^2` is compared against the null `1/(n-1) = 0.00392` at n=256, and
sign BER against 0.5, at each k, per receiver seed.

**Dimensionality is the limiter** if at k=8, in at least 2 of 3 receiver seeds,
`(1/k) sum_j rho_j^2` exceeds the null by more than the between-seed spread, and the
effect shrinks monotonically as k grows.

**Dimensionality is not the limiter** if k=8 sits at the null like k=64. In that case the
remaining suspects are the prompt-free receiver protocol (H(Y) with no prompt) and the
target/frame — and the honest next step is a falsification experiment on those, not a
larger run of the same design.

A smaller k makes the AGGREGATE readouts noisier, not cleaner: BER's standard error is
`0.5/sqrt(n*k)` and the spread of the mean over k coordinates scales as `1/sqrt(k)`. So a
k=8 point that merely looks different is not evidence; it has to clear its own spread.

## Result

**Verdict: dimensionality is not the limiter.** 18 runs, commit `5896363`, scope
`powercurve`, 0 failures. Read in the registered order: D_spectral at k=8, three receiver
seeds separately, before anything else.

### First priority — D_spectral, k=8, per receiver seed

| receiver seed | sign BER | mean rho | R2_lin | R2/null | train loss |
|---|---|---|---|---|---|
| 0 | 0.4931 | +0.01220 | 0.00218 | 0.556 | 0.1655 |
| 1 | 0.5014 | +0.02514 | 0.00353 | 0.899 | 0.1582 |
| 2 | 0.4882 | +0.02310 | 0.00399 | 1.018 | 0.1451 |
| **mean** | | | **0.00323** | **0.825** | |

`R2_lin = (1/k) sum_j rho_j^2`; null `1/(n-1) = 0.00392` at n=256. Between-seed
sd = 0.00094, and the mean sits 0.00069 **below** the null — 0.73 spreads on the wrong
side. Seeds above null: 1 of 3.

The registered criterion needed k=8 to exceed its own null by more than its own
between-seed spread in at least 2 of 3 seeds. It does not.

### All three k

| arm | k | R2_lin mean | R2/null | between-seed sd | above null |
|---|---|---|---|---|---|
| D_spectral | 8 | 0.00323 | 0.825 | 0.00094 | 1/3 |
| D_spectral | 16 | 0.00488 | 1.244 | 0.00134 | 2/3 |
| D_spectral | 64 | 0.00406 | 1.034 | 0.00120 | 2/3 |
| C2_haar | 8 | 0.00556 | 1.419 | 0.00315 | 2/3 |
| C2_haar | 16 | 0.00348 | 0.886 | 0.00111 | 1/3 |
| C2_haar | 64 | 0.00408 | 1.041 | 0.00051 | 2/3 |

The second registered condition — a monotone shrink `k=8 > k=16 > k=64` — fails for both
arms (D_spectral 0.825, 1.244, 1.034; C2_haar 1.419, 0.886, 1.041). Sign BER stays inside
0.4856–0.5040 at every k and seed.

The one cell that departs from null noticeably is **C2_haar** at k=8, seed 2
(R2/null = 2.32, mean rho +0.054) — the *random reference*, and its sign BER is 0.5040,
worse than chance. A single run over 80 rho values fluctuating, in the null arm, pointing
the wrong way for the FIBER claim.

### Consequence

"64 simultaneous targets is too hard" has no support. The diagnostic gave that hypothesis
the most favourable case available — the 8 top-ranked directions of the same spectral
fit, nested subspaces, constant optimisation budget — and the readouts stayed at the
noise value.

Combined with the learning curve, two hypotheses are now out at this scale: sample count
at fixed compute, and task dimensionality. Neither 10k nor a smaller k is the indicated
next step. The registered follow-up is the receiver's information set — see
`reports/blind_vs_sideinfo.md`.
