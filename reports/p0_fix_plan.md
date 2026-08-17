# P0 FIX PLAN

Written against HEAD `71f3331` in response to the external audit. **No code changed yet.**
The Phase-2/3 sweep is stopped: the auto-chain that would have launched `run_pilot.sh`
at cache completion was killed. Phase-1 caching was allowed to finish (see cache
validity below).

Three of the audit's claims were verified empirically on CPU before writing this plan,
because two of them change what the fix has to be.

| Claim | Measurement | Verdict |
|---|---|---|
| BCE on `1[W>0]` does not train `Q` | frame `.grad` is **`None`** under BCE-only; `1.2e-2` under MSE-only | confirmed, and stronger than stated: the autograd graph is *severed*, not merely flat |
| `Cov(f) ⋠ C_obs` for approximate `f` | `f = 3m`: `max eig(Cov − C_obs) = +7.97`; `max eig(C_cert − C_obs) = +0.011` | confirmed |
| `qr(gaussian)` is Haar | `P(diag(R) > 0) = 0.515 / 0.514 / 0.478` over 2000 draws (±0.011) | **not demonstrated** — LAPACK looks roughly sign-symmetric here. See P0-2. |

## New finding not in the audit: the eigensolver must be algebraically-largest

`C_cert` is symmetric but **indefinite**. Measured for `f = 3m`:

```
eigenvalues of C_cert : min = -2.997 , max = +0.014      =>  |lambda_min| >> lambda_max
```

The current estimator (`fit_randomized`, `observability.py:127`) is a randomized **SVD**,
which returns the largest **singular** values — i.e. the largest **magnitude** eigenvalues
of a symmetric matrix. Applied to `C_cert` it would return the *most negative* directions
as the "top-k": exactly the directions the teacher is worst on. Under a bad teacher this
silently inverts the experiment.

So P0-1 needs a symmetric eigensolver targeting the **algebraically** largest eigenvalues
(shifted randomized subspace iteration, cross-checked against `scipy.sparse.linalg.eigsh(which="LA")`),
not a drop-in replacement of the matrix inside the existing SVD path.

---

## Phase-1 cache validity — valid for all seven items

No P0 fix changes `z`, the prompt assignment, the generator config, or the clean carrier
images, so **the cached images and latents remain valid and will not be regenerated**.

One item touches metadata: P0-1 subdivides discovery split A into `A_teacher` /
`A_operator`, which adds a field to `index.jsonl`. Images are addressed by
`(split, shard, offset)` and latents by `(SALT, tag, "latent", split, i)`; the new field is
derived from `i` alone and touches neither. A regression test will pin
`(sample_id, split, index, shard, offset, latent_seed, prompt)` against the current index so
that a rebuild provably re-addresses the same cached shards.

---

## P0-1 — certified observability operator

**Existing.** `spectrum/observability.py`: `trace_c_obs` (L75) returns `‖M‖²_F / N`, the
**uncentered second moment** of teacher outputs; `fit_gram` (L108) and `fit_randomized`
(L127) take the SVD of `M` directly; `Spectrum.evaluate_on` measures held-out variance of
the projections. `scripts/fit_observability_spectrum.py` trains the teacher on split A and
builds `M` from **the same split A**.

**Problems.** (a) no centering, so any teacher output bias `m̄` injects a rank-1 direction
`m̄m̄ᵀ`; (b) `Cov(f)` is not a lower bound on `C_obs` for approximate `f` (measured above);
(c) teacher fitting and operator estimation share samples; (d) the solver picks the wrong
sign of eigenvalue for the replacement operator (new finding above).

**Fix.** Implement

```
C_cert(f) = E[ z_c f_cᵀ + f_c z_cᵀ − f_c f_cᵀ ]
          = C_obs − E[(m−f)(m−f)ᵀ]  ⪯  C_obs
vᵀ C_cert v = Var(vᵀZ) − E[(vᵀZ − vᵀf(Y))²]  = 1 − MSE_v   for Z ~ N(0,I)
```

matrix-free: `C_cert V = (Zcᵀ(Fc V) + Fcᵀ(Zc V) − Fcᵀ(Fc V)) / N`, never forming 16384².
Top-k by shifted randomized subspace iteration on `C_cert + σI`, `σ ≥ |λ_min|` estimated
first, so "largest algebraic" and "largest magnitude" coincide.

Split chain becomes, with no sample in two roles:

```
A_teacher  -> teacher fitting
A_operator -> coordinate discovery (C_cert directions), then FROZEN
B          -> fresh evaluation extractor  (unchanged)
val        -> ALL model/hyperparameter selection   (P0-3)
test       -> locked evaluation only               (P0-3)
```

**Files.** new `src/fiber/spectrum/certified.py`; modify `src/fiber/spectrum/observability.py`
(keep the `Cov(f)` path, renamed to what it is — a diagnostic `lambda_var` — with centering
added); `src/fiber/diffusion/cache_dataset.py` (A sub-split + index field);
`scripts/fit_observability_spectrum.py`; `src/fiber/transforms/spectral.py`;
`configs/linear_fiber.yaml`; `PLAN.md` §3.

**Tests** (`tests/test_certified_operator.py`): exact teacher ⇒ `C_cert == C_obs`;
`f = 3m` ⇒ `Cov(f) ⊀ C_obs` but `C_cert ⪯ C_obs`; `f = 0` ⇒ `C_cert = 0`; `f = m` ⇒
`vᵀC_cert v = 1 − MMSE_v`; a constant teacher bias is removed by centering; matrix-free ==
dense on small `d`; **top-k under an indefinite operator returns the algebraically largest,
not the largest magnitude**; `A_teacher ∩ A_operator = ∅`; validity diagnostic
`mean|λ_var − λ_skill|` is computed and gated.

**Naming.** `Tr(C_cert)` becomes `D_obs` = *Certified Observability Mass*. Not Shannon
capacity, not `I(Z;Y)`, not "true `C_obs`".

---

## P0-2 — genuine random subspace baseline

**Existing.** Random arms are `SignedPermutationFrame` (local: one latent pixel per
coordinate) and `HadamardFrame` (rows are exactly `±1/√d`). Neither samples a uniformly
random `k`-subspace.

**Fix.** `HaarRandomFrame`: `A ~ N(0,1)^{d×k}`, `Q,R = qr(A)`, `R ← Q·sign(diag(R))`, rows
`= Qᵀ`. On the sign correction: my measurement did **not** show a LAPACK bias
(0.478–0.515), so I will not claim one — but Haar-ness must not depend on an undocumented
LAPACK convention, and the correction is free, so it goes in and is tested.

`RandomHouseholderFrame`: identical architecture and reflector count to the learned arm,
parameters random and frozen — the architecture-matched control that separates "the
Householder parameterisation helps" from "learning selects better directions".

Gate reference policy: **Haar is the primary random reference**; signed permutation,
Hadamard and random Householder are reported controls. Identity stays sanity-only.

**Files.** new `src/fiber/transforms/haar.py`; `src/fiber/transforms/householder.py`
(`frozen=True` variant); `registry.py`; `configs/linear_fiber.yaml` (arms `C2_haar`,
`C3_rand_householder`); `scripts/eval_coordinates.py` (reference policy).

**Tests.** `‖RRᵀ−I‖_∞ < 1e-5`; `RZ ~ N(0,I_k)`; **isotropy**: over many draws,
`E[R₀ᵢR₀ⱼ] ≈ δᵢⱼ/d` (Hadamard fails this by construction, Haar passes); rows are dense,
unlike signed permutation; the default gate reference resolves to Haar.

---

## P0-3 — remove test-set model selection

**Existing.** `scripts/eval_coordinates.py:180` selects, per channel group, the derived run
with the lowest **test** BER, then bootstraps on the same test data. The Bonferroni factor
counts derived families only — not seeds, not channel groups. This is leakage and the
correction does not repair it.

**Fix.** Split into two phases with a lock file.

```
scripts/select_method.py  --split val   ->  reports/selection_<tag>.json
    { family, k, hyperparameters, random_reference, seeds_used,
      commit_sha, config_fingerprint, written_at }
scripts/eval_coordinates.py --split test
    must LOAD that file; refuses to run without it; evaluates only the locked method.
```

Seeds become replications, never a selection axis: metrics are averaged over seeds before
the paired sample bootstrap, plus a hierarchical bootstrap (resample seeds, then samples).

**My caveat, which the audit does not cover.** With 2–3 training seeds a seed-level
bootstrap has essentially no resolution — the seed variance is estimated from 3 points and
the resulting CI is not trustworthy. I will report the sample-level paired CI (on
seed-averaged metrics) as primary, the seed spread as an explicit second number, and
require **≥5 seeds for the locked method** in the full run before any hierarchical CI is
quoted as evidence.

**Files.** new `scripts/select_method.py`; `scripts/eval_coordinates.py` (gate rewrite);
`src/fiber/metrics/bootstrap.py` (seed aggregation + hierarchical bootstrap); `PLAN.md` §7.

**Tests** (`tests/test_locked_selection.py`): synthetic runs constructed so the **val
winner ≠ test winner**, asserting the reported test model is the val winner; test
evaluation without a lock file raises; the lock records commit SHA and config fingerprint;
seed aggregation is a mean, and a `min` over seeds anywhere in the gate path fails the test.

---

## P0-4 — honest learned-Q objective

**Existing.** `training/loops.py:87`, `bits = (w_true > 0).float()`. Measured: the frame's
`.grad` is `None` under BCE-only — `>` returns a bool tensor with no `grad_fn`, so the graph
is severed, not merely zero-gradient a.e.

**Fix.** Level 1 takes the audit's route A. Discovery optimises **MSE only**; the sign head
is trained after `Q` is frozen and used for communication evaluation. `train_extractor`
gains an explicit `discovery_objective` and **raises** if `learn_frame=True` is combined
with a non-zero sign weight, so the interpretation cannot silently regress. PLAN.md §3.4(2)
and §5.2 are rewritten: spectral and learned are both estimating MMSE-observable
directions by different means, and Gate 3B compares *operator estimation vs differentiable
subspace optimisation* — the "learned wins because it optimises sign BER" prediction is
withdrawn. Soft-sign (`σ(W/τ)`) is recorded as a separate future arm, outside the Level-1
gate.

**Files.** `src/fiber/training/loops.py`; `scripts/train_coordinates.py`; `PLAN.md`; `README.md`.

**Tests.** frame gradient is `None`/zero under a hard-sign target and non-zero under MSE;
`learn_frame=True` with sign weight > 0 raises.

---

## P0-5 — teacher architecture confound

**Existing.** `models/extractor.py:62`: `Teacher` is ResNet18 → global average pool →
`Linear(512, 16384)`, so `rank(Cov(f)) ≤ 512` regardless of the channel, and GAP is biased
toward global/semantic content — which is the R1 locality confound relocated into the
measuring instrument.

**Fix.** Keep it as `GlobalTeacher`; add `SpatialTeacher`: conv/residual pyramid
`512×512 → 64×64`, spatial residual blocks, `3×3` head → `[4,64,64]`, **no GAP**. Fit
`C_cert` with both, report `D_obs_global`, `D_obs_spatial` and the top-k principal-angle
alignment, plus downstream BER from each. Disagreement is reported as
architecture-dependent, not hidden. Terminology until robustness is shown:
**decoder-certified** observability geometry, never *intrinsic*.

**My addition, which the audit stops one step short of.** The **extractor** has the same
bottleneck: ResNet18 → GAP → `Linear(512, k)`. If GAP favours global frames, that bias sits
directly on the BER comparison, not just on the spectrum — so a global arm could beat Haar
partly because the *receiver* cannot read local structure. I will add a spatial-extractor
control run for the locked method and for Haar at `k=64` (two extra runs, ~20 min on
pilot), and report BER under both extractor architectures. If the ranking flips, Gate 3A's
result is receiver-architecture-dependent and must be reported as such.

**Files.** new `src/fiber/models/spatial_teacher.py`; `scripts/fit_observability_spectrum.py`
(`--teacher {global,spatial}`); new `scripts/compare_teachers.py`; `configs/linear_fiber.yaml`.

**Tests.** `Cov(GlobalTeacher output)` has rank ≤ 512 for `N > 512` while `SpatialTeacher`
exceeds it; `SpatialTeacher` contains no global pooling layer; output shape `[B,4,64,64]`;
alignment metric is symmetric and equals 1.0 for identical subspaces.

---

## P0-6 — fresh stochastic channel draws in training

**Existing.** `channels/registry.py:52,57`: the seed is
`blake2s(sample_id | attack | severity | split_salt)`. `FiberDataset.attack_for` uses
`epoch_salt` only to choose **which** attack; the realisation itself never changes. Under
`noise005` a given sample sees the identical Gaussian field in every epoch.

**Fix.** Add `draw_salt` to `attack_seed` / `ChannelBank.apply`. Training passes the epoch
identifier; evaluation passes the fixed `"eval-v1"`. Deterministic attacks ignore it.

**Files.** `src/fiber/channels/registry.py`; `src/fiber/diffusion/cache_dataset.py`;
`src/fiber/training/loops.py`.

**Tests.** `noise005(epoch=1) != noise005(epoch=2)`; `noise005("eval-v1")` byte-identical
across calls **and across dataloader orderings/worker counts**; `jpeg50` unaffected by
`draw_salt`; every arm sees the identical evaluation corruption (otherwise the paired
bootstrap is comparing different channels).

---

## P0-7 — subspace quality vs basis quality

**Existing.** Only the raw spectral eigenbasis exists; nothing distinguishes "better
subspace" from "better basis inside the same subspace".

**Fix.** `RotatedFrame(base, A)` with `A ∈ O(k)` parameterised as `expm(S − Sᵀ)`
(`torch.linalg.matrix_exp`), exactly orthogonal for any parameter value and differentiable —
the same structural discipline as the Householder arm, at `k ≤ 256` cost. Arms: **D1** raw
eigenbasis, **D2** subspace + random `O(k)` rotations (multiple seeds), **D3** subspace +
learned `O(k)` rotation with the ambient subspace frozen. Report a **subspace score**
(captured certified observability) and a **basis score** (sign BER) separately.

**Files.** new `src/fiber/transforms/rotation.py`; `registry.py`; `configs/linear_fiber.yaml`;
`scripts/eval_coordinates.py` (two-number reporting).

**Tests.** rotation preserves the subspace (alignment `= 1.0` to 1e-6) and
`Tr(V C Vᵀ)`; orthonormality survives gradient steps; the rotation actually changes the sign
bits (otherwise the control is vacuous).

---

## Secondary cleanups

| # | Change |
|---|---|
| A | COCO-val captions are **held-out prompt instances**, not a held-out *domain*. Rename everywhere; record a genuine OOD prompt source as future work. |
| B | Per-attack figures become **fixed-decoder operational spectrum under attack t**, never `Cov(E[Z|Y,T=t])`. |
| C | `TAG=pilot` may only emit `PROVISIONAL_PASS` / `PROVISIONAL_FAIL` / `INCONCLUSIVE`. `KILL` and `PASS` require the full protocol. Enforced in `gate3a_verdict`, with a test. |
| D | `environment.yml` + `requirements-lock.txt` pinned to the measured versions (python 3.11.15, torch 2.4.1+cu121, torchvision 0.19.1, diffusers 0.26.3, transformers 4.38.2, numpy 1.26.4, scipy 1.17.1, pillow 12.3.0, opencv 4.11.0, pyyaml). |
| E | GitHub Actions running the CPU suite on push/PR; GPU/cache-dependent tests marked and skipped. |
| F | Re-run Gate 0 after the P0 pass and record the exact commit SHA + config fingerprint (currently `uncommitted`). |

---

## Cost of the repaired protocol — worth deciding before it runs

The audit roughly doubles the arm count. At the measured rates (0.81 img/s generation;
~9.5 min per arm run on the pilot's ~1000-image split B; the full cache's split B is ~5×
larger, so ~45 min per arm run):

| | arms × seeds | full-run GPU |
|---|---|---|
| random reference + controls (Haar 8, signperm 3, Hadamard 3, rand-Householder 3) | 17 | ~13 h |
| data-derived (D1, D2, D3, learned; ≥5 seeds for the locked method) | 14–17 | ~11 h |
| teachers (2 architectures × 3 seeds) | 6 | ~6 h |
| spatial-extractor control (my P0-5 addition) | 2 | ~1.5 h |
| **total Phase 2/3, full cache** | | **~31 h** |

PLAN.md §8 budgeted ~16 h for Phase 3. If that is a hard ceiling, the honest lever is
**fewer channel groups or fewer k values**, not fewer random draws or seeds — cutting the
random reference is precisely what the audit forbids. I will surface this trade-off with
numbers rather than silently trimming.

---

## Order and deliverables

Implemented as small commits in the audit's order — P0-1 → P0-7 → secondary — each with:
files changed, the mathematical invariant it enforces, tests added, tests passed, cache
validity, and whether the experimental protocol changed. Final report
`reports/p0_audit_fix.md` with the ten required sections, ending in exactly one of
`READY_FOR_PHASE2_3` / `NOT_READY_FOR_PHASE2_3`.

No GPU sweep until that report says READY.


---

## Deferred to pre-Gate-3 cleanup (recorded, not blocking P0-2)

Raised in review after P0-1.1 was accepted; neither changes a conclusion, both change
what may be *claimed*.

1. **DONE — numerical positive rank is not statistical certification.** `tau = max(1e-9,
   k·eps·max|mu|)` only excludes floating-point noise. Even after the inner cross-fit,
   a decoder with zero true skill still shows `D = 0.44` at `N=64, k=32` purely from
   clipping. Before Gate 3, split the reported rank into `numerical_positive_rank`
   (already renamed) and a `statistically_certified_rank` requiring a one-sided 95%
   lower confidence bound above zero, bootstrapped on the measurement half with the
   rotation held fixed from the discovery half. The main table should then quote
   `D_cert^LCB`. Until that exists, "certified" means "certified by this decoder up to
   numerical noise", and the reports say so.

2. **Unbiased denominator — DONE.** The empirical operator is sample-centered, so
   `N-1` is the unbiased denominator; all of `CertifiedObservabilityOperator`,
   `project_operator`, `quadratic_form`, `variance_form` and the `range_eigh` path now
   take `ddof=1`. The change is a factor `N/(N-1)` and moves no conclusion.

3. **DONE — symmetric two-fold inner cross-fit.** The inner cross-fit currently
   measures on one half only. Averaging `D_{1->2}` and `D_{2->1}` would use every
   report sample for measurement and reduce dependence on the split seed. Polish after
   P0-7.


---

## Finding while closing P0-2: arm E initialises at (almost exactly) identity

`scripts/diagnose_frames.py`, at the configured size rather than a toy — 100 draws,
`d = 16384`, `m = 128`:

| frame | `E[r₀²]` | `E[max_i r_i²]` | participation ratio |
|---|---|---|---|
| signed permutation | 0 | 1.000 | 1.0 |
| **random Householder, m=128** | **0.969** | **0.969** | **1.1** |
| Haar | 5.82e-05 | 1.02e-03 | 5470 |
| Hadamard | 6.10e-05 | 6.10e-05 | 16384 |
| uniform ideal | 1/d = 6.10e-05 | 2ln d/d = 1.18e-03 | d/3 = 5461 |

Each reflection displaces `e₀` by about `2/√d = 0.016`, so `m` random reflections move
it by only `√m·2/√d = 0.18`; a Haar-like draw would need `m ~ d/4 = 4096`.

Two consequences.

**Arm C3 is nearly the identity arm.** It remains a valid architecture-matched control
for E — it *is* E at initialisation — but it is not a random frame in any useful sense,
and R1 applies to it. Its expected behaviour is arm A's, and the report must say so
rather than let a reader read it as a random baseline.

**Arm E starts at the one point R1 says is worst.** This is about the initialisation,
not capacity: `m ≥ k` Householder reflections can represent any `k`-frame exactly. But
gradient descent begins essentially at identity, so an arm-E failure would be
confounded with a bad starting point rather than evidence about learning.

**Proposed fix, to land with P0-4:** parameterise arm E as `R_E = Householder_φ(H)`
with `H` a *frozen Haar frame*. At initialisation `Householder_φ ≈ I`, so `R_E ≈ H` —
arm E starts exactly at the gate denominator, and any improvement over Haar is
unambiguously due to learning rather than to having escaped a bad init. Arm C3 then
coincides with C2 at initialisation, which is itself the honest statement: at init the
architecture adds nothing.


---

## Pre-Gate blockers (recorded after the P0-4 review; formal Gate 3 may not run until closed)

### B1 — P0-3.2, physical test isolation

`train_coordinates.py` evaluates `val`, `test` and `test_heldout_prompts` in one pass, so
every candidate's test predictions exist on disk **before** `select_method.py` runs. The
selection script provably does not read them (`_forbid_test`), and the lock hashes them,
but the strongest available answer to *"how do you know the test set was not inspected
before selection?"* is currently "the script does not read it" rather than "those outputs
did not exist yet".

Required before Gate 3:

* pre-lock runs evaluate `val` ONLY, and must contain no `test|...` arrays at all;
* `train_coordinates.py` saves the frame and extractor checkpoints;
* a new `evaluate_locked.py` loads the checkpoints named by `selection.json` and computes
  the test splits for the first time, hashing its own outputs.

This is a workflow change, not a fix to the current logic, and it is where the audit
trail becomes falsifiable rather than merely stated.

### B2 — Strict completeness of preregistered seed sets

Selection averages whatever seed files exist. If a registered seed crashes or is deleted,
the average silently proceeds over the survivors — survivorship bias with no warning.
Before Gate 3, `select_method.py` must read the registered `seeds:` list from the config
and hard-fail unless every one is present, for the derived arm and for all 8 Haar draws.

### B3 — Householder reflector capacity versus k

Arm E uses `m = 128` reflectors at every `k`, and `k` sweeps to 256. The claim that
`m >= k` reflections can represent any `k`-frame was removed for lack of a citation, so
its converse cannot be assumed either: at `k = 256` the reachable family may be strictly
smaller than the set of `k`-frames, and `BER_learned > BER_spectral` there could reflect
capacity rather than geometry.

Cheap synthetic audit, no diffusion: draw `R* ~ Haar(k, d)`, optimise a Householder frame
against the subspace distance `‖P_phi − P*‖_F` with `P = RᵀR`, and tabulate the achievable
fit over `k ∈ {16, 64, 128, 256}` and `m ∈ {64, 128, 256, 512}`. If `m = 128` cannot fit
`k = 256`, either scale `m(k)` or restrict the locked gate to `k = 64` and report the rest
as a capacity-sensitive ablation.


---

## Pre-full-run blocker: the selection lock is not mechanically write-once

**CLOSED `83006b8`.** `write_once()` (tmp + `os.link`), 3 tests, negative control.

Raised while the provisional triage was running, so it is recorded rather than fixed —
editing source would abort the run.

`evaluate_locked.py` refuses to overwrite an official test evaluation, but
`select_method.py` ends with a plain `out.write_text(...)`. Running it twice on the same
tag silently replaces the lock. And now that `reports/` sits under `ARTIFACT_PREFIXES`,
`require_clean()` will not flag that overwrite either: the two changes are individually
right and jointly leave the lock itself unprotected.

This does not invalidate the running triage — the driver selects once, moves straight
into the post-lock cache, and triage carries no PASS/KILL authority — but the full
protocol must not start until:

* `select_method.py` refuses when `reports/selection_<tag>.json` exists, and writes with
  exclusive create (`open(..., "x")`) so a concurrent run cannot race it;
* two adversarial tests cover a second selection on the same tag, and a silent
  overwrite;
* optionally, the first post-lock cache records a `selection_sha` witness that the
  cache generator and the evaluator both re-check, so a lock edited *in place* after
  creation is caught as well as one replaced wholesale.

## Pre-full-run blocker: D_cert^LCB is not a lower bound

**CLOSED `1f5a20c`.** `sum_j max(L_fj, 0)`, max over folds by majorisation. Null 0.548 -> 0.001; power retained. 4 tests.

The current estimator takes, per bootstrap replicate, `Σ_j max(μ̂*_j, 0)` and then the
α-quantile of that. Near the boundary `μ_j = 0` the positive part rectifies noise into
mass in *every* replicate, so the low quantile of a non-smooth positive-part functional
carries no one-sided coverage. Demonstrated with terms whose true mass is exactly zero
at `k = 64, N = 256`:

```
true positive mass                       0.000
quantile of rectified mass  (current)    1.849     <- claims mass where there is none
sum of max(directional LCB, 0) (fix)     0.000
```

The fix uses the simultaneous directional bounds the code already computes. On the
simultaneous event `L_fj ≤ μ_j` for every `j`, so

```
Σ_j max(L_fj, 0)  ≤  Σ_j max(μ_j, 0)  ≤  tr(C_V)_+
```

the last step because the diagonal of `C_V` in any orthonormal basis is majorized by its
eigenvalues and `max(·, 0)` is convex. The Bonferroni over `k` and folds is already in
place; what changes is only that the rectification happens *after* the bound rather than
inside the bootstrap.

**Consequence for the triage numbers.** The spatial teacher's reported
`D_cert^LCB = 6.619` is not a valid lower bound and must not be quoted as one. Its point
estimate `D_cert = 7.348` is unaffected — that is a plug-in estimate, not a bound.

## Pre-full-run blocker: trace_C_V should use every held-out sample

**CLOSED `c677a88`.** Full held-out; sd ratio 1.412 vs sqrt(2). 2 tests.

`trace_C_V` is currently `mu.sum()`, and with the inner cross-fit on, `mu` comes from one
measurement fold. The trace is linear and basis-invariant, so it needs no inner
eigenvector cross-fit at all and should be computed over the full held-out set. The
triage's −41.8 / −47.2 / +7.12 remain diagnostic in sign and magnitude; the formal table
should not reuse a half-split trace.

## Correction to how validity FAIL was described

`C_cert(f) = C_obs − Cov(m−f) ⪯ C_obs` holds for **any** decoder `f`, so a failed
validity check does not void the `C_cert` numbers. What it does mean is that the bound is
probably loose and that the geometry may not be called intrinsic or architecture-stable.
An earlier reading of the triage output overstated this as "no result can be a
conclusion".

## Top priority after the triage: an end-to-end dry run

**CLOSED `d2036b1`.** `tests/test_end_to_end_dryrun.py`, 44 s, 8 attacks, one of which (lock edited before the locked evaluation) was ACCEPTED and is now fail-closed. 11 tests.

Five of the six defects found once the pipeline was actually executed were
cross-component contract failures — a renamed field read elsewhere, failure that did not
propagate, a variable used before assignment, a guard blocking the pipeline it guards, a
parser shifting paths that a new filter had started to depend on. None was reachable
from any single component's unit tests, and there were 302 of those.

The missing test is a synthetic end-to-end run over the whole chain —
`cache → train → select → lock → materialise test → evaluate_locked → gate` — that then
attacks it: a second selection, a dirty tree, an edited lock, a swapped cache, a missing
checkpoint. It belongs in the suite that runs on every push, not in a checklist.


---

# Status after the four pre-full-run blockers (2026-08-17)

All four closed; see `reports/pre_full_run_blockers.md`. 322 tests. HEAD `d2036b1`.

The blockers were protocol, not science. They do not change the triage's finding,
which is that **no arm beat chance** and the experiment therefore had no resolving
power to discriminate arms at pilot scale.

Next, and explicitly NOT part of any gate: an exploratory power diagnostic
(learning curve). It is registered here before it is run, so its status cannot be
revised afterwards.

---

## Pre-full mechanical blocker: the spectrum's provenance is recorded but never enforced

`fit_observability_spectrum.py` now refuses a dirty tree and stamps `git_commit` /
`git_dirty` into the frame checkpoint. The **consumer** does not check either.

`build_arm_frame` resolves `spectrum/{tag}_seed{seed}.pt` and hands it to `SpectralFrame`,
which validates `operator == "certified"`, warns on `validity_pass is False`, and takes
`eigenvectors`. Nothing reads `git_dirty` or `git_commit`. So this path is legal:

    fit_observability_spectrum.py --allow-dirty      ->  git_dirty = True in the .pt
    (commit the tree)
    train_coordinates.py --scope gate                ->  run summary git_dirty = False

and the Gate run's own clean provenance masks the dirty evidence it was built on. The
summary records only `spectrum_file`, a path, so nothing downstream can even detect it
after the fact.

**Required before any full Gate run**, in gate scope only:

* refuse unless `blob["git_dirty"] is False`
* refuse unless `blob["git_commit"] == prov["git_commit"]`
* refuse unless `blob["tag"] == args.tag` and `blob["seed"] == args.seed`
* record `spectrum_git_commit`, `spectrum_digest`, `spectrum_cache_tag` in the arm
  summary so they propagate into the selection lock

Exploratory scopes (`powercurve`, `receiver_control`, `p0_7_basis`) must keep working
against a stale frame: the k diagnostic reuses `triage1_seed0` from `48e8351`
deliberately, and that reuse is registered in `reports/powercurve.md`.

## Wording, fixed: the learning curve is a fixed-compute result

"Ruled out data volume" overstated it. The curve holds optimisation compute at 2320
steps, which is what makes sample count the isolated variable and is also the limit of
the claim. A 10k run under the full protocol's 40 epochs carries 5.0x the optimisation
compute (extractor trains on split B only, 46.4% of train: 10000 -> B ~ 4640 ->
40*ceil(4640/16) = 11600 steps vs 2320 here), so it would differ in two ways at once.
Supported: *at fixed optimisation compute, 100 -> 928 unique samples produces no
reproducible improvement, and that does not justify the more expensive 10k run.*

## Wording, fixed: D_cert^LCB is a bootstrap percentile bound

The mass-from-directional-bounds algebra is exact, but the underlying `L_fj` are
bootstrap percentile quantiles with no BCa correction, and `_per_sample_contributions`
centres with the full measurement-half means before resampling, so a replicate reuses
`A_bar`, `B_bar` instead of re-estimating them. Reported as `bound_type` in the JSON.
Not a blocker at current effect sizes; must not be called analytic.
