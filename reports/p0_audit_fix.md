# P0 Audit — fix report

Response to the external audit of the FIBER Phase 0–3 skeleton. Every item below was
raised as a scientific confound that could invalidate the Phase-3 claim, not as a code
defect. The report is organised by the audit's own numbering.

**The claim this report supports is narrow.** It says the protocol no longer contains
the identified confounds and that each fix is pinned by a test. It says nothing about
whether the FIBER hypothesis holds — no Phase-2/3 experiment has been run.

Verification is mechanical: `reports/invariants.yaml` maps 37 protocol invariants to
98 named tests, and `scripts/audit_invariants.py` refuses to pass if any cited test has
been renamed or deleted. Full suite: **292 tests**.

Three defects were found *by this audit process itself* after the fixes were first
declared complete, and are recorded in the sections below rather than quietly repaired:
a test-cache binding that proved only when the manifest was written, a completeness
check that exempted an arm with no runs at all (exercised for real by a driver typo),
and a "rank" that counted positive quadratic forms.

---

## 1. P0-1 — observability estimator

**Was.** `Ĉ_obs = (1/N) Σ m̂ m̂ᵀ`, the uncentered second moment of a teacher's output,
described as "a lower bound limited by teacher capacity".

**Problem.** That is not a lower bound. For `f ≠ E[Z|Y]` there is no PSD ordering
`Cov(f) ⪯ C_obs`; measured on the synthetic channel, `f = 3m` gives
`max eig(Cov(f) − C_obs) = +7.97`, and `+23.3` at `f = 5m`.

**Now.** The decoder-certified operator

```
C_cert(f) = E[z_c f_cᵀ + f_c z_cᵀ − f_c f_cᵀ] = C_obs − Cov(m − f) ⪯ C_obs
vᵀ C_cert v = Var(vᵀZ) − E[(vᵀz_c − vᵀf_c)²]      ( = 1 − MSE_v for Z ~ N(0,I) )
```

so a weak decoder understates observability and can never manufacture it — measured
within `+0.07` of the bound at every teacher scale from 0.25 to 5.0.

Three things the audit's prescription did not anticipate, each measured:

- **`C_cert` is indefinite** (range `[−3.00, +0.014]` at `f = 3m`). Every SVD-based
  solver ranks by `|λ|` and would return the directions the decoder is *worst* on as the
  top-k. The solver is algebraically largest, and a test asserts magnitude selection
  picks a negative direction where the certified one does not.
- **ARPACK stalls** on the degenerate null space (`d = 16384 ≫ 2N` leaves `d − 2N` exact
  zeros; observed *"No convergence, 8/11 converged"*). Default is an exact
  range-restricted eigendecomposition on the `2N`-dimensional range, which also returns
  the complete signed spectrum. `eigsh(which="LA")` is kept as an independent
  cross-check and agrees to `1e-6`. Cost at production size: 2.7 s / 4.9 s / 102 s and
  1.3 / 2.1 / 4.7 GB at `N_operator` 500 / 1000 / 2500 — no top-k solver needed.
- **`Tr(C_cert)` can be negative**, so the headline is the positive mass, with `D⁻`
  reported separately as a misspecification diagnostic.

Centering is mandatory (an uncentered estimate turns a decoder bias into the leading
direction — tested by asserting the spurious direction *is* the bias direction), the
denominator is `N−1`, and split A is subdivided so `A_teacher ∩ A_operator = ∅`.

**Smoke evidence on real data.** A deliberately undertrained teacher (2 epochs, 64
samples): the old estimator reports `λ_var = +6.32` and a mass of 16.4; the certified
operator reports `λ_skill = −8.22` and `D_cert = 0.125`, and the validity gate fails.
That is the failure mode this item exists to prevent, on the actual pipeline.

## 1b. P0-1.1 — the headline is a property of the subspace

Clipping the *diagonal* of `V C_cert^held Vᵀ` is basis dependent: `C_V = [[−1,2],[2,−1]]`
clips to 0 while its eigenvalues are `(1, −3)`. On pilot data the two differ by 24×.
The headline is now `D_cert(V) = Σ_j max(μ_j, 0)` with `μ = eig(V C_cert^held Vᵀ)`.

Self-audit of that fix found the same error one level down: summing *positive*
eigenvalues is a max-type functional, so choosing the rotation and measuring it on the
same samples rectifies noise into mass — a zero-skill decoder scored 3.58 at
`N=64, k=32` where the truth is 0. The rotation is now chosen on one half and measured
on the other, **symmetrically**, and `certified` rests on a **one-sided bootstrap lower
bound** rather than on the numerical tolerance. Calibration: null gives
`D = 0.0000, D^LCB = 0.0000`; the exact conditional mean gives `D = 15.48,
D^LCB = 15.23`.

**A count of positive directions is not a rank.** `C = [[1,2],[2,1]]` has both
diagonals positive and eigenvalues `(3, −1)`, so its positive inertia is 1 where a
direction count would say 2. The count is named
`certified_positive_direction_count`, and positive inertia is certified separately by
bounding the whole restricted matrix: a bootstrap spectral-norm radius `ε` plus Weyl
gives `λ_j(C_V) ≥ λ_j(Ĉ_V) − ε` simultaneously for every `j`, so
`#{j : λ_j(Ĉ_V) − ε > 0}` lower-bounds the positive eigenvalue count. Eigenvalues are
rotation-invariant, so this needs no cross-fitted rotation and one radius covers all
`k` at once. Tested against a data-realised matrix whose diagonal is `(0.4, 0.4)` while
its eigenvalues are `(1, −0.2)`.

Multiplicity spans both levels — `α/(2k)` per direction per fold, by a union bound that
needs no independence between folds — and the per-sample terms are scaled by `N/(N−1)`
so that "the column mean equals `vᵀC_cert v`" is exact rather than nearly so.

## 2. P0-2 — random subspace baselines

Signed permutation is local and Hadamard is structured (`±1/√d` in every entry), so
beating either showed only that the derived directions beat a family.
`HaarRandomFrame` (QR with the sign correction) is the gate denominator; signed
permutation, Hadamard and the frozen Householder control are reported beside it, and a
control beating the locked arm is flagged in the report rather than left to the reader.

Isotropy does **not** separate Haar from Hadamard — both have `E[rᵢrⱼ] = δᵢⱼ/d` — so the
test uses the spread of a squared entry, `Beta(1/2,(d−1)/2)` against identically zero.
The four families also quantify R1's locality axis directly: participation ratio 1.0
(signed permutation), 1.1 (frozen Householder), 5470 (Haar), 16384 (Hadamard).

## 3. P0-3 — locked model selection

The gate previously chose, per channel group, the derived run with the lowest **test**
BER and bootstrapped on that same data. The pipeline is now
`VAL → selection.json → LOCK → TEST`, with `select_method.py` reading `results[val]`
only (`_forbid_test()` raises at runtime) and `eval_coordinates.py` refusing to run
without the artifact and refusing `--split val`.

The lock names **exact runs with content hashes** — identifying `(arm, k)` let a run
dropped into the directory afterwards join the average through the filesystem — covering
`selected_runs`, `reference_runs` and `context_runs`, because the invariant is that
nothing added after the lock changes the output, not merely that the gate statistic
survives. Seeds are replications, aggregated hierarchically so a structural seed is not
weighted by how many receiver repetitions it happens to have.

The adversarial test constructs a case where D_spectral wins on val (0.40 vs 0.45) and
E_learned wins on test by a wide margin (0.30 vs 0.46), and asserts the gate reports
D_spectral at its true test BER of 0.46.

## 4. P0-4 — learned-Q objective

`1[W>0]` returns a bool tensor, so the autograd graph ends there and the frame's
gradient is `None` — absent, not small. Discovery is MSE-only and `train_extractor`
**raises** if `learn_frame=True` is combined with a non-zero sign weight. PLAN withdraws
the Rev-2 prediction that the learned arm should win *because* it optimises sign BER.

Measurement then showed arm E would have started at **identity**: a raw random-reflector
product has `E[r₀²] = 0.969` and participation ratio 1.1 at `d = 16384, m = 128`, which
is the one starting point R1 says loses for a trivial reason. `R_E = Q_φ H` on a frozen
Haar frame with reflectors initialised in identical pairs gives `Q_φ(0) = I` exactly —
measured `‖R_E(0) − H‖_∞ = 2.2e-8` — so arm E begins *at* the denominator, and a test
confirms one gradient step leaves it, reduces the loss and preserves orthogonality.

## 5. P0-5 — teacher and receiver architecture

The default teacher's centered outputs are `W(h − h̄)` with `h ∈ R^512`: the first 512
principal directions hold `> 1 − 1e-9` of its output variance regardless of the channel.
`SharedTrunkSpatialTeacher` keeps the identical trunk and changes only the head, so the
isolated variable is whether spatial position survives; an independent conv pyramid is a
second opinion. The same critique applies to the **receiver**, so the locked method and
the Haar reference are also evaluated with a no-GAP extractor (shared trunk, capacity
ratio 1.09).

`compare_teachers.py` reports `D_cert` per teacher, the principal-angle spectrum (a mean
hides "a few directions agree strongly" versus "all agree weakly"), a 2×2 cross-decoder
certificate, and parameter counts. **Architecture dependence may only be declared when
both teachers pass validity** — a decoder that failed it has an eigenspace that may be
noise, so a low alignment cannot separate "architecture-dependent" from "not fitted".

## 6. P0-6 — stochastic channel draws

The attack seed omitted the epoch, so a sample meeting `noise005` twice received the
identical Gaussian field — a finite deterministic corruption table. `draw_salt` is the
epoch while training and the constant `"eval-v1"` while evaluating. The evaluation side
matters as much: the paired bootstrap differences arms on the same
`(z_i, prompt_i, attack_i)`, so arms seeing different realisations would be compared
across different channels. Loader-order invariance is tested, since the draw is keyed by
`sample_id` rather than by position in a batch.

## 7. P0-7 — subspace versus basis

`span(AV) = span(V)`, so every certified quantity is unchanged by an in-subspace
rotation — asserted to `1e-9` on `D_cert` and `1e-10` on the restricted operator's
eigenvalues — while the sign bits move. D1/D2/D3 hold the subspace fixed and vary only
the basis; `expm(S − Sᵀ)` is `SO(k)`, stated as such, while D2's QR covers both
components of `O(k)`.

`V` is a buffer, never a parameter (`V.grad is None`, `S.grad ≠ None`), `A_φ(0) = I`, and
discovery uses `tanh(W/τ)` because a hard target severs the gradient to a rotation
exactly as it does to a frame. `τ` is an arm hyperparameter and therefore a val-locked
choice. Neither basis arm can win Gate 3A, and D1 is the **single** spectral run at the
rotations' `base_seed` — averaging D1 over its seeds would compare several subspaces
against rotations of one while claiming the subspace was fixed.

## 8. Secondary cleanups

| | |
|---|---|
| Prompts | COCO-val is *held-out prompt instances*, not a held-out domain |
| Per-attack | *fixed-decoder operational spectrum under attack t*, never `Cov(E[Z\|Y,T=t])` |
| Pilot verdicts | `PROVISIONAL_PASS` / `PROVISIONAL_FAIL` / `INCONCLUSIVE` only |
| Naming | `D_obs` is certified observability mass, not Shannon capacity |
| Provenance | artifacts refuse to be produced from a dirty tree or outside a repository |
| B0 | run identity is (arm, k, structure seed, receiver arch, receiver seed, scope) |
| B1 | no test image is materialised or accessed before the lock; the binding is written into each shard at generation, so a skipped shard cannot be covered by a freshly written manifest; the official evaluation is write-once and the test execution protocol is itself locked |
| B2 | a missing registered seed stops the lock, and a wholly absent required arm counts as all its seeds missing rather than exempting itself; "required" is declared in `fiber.required_arms`, not inferred from which arms produced files |

`environment.yml` / lock file and CI remain open (see §9).

## 9. Remaining risks and open items

1. **B3 is complete** (`reports/b3_capacity.md`, swept from `bdd02ba` on a clean tree,
   no cell hitting the step limit). The pre-registered rule selected
   `m*_B3(k) = {16: 64, 64: 128, 128: 256, 256: 256}`; production applies a floor, so
   `m_configured(k) = max(128, m*_B3(k)) = {16: 128, 64: 128, 128: 256, 256: 256}` — the
   two differ only at `k = 16`. **The main setting `k = 64, m = 128` is sufficient and
   unchanged.** The collapse of the three dimension-infeasible cells is attributable to
   insufficient generic coverage; for the two dimension-feasible marginal cells the gap
   is **not identified** as capacity versus optimisation, and the larger `m` is chosen
   conservatively by the rule rather than by an unsupported attribution. The generic Haar target is a stress test, so a
   below-threshold cell does not predict that arm E cannot find the *channel-optimal*
   subspace — it is a necessity argument about capacity.
2. **The pilot cache predates the lock.** `evaluate_locked.py` detects this and
   downgrades the recorded claim from *no test sample existed* to *no test sample was
   accessed*. The full run, cached under `--post-lock`, supports the stronger form.
3. **Hierarchical CIs are thin at pilot seed counts.** Reported beside the paired
   interval, not instead of it; ≥5 seeds for the locked method before a seed-level
   interval is quoted as evidence.
4. **No dependency lock file and no CI.** Versions are recorded in reports; pinning and
   a GitHub Actions run over the CPU suite are not yet in place.
5. **Nothing has been measured about the channel.** Every number in this report is a
   protocol calibration or a synthetic check.
6. **The inertia radius is a bootstrap estimate**, not an analytic concentration bound,
   so `certified_positive_inertia` is an approximate certificate. It is labelled as such
   in the output.

## 10. Restarting Phase 2/3

```bash
conda activate invcisd && pip install -e .
pytest -q                                   # 284 tests
python scripts/audit_invariants.py          # invariant map -> live tests

# pre-lock: train/val only. Caching a test split without --post-lock is refused.
python scripts/cache_native_dataset.py --pilot --splits train val --batch 12
python scripts/fit_observability_spectrum.py --tag pilot --seed 0 --per-attack
python scripts/compare_teachers.py --tag pilot --seed 0 --k 64
bash scripts/run_pilot.sh                   # arms, val only, declared seed sets

# lock, then compute test for the first time
python scripts/select_method.py --tag pilot --registered-seeds ...
python scripts/cache_native_dataset.py --pilot --splits test test_heldout_prompts \
       --post-lock reports/selection_pilot.json
python scripts/evaluate_locked.py --tag pilot
python scripts/eval_coordinates.py --tag pilot --split test
```

The triage run — `k = 64`, five representative channels, 3 Haar draws, 1 Hadamard, 1
frozen Householder, 2 certified-spectrum seeds, 2 learned-Q seeds, both teachers — has
no `PASS`/`KILL` authority and exists only to decide whether the ~31 GPU-hour full
protocol is worth starting.

---

## Verdict

**READY_FOR_PHASE2_3**

Every identified confound is closed and pinned by a named test that the audit script
verifies still exists. The protocol blockers open at the last two reviews — physical
test isolation, seed completeness, and the three defects this process found after the
fixes were first declared complete — are implemented and adversarially tested. B3 is
complete and `m(k)` was set by the rule registered before the table was seen, leaving
the main setting unchanged.

This authorises the **triage run only**: `k = 64`, the five representative channel
groups, 3 Haar draws, 1 Hadamard, 1 frozen Householder, 2 certified-spectrum seeds, 2
learned-Q seeds, both teacher architectures. Triage has no `PASS`/`KILL` authority. The
full ~31 GPU-hour protocol starts only if triage shows a reproducible
derived-versus-Haar signal and no catastrophic teacher-architecture contradiction.

What READY does **not** mean: nothing has been measured about the channel. Every number
in this report is a protocol calibration or a synthetic check, and the pilot's own
results will be `PROVISIONAL_*` by construction.
