# Pre-full-run blockers — audit

**Verdict: all four implemented, unit-tested and negative-controlled.**
Scope: the four items recorded in `reports/p0_fix_plan.md` after the `triage1` run.
No scientific hyperparameter was changed. No cached image was invalidated.

| # | blocker | commit | tests | negative control |
|---|---|---|---|---|
| 1 | selection lock mechanically write-once | `83006b8` | 3 | reverting to `write_text()` fails the adversarial test |
| 2 | `D_cert^LCB` from simultaneous directional bounds | `1f5a20c` | 4 | old estimator reports 0.548 where the truth is 0 |
| 3 | `trace_C_V` over the full held-out set | `c677a88` | 2 | sd ratio 1.412 vs the theoretical sqrt(2) |
| 4 | synthetic stateful end-to-end dry run | this commit | 11 | reverting the fix it found fails the attack test |

---

## 1. The lock was write-once by convention only

`out.write_text()` replaces an existing lock, and `require_clean()` cannot see it:
`reports/` is in `ARTIFACT_PREFIXES`, so rewriting the lock leaves the working tree
exactly as clean as writing it the first time. The attack is one command — lock, read
the test number, re-run selection over a tweaked candidate set.

`write_once()` uses tmp + `os.link` rather than an `O_EXCL` open, separating the two
properties: `link` is atomic (an interrupted write leaves a temp file, never a truncated
lock that still parses) and fails if the destination exists.

The adversarial test runs a second selection over a candidate set in which a **different
arm wins on val**, and asserts both a non-zero exit and unchanged lock bytes.

## 2. `D_cert^LCB` could not reach zero

```python
masses.append(np.quantile(np.clip(boot, 0, None).sum(axis=1), alpha / n_folds))
```

`sum_j max(b_j, 0)` is non-negative by construction, so under a null where the true
`mu_j` sit at zero its entire bootstrap distribution lives on the positive half-line and
every quantile of it is positive.

Measured with `f` independent of `Z` at scale `eps` (true `C_cert = -eps^2 I`, true
certified mass exactly 0):

| eps | k | old `D_cert_LCB` | point estimate | new |
|---|---|---|---|---|
| 0.10 | 128 | **0.548** | 0.423 | 0.001 |
| 0.03 | 128 | 0.259 | 0.224 | 0.001 |
| 0.10 | 32 | 0.084 | 0.095 | 0.000 |

The `eps=0.1, k=128` row is the bug in its self-evident form: a *lower bound* of 0.548
against its own point estimate of 0.423.

The corrected bound is derived from the directional ones. On
`{mu_fj >= L_fj for all j}` — which the Bonferroni correction already buys at
`1 - alpha/n_folds` — monotonicity of `max(.,0)` gives
`sum_j max(L_fj, 0) <= sum_j max(mu_fj, 0)`. No second alpha spend.

Across folds the headline takes **max**, not mean. Each fold measures in a different
rotated frame, but both bound the same target: for any orthonormal `U` inside `V`,
`sum_j max(u_j' C u_j, 0) <= sum_j max(lambda_j, 0)`, because the diagonal is majorised
by the spectrum and `sum max(.,0)` is Schur-convex. A union bound makes both hold at
once, and the larger of two simultaneously valid lower bounds is a lower bound.

Power is retained — three of the four tests would pass on a bound that is always zero,
so the fourth checks it: `alpha=0.5, sigma=0.3` (true 21.1) gives LCB 13.3.

## 3. The trace was paying for a cross-fit it does not need

`trace_C_V` was fold 0's measured diagonal, i.e. half the held-out set. The inner
cross-fit exists because `sum_j max(.,0)` is a max-type functional. The trace is linear
(unbiased) and rotation-invariant (nothing to select), so splitting it bought no
protection and cost half the sample. Over 150 null draws at `n=256, k=32`: sd 0.391 full
vs 0.551 fold-0, ratio **1.412** against `sqrt(2) = 1.414`, both unbiased at −8.04/−8.07.

This matters because the trace is the one statistic in the certificate that can be
negative, and it answers "is this decoder net-positive at all".

## 4. End-to-end dry run — and the defect it found

`tests/test_end_to_end_dryrun.py` snapshots the working tree into a throwaway git repo,
commits it, and runs the real scripts:

    cache -> spectrum -> train -> select -> LOCK -> materialise test -> locked eval -> gate

on synthetic pixels (`cache_native_dataset.py --synthetic`), d=256, k=8, ~250 images at
64x64, 2 epochs, CPU, **44 seconds**. A snapshot rather than a git worktree at HEAD:
a worktree would test the last commit rather than the code being edited.

Attacks, all refused:

| attack | refused by |
|---|---|
| second selection on a locked tag | `write_once` (blocker 1) |
| selection from a dirty tree | `require_clean` |
| locked evaluation from a dirty tree | `require_clean` |
| swapped cache namespace | `resolve_cache_tag` |
| repeat official test evaluation | `test_eval_<tag>.json` write-once |
| locked arm's checkpoint deleted | artifact verification, before any output |
| lock edited **after** the evaluation | gate compares `selection_sha` |
| lock edited **before** the evaluation | **was accepted — fixed in this commit** |

### The defect

Editing `selection_<tag>.json` before `evaluate_locked.py` exited **0**. The only trace
was `test_cache_post_lock` silently flipping from `True` to `False`.

The test pixels carry a cryptographic binding to the lock bytes they were generated
under. If the lock no longer hashes to that, the lock changed after the test set was
materialised — which is exactly the tamper signature, not a claim to be quietly
downgraded. `evaluate_locked.py` now fails closed on the mismatch.

Having **no** `test_cache_manifest.json` at all stays legitimate: that is test pixels
generated pre-lock, the weaker "never accessed" claim, and it is reported as such. Both
paths are asserted.

### Containment of synthetic artifacts

`synthetic_pixels: true` is stamped in the cache manifest **and in every per-shard
marker** (a manifest can be rewritten over skipped shards; a marker written at generation
time cannot). A test asserts the stamp on every marker.

The dry-run config is **derived from `configs/linear_fiber.yaml`** and shrunk, not
hand-copied: a copied fixture drifts, and a key added to production would silently stop
being covered.

---

## Noted, not changed

* `fit_observability_spectrum.py` does **not** call `require_clean()`, unlike training,
  selection, the locked evaluation and the teacher comparison. It produces the frozen
  `D_spectral` subspace, which is scientific evidence. It records the commit but does not
  refuse a dirty tree. Out of scope for these four blockers; flagged for a decision.
* `lcb_per_direction` averages the two folds' bounds index by index, but direction `j`
  means "the j-th direction of that fold's own fitted rotation". These agree only to the
  extent that both halves estimate the same population eigenvectors. The headline mass no
  longer depends on this (it takes the max over per-fold masses), but
  `certified_positive_direction_count` still does.
* The spectrum checkpoint dict has a duplicate `"k"` key. Cosmetic.
