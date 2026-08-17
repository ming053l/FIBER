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

The **distribution** of per-coordinate correlations is indistinguishable from noise:
not one coordinate stands out. If the channel carried recoverable information that more
data would unlock, some coordinates would already be ahead at N=928. What remains is a
uniform mean shift of +0.0037 (0.06 of the null sd; run-level t=3.32 on 4 df) which is
spread evenly over all coordinates rather than concentrated in any — the signature of a
faint global effect, not of per-coordinate recovery.

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

*(to be filled in after the run)*
