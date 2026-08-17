# Paper v1 scope — frozen 2026-08-18

**No new main-line branches from here.** This file exists because the failure mode is
already visible in this project's history: every caveat found so far has been real, and
following each one has repeatedly opened another. The result of running that loop to
completion is a paper where every one of twenty threads is half-done and each invites
"why only this much?".

## The question

> **What latent information remains observable through a frozen generative channel, and
> how does that depend on what the receiver knows?**

Everything in v1 serves that sentence. Anything that does not is future work.

## The four contributions

1. **The object.** `C_obs = Cov(E[Z|Y])`, and the receiver-conditional form
   `C_obs(I_R)`. Claim: latent information is lost *directionally* and
   *observer-conditionally*.
2. **The estimator.** `C_cert(f) = C_obs - Cov(m - f) <= C_obs`, decoder-certified, with
   the cross-fit and the bootstrap bound. Synthetic Gaussian worlds validate the
   estimator; one such world is enough.
3. **The phenomenon.** Blind vs side information — Phase A, four arms, primary contrast
   `S_correct` vs `S_shuffled`.
4. **The consequence.** More certified observability implies better held-out
   recoverability. Ordering only; no watermarking benchmark.

Shape: **define → certify → intervene → predict.**

## Out of scope for v1

Each of these is defensible work. None is needed for the sentence above.

| deferred | why it is out |
|---|---|
| 10k cache and run | no evidence supports the cost; `reports/powercurve.md` |
| further `k` sweeps | the dimensionality bottleneck was falsified; done |
| direct `C_gain(h)` certificate | Phase B, and only if Phase A is strongly positive |
| fusion ablations (early / cross-attention) | one preregistered late fusion, no ablation |
| prompt-compression ladder `Y ⊂ (Y, q(S)) ⊂ (Y, S)` | second paper |
| multiple generators | make SD-v1.5 clean first |
| CFG sweep | only if a core claim needs it |
| watermarking benchmarks | a whole separate literature |

## The rule for anything found from here

| category | test | action |
|---|---|---|
| **Blocker** | the current main result is invalid without it | fix now |
| **Necessary control** | a reviewer cannot interpret the main result without it | do it |
| **Interesting extension** | the core claim still stands without it | discussion / appendix / future work — **no new main line** |

Recent examples, for calibration: the conditioning stale-marker binding was a *blocker*;
`S_shuffled` is a *necessary control*; the direct gain certificate is an *extension*.

## Phase A is the last big directional decision

**If `S_correct >> S_shuffled`:** the paper becomes *Receiver-Conditional Observability in
Frozen Generative Channels*, and the remaining effort goes into making that one phenomenon
solid.

**OUTCOME, 2026-08-18: the second branch.** `reports/phase_a.md`. Primary contrast
ΔBER = −0.00216 ± 0.00298, CI95 [−0.0053, +0.0010], p = 0.135; both placebo gates inside
±0.007; all four arms at the `R2_lin` noise value. Phase B and Phase A-2 are therefore
**not opened**, per the gate written here before the run.

**If `S_correct ≈ S_shuffled`:** stop looking for a sixth explanation. The paper becomes
*Certified Observability and the Limits of Latent Recoverability in Frozen Generative
Channels* — observability is definable, decoder-certifiable, shows anisotropy and
architecture dependence, and blind operational recovery is nonetheless very limited under
a realistic receiver, with sample count, output dimensionality and receiver conditioning
all falsified as explanations under controlled diagnostics.

Both are complete scientific results. Only the flavour differs. What must not happen is a
seventh hypothesis.
