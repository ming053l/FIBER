# B3 — reflector capacity of arm E

Does the configured reflector count let `R_E = Q_phi H` reach an arbitrary k-dimensional subspace? If not, `BER_learned > BER_spectral` at that k would reflect PARAMETERISATION CAPACITY rather than channel geometry, and Gate 3B would not mean what it says.

Fitted with the production `HouseholderFrame` itself — d = 16384, frozen Haar base, paired identity initialisation, real reflector ordering — not a separate generic fitter.

## Metric

For orthonormal k-frames `R`, `T`: `||R'R - T'T||_F^2 = 2k - 2||RT'||_F^2` and `||T'T||_F = sqrt(k)`, so with `A_sub = ||RT'||_F^2 / k = (1/k) sum_i cos^2(theta_i)`,

```
E_sub = ||R'R - T'T||_F / ||T'T||_F = sqrt(2(1 - A_sub))
```

The projector error and the mean squared principal cosine are the same quantity. Only `R T'` (k x k) is ever formed; the d x d projector never is.

## Decision rule (registered before the sweep finished)

`A_sub >= 0.99` sufficient, `>= 0.95` marginal, otherwise insufficient, using the **minimum over seeds**: an m that works on one draw and not another is not a capacity a locked experiment can rely on. `m(k)` is the smallest sufficient m.

## Necessary dimensional condition

`dim Gr(k,d) = k(d-k)`; a reflector direction carries at most `d-1` degrees of freedom, so generic coverage needs `m >= k(d-k)/(d-1)` — at d = 16384 that is 16.0 / 63.8 / 127.0 / 252.0 for k = 16 / 64 / 128 / 256. Necessary, **not** sufficient: measured below, roughly 4x the bound is needed in practice.

## Generic Haar target — the coverage stress test

| k \ m | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | 0.9445 | 0.9992 | 1.0000 | 1.0000 |
| **64** | 0.7369 | 0.9373 | 1.0000 | 1.0000 |
| **128** | 0.5016 | 0.8044 | 0.9646 | 1.0000 |
| **256** | 0.2597 | 0.5041 | 0.8693 | 0.9880 |

## Reachable target — same family, same frozen base

| k \ m | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | 1.0000 | 1.0000 | 1.0000 | 0.9998 |
| **64** | 0.9984 | 1.0000 | 1.0000 | 1.0000 |
| **128** | 0.9975 | 0.9972 | 1.0000 | 1.0000 |
| **256** | 0.9982 | 0.9971 | 0.9964 | 1.0000 |

## Attribution

Every reachable cell is **>= 0.9964**, including `(k=256, m=64)` at 0.9982. So whenever the target provably lies in the family, the optimiser finds it at the same 250-step budget. The generic shortfall is therefore **coverage**, not optimisation, initialisation or budget — which is exactly what this control exists to separate.

## Recommended m(k)

| k | recommended m | why |
|---|---|---|
| 16 | 128 | 128 -> 0.9992 |
| 64 | 256 | 128 -> 0.9373 insufficient; 256 -> 1.0000 |
| 128 | 512 | 256 -> 0.9646 marginal; 512 -> 1.0000 |
| 256 | **none in range** | 512 -> 0.9880 still marginal; no swept m qualifies |

That is `m(k) = max(128, 4k)`, with k = 256 extrapolating to 1024 (untested).

## Consequence for the configured experiment

`E_learned` uses `m = 128` at every k, so at the main **k = 64 it is insufficient by this rule** (0.9373). Options: scale `m(k) = max(128, 4k)`, or lock the gate at k = 64 with m = 256 and report other k as capacity-sensitive ablations marked as parameterisation-limited.

**Limit of the claim.** The generic Haar target is a stress test. The channel-optimal subspace may belong to a lower-dimensional family, so failing generic coverage at m = 128 does not predict that arm E cannot find *that* subspace. This is a necessity argument about capacity, not a prediction of failure.

## The first sweep was invalid — it measured the step budget

A convergence control at the decisive cell overturned the reading above:

```
k = 64, m = 128, generic target
    250 steps -> A_sub 0.9373        (the sweep's value, classified "insufficient")
   2000 steps -> A_sub 1.0000        (min over seeds 1.0000)
```

So 250 Adam steps was the binding constraint, not the reachable family, and the whole
first table measures how far 250 steps get rather than what the parameterisation can
represent. **The `m(k) = max(128, 4k)` recommendation derived from it is withdrawn.**

The reachable column being ~1.0 everywhere did *not* imply the generic targets had
converged: the two have different landscapes, and a reachable target starts far closer
to the paired-identity initialisation.

The fix is not a larger fixed budget, which would just be another arbitrary number.
The audit now optimises to a **plateau** — no improvement above `tol` over `patience`
steps — and records `converged` and `steps_used` per cell. A cell that merely hit the
step limit is classified `not_converged`, and can neither be recommended nor rule out a
smaller `m`.

_Table above retained as the record of the invalid run; the corrected sweep replaces it._
