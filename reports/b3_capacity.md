# B3 — reflector capacity of arm E

Does arm E's reflector count let `R_E = Q_φ H` reach an arbitrary k-dimensional subspace? If not, `BER_learned > BER_spectral` at that k would reflect **parameterisation capacity** rather than channel geometry, and Gate 3B would not mean what it says.

Fitted with the production `HouseholderFrame` itself — d = 16384, frozen Haar base, paired identity initialisation, real reflector ordering — from commit `bdd02ba` on a clean tree. Optimised to a plateau, not a fixed budget; **no cell hit the 6000-step limit**.

## Metric

For orthonormal k-frames `R`, `T`: `‖RᵀR − TᵀT‖²_F = 2k − 2‖RTᵀ‖²_F` and `‖TᵀT‖_F = √k`, so with `A_sub = ‖RTᵀ‖²_F / k = (1/k) Σᵢ cos²θᵢ`,

```
E_sub = ‖RᵀR − TᵀT‖_F / ‖TᵀT‖_F = √(2(1 − A_sub))
```

The projector error and the mean squared principal cosine are the same quantity. Only `R Tᵀ` (k×k) is ever formed. The capacity metric is `A_best` over the trajectory — the question is whether the subspace was ever reached — with `A_final` kept as an optimisation-stability diagnostic.

## Decision rule, registered before the sweep finished

`A_best ≥ 0.99` sufficient, `≥ 0.95` marginal, otherwise below threshold, using the **minimum over seeds**: an `m` that works on one draw and not another is not a capacity a locked experiment can rely on. `m(k)` is the smallest sufficient `m`.

A below-threshold fit is called a capacity limit **only** where the Grassmann count `m(d−1) ≥ k(d−k)` already forbids generic coverage (16.0 / 63.8 / 127.0 / 252.0 for k = 16 / 64 / 128 / 256). Otherwise it stays `ambiguous_capacity_vs_optimization`: a plateau shows where this optimiser stopped, not that no parameter setting reaches the target.

## Generic Haar target — the coverage stress test

| k \ m | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **64** | 0.9766 ~ | **1.0000** | **1.0000** | **1.0000** |
| **128** | 0.5066 ✗ | 0.9883 ~ | **1.0000** | **1.0000** |
| **256** | 0.2650 ✗ | 0.5136 ✗ | **0.9941** | **1.0000** |

**bold** = empirically sufficient, `~` = marginal, `✗` = dimension count forbids generic coverage.

## Reachable target — same family, same frozen base

| k \ m | 64 | 128 | 256 | 512 |
|---|---|---|---|---|
| **16** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **64** | **0.9999** | **1.0000** | **1.0000** | **1.0000** |
| **128** | **1.0000** | **0.9999** | **1.0000** | **1.0000** |
| **256** | **1.0000** | **1.0000** | **0.9999** | **1.0000** |

Every reachable cell is **≥ 0.9999**: whenever the target provably lies in the family, the optimiser finds it. `(k=256, m=128)` fits a reachable target to 1.0000 while failing the generic dimension count, which makes the scope of the counting argument concrete — it bounds coverage of the whole Grassmannian, never the reachability of one target.

**How far this attributes the generic shortfalls.** The collapse of the three dimension-infeasible cells — `(128, 64)`, `(256, 64)`, `(256, 128)` — is attributable to insufficient generic coverage, since coverage is analytically impossible there. For the dimension-*feasible* cells that fall short, `(64, 64)` at 0.9766 and `(128, 128)` at 0.9883, the remaining gap is **not identified** as capacity versus optimisation: a reachable target reaching 1.0 shows the optimiser can fit *some* family-internal targets, not that the generic landscape is free of optimisation difficulty. Those cells stay `marginal_fit`, and the larger `m` is selected conservatively by the pre-registered empirical rule rather than by an attribution the data does not support.

## Recommended m(k)

| k | `m*_B3(k)` | by class |
|---|---|---|
| 16 | **64** | 64:empirically_sufficient, 128:empirically_sufficient, 256:empirically_sufficient, 512:empirically_sufficient |
| 64 | **128** | 64:marginal_fit, 128:empirically_sufficient, 256:empirically_sufficient, 512:empirically_sufficient |
| 128 | **256** | 64:structurally_insufficient, 128:marginal_fit, 256:empirically_sufficient, 512:empirically_sufficient |
| 256 | **256** | 64:structurally_insufficient, 128:structurally_insufficient, 256:empirically_sufficient, 512:empirically_sufficient |

The rule selects `m*_B3(k) = {16: 64, 64: 128, 128: 256, 256: 256}`. Production keeps a floor of 128, so what is configured is

```
m_configured(k) = max(128, m*_B3(k)) = {16: 128, 64: 128, 128: 256, 256: 256}
```

The two differ only at `k = 16`, where the floor is more conservative than the rule requires. **The main setting `k = 64, m = 128` is sufficient and is not changed.** Only the `k` sweep needs more reflectors, and `k = 128` is the one cell where the configured 128 is merely marginal (0.9883).

## What this does and does not establish

The generic Haar target is a **stress test**. The channel-optimal subspace may belong to a lower-dimensional family, so a below-threshold generic fit does not predict that arm E cannot find *that* subspace. This is a necessity argument about capacity, not a prediction of failure. The dimension count is analytic; the sufficiency verdicts are empirical over two seeds.

## Correction: the first sweep measured its own step budget

An earlier 250-step sweep put `(k=64, m=128)` at 0.9373 and classified the configured main setting as *insufficient*, which produced an `m(k) = max(128, 4k)` recommendation. A convergence control overturned it — the same cell reaches 1.0000 by 2000 steps — so that table measured how far 250 Adam steps get, not what the parameterisation can represent. It is kept as `reflector_capacity_pretaxonomy.json`. The reachable column sitting at ~1.0 did *not* imply the generic targets had converged: the two have different landscapes and a reachable target starts far closer to the paired-identity initialisation, which is exactly why the control was run rather than inferred.
