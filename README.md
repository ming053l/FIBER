# FIBER

**Learning Measure-Preserving Robust Coordinates for Full-Image Diffusion Steganography**

Status: **Phase 0 — not yet implemented.** See [PLAN.md](PLAN.md) for the implementation
plan, acceptance criteria and risk register.

---

## The scientific question

Everything before Phase 6 exists to answer exactly one question:

> Can a **measure-preserving change of coordinates** increase the information
> recoverable through a **frozen** diffusion channel?

Formally, with a frozen generator `G` and a channel `T`:

```
Z ~ N(0, I)  ->  X = G(Z)  ->  Y = T(X)
```

we look for a transform `W = F(Z)` with `F` measure-preserving (so
`P(W) = P(Z)` exactly, by construction) such that a designated subset `W_R`
of `k` coordinates maximises `I(W_R; Y)`.

The target is **not** `d(x_s, x_h) ↑`. The target is
`P(Z_stego) == P(Z_native) == N(0, I)` **by construction**, while `I(W_R; Y)` grows.

## What this is not

FIBER is deliberately *not* the InvCISD line of work. Forbidden by design:

| Forbidden | Why |
|---|---|
| `ℓ_h = ℓ_r + f` or any secret residual | FIBER is a reparameterisation, not a perturbation |
| Fine-tuning the generator / VAE / text encoder | The channel must be *native*, not manufactured |
| Distribution matching by KL / MMD / GAN penalty | Measure preservation must be structural, not approximate |
| Extractor seeing `z`, the clean image, or the reference | Receiver observes `Y` only |
| Rejection-sampling latents for robustness | Changes `q(z) ≠ p(z)` |

Empirical distribution tests are **bug detection**, never the security mechanism.

## Two levels

**Level 1 — Linear.** `W = QZ`, `QᵀQ = I`. Since `N(0, I)` is rotationally
invariant, *any* orthogonal `Q` preserves the measure exactly. So Level 1 asks a
sharp question: **which k-dimensional subspace of latent space survives the
channel best?** This is the kill-or-go experiment.

Its central object is not a learned `Q` but the **Generative Observability
Spectrum**. With `C_obs ≜ Cov(E[Z|Y])`, the Bayes-optimal k-dimensional linear
readout is the top-k eigenspace of `C_obs`, and because `C_obs ⪯ I`,

```
λ_j = 1 − MMSE_j ∈ [0,1]        (fraction of direction j's variance that survives)
Tr(C_obs) = Σ λ_j               (effective number of recoverable dimensions)
```

`Tr(C_obs)` answers *"how many dimensions of information can this generative
channel carry?"* — a bounded, interpretable number that BER is downstream of.
Derivation and caveats: PLAN.md §3.

**Level 2 — Nonlinear.** `U = Φ(Z)` (probability integral transform),
`W = P_φ(U)` with a volume-preserving torus coupling flow, `|det J| = 1`.
Only attempted if Level 1 shows signal.

## Gate discipline

Each phase ends in a gate. A failed gate stops the project or forces a
falsification experiment — it never gets papered over with ECC, a bigger
extractor, or more augmentation. See PLAN.md §7.

The Level-1 gate is deliberately split, because the scientific claim and the
methodological claim are different questions:

- **Gate 3A (science).** `min(BER_spectral, BER_learned) < BER_random`, by a
  paired-bootstrap margin. This decides whether the FIBER hypothesis holds.
  **The only kill condition is `data-derived ≈ random`.**
- **Gate 3B (framing).** `learned` vs `spectral`. Decides whether the story is
  *learning* robust coordinates or *characterising* a discoverable spectrum.
  Never a kill gate — a spectral win is still a win for the hypothesis.

The first real success condition is **not** "full image recovered". It is:

> Under identical `Z`, generator, output distribution and extractor capacity,
> a data-derived measure-preserving coordinate system is significantly more
> recoverable than a random one.

Comparisons are against **random** subspaces, never identity: identity loses to
any global transform for a trivial locality reason (PLAN.md R1).

## Layout

```
FIBER/
├── configs/            # sd15, channels, linear_fiber, nonlinear_fiber
├── src/fiber/
│   ├── diffusion/      # frozen generator wrapper, cached dataset builder
│   ├── channels/       # jpeg / resize / noise / blur (non-differentiable, real codecs)
│   ├── transforms/     # identity, hadamard, orthogonal, torus_coupling, fiber_flow
│   ├── models/         # extractor
│   ├── coding/         # partition, ecc, crypto
│   ├── metrics/        # ber, distribution, reconstruction
│   └── utils/
├── scripts/            # cache_native_dataset, train_*, eval_*
├── tests/              # invertibility + distribution unit tests
└── reports/            # one markdown per phase, committed
```

Data and generated images live on `/ssd2/ming/FIBER` (`/ssd1` has ~120 GB free);
code stays here and reaches the data through the config, not through symlinks.

## Related work in this tree

`/ssd1/ming/STEGO` holds a full reproduction of the InvCISD baseline
(SD-v1.5 + EDICT + IP-Adapter, two-stage training). FIBER reuses its model
weights, its conda env and its COCO/prompt assets — see PLAN.md §Reuse.
