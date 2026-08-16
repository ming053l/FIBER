# FIBER

**Learning Measure-Preserving Robust Coordinates for Full-Image Diffusion Steganography**

Status: **Gate 0 PASSED** ([reports/phase0.md](reports/phase0.md)); Phase 1 caching in
progress. See [PLAN.md](PLAN.md) for the implementation plan, acceptance criteria and
risk register.

| Phase | State |
|---|---|
| §3.5 synthetic spectrum validation (no GPU) | done — `tests/test_spectrum_synthetic.py` |
| Gate 0 — environment | **PASS**: DDIM, `eta=0`, `init_noise_sigma=1.0`, `z` round-trips with max abs error `0.0`, 100 samples zero NaN, generation bit-identical across runs (8/8 pairs) |
| Phase 1 — cached native dataset | pilot 2k/256/256/256 caching; `scripts/cache_native_dataset.py` |
| Phase 2 — denominators + first `Ĉ_obs` | code ready, not run |
| Phase 3 — 5 arms, cross-fit, gates 3A/3B | code ready, not run |

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
├── configs/            # sd15, channels, linear_fiber
├── src/fiber/
│   ├── diffusion/      # frozen G, prompt pool, cache builder + dataset, DDIM inversion
│   ├── channels/       # jpeg / webp / resize / noise / blur — real codecs, seeded
│   ├── transforms/     # identity, signperm, hadamard, spectral, householder
│   ├── spectrum/       # C_obs estimator (randomized SVD / exact Gram, cross-fit)
│   ├── models/         # extractor (2 heads) + teacher (E[Z|Y])
│   ├── training/       # cross-fit train/eval loops
│   ├── metrics/        # sign BER, paired bootstrap, gate thresholds
│   └── utils/          # config, seeding, logging
├── scripts/            # phase0_env_audit, cache_native_dataset,
│                       # fit_observability_spectrum, train_coordinates,
│                       # eval_coordinates, ddim_reference
├── tests/              # 112 tests, all CPU except the cached-dataset ones
└── reports/            # one markdown per phase, committed
```

### Transforms are k-frames, not matrices

`d = 16384`, so a dense `Q` is a 1 GB object — and only the `k` robust
coordinates are ever read. Every arm is therefore a **frame** `R ∈ R^{k×d}` with
`R Rᵀ = I_k`, exposing `project(z) = Rz` and its adjoint `expand(a) = Rᵀa`. Any
such `R` extends to an orthogonal `Q`, so measure preservation is unaffected,
and the Gate 3A orthonormality check `‖R Rᵀ − I‖_∞ < 1e-5` costs `O(k²d)`
without ever forming `Q`.

### Paths are machine-specific

`configs/sd15.yaml` points at a local SD-v1.5 checkpoint, a COCO caption dump and
two SSDs (`/ssd1` for code and reports, `/ssd2` for images and latents). Nothing
is symlinked — every path is read from the config, so a different machine needs
edits to `configs/sd15.yaml` (`model.checkpoint`, `paths.*`) and
`configs/linear_fiber.yaml` (`dataset.prompts.*`) and nothing else.

### Running it

```bash
conda activate invcisd
pip install -e .
pytest -q                                                    # 112 tests, ~40 s, no GPU
python scripts/phase0_env_audit.py                           # Gate 0
python scripts/cache_native_dataset.py --pilot               # Phase 1 (~1 h)
python scripts/fit_observability_spectrum.py --tag pilot --per-attack
python scripts/train_coordinates.py --tag pilot --arm C_hadamard --k 64 --seed 0
python scripts/eval_coordinates.py --tag pilot --k 64        # gates 3A / 3B
```

Data and generated images live on `/ssd2/ming/FIBER` (`/ssd1` has ~120 GB free);
code stays here and reaches the data through the config, not through symlinks.

## Related work in this tree

`/ssd1/ming/STEGO` holds a full reproduction of the InvCISD baseline
(SD-v1.5 + EDICT + IP-Adapter, two-stage training). FIBER reuses its model
weights, its conda env and its COCO/prompt assets — see PLAN.md §Reuse.

## License

Apache-2.0 — see [LICENSE](LICENSE).
