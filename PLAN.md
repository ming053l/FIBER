# FIBER — Implementation Plan (Phases 0–3)

Revision 2. Written before any implementation; nothing here has been run yet.

Rev 2 changes: Gate 3 split into a scientific gate (3A) and a methodological gate (3B);
Arm D promoted from "a spectral baseline" to the **Generative Observability Spectrum**,
which is now the central quantity of the phase; the broken re-randomisation control
replaced by a **cross-fit** protocol; Gaussian heads instead of circular heads for
Level 1; DDIM inversion demoted to a prompt-assisted reference; R5 rewritten; VAE
scaling separated from the diffusion latent; guidance framed as a hypothesis with a
mandatory sweep; random arms get 5–10 draws; paired bootstrap replaces CI overlap.

---

## 1. Repository audit — findings

### 1.1 Existing diffusion entry point

`/ssd1/ming/STEGO/16684_Secret_Stego_Dissimilari_Code And Data Supplement/code/models/ODESolve.py`

| What | Where | Note |
|---|---|---|
| SD-v1.5 pipeline load | `ODESolve.__init__`, `model_id = pretrained_models/sd-v1-5` | `torch_dtype=float16`, `safety_checker=None` |
| Second model (picx-real) | same, `ref_key`, only when `single_model=False` | **scheduler forced to `sde-dpmsolver++` — stochastic; FIBER must not inherit this** |
| Text→image generation | `self.model_ref(prompt, h, w, num_inference_steps=…, …)` | plain `StableDiffusionPipeline.__call__` |
| VAE encode | `image2latent()` | `latent_dist.mean * 0.18215`, fp16 |
| VAE decode | `latent2image()` | `1/0.18215` then `vae.decode` |
| EDICT inversion | `edict_noise` / `edict_denoise` | coupled pair, 2 UNet calls per timestep |
| DDIM inversion | `models/ODESolve_ddim.py` | single chain |

### 1.2 Can we specify the initial Gaussian latent? **Yes, cleanly.**

`diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`

- `__call__(..., latents: Optional[torch.FloatTensor] = None, ...)` — line 817
- `prepare_latents(...)` line 639: `if latents is None: randn_tensor(...) else: latents.to(device)`,
  then `latents = latents * self.scheduler.init_noise_sigma`

So `pipe(prompt, latents=z, ...)` is exactly `X = G(Z)`.

**Caveat:** `init_noise_sigma` must be 1.0 or `z` is silently rescaled. PNDM and DDIM
give 1.0; Euler/LMS do not. FIBER pins DDIM and asserts the value at startup.

### 1.3 Two different objects both called "latent" — do not conflate

| | Symbol | Shape | Scaling |
|---|---|---|---|
| **Initial diffusion noise** — what FIBER operates on | `z_T ~ N(0,I)` | `[4,64,64]`, d = 16384 | **none** |
| VAE latent — internal to encode/decode | `ℓ` | `[4,64,64]` | `× 0.18215` on encode |

`0.18215` belongs to the VAE, *not* to `z`. Multiplying `z` by it would be a silent,
hard-to-find bug, so the config keeps them in separate blocks (`latent:` vs `vae:`).

Checkpoint `sd-legacy/stable-diffusion-v1-5`; 512×512; `vae_scale_factor = 8`;
fp16 on the TITAN RTX (sm75, no bf16). d = 16384 = 2¹⁴ — a power of two, so
Walsh–Hadamard needs no dense matrix.

### 1.4 Code we reuse

| Asset | Path | Use |
|---|---|---|
| SD-v1.5 weights | `/ssd2/ming/STEGO/pretrained_models/sd-v1-5` | frozen generator |
| conda env `invcisd` | torch 2.4.1+cu121, diffusers 0.26.3 | same env |
| Batched EDICT inversion | `/ssd1/ming/STEGO/scripts/cache_latents.py` | template for the prompt-assisted reference |
| DDIM inversion | `…/code/models/ODESolve_ddim.py` | same |
| COCO captions | `/ssd2/ming/STEGO/datasets/COCO/annotations/captions_{train,val}2017.json` | ~590 k prompts |
| COCO images 512² | `…/datasets/Unistega_{full,test}` | Phase 10 secret images only |

Not reused: `options/options.py`; `LIM_module`/`INV_block` (the residual design FIBER
rejects); `text_prompt.yaml` (100 prompts — far too few, see R6).

### 1.5 New files (Phase 0–3 only)

```
configs/sd15.yaml, channels.yaml, linear_fiber.yaml
src/fiber/diffusion/generator.py        # frozen G, latents= injection, determinism control
src/fiber/diffusion/cache_dataset.py    # (z, prompt, seed, clean image) shards
src/fiber/channels/{jpeg,resize,noise,blur,registry}.py
src/fiber/transforms/{base,identity,signperm,hadamard,spectral,householder}.py
src/fiber/models/{extractor,teacher}.py # teacher = E[Z|Y] estimator for C_obs
src/fiber/spectrum/observability.py     # randomized SVD of C_obs
src/fiber/metrics/{ber,bootstrap}.py
src/fiber/utils/{seeding,logging,config}.py
scripts/cache_native_dataset.py, fit_observability_spectrum.py,
        train_coordinates.py, eval_coordinates.py
tests/test_orthogonality.py, test_channel_determinism.py, test_split_disjoint.py,
      test_spectrum_synthetic.py
```

---

## 2. Deviations from the master prompt (flagged, not silent)

**(a) Attacks applied on-the-fly, not cached as images.** 10 k × 9 variants × ~400 KB
≈ 36 GB, and it freezes the attack set. We cache only `(z, prompt, seed, clean_image)`
(~4 GB) and apply attacks in the dataloader with a seed from
`blake2s(sample_id|attack|severity)`. Exactly reproducible, real codecs, ~1 ms/image,
and Phase 3 can add a held-out attack without regenerating. `X` is still a cached
constant, so nothing needs to be differentiable.

**(b) DDIM 25 steps.** Deterministic, `init_noise_sigma = 1`, and it makes DDIM
inversion available as a reference (§5.3).

**(c) Guidance: pilot at 3.0, but the paper table sweeps `{1.0, 3.0, 7.5}`.** See R3 —
"high CFG destroys z-information" is a *hypothesis we test*, not a premise we build on.

---

## 3. The central object: the Generative Observability Spectrum

This replaces "Arm D is a spectral baseline" and is, in my view, the strongest
contribution available in Level 1.

### 3.1 Derivation

Take a unit direction `v`, and the scalar coordinate `s = vᵀZ`. Since `Z ~ N(0,I)` and
`‖v‖ = 1`, `Var(s) = 1`. The Bayes-optimal (MMSE) estimate of `s` from the received
image is

```
ŝ(Y) = E[s | Y] = vᵀ E[Z | Y] = vᵀ m(Y)
```

By the law of total variance,

```
E[(s − ŝ)²] = Var(s) − Var(ŝ) = 1 − vᵀ C_obs v ,      C_obs ≜ Cov(E[Z | Y])
```

For `k` orthonormal directions `V` (`VVᵀ = I_k`) the total Bayes error is
`k − Tr(V C_obs Vᵀ)`, so by Rayleigh–Ritz

```
V* = top-k eigenvectors of C_obs                                   (★)
```

### 3.2 Why this is worth more than a learned Q

`E[m(Y)] = E[E[Z|Y]] = E[Z] = 0`, so `C_obs = E[m mᵀ]`; and `Var(vᵀm) ≤ Var(vᵀZ) = 1`
gives `C_obs ⪯ I`. Therefore

```
λ_j = 1 − MMSE_j  ∈ [0, 1]
```

**Each eigenvalue is literally the fraction of that direction's variance that survives
the generative channel.** And

```
Tr(C_obs) = Σ_j λ_j  =  the effective number of recoverable dimensions
```

a bounded, interpretable scalar answering *"how many dimensions of information can this
generative channel actually carry?"* This should be the paper's headline number; BER is
downstream of it.

The spectrum `λ_1 ≥ … ≥ λ_d` also gives the whole picture at once: if it is flat, there
is no anisotropy and FIBER is dead; if it decays fast, FIBER's premise holds and the
decay rate predicts the achievable rate.

### 3.3 Estimation

`E[Z|Y]` is unknown, so train a teacher `M_θ(Y) ≈ E[Z|Y]`, then

```
Ĉ_obs = (1/N) Σ_i m̂_i m̂_iᵀ
```

**The teacher must be trained with MSE.** L2 regression converges to the conditional
mean; L1 converges to the conditional *median*, which is not what (★) requires. This is
a silent-failure bug and is asserted in `tests/test_spectrum_synthetic.py`.

Never form the `16384 × 16384` matrix. With `M = [m̂_1 … m̂_N] ∈ R^{d×N}` and N ≪ d,
`Ĉ = MMᵀ/N` has rank ≤ N, so take the economy SVD of `M` (or randomized SVD / Lanczos)
and read the left singular vectors.

### 3.4 Three honest caveats, to be stated in the report

1. **`Ĉ_obs` is a lower bound**, limited by the teacher's capacity. It is the
   observability spectrum *given the teacher class*, not the true one.
2. **(★) is MMSE-optimal, not sign-BER-optimal.** Our communication metric is sign BER;
   the two coincide only when the posterior is near-Gaussian. This is exactly the gap a
   learned `Q` can exploit, and is the principled reason to expect
   `learned ≥ spectral` in Gate 3B — it turns 3B into an experiment with a prediction
   rather than a coin flip.
3. **`C_obs` depends on the channel.** We fit one spectrum on the training attack
   mixture for the main arm, and report per-attack spectra as an analysis figure (how
   observability geometry changes with channel severity is itself a result).

### 3.5 Synthetic validation before touching the GPU

`tests/test_spectrum_synthetic.py`: build a toy channel `Y = A Z + noise` with a known
`A` of designed spectrum; check the recovered eigenvalues/eigenvectors match the
analytic `C_obs`. This validates the estimator with zero GPU time.

---

## 4. Cross-fit protocol (replaces the broken re-randomisation control)

The Rev-1 control — keep `Q_φ`, read out a different random k-subset with the *same*
extractor — was invalid: the extractor was trained to predict coordinates `1:k`, so
asking it for `100:164` fails regardless of observability. It tests nothing.

Every data-derived arm instead uses:

```
Discovery   : on split A, fit Q      (joint (Q,H) for learned; teacher + SVD for spectral)
Freeze      : Q is frozen
Discard     : throw the discovery extractor away entirely
Re-fit      : fresh extractor H', identical init/capacity/schedule, trained on split B only
Evaluate    : held-out test split
```

**Every arm — including the random ones — trains its evaluation extractor on split B
only**, so the extractor-training budget is identical across arms. The only asymmetry
left is that data-derived arms also saw split A during discovery, which is intrinsic to
being data-derived and is stated as such.

This kills the "the gain is just Q–extractor co-adaptation" objection, so it is a main
experiment, not a supplementary control.

---

## 5. Phase 3 design

### 5.1 Arms

| Arm | Transform | Draws |
|---|---|---|
| A | `Q = I` | 1 (sanity only — see R1) |
| B | random signed permutation | **5–10 seeds** |
| C | structured orthogonal `P₂ H D P₁` (Walsh–Hadamard) | **5–10 seeds** |
| D | **Observability Spectrum**, top-k eigenvectors of `Ĉ_obs` | 3 seeds (teacher init) |
| E | learned `Q_φ`, Householder product | 3 seeds |

Random arms report `E_Q[BER]`, the spread, and the best draw. Claiming
"learned > random" from a single random subspace is indefensible.

### 5.2 Heads — Gaussian, not circular

In Level 1 `W = QZ ~ N(0,I)`; it is **not** a torus variable. A circular head
`(cos 2πW, sin 2πW)` would alias `W` and `W+1` onto the same target, discarding the
integer part. Circular heads belong in Phase 4/5, after `U = Φ(Z)`.

| Head | Target | Loss | Role |
|---|---|---|---|
| regression | `Ŵ` | MSE | diagnostic; also the teacher's objective (§3.3) |
| **sign** | `b_j = 1[W_j > 0]` | BCE | **primary communication metric — sign BER** |

Pearson `ρ(W_j, Ŵ_j)` and per-coordinate MSE are diagnostics only.

### 5.3 DDIM inversion is a reference, not a baseline

Rule D says the extractor sees only `Y`. DDIM inversion needs the conditioning prompt
`c`, so under the MVP protocol (`H(Y)`, prompt **not** given to the receiver) it is not
a fair comparison. It is plotted as a dashed line labelled

> prompt-assisted reference — violates the receiver protocol

Its only job is diagnostic: if prompt-assisted inversion recovers `z` but the CNN
extractor cannot, the extractor is underpowered rather than the channel being empty.
If we later adopt the "prompt is receiver-public" threat model, every extractor becomes
`H(Y, c)` and inversion is promoted to a real baseline. That is a threat-model decision,
recorded explicitly, not an implementation detail.

---

## 6. Risk register

### R1 — Random orthogonal beats identity for a *trivial* reason
A single `z_j` is one value of 16 384, spatially local; a Hadamard row is a **global**
functional, spatially spread, effectively low-frequency, and survives JPEG/resize far
better. The identity→random gap is expected and **is not evidence for FIBER**.
→ Gates compare against **random**, never identity. Reports must not headline identity.

### R2 — (resolved by promotion) spectral vs learned
Rev 1 listed "a spectral method may match learned `Q`" as a risk *and* made
`learned > spectral` a PASS condition — a contradiction. Resolved by §3 and Gate 3A/3B:
a spectral win is a *positive* result for the hypothesis and only changes the framing.

### R3 — Guidance scale is a confound, and lowering it looks like rigging
At CFG 7.5 the prompt dominates and `I(W_R;Y)` may be too small to see; at 1.0 images
are poor. Choosing 3.0 without justification invites "you tuned CFG until the hidden
channel worked".
→ Pilot at 3.0; **the paper's main table sweeps `{1.0, 3.0, 7.5}`** and co-reports
generation quality and prompt alignment (CLIPScore + FID/NIQE) at each. The predicted —
and if true, publishable — result is monotone: `CFG ↑ ⇒ prompt information ↑ ⇒ latent
observability ↓`, i.e. `Tr(C_obs)` decreasing in guidance strength.

### R4 — Q–extractor co-adaptation
→ Handled by the cross-fit protocol (§4), promoted to a main experiment.

### R5 — Generation reproducibility is an audit, **not** a BER floor
Rev 1 claimed fp16 run-to-run variation is the irreducible BER floor. That is wrong:
`X` is generated once and cached, and every extractor reads the same cached `X`, so
generation nondeterminism never enters the extractor's error.
→ Gate 0 still measures `PSNR(G(z)⁽¹⁾, G(z)⁽²⁾)`, labelled **generation reproducibility
audit**. It would only become channel noise in a variant where the receiver must
regenerate the carrier — which is exactly InvCISD's situation, and notably *not*
FIBER's. Worth one sentence in the paper as a structural advantage.

### R6 — Extractor shortcut through prompt memorisation
100 prompts would let the extractor memorise prompt→image structure and infer `z`
indirectly. → COCO captions (~590 k), prompt sets disjoint across splits, and the
held-out-prompt-domain number reported as first-class.

### R7 — Identity may be at chance even on clean images
→ Sweep `k ∈ {16,32,64,128,256}`; use the prompt-assisted reference (§5.3) to separate
"weak channel" from "weak extractor". Note `Tr(Ĉ_obs)` answers this directly and does
not depend on the identity arm at all.

### R8 — The optimal subspace may be content-dependent
Flat sky and dense texture have different JPEG survivability, but a content-adaptive `Q`
breaks the shared-key model. → Limitation; ablate per-prompt-cluster `Q` only if 3A passes.

### R9 — Compute contention
The GPU runs InvCISD stage-2 until ~20:45, then ~40 min of evaluation. Phase 1 caching
(~4.5 h) queues behind it.

---

## 7. Acceptance criteria

### Gate 0 — Environment
| # | Criterion |
|---|---|
| 0.1 | `PSNR(G(z)⁽¹⁾, G(z)⁽²⁾)` measured and reported as a **reproducibility audit** (R5) |
| 0.2 | `assert scheduler.init_noise_sigma == 1.0`; `z` round-trips through `prepare_latents` unmodified |
| 0.3 | `z` persisted and reloadable bit-exactly |
| 0.4 | `[4,64,64]`, d = 16384; VAE `0.18215` documented as **not** applying to `z` |
| 0.5 | Generation is `no_grad`; 100 samples, zero NaN |
| 0.6 | Scheduler deterministic: DDIM, `eta = 0.0`; explicit assert that nothing sets `sde-*` |

### Gate 1 — Cached native channel dataset
Pilot 2 k/256/256 then 10 k/1 k/1 k · seeds, latents **and prompts** disjoint across
splits (asserted) · **split A / split B disjoint within train** (cross-fit, §4) ·
attack registry complete · attacks byte-reproducible · prompt sampling rule recorded.

### Gate 2 — Denominators
Identity + random BER for `k ∈ {16,32,64,128,256}` across all attacks · per-coordinate
`ρ` distribution · prompt-assisted reference as a dashed line · held-out latents **and**
prompts · ≥3 seeds.
Also: **first estimate of `Ĉ_obs` and `Tr(Ĉ_obs)`** — if the spectrum is flat here, we
learn it before spending Phase 3.

### Gate 3A — Existence of observable coordinates *(the scientific gate)*

```
min( BER_spectral , BER_learned )  <  BER_random
```

with, in **≥3 of 5** channel groups, all of:

| | Threshold |
|---|---|
| paired bootstrap on `Δ_i = BER_i^random − BER_i^derived` | `CI₉₅(ΔBER) > 0` |
| relative reduction | ≥ **20 %** |
| absolute reduction | ≥ **0.02** |

Both thresholds are required so that `0.005 → 0.004` cannot be sold as "20 % better".
Paired bootstrap, not CI overlap: every arm is evaluated on the *same*
`(z_i, prompt_i, attack_i)`, so the paired difference is the correct statistic.
Held-out latents **and** held-out prompts. Orthogonality asserted: `‖QᵀQ − I‖_∞ < 1e-5`.

**PASS ⇒ the FIBER hypothesis holds.**
**The only kill condition is `data-derived ≈ random`** — that means measure-preserving
coordinate selection has no signal at all.

### Gate 3B — Learning vs characterisation *(framing, never a kill gate)*

Compare `BER_learned` against `BER_spectral` under the same cross-fit protocol.

| Outcome | Story |
|---|---|
| learned > spectral | "learning robust communication coordinates" — and §3.4(2) explains why |
| learned ≈ spectral | "the **Generative Observability Spectrum** is discoverable in closed form" |
| spectral > learned | believe the spectral result; the optimisation is the weak part |

None of these stops the project.

---

## 8. Execution order

```
§3.5 synthetic spectrum test   (CPU, minutes)   — validate the estimator first
Phase 0  (CPU + short GPU)   ~2 h  — after InvCISD releases the GPU
Phase 1  (GPU)               ~5 h  — pilot 2 k, inspect, then 10 k
Phase 2  (GPU)               ~6 h  — denominators + first Ĉ_obs
Phase 3  (GPU)              ~16 h  — 5 arms, cross-fit, 5–10 random draws
                                      ↓
                             GATE 3A — kill or go
                             GATE 3B — framing
```

Nothing beyond Phase 3 is planned in detail until 3A reports.
