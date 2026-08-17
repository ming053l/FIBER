# Phase A — blind vs side information

**Verdict: a clean Level-1 operational null.** 24 runs, commit `9a69cc0`, 0 failures.
Preregistered in `reports/blind_vs_sideinfo.md`; nothing below was decided after seeing a
number.

## Mechanical audit, run before any effect size was read

| # | check | result |
|---|---|---|
| 1 | 24/24 artifacts, every (mode, seed) cell | PASS |
| 2 | single git commit `9a69cc0`, clean tree in every run | PASS |
| 3 | one frame digest `120239f1701fc9db` across all runs, orthonormal | PASS |
| 4 | per receiver seed, the three side arms share one `init_state_sha`; blind differs; the six seeds differ from each other | PASS |
| 5 | donor map fixed across receiver seeds, role-local, 0 fixed points, 0 same-prompt pairs | PASS |
| 6 | one conditioning manifest, one protocol version | PASS |
| 7 | correct vs shuffled evaluated on identical sample ids in identical order | PASS |

## Both placebo gates pass

| step | mean ΔBER | CI90 | verdict |
|---|---|---|---|
| `f(Y)` → `f(Y, S_null)` — architecture and capacity | +0.00297 | [+0.00064, +0.00530] | inside ±0.007 |
| `f(Y, S_null)` → `f(Y, S_shuffled)` — the `S` marginal | −0.00058 | [−0.00361, +0.00245] | inside ±0.007 |

Six receiver seeds gave the equivalence test enough precision to pass: CI90 half-widths
0.0023 and 0.0030 against the 0.007 margin. The nuisance steps are practically negligible,
which is what leaves the primary comparison isolating the pairing.

## The primary contrast does not move

    Delta BER (correct - shuffled) = -0.00216 +- 0.00298
    CI95 = [-0.00529, +0.00096],  p = 0.135,  4/6 seeds favour correct
    Delta R2_lin = -0.00047        (the secondary readout points the other way)

| arm | BER | R2_lin | train loss |
|---|---|---|---|
| blind | 0.4983 | 0.00344 | 0.588 |
| `S_null` | 0.5013 | 0.00366 | 0.552 |
| `S_shuffled` | 0.5007 | 0.00425 | 0.508 |
| `S_correct` | 0.4986 | 0.00378 | 0.508 |

Null for `R2_lin` is `1/(n-1) = 0.00392`; every arm sits on it. Two independent readouts
would both have to lack power for "BER simply missed it" to hold.

## Per-attack breakdown

The aggregation across the ten perturbations was **not** preregistered — the registered
primary says only "paired across receiver seeds" — so it is kept as computed and the
breakdown is reported rather than chosen after the fact.

Per-attack means span [−0.00305, −0.00137], sd 0.00054: no heterogeneity, so the average
hides nothing. Smallest per-attack `p` is 0.020 against a Bonferroni threshold of 0.005;
none survives.

All ten attacks point the same way, and that is **not** independent evidence: the ten
perturbations are ten views of the same 256 images decoded by the same receivers. The
receiver seed remains the unit of independence, giving 4/6 and `p = 0.135`.

## What this establishes, and what it does not

> At the tested observer, data scale and neutral frame, sample-specific conditioning
> provides no detectable operational recovery benefit relative to a capacity- and
> marginal-matched placebo.

It is **not** evidence that `C_obs(Y,S) = C_obs(Y)`, and it does not touch the
monotonicity theorem, which concerns Bayes quantities rather than a trained receiver. The
three levels registered in `reports/blind_vs_sideinfo.md` stay separate: Bayes,
decoder-certified, operational. Only the third was measured.

**No equivalence claim is made for the primary.** The ±0.007 margin was registered for the
placebo chain. The primary interval happens to fall inside it, and promoting a
non-significant difference to an equivalence claim after the fact is precisely what the
preregistration exists to prevent.

## Recorded, not interpreted

Training loss falls across the three side arms (0.552, 0.508, 0.508) while held-out
recovery does not move. Access to varying conditioning embeddings changes the training
fit; correct sample-wise pairing brings no corresponding held-out advantage. No mechanism
is claimed from this.

## Consequence

Phase A was registered as the last large directional decision, with both outcomes written
down in advance. This is the negative branch, so the project converges rather than opening
a sixth hypothesis:

* **Phase B is not opened.** The direct `C_gain(h)` certificate was gated on a strong
  positive Phase A.
* **`Delta D_dec` is not pursued.** It was Phase A-2, gated on the same condition.
* The paper becomes *Certified Observability and the Limits of Latent Recoverability in
  Frozen Generative Channels*, as registered in `reports/scope_v1.md`.
