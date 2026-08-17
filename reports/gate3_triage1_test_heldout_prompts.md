# FIBER — locked evaluation, `triage1`, split `test_heldout_prompts`

Method locked on **val** by `selection_triage1.json` (commit `48e8351`, config `a9c8c73425575c95`):

- locked arm: **D_spectral**, k = 64, seeds [0, 1] (replications, never selected over)
- denominator: **C2_haar** (haar), seeds [0, 1, 2]
- exact locked runs: `D_spectral_k64_s0_rxresnet18_r0_scgate, D_spectral_k64_s1_rxresnet18_r1_scgate`
- exact reference runs: `C2_haar_k64_s0_rxresnet18_r0_scgate, C2_haar_k64_s1_rxresnet18_r1_scgate, C2_haar_k64_s2_rxresnet18_r2_scgate`

This script cannot choose any of the above; it reads them.

## Per-arm sign BER (descriptive)

| arm | seed | clean | jpeg90 | jpeg70 | jpeg50 | resize075 | resize050 | noise005 | noise010 | blur10 | blur20 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2_haar | 0 | 0.4979 | nan | nan | 0.4975 | nan | 0.4979 | nan | 0.4952 | nan | 0.4980 | 0.4973 |
| C2_haar | 1 | 0.4982 | nan | nan | 0.4965 | nan | 0.5024 | nan | 0.5022 | nan | 0.5045 | 0.5008 |
| C2_haar | 2 | 0.4991 | nan | nan | 0.5006 | nan | 0.4996 | nan | 0.4998 | nan | 0.4956 | 0.4989 |
| C3_frozen_hh | 0 | 0.4963 | nan | nan | 0.4970 | nan | 0.4960 | nan | 0.4980 | nan | 0.4957 | 0.4966 |
| C_hadamard | 0 | 0.4958 | nan | nan | 0.4963 | nan | 0.4948 | nan | 0.4919 | nan | 0.4930 | 0.4944 |
| D_spectral **[locked]** | 0 | 0.4996 | nan | nan | 0.4988 | nan | 0.4982 | nan | 0.4976 | nan | 0.4956 | 0.4980 |
| D_spectral **[locked]** | 1 | 0.4968 | nan | nan | 0.4963 | nan | 0.4903 | nan | 0.4981 | nan | 0.4907 | 0.4944 |
| E_learned | 0 | 0.5003 | nan | nan | 0.4976 | nan | 0.5007 | nan | 0.4980 | nan | 0.5022 | 0.4998 |
| E_learned | 1 | 0.5020 | nan | nan | 0.5042 | nan | 0.5057 | nan | 0.5045 | nan | 0.5051 | 0.5043 |

## Gate 3A — locked method vs Haar (k = 64)

**Verdict: INCONCLUSIVE** (0/3 channel groups, provisional: a pilot may not close the question)

| group | Haar E_Q[BER] | spread | locked BER | ΔBER | CI95 (paired) | CI95 (hierarchical) | rel | pass |
|---|---|---|---|---|---|---|---|---|
| blur | 0.4994 | ±0.0038 | 0.4931 | +0.0062 | [-0.0007, +0.0132] | [-0.0041, +0.0163] | 1.2% | no |
| identity | 0.4984 | ±0.0005 | 0.4982 | +0.0002 | [-0.0066, +0.0068] | [-0.0084, +0.0088] | 0.0% | no |
| jpeg | 0.4982 | ±0.0017 | 0.4976 | +0.0007 | [-0.0060, +0.0073] | [-0.0079, +0.0091] | 0.1% | no |
| noise | 0.4990 | ±0.0029 | 0.4979 | +0.0012 | [-0.0049, +0.0073] | [-0.0079, +0.0097] | 0.2% | no |
| resize | 0.5000 | ±0.0019 | 0.4942 | +0.0057 | [-0.0012, +0.0126] | [-0.0044, +0.0163] | 1.1% | no |

The paired interval is conditional on the Haar draws actually made; the hierarchical one resamples draws and seeds too, which is the interval that matches a claim about `E_{Q~Haar}[BER]`. Gate thresholds use the paired interval and the hierarchical one is reported beside it.


> **A control beats the locked arm.** If a structured random family is stronger than the data-derived one, the Haar comparison alone overstates the result.

>   - blur: C_hadamard 0.4930
>   - identity: C_hadamard 0.4958
>   - jpeg: C_hadamard 0.4963
>   - noise: C_hadamard 0.4919


## Gate 3B — learning vs characterisation (framing only)

- **blur**: certified-spectral 0.4931 vs learned 0.5037 (Δ -0.0105, CI95 [-0.0179, -0.0029])
- **identity**: certified-spectral 0.4982 vs learned 0.5011 (Δ -0.0029, CI95 [-0.0102, +0.0047])
- **jpeg**: certified-spectral 0.4976 vs learned 0.5009 (Δ -0.0033, CI95 [-0.0105, +0.0045])
- **noise**: certified-spectral 0.4979 vs learned 0.5013 (Δ -0.0034, CI95 [-0.0101, +0.0038])
- **resize**: certified-spectral 0.4942 vs learned 0.5032 (Δ -0.0090, CI95 [-0.0163, -0.0012])

Story: _trust the certified spectral result; the optimisation is the weak part_. Never a kill gate. Both families are compared as seed averages — best-seed selection is not available here.

