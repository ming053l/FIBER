# FIBER — locked evaluation, `triage1`, split `test`

Method locked on **val** by `selection_triage1.json` (commit `48e8351`, config `a9c8c73425575c95`):

- locked arm: **D_spectral**, k = 64, seeds [0, 1] (replications, never selected over)
- denominator: **C2_haar** (haar), seeds [0, 1, 2]
- exact locked runs: `D_spectral_k64_s0_rxresnet18_r0_scgate, D_spectral_k64_s1_rxresnet18_r1_scgate`
- exact reference runs: `C2_haar_k64_s0_rxresnet18_r0_scgate, C2_haar_k64_s1_rxresnet18_r1_scgate, C2_haar_k64_s2_rxresnet18_r2_scgate`

This script cannot choose any of the above; it reads them.

## Per-arm sign BER (descriptive)

| arm | seed | clean | jpeg90 | jpeg70 | jpeg50 | resize075 | resize050 | noise005 | noise010 | blur10 | blur20 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C2_haar | 0 | 0.5015 | nan | nan | 0.5045 | nan | 0.5013 | nan | 0.5009 | nan | 0.5001 | 0.5017 |
| C2_haar | 1 | 0.4978 | nan | nan | 0.4974 | nan | 0.4995 | nan | 0.4998 | nan | 0.5020 | 0.4993 |
| C2_haar | 2 | 0.4997 | nan | nan | 0.5005 | nan | 0.5010 | nan | 0.4965 | nan | 0.4991 | 0.4994 |
| C3_frozen_hh | 0 | 0.4957 | nan | nan | 0.4979 | nan | 0.4965 | nan | 0.4969 | nan | 0.4982 | 0.4970 |
| C_hadamard | 0 | 0.4949 | nan | nan | 0.4968 | nan | 0.4979 | nan | 0.4922 | nan | 0.4929 | 0.4949 |
| D_spectral **[locked]** | 0 | 0.5061 | nan | nan | 0.5044 | nan | 0.5056 | nan | 0.5018 | nan | 0.5026 | 0.5041 |
| D_spectral **[locked]** | 1 | 0.4970 | nan | nan | 0.4966 | nan | 0.5005 | nan | 0.4966 | nan | 0.5032 | 0.4988 |
| E_learned | 0 | 0.5004 | nan | nan | 0.5001 | nan | 0.5028 | nan | 0.5007 | nan | 0.5042 | 0.5016 |
| E_learned | 1 | 0.4963 | nan | nan | 0.5003 | nan | 0.4964 | nan | 0.4991 | nan | 0.4984 | 0.4981 |

## Gate 3A — locked method vs Haar (k = 64)

**Verdict: PROVISIONAL_FAIL** (0/3 channel groups, provisional: a pilot may not close the question)

| group | Haar E_Q[BER] | spread | locked BER | ΔBER | CI95 (paired) | CI95 (hierarchical) | rel | pass |
|---|---|---|---|---|---|---|---|---|
| blur | 0.5004 | ±0.0012 | 0.5029 | -0.0025 | [-0.0096, +0.0044] | [-0.0117, +0.0062] | -0.5% | no |
| identity | 0.4997 | ±0.0015 | 0.5016 | -0.0019 | [-0.0087, +0.0049] | [-0.0129, +0.0095] | -0.4% | no |
| jpeg | 0.5008 | ±0.0029 | 0.5005 | +0.0003 | [-0.0065, +0.0070] | [-0.0109, +0.0115] | 0.1% | no |
| noise | 0.4991 | ±0.0019 | 0.4992 | -0.0001 | [-0.0072, +0.0070] | [-0.0102, +0.0096] | -0.0% | no |
| resize | 0.5006 | ±0.0008 | 0.5031 | -0.0025 | [-0.0097, +0.0047] | [-0.0121, +0.0074] | -0.5% | no |

The paired interval is conditional on the Haar draws actually made; the hierarchical one resamples draws and seeds too, which is the interval that matches a claim about `E_{Q~Haar}[BER]`. Gate thresholds use the paired interval and the hierarchical one is reported beside it.


> **A control beats the locked arm.** If a structured random family is stronger than the data-derived one, the Haar comparison alone overstates the result.

>   - blur: C_hadamard 0.4929
>   - identity: C_hadamard 0.4949
>   - jpeg: C_hadamard 0.4968
>   - noise: C_hadamard 0.4922
>   - resize: C_hadamard 0.4979


## Gate 3B — learning vs characterisation (framing only)

- **blur**: certified-spectral 0.5029 vs learned 0.5013 (Δ +0.0016, CI95 [-0.0063, +0.0099])
- **identity**: certified-spectral 0.5016 vs learned 0.4983 (Δ +0.0032, CI95 [-0.0049, +0.0116])
- **jpeg**: certified-spectral 0.5005 vs learned 0.5002 (Δ +0.0003, CI95 [-0.0078, +0.0088])
- **noise**: certified-spectral 0.4992 vs learned 0.4999 (Δ -0.0008, CI95 [-0.0098, +0.0078])
- **resize**: certified-spectral 0.5031 vs learned 0.4996 (Δ +0.0034, CI95 [-0.0041, +0.0111])

Story: _the observability geometry is discoverable in closed form_. Never a kill gate. Both families are compared as seed averages — best-seed selection is not available here.

