# Practical Extension P1 — receiver-resolved certified geometry

**Preregistered. Opened after Phase A, not as its continuation.** Phase A's negative
branch stands unchanged: at the tested observer, sample-specific conditioning gave no
detectable operational benefit, and Phase B and Phase A-2 remain closed. This extension is
opened because a diagnostic that already existed in the instrumentation produced a large
measured effect, not because the null was unwelcome.

## The trigger, measured

`scripts/ddim_reference.py` had never been run in this project; the figure quoted in the
design note was carried in from elsewhere. Run on the Phase A frame (`C2_haar` seed 0,
$k=64$), the val split, all ten evaluation attacks at $n=256$ each:

| attack | blind CNN | CNN + $S_{\rm correct}$ | DDIM inversion |
|---|---|---|---|
| clean | 0.4983 | 0.4996 | **0.1287** |
| jpeg90 / jpeg70 / jpeg50 | 0.4982 / 0.4979 / 0.4996 | 0.4988 / 0.4990 / 0.4984 | 0.1520 / 0.1960 / 0.2253 |
| resize075 / resize050 | 0.4982 / 0.4972 | 0.4988 / 0.4984 | 0.1526 / 0.1909 |
| noise005 / noise010 | 0.4998 / 0.4992 | 0.4982 / 0.4972 | 0.2502 / **0.3134** |
| blur10 / blur20 | 0.4970 / 0.4979 | 0.4986 / 0.4987 | 0.1695 / 0.2382 |
| **mean, same rule both sides** | **0.4983** | **0.4986** | **0.2017** |

## What this does and does not show

It shows the channel retains recoverable latent sign information: a receiver exists that
reaches $0.20$ where the learned one sits at chance, on the same frame and the same
images.

It does **not** isolate the information set. The inversion receiver differs from
$f(Y,S_{\rm correct})$ in two ways at once --- it is given the conditioning *and* it runs
the frozen generator backwards --- and Phase A already showed that supplying the
conditioning to a learned CNN changes nothing. The supported statement is therefore about
the **estimator class**:

> The same side-informed recovery problem is nearly impossible for the learned CNN
> receiver and substantially recoverable by a model-based inversion receiver.

It also does **not** show that FIBER can select directions. Recoverability and
*differential* recoverability are separate claims, and only the second makes the framework
a design tool.

## The question P1 asks

> Does the inversion receiver induce a certified geometry that is both valid and
> anisotropic?

The parent frame stays the Phase A one --- $R\in\mathbb R^{64\times d}$, $RR^\top=I$,
`C2_haar` seed 0 --- so no new frame is searched. With $\hat z=f_{\rm inv}(Y,S;G)$,
$c=Rz$ and $\hat c=R\hat z$, we estimate inside that fixed 64-dimensional space
\[
  C_{\rm cert}^{\rm inv}
  = \mathbb E\bigl[c_c\hat c_c^\top+\hat c_c c_c^\top-\hat c_c\hat c_c^\top\bigr],
\]
with direction discovery and certification on disjoint splits, exactly as in
\S3.2. The inversion receiver requires no training, so the teacher/operator split is not
needed to keep fitting noise out --- but the discovery/certification split still is,
because choosing directions and measuring them on the same samples rectifies noise
whatever produced the decoder.

## Design decisions, fixed before running

**Channel condition.** P1-A runs on the **clean** channel first. That is the inversion
receiver's most favourable condition (BER $0.1287$, its best of ten), so a failure to
certify there is decisive and cheap, while success there claims nothing yet about attacked
channels. If P1-A fails on clean it will not pass under attack, and we stop.

**Splits.** Discovery on `train / A_operator` ($n=537$), certification on `val` ($n=256$);
disjoint by construction, neither is test, and the script asserts the sample ids do not
intersect. Because the parent frame is fixed a priori, P1-A selects nothing and therefore
needs no inner cross-fit --- the certificate is a sample mean, bootstrapped directly with
a Bonferroni correction over the $64$ coordinates. P1-B *does* select, so top and bottom
are chosen on discovery and measured on certification.

## Gates, fixed before running

**P1-A --- can it certify?** At least some directions must satisfy
$\underline{\lambda}_j>0$ with the one-sided bootstrap bound. A second $0/64$ would be a
result in itself: *operationally recoverable, yet still not directionally certifiable by
this estimator*, and FIBER would not be a coordinate-selection tool.

**P1-B --- is there anything to select?** A certified but near-isotropic spectrum has no
practical selection value: recoverable is not the same as *there are better coordinates to
pick*. We require separation with uncertainty, $\Dcert(V_{\rm top})>\Dcert(V_{\rm bottom})$
and $\Dcert(V_{\rm top})>\Dcert(V_{\rm Haar})$.

**P2 --- does it transmit better?** Only if P1-A and P1-B both pass. Freeze $V_{\rm top}$
from the discovery split and compare on fresh held-out data,
$\mathrm{BER}_{\rm inv}(V_{\rm top})<\mathrm{BER}_{\rm inv}(V_{\rm Haar})$.

The unit of independence is the **base image**, not the attack. Per image $i$, average the
paired difference over the ten attacks first,
$\Delta_i=\frac1{10}\sum_a[\mathrm{BER}^{\rm top}_{i,a}-\mathrm{BER}^{\rm haar}_{i,a}]$,
then bootstrap over $i$. Ten perturbations of one image are not ten independent
observations --- the same dependence that made an earlier permutation null
anti-conservative by a factor of five.

No sign-reflection watermark is implemented. With $c=Rz$, $c'_j=b_j|c_j|$ and $b_j$
uniform and independent, $c'_j\sim\mathcal N(0,1)$ and $z'\sim\mathcal N(0,I)$, so the
payload's bit error rate *is* the coordinate sign BER already measured. Implementing the
encoder would re-measure a quantity already in hand.

## What is not opened

No new generator, no VAE-only channel, no additional teacher architecture, no error
correction, no new frame search. One intervention: the receiver.
