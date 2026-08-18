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


---

# P1 result — clean channel, commit `13aeb9a`

Discovery `train/A_operator` $n=537$, certification `val` $n=256$, parent frame
`C2_haar` seed 0 (digest `120239f1701fc9db`, the Phase A frame), 2000 bootstrap draws,
Bonferroni over 64.

## P1-A — PASS, decisively

**64 of 64** fixed-frame coordinates have a one-sided bound above zero.

| | min | mean | max |
|---|---|---|---|
| $\hat\lambda_j$ | 0.7094 | 0.8517 | 1.0593 |
| $\mathrm{LCB}_j$ | **0.4727** | 0.5956 | 0.7414 |

The lowest bound in the whole frame is $0.47$. On the same channel, the same frame and
the same images, the learned CNN decoder certified $0/64$ and failed its validity gate.

The supported statement is about the **estimator class**, not the population operator:

> The amount of observability that can be certified changes dramatically with the
> receiver and estimator class, even on the same frozen channel and the same fixed latent
> frame.

It is not a measurement of $\Cobs(\mathcal I_R)$ itself. Two different $\Ccert(f)$ are
being compared, and the CNN side additionally failed teacher validity.

## P1-B (coordinate ranking) — FAIL, and in the reverse direction

Top-8 coordinates chosen on discovery, measured on certification:

    D_top = 6.6206   <   D_bottom = 6.7040   <   D_random-8 = 6.8383

The discovery-selected best coordinates are the **worst** of the three on fresh data. The
mechanism is visible and measured:

    corr(lambda_discovery, lambda_certification) = +0.0396
    top-8:     0.9302 (disc) -> 0.8276 (cert)     shrinkage +0.1027
    bottom-8:  0.7270 (disc) -> 0.8380 (cert)     shrinkage -0.1110

Both extremes regress to the global mean $0.8517$. This is a winner's curse: the ranking
carries almost no information across splits, so the discovery top-8 are simply the eight
that fluctuated highest. Without the discovery/certification split, $D_{\rm top}$ would
have been reported as $0.9302\times8\approx7.44$ --- clearly above the random subset --- and
a practical claim would have rested entirely on selection bias.

Supported: **no reproducible coordinate-level anisotropy in the preregistered Haar frame.**

## What P1-B does NOT establish

It reads the **diagonal** of the restricted operator in a fixed frame. A flat diagonal in
a random basis does not imply an isotropic operator: $r_j^\top C r_j=\sum_m\lambda_m
(u_m\!\cdot\! r_j)^2$, and in high dimension $(u_m\!\cdot\! r_j)^2$ concentrates near $1/d$,
so any spectrum is flattened by the choice of basis. FIBER is an eigenspace framework, and
ruling out a coordinate ranking does not rule out a reproducible eigen-direction.

**P2 is not opened.** Its precondition was P1-A *and* P1-B, and P1-B failed as
preregistered.

---

# P1-C — operator-level anisotropy diagnostic

**Preregistered. A different question from P1-B, not a repair of it.** P1-B was defined in
advance on the *coordinates* of the fixed parent frame and failed; that result is final and
is not revisited. P1-C asks what P1-B is structurally unable to answer: whether the
off-diagonal structure of the same restricted operator defines a reproducible
**eigen-direction**.

No new data: it runs on the coordinates saved by the P1 run, so it costs no further
inversion.

    discovery:      C_disc = eig -> U_disc, take the top-8 and bottom-8 eigenvectors
    certification:  measure D_cert(U_top), D_cert(U_bottom), D_cert(U_random) on val
                    WITHOUT re-selecting

**Gate P1-C.** A reproducible eigenspace requires
$D_{\rm cert}(U_{\rm top})>D_{\rm cert}(U_{\rm random})$ on the certification split, with
the difference exceeding the spread over random 8-dimensional subspaces drawn in the same
parent space. The random reference is drawn many times, not once, because a single draw
cannot show that the top subspace beats a *typical* one.

If this fails too, the honest conclusion is that at this scale the inversion receiver
exposes broad recoverability with no reproducible directional structure to exploit --- and
the coordinate-selection use case does not close, for a reason that is now measured at the
operator level rather than assumed from a diagonal.
