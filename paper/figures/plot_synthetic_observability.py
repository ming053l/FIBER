#!/usr/bin/env python
"""Framework validation on a channel whose observability is known in closed form.

    Z ~ N(0, I_d),   Y = A Z + eps,   eps ~ N(0, Sigma),   A, Sigma diagonal

Then J = A' Sigma^-1 A and C_obs = J (I+J)^-1 are diagonal too, so direction j has

    lambda_j = a_j^2 / (a_j^2 + sigma_j^2)          TRUE observability
    s_j      = |a_j|                                forward sensitivity

and the two can be set independently. We choose them to disagree: a quiet, clean
direction (a=1, sigma=0.2) is highly observable, while a loud, noisy one (a=5, sigma=20)
moves the observation five times as far and is almost unrecoverable. We call this the
SENSITIVITY-OBSERVABILITY MISMATCH.

Four panels, three different questions:
  (a) does the estimator recover a known geometry?   analytic vs C_cert and its LCB
  (b) does the geometry predict recovery?            alpha-sweep with a FRESH decoder
  (c) why can it disagree with sensitivity?          matched displacement, different overlap
  (d) how often does it disagree?                    s_j against lambda_j

The certificate is computed with the shipped estimator (fiber.spectrum.certified), not a
reimplementation, so this figure also exercises the code the paper describes.

NOTE ON NAMING: directions are labelled by TRUE observability, which is available here
because the channel is synthetic. They are never called "high-certified" -- that is an
empirical claim requiring a lower bound above zero on real data.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/ssd1/ming/FIBER/src")
from fiber.spectrum.certified import (_per_sample_contributions,  # noqa: E402
                                      project_operator)

D = 16
ACCENT, ACCENT2, GREY = "#c43a5a", "#2b6cb0", "#8a8a8a"


def channel():
    """a_j and sigma_j chosen so forward sensitivity runs opposite to observability."""
    a = np.geomspace(1.0, 5.0, D)                 # forward gain: increasing
    sigma = np.geomspace(0.2, 20.0, D)            # noise: increasing faster
    lam = a ** 2 / (a ** 2 + sigma ** 2)          # true observability, closed form
    return a, sigma, lam


def sample(a, sigma, n, rng):
    Z = rng.standard_normal((n, D))
    Y = Z * a + rng.standard_normal((n, D)) * sigma
    return Z, Y


def fit_decoder(Z, Y):
    """A plain least-squares decoder; deliberately not the analytic Bayes rule, so the
    certificate is estimated from something that had to be learned."""
    return np.linalg.lstsq(Y, Z, rcond=None)[0]          # [D, D]


def main(out="figures/synthetic_observability.pdf"):
    rng = np.random.default_rng(0)
    a, sigma, lam = channel()
    hi, lo = int(np.argmax(lam)), int(np.argmin(lam))

    # ---- estimate C_cert with the shipped estimator, on a split disjoint from fitting
    Zf, Yf = sample(a, sigma, 4000, rng)
    W = fit_decoder(Zf, Yf)
    Zo, Yo = sample(a, sigma, 4000, rng)
    F = Yo @ W
    V = np.eye(D)
    C_hat = project_operator(Zo, F, V, center=True)
    # Per-coordinate lower bounds in a FIXED, a-priori basis. No within-subspace rotation
    # is chosen here, so no inner cross-fit is needed and none is used: the certificate is
    # a sample mean of the per-sample terms and is bootstrapped directly. (The rotated
    # per-fold bounds that subspace_certificate returns index each fold's OWN frame and
    # could not be plotted against a coordinate axis.)
    terms = _per_sample_contributions(Zo, F, V, center=True)
    bidx = rng.integers(0, terms.shape[0], size=(600, terms.shape[0]))
    boot = terms[bidx].mean(axis=1)
    lcb = np.quantile(boot, 0.05 / D, axis=0)      # Bonferroni over the D coordinates

    fig, ax = plt.subplots(1, 4, figsize=(13.6, 2.75))

    # ---- (a) known geometry vs certificate ----------------------------------------
    j = np.arange(D)
    ax[0].plot(j, lam, "o-", color="0.25", ms=3.5, lw=1.3, label=r"analytic $\lambda_j$")
    ax[0].plot(j, np.diag(C_hat), "s--", color=ACCENT, ms=3.2, lw=1.2,
               label=r"$C_{\rm cert}$ estimate")
    ax[0].fill_between(j, lcb, np.diag(C_hat), color=ACCENT, alpha=0.15, lw=0)
    ax[0].axhline(0, color="0.8", lw=0.8)
    ax[0].set_xlabel("direction $j$"); ax[0].set_ylabel(r"observability")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("(a) the certificate recovers\nthe known geometry", fontsize=8.5)

    # ---- (b) alpha-sweep with the fresh decoder ------------------------------------
    alphas = np.linspace(-2, 2, 9)
    for name, idx, col in (("high", hi, ACCENT), ("low", lo, GREY)):
        mu, sd = [], []
        for al in alphas:
            z = rng.standard_normal((600, D)); z[:, idx] += al
            y = z * a + rng.standard_normal((600, D)) * sigma
            est = (y @ W)[:, idx] - (z[:, idx] - al)      # isolate the injected step
            mu.append(est.mean()); sd.append(est.std())
        mu, sd = np.asarray(mu), np.asarray(sd)
        ax[1].plot(alphas, mu, "o-", color=col, ms=3.2, lw=1.4,
                   label=rf"$\lambda={lam[idx]:.2f}$")
        ax[1].fill_between(alphas, mu - sd, mu + sd, color=col, alpha=0.16, lw=0)
    ax[1].plot(alphas, alphas, ":", color="0.75", lw=1.0, zorder=0)
    ax[1].set_xlabel(r"injected step $\alpha$"); ax[1].set_ylabel(r"recovered $\hat\alpha$")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    ax[1].set_title("(b) a fresh decoder recovers\nthe observable direction", fontsize=8.5)

    # ---- (c) matched displacement, different ambiguity -----------------------------
    # Plotted in units of that direction's own noise: the displacement is the SAME in
    # observation space, but discriminability is delta/sigma, which differs by 100x. A
    # shared absolute axis would compress the clean row into a line and hide the point.
    delta = 2.0
    gs = ax[2].get_subplotspec().subgridspec(2, 1, hspace=0.55)
    fig.delaxes(ax[2])
    for row, (idx, col, lbl) in enumerate(((hi, ACCENT, "high"), (lo, GREY, "low"))):
        sub = fig.add_subplot(gs[row])
        sg = sigma[idx]
        xs = np.linspace(-4 * sg + min(0, delta), 4 * sg + max(0, delta), 400)
        gauss = lambda m: np.exp(-((xs - m) ** 2) / (2 * sg ** 2))
        sub.fill_between(xs, gauss(0), color="0.78", alpha=0.75, lw=0)
        sub.fill_between(xs, gauss(delta), color=col, alpha=0.45, lw=0)
        sub.annotate("", xy=(delta, 1.16), xytext=(0, 1.16),
                     arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.8))
        sub.text(delta / 2, 1.24, rf"$\Delta/\sigma={delta/sg:.2f}$", fontsize=7,
                 ha="center", color="0.35")
        sub.text(0.02, 0.78, rf"$\lambda={lam[idx]:.2f}$", transform=sub.transAxes,
                 fontsize=7.5, color=col)
        sub.set_ylim(0, 1.55); sub.set_yticks([])
        sub.tick_params(labelsize=7)
        sub.spines[["top", "right", "left"]].set_visible(False)
        if row == 0:
            sub.set_title("(c) same displacement,\ndifferent overlap", fontsize=8.5)
        else:
            sub.set_xlabel(r"observation $Y_j$", fontsize=8)

    # ---- (d) sensitivity-observability mismatch ------------------------------------
    ax[3].scatter(a, lam, s=16, color="0.72", edgecolor="none")
    for idx, col, lbl in ((hi, ACCENT, "high"), (lo, GREY, "low")):
        ax[3].scatter([a[idx]], [lam[idx]], s=46, color=col, zorder=3)
        ax[3].annotate(rf"$v_{{\rm {lbl}}}$", (a[idx], lam[idx]),
                       textcoords="offset points", xytext=(6, 2), fontsize=8, color=col)
    r = np.corrcoef(a, lam)[0, 1]
    ax[3].set_xlabel(r"forward sensitivity $s_j=|a_j|$")
    ax[3].set_ylabel(r"true observability $\lambda_j$")
    ax[3].set_title(rf"(d) mismatch ($r={r:+.2f}$)", fontsize=8.5)

    for x in (ax[0], ax[1], ax[3]):
        x.spines[["top", "right"]].set_visible(False)
        x.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")

    print(f"wrote {out}")
    print(f"  true lambda: high {lam[hi]:.4f} (a={a[hi]:.2f}, s={sigma[hi]:.2f})  "
          f"low {lam[lo]:.4f} (a={a[lo]:.2f}, s={sigma[lo]:.2f})")
    print(f"  sensitivity: high {a[hi]:.2f}  low {a[lo]:.2f}   -> low moves Y "
          f"{a[lo]/a[hi]:.1f}x more")
    # C_cert <= C_obs is a POPULATION statement; the finite-sample point estimate
    # fluctuates around it, so what must hold is that the BOUND covers the truth.
    print(f"  lower bound below the truth in {int((lcb <= lam).sum())}/{D} directions "
          f"(Bonferroni 5% over {D}); worst overshoot {float((lcb - lam).max()):+.5f}")
    print(f"  point estimate: mean |C_cert - lambda| = "
          f"{float(np.abs(np.diag(C_hat) - lam).mean()):.4f}")
    print(f"  corr(sensitivity, observability) = {r:+.3f}")


if __name__ == "__main__":
    main()
