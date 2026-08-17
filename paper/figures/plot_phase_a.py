#!/usr/bin/env python
"""Phase A in one figure: the controls are clean, the primary does not move, and every
arm stays at chance.

(A) the preregistered primary contrast, paired across receiver seeds
(B) the two preregistered placebo equivalence gates against their +-0.007 band
(C) all four arms against the empirical null for R2_lin

The equivalence band is drawn only on (B). It was registered for the placebo chain, and
applying the same margin to the primary after the fact would upgrade a non-significant
difference into an equivalence claim that was never preregistered.
"""
from __future__ import annotations

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RES = "/ssd2/ming/FIBER/results/sideinfo1"
MODES = ("blind", "null", "shuffled", "correct")
EPS, NSEED, NVAL = 0.007, 6, 256
ACCENT, GREY = "#c43a5a", "#8a8a8a"


def load():
    R = {}
    for f in sorted(glob.glob(f"{RES}/*_scsideinfo.json")):
        d = json.load(open(f))
        r = d["results"]["val"]
        npz = np.load(f.replace(".json", ".npz"), allow_pickle=True)
        rho = np.concatenate([np.asarray(npz[k], dtype=float) for k in npz.keys()
                              if k.startswith("val|") and k.endswith("|pearson")])
        R[(d["side_mode"], d["receiver_seed"])] = {
            "ber": float(np.mean([x["sign_ber"] for x in r.values()])),
            "r2": float((rho ** 2).mean())}
    return R


def ci(d, level=0.95):
    return stats.t.interval(level, len(d) - 1, loc=d.mean(),
                            scale=d.std(ddof=1) / np.sqrt(len(d)))


def main(out="figures/phase_a.pdf"):
    R = load()
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 2.8))

    # ---- (A) primary contrast -------------------------------------------------------
    d = np.array([R[("correct", s)]["ber"] - R[("shuffled", s)]["ber"]
                  for s in range(NSEED)])
    lo, hi = ci(d)
    ax[0].axhline(0, color="0.75", lw=0.9)
    ax[0].scatter(range(NSEED), d, s=26, color=GREY, zorder=3, label="receiver seed")
    ax[0].errorbar([NSEED + 0.7], [d.mean()], yerr=[[d.mean() - lo], [hi - d.mean()]],
                   fmt="o", color=ACCENT, ms=6, capsize=4, lw=1.6, zorder=4)
    ax[0].text(NSEED + 1.15, d.mean(), "mean\n95% CI", fontsize=7.5, color=ACCENT,
               va="center")
    ax[0].set_xticks(list(range(NSEED)) + [NSEED + 0.7])
    ax[0].set_xticklabels([str(s) for s in range(NSEED)] + [""], fontsize=7.5)
    ax[0].set_xlabel("receiver seed", fontsize=8.5)
    ax[0].set_ylabel(r"$\Delta$BER (correct $-$ shuffled)", fontsize=8.5)
    ax[0].set_title(rf"(A) primary contrast: $p={stats.ttest_1samp(d,0).pvalue:.2f}$",
                    fontsize=9)
    ax[0].set_xlim(-0.7, NSEED + 2.4)

    # ---- (B) the two placebo gates --------------------------------------------------
    pairs = [("blind", "null", "architecture"), ("null", "shuffled", r"$S$ marginal")]
    ax[1].axvspan(-EPS, EPS, color=ACCENT, alpha=0.10, lw=0)
    ax[1].axvline(0, color="0.75", lw=0.9)
    for i, (a, b, lab) in enumerate(pairs):
        dd = np.array([R[(b, s)]["ber"] - R[(a, s)]["ber"] for s in range(NSEED)])
        l, h = ci(dd, 0.90)
        y = 1 - i
        ax[1].plot([l, h], [y, y], color=GREY, lw=2.2, solid_capstyle="round")
        ax[1].scatter([dd.mean()], [y], s=30, color=GREY, zorder=3)
        ax[1].text(-EPS * 0.96, y + 0.22, lab, fontsize=8, color="0.35")
    ax[1].set_yticks([]); ax[1].set_ylim(-0.6, 1.7)
    ax[1].set_xlim(-EPS * 1.45, EPS * 1.45)
    ax[1].set_xticks([-EPS, -EPS/2, 0, EPS/2, EPS])
    ax[1].set_xticklabels([f"{v:+.3f}" for v in (-EPS, -EPS/2, 0, EPS/2, EPS)],
                          fontsize=7.5)
    ax[1].set_xlabel(r"$\Delta$BER with 90% CI", fontsize=8.5)
    ax[1].set_title(rf"(B) placebo gates inside $\pm{EPS}$", fontsize=9)

    # ---- (C) all four arms against the null -----------------------------------------
    null = 1.0 / (NVAL - 1)
    ax[2].axhline(null, color=ACCENT, lw=1.1, ls="--")
    ax[2].text(0.02, null, r"null $1/(n-1)$", fontsize=7.5, color=ACCENT,
               va="bottom", ha="left", transform=ax[2].get_yaxis_transform())
    for i, m in enumerate(MODES):
        v = np.array([R[(m, s)]["r2"] for s in range(NSEED)])
        ax[2].scatter(np.full(NSEED, i) + np.linspace(-0.11, 0.11, NSEED), v, s=15,
                      color="0.72", edgecolor="none")
        ax[2].plot([i - 0.22, i + 0.22], [v.mean()] * 2, color=GREY, lw=2)
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels(["blind", r"$S_{\rm null}$", r"$S_{\rm shuf}$",
                           r"$S_{\rm corr}$"], fontsize=8)
    ax[2].set_ylabel(r"$R^2_{\rm lin}$", fontsize=8.5)
    ax[2].set_title("(C) every arm at the noise value", fontsize=9)
    ax[2].set_xlim(-0.5, 3.5)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  primary dBER {d.mean():+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]  "
          f"p={stats.ttest_1samp(d,0).pvalue:.3f}")
    for a, b, lab in pairs:
        dd = np.array([R[(b, s)]["ber"] - R[(a, s)]["ber"] for s in range(NSEED)])
        l, h = ci(dd, 0.90)
        print(f"  {a}->{b}: {dd.mean():+.5f} CI90 [{l:+.5f}, {h:+.5f}] "
              f"{'inside' if l > -EPS and h < EPS else 'OUTSIDE'} the band")
    for m in MODES:
        v = np.array([R[(m, s)]["r2"] for s in range(NSEED)])
        print(f"  {m:9s} R2 {v.mean():.5f} (null {null:.5f})")


if __name__ == "__main__":
    main()
