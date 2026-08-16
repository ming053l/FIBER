#!/usr/bin/env python
"""Aggregate arm results -> denominators table (Phase 2) and Gates 3A/3B.

Comparisons are against RANDOM, never identity (PLAN.md R1): a single latent
coordinate is spatially local while a Hadamard row is a global functional, so
identity loses to any global transform for a trivial reason that is not evidence
for FIBER. Identity is printed for context and excluded from every gate.

The random reference is E_Q[BER] over ALL random draws (arms B and C, every
seed), with the spread and the best single draw reported alongside: claiming
"learned > random" from one random subspace is indefensible.

    python scripts/eval_coordinates.py --tag pilot --k 64
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fiber.channels import ChannelBank
from fiber.metrics import gate3a_condition, gate3a_verdict, paired_bootstrap
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger

log = get_logger("eval")

RANDOM_TYPES = {"signed_permutation", "hadamard"}
DERIVED_TYPES = {"spectral_topk", "householder"}
SANITY_TYPES = {"identity"}
REFERENCE_TYPES = {"ddim_inversion_reference"}


def load_runs(out_dir: Path, k: int | None) -> list[dict]:
    runs = []
    for jf in sorted(out_dir.glob("*.json")):
        meta = json.loads(jf.read_text())
        if k is not None and meta["k"] != k:
            continue
        npz = jf.with_suffix(".npz")
        if not npz.exists():
            log.warning("%s has no arrays, skipping", jf.name)
            continue
        meta["_npz"] = npz
        runs.append(meta)
    return runs


def per_sample(run: dict, split: str, attack: str):
    with np.load(run["_npz"], allow_pickle=False) as z:
        key = f"{split}|{attack}|per_sample"
        if key not in z:
            return None, None
        return z[key], z[f"{split}|{attack}|sample_ids"]


def group_matrix(runs: list[dict], split: str, attacks: list[str], bank: ChannelBank):
    """{run_stem: {group: per_sample_ber}} averaged over the attacks in a group,
    aligned across runs by sample_id."""
    groups = defaultdict(list)
    for a in attacks:
        groups[bank.group_of(a)].append(a)
    out, ref_ids = {}, None
    for run in runs:
        stem = f"{run['arm']}_s{run['seed']}"
        per_group = {}
        for g, atts in groups.items():
            cols = []
            for a in atts:
                v, ids = per_sample(run, split, a)
                if v is None:
                    continue
                if ref_ids is None:
                    ref_ids = ids
                elif not np.array_equal(ids, ref_ids):
                    raise RuntimeError(f"sample alignment broken in {run['_npz'].name}/{a}: "
                                       "the paired bootstrap requires identical ordering")
                cols.append(v)
            if cols:
                per_group[g] = np.mean(cols, axis=0)
        out[stem] = {"per_group": per_group, "run": run}
    return out, sorted(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bank = ChannelBank(cfg)
    out_dir = Path(cfg["paths"]["data_root"]) / "results" / args.tag
    runs = load_runs(out_dir, args.k)
    if not runs:
        raise SystemExit(f"no runs in {out_dir} (k={args.k})")
    ks = sorted({r["k"] for r in runs})
    log.info("%d runs, k in %s", len(runs), ks)

    report: dict = {"tag": args.tag, "split": args.split, "k": args.k, "ks_present": ks}
    lines: list[str] = []
    lines.append(f"# FIBER results — `{args.tag}`, split `{args.split}`\n")

    # ---------------- Phase 2: denominators, per k and per attack --------
    lines.append("## Denominators (Phase 2)\n")
    lines.append("Mean sign BER per attack. Chance is 0.5 by construction "
                 "(`W ~ N(0,I)` so the sign bits are uniform).\n")
    for k in ks:
        rk = [r for r in runs if r["k"] == k]
        attacks = bank.eval
        lines.append(f"\n### k = {k}\n")
        header = "| arm | seed | " + " | ".join(attacks) + " | mean |"
        lines.append(header)
        lines.append("|" + "---|" * (len(attacks) + 3))
        has_ref = any(r["type"] in REFERENCE_TYPES for r in rk)
        for r in sorted(rk, key=lambda r: (r["arm"], r["seed"])):
            res = r["results"].get(args.split, {})
            vals = [res.get(a, {}).get("sign_ber", float("nan")) for a in attacks]
            mark = " †" if r["type"] in REFERENCE_TYPES else ""
            lines.append(f"| {r['arm']}{mark} | {r['seed']} | " +
                         " | ".join(f"{v:.4f}" for v in vals) +
                         f" | {np.nanmean(vals):.4f} |")
        if has_ref:
            lines.append("\n† prompt-assisted DDIM inversion — **violates the receiver "
                         "protocol** (it uses the prompt). Diagnostic only: it separates "
                         "'weak channel' from 'weak extractor'. Never a baseline, never in a gate.")

    # ---------------- Gates 3A / 3B -------------------------------------
    k_gate = args.k or (ks[-1] if len(ks) == 1 else int(cfg["fiber"]["robust_dims"]))
    gate_runs = [r for r in runs if r["k"] == k_gate]
    mats, groups = group_matrix(gate_runs, args.split, bank.eval, bank)

    def stems_of(types):
        return [s for s, m in mats.items() if m["run"]["type"] in types]

    # Reference runs are excluded from every gate by construction: they are not in
    # RANDOM_TYPES or DERIVED_TYPES, and this assert keeps it that way.
    assert not (RANDOM_TYPES | DERIVED_TYPES) & REFERENCE_TYPES

    random_stems, derived_stems = stems_of(RANDOM_TYPES), stems_of(DERIVED_TYPES)
    lines.append(f"\n## Gate 3A — existence of observable coordinates (k = {k_gate})\n")

    if not random_stems or not derived_stems:
        lines.append("_Not enough arms yet: Gate 3A needs at least one random draw "
                     "and one data-derived arm._\n")
        report["gate3a"] = {"verdict": "NOT_RUN", "random_arms": random_stems,
                            "derived_arms": derived_stems}
    else:
        n_derived_arms = len({mats[s]["run"]["arm"] for s in derived_stems})
        alpha = 0.05 / max(n_derived_arms, 1)     # the gate takes min over arms
        by_family = defaultdict(list)
        for s in random_stems:
            by_family[mats[s]["run"]["arm"]].append(s)

        conditions, per_arm = {}, defaultdict(dict)
        for g in groups:
            rand_stack = np.stack([mats[s]["per_group"][g] for s in random_stems])
            # The reference is the STRONGEST random family, not the pool average.
            # Arms B and C are different families -- a signed permutation is local,
            # a Hadamard row is global -- and R1 says the local one loses for a
            # trivial reason. Averaging it into the reference would hand the derived
            # arms a free win. Per family we average over its seeds (E_Q[BER]); the
            # gate then runs against whichever family is hardest to beat.
            family_means = {a: np.mean([mats[s]["per_group"][g] for s in ss], axis=0)
                            for a, ss in by_family.items()}
            ref_family = min(family_means, key=lambda a: family_means[a].mean())
            reference = family_means[ref_family]
            best_draw = float(rand_stack.mean(axis=1).min())
            for s in derived_stems:
                res = paired_bootstrap(reference, mats[s]["per_group"][g],
                                       resamples=int(cfg["eval"]["bootstrap_resamples"]),
                                       seed=0, alpha=alpha)
                per_arm[g][s] = gate3a_condition(res, cfg["gate3a"])
            best = min(per_arm[g], key=lambda s: mats[s]["per_group"][g].mean())
            conditions[g] = {**per_arm[g][best], "selected_arm": best,
                             "reference_family": ref_family,
                             "random_pooled_mean": float(rand_stack.mean()),
                             "random_mean": float(reference.mean()),
                             "random_spread": float(rand_stack.mean(axis=1).std()),
                             "random_best_draw": best_draw,
                             "n_random_draws": len(random_stems)}
        verdict = gate3a_verdict(conditions, cfg["gate3a"])
        report["gate3a"] = {**verdict, "alpha_per_arm": alpha, "conditions": conditions}

        lines.append(f"**Verdict: {verdict['verdict']}** "
                     f"({verdict['n_passed']}/{verdict['n_required']} channel groups, "
                     f"Bonferroni alpha = {alpha:.3f} over {n_derived_arms} derived arms)\n")
        lines.append("| group | reference family | E_Q[BER] | spread | best draw | derived | BER | ΔBER | CI95 | rel | pass |")
        lines.append("|" + "---|" * 11)
        for g, c in conditions.items():
            lines.append(
                f"| {g} | {c['reference_family']} | {c['random_mean']:.4f} | ±{c['random_spread']:.4f} | "
                f"{c['random_best_draw']:.4f} | {c['selected_arm']} | {c['treatment']:.4f} | "
                f"{c['mean_delta']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                f"{c['relative_reduction']*100:.1f}% | {'yes' if c['passed'] else 'no'} |")
        lines.append("\nThe random reference is the **strongest random family** per group "
                     "(seed-averaged), not the pool average: arm B is local and arm C is "
                     "global, and R1 says the local one loses for a trivial reason. The best "
                     "single draw is shown for context but is not the reference — a gate "
                     "against one lucky draw would be indefensible in either direction.\n")
        lines.append("\nThe gate needs all three of `CI95(Δ) > 0`, relative reduction ≥ "
                     f"{cfg['gate3a']['target_relative_reduction']*100:.0f}%, absolute "
                     f"reduction ≥ {cfg['gate3a']['min_absolute_reduction']}, in ≥ "
                     f"{cfg['gate3a']['min_conditions_improved']} of 5 groups. The only kill "
                     "condition is data-derived ≈ random.\n")

        # ---- Gate 3B: framing only, never a kill gate
        spectral = [s for s in derived_stems if mats[s]["run"]["type"] == "spectral_topk"]
        learned = [s for s in derived_stems if mats[s]["run"]["type"] == "householder"]
        lines.append("\n## Gate 3B — learning vs characterisation (framing only)\n")
        if spectral and learned:
            g3b = {}
            for g in groups:
                sp = np.mean([mats[s]["per_group"][g] for s in spectral], axis=0)
                le = np.mean([mats[s]["per_group"][g] for s in learned], axis=0)
                r = paired_bootstrap(sp, le, resamples=2000, seed=0)   # + => learned wins
                g3b[g] = r.as_dict()
                lines.append(f"- **{g}**: spectral {r.baseline:.4f} vs learned "
                             f"{r.treatment:.4f} (Δ {r.mean_delta:+.4f}, "
                             f"CI95 [{r.ci_low:+.4f}, {r.ci_high:+.4f}])")
            wins = sum(1 for r in g3b.values() if r["ci_low"] > 0)
            losses = sum(1 for r in g3b.values() if r["ci_high"] < 0)
            story = ("learning robust communication coordinates" if wins > losses else
                     "the Generative Observability Spectrum is discoverable in closed form"
                     if wins == losses else
                     "believe the spectral result; the optimisation is the weak part")
            lines.append(f"\nStory: _{story}_. None of these outcomes stops the project.\n")
            report["gate3b"] = {"per_group": g3b, "story": story}
        else:
            lines.append("_Needs both a spectral and a learned arm._\n")

    # identity, for context only
    ident = [r for r in gate_runs if r["type"] in SANITY_TYPES]
    if ident:
        m = np.mean([np.mean([v["sign_ber"] for v in r["results"][args.split].values()])
                     for r in ident])
        lines.append(f"\n> Identity arm mean sign BER: {m:.4f} — sanity only. Identity loses "
                     "to any global transform for a trivial locality reason (R1) and is "
                     "excluded from every gate.\n")

    out = Path(args.out or (Path(cfg["paths"]["reports_dir"]) /
                            f"results_{args.tag}_{args.split}.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
