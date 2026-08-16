#!/usr/bin/env python
"""Evaluate the LOCKED method on the test split (P0-3).

This script is deliberately dumb. It cannot choose the family, the seed, k, the
hyperparameters or the reference: all of that is decided on `val` by
`scripts/select_method.py` and frozen in `reports/selection_<tag>.json`, which this
script requires. Without that artifact it refuses to run.

Comparisons are against RANDOM, never identity (PLAN.md R1). The denominator is the
Haar family, seed-averaged (P0-2): a uniformly random k-dimensional subspace is the
null the claim is against, whereas beating Hadamard would only say the directions beat
one structured family. Signed permutation, Hadamard and random-Householder are
controls, reported beside the gate and flagged if any of them beats the locked arm.

Seeds are replications, never a selection axis. Two intervals are reported: the paired
sample bootstrap on seed-averaged metrics (conditional on the draws actually made) and
a hierarchical bootstrap that also resamples the Haar draws and training seeds, which
is the interval matching the claim about E_{Q~Haar}[BER].

    python scripts/select_method.py   --tag pilot        # first, on val
    python scripts/eval_coordinates.py --tag pilot       # then, locked, on test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fiber.channels import ChannelBank
from fiber.metrics import (gate3a_condition, gate3a_verdict, hierarchical_paired_bootstrap,
                           paired_bootstrap)
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_method import config_fingerprint, file_digest  # noqa: E402

log = get_logger("eval")

RANDOM_TYPES = {"haar", "signed_permutation", "hadamard", "random_householder"}
DERIVED_TYPES = {"spectral_topk", "householder"}      # Gate 3A / 3B families
BASIS_ONLY_TYPES = {"rotated_random", "rotated_learned"}   # P0-7 only, never a gate
# P0-7: same subspace, different basis. Reported in their own section, never as a
# random-SUBSPACE control and never as the gate denominator.
BASIS_TYPES = {"spectral_topk", "rotated_random", "rotated_learned"}
SANITY_TYPES = {"identity"}
REFERENCE_TYPES = {"ddim_inversion_reference"}
assert not (RANDOM_TYPES | DERIVED_TYPES) & REFERENCE_TYPES


def load_locked_runs(out_dir: Path, manifest: list[dict], role: str) -> list[dict]:
    """Load EXACTLY the runs the lock names, verifying their content hashes.

    Globbing the directory instead would mean a run added after the lock silently
    joins the average -- the same post-hoc mutation the lock exists to prevent, just
    arriving through the filesystem rather than through an argmin.
    """
    runs = []
    for entry in manifest:
        jf = out_dir / f"{entry['stem']}.json"
        npz = jf.with_suffix(".npz")
        if not jf.exists() or not npz.exists():
            raise SystemExit(f"locked {role} run {entry['stem']} is missing from {out_dir}")
        for path, key in ((jf, "json_sha"), (npz, "npz_sha")):
            if entry.get(key) and file_digest(path) != entry[key]:
                raise SystemExit(
                    f"{path.name} changed since the lock ({key} mismatch). The selection "
                    "artifact refers to specific run CONTENT; re-select if the runs were "
                    "regenerated.")
        meta = json.loads(jf.read_text())
        meta["_npz"] = npz
        runs.append(meta)
    return runs


def per_sample(run: dict, split: str, attack: str):
    with np.load(run["_npz"], allow_pickle=False) as z:
        key = f"{split}|{attack}|per_sample"
        if key not in z:
            return None, None
        return z[key], z[f"{split}|{attack}|sample_ids"]


def group_matrix(runs, split, attacks, bank):
    """{run_stem: {group: per-sample BER}} averaged over the attacks in a group and
    aligned across runs by sample_id."""
    groups = defaultdict(list)
    for a in attacks:
        groups[bank.group_of(a)].append(a)
    out, ref_ids = {}, None
    for run in runs:
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
        # key by the file stem: with the receiver seed decoupled from the arm seed
        # (P0-7.1), (arm, seed) is no longer unique and runs would overwrite each other
        out[Path(run["_npz"]).stem] = {"per_group": per_group, "run": run}
    return out, sorted(groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--split", default="test")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "val":
        raise SystemExit("val is the SELECTION split; run scripts/select_method.py. "
                         "Evaluating the gate on val would defeat the lock (P0-3).")

    cfg = load_config(args.config)
    bank = ChannelBank(cfg)
    sel_path = Path(args.selection or (Path(cfg["paths"]["reports_dir"]) /
                                       f"selection_{args.tag}.json"))
    if not sel_path.exists():
        raise SystemExit(
            f"{sel_path} not found. Test evaluation is locked to a val-selected method "
            "(P0-3): run scripts/select_method.py --tag " + args.tag + " first.")
    sel = json.loads(sel_path.read_text())
    locked_arm = sel["selected"]["arm"]
    locked_k = int(sel["selected"]["k"])
    ref_arm = sel["random_reference"]["arm"]

    # A lock that survives a protocol edit is not a lock.
    now_fp = config_fingerprint(cfg)
    if sel.get("config_fingerprint") and sel["config_fingerprint"] != now_fp:
        raise SystemExit(
            f"protocol config changed since selection ({sel['config_fingerprint']} -> "
            f"{now_fp}). Re-run scripts/select_method.py on val; the lock does not carry "
            "across a protocol change.")

    out_dir = Path(args.results_dir or (Path(cfg["paths"]["data_root"]) / "results" / args.tag))
    if "selected_runs" not in sel or "reference_runs" not in sel:
        raise SystemExit("selection artifact predates the exact-run lock (P0-3.1): "
                         "re-run scripts/select_method.py")
    locked_runs = load_locked_runs(out_dir, sel["selected_runs"], "selected")
    ref_runs = load_locked_runs(out_dir, sel["reference_runs"], "reference")
    # Controls and sanity arms are locked too. Discovering them by glob would let a run
    # added after the lock change the report -- the gate number would survive, but the
    # control table and the "a control beat the locked arm" flag would not.
    context_runs = load_locked_runs(out_dir, sel.get("context_runs", []), "context")

    mats, groups = group_matrix(locked_runs + ref_runs + context_runs, args.split,
                                bank.eval, bank)
    locked_set = {Path(r["_npz"]).stem for r in locked_runs}
    ref_set = {Path(r["_npz"]).stem for r in ref_runs}

    def stems_where(pred):
        return [s for s, m in mats.items() if pred(m["run"])]

    locked_stems = [s for s in mats if s in locked_set]
    haar_stems = [s for s in mats if s in ref_set]
    if not locked_stems:
        raise SystemExit(f"locked arm {locked_arm} has no runs on {args.split}")
    if not haar_stems:
        raise SystemExit(f"reference arm {ref_arm} has no runs on {args.split}")
    control_stems = [s for s in mats
                     if s not in locked_set and s not in ref_set
                     and mats[s]["run"]["type"] in RANDOM_TYPES]

    report = {"tag": args.tag, "split": args.split, "k": locked_k,
              "selection": {"file": str(sel_path), "arm": locked_arm,
                            "commit": sel.get("commit"),
                            "config_fingerprint": sel.get("config_fingerprint"),
                            "config_fingerprint_now": now_fp,
                            "locked_runs": [e["stem"] for e in sel["selected_runs"]],
                            "locked_reference_runs": [e["stem"] for e in sel["reference_runs"]]},
              "n_locked_seeds": len(locked_stems), "n_haar_draws": len(haar_stems)}
    lines = [f"# FIBER — locked evaluation, `{args.tag}`, split `{args.split}`\n",
             f"Method locked on **val** by `{sel_path.name}` "
             f"(commit `{sel.get('commit')}`, config `{sel.get('config_fingerprint')}`):\n",
             f"- locked arm: **{locked_arm}**, k = {locked_k}, "
             f"seeds {sel['selected']['seeds']} (replications, never selected over)",
             f"- denominator: **{ref_arm}** ({sel['random_reference']['type']}), "
             f"seeds {sel['random_reference']['seeds']}",
             f"- exact locked runs: `{', '.join(e['stem'] for e in sel['selected_runs'])}`",
             f"- exact reference runs: `{', '.join(e['stem'] for e in sel['reference_runs'])}`\n",
             "This script cannot choose any of the above; it reads them.\n"]

    # ---------------- descriptive table (no selection happens here) -------
    lines.append("## Per-arm sign BER (descriptive)\n")
    header = "| arm | seed | " + " | ".join(bank.eval) + " | mean |"
    lines.append(header)
    lines.append("|" + "---|" * (len(bank.eval) + 3))
    for stem in sorted(mats):
        r = mats[stem]["run"]
        res = r["results"].get(args.split, {})
        vals = [res.get(a, {}).get("sign_ber", float("nan")) for a in bank.eval]
        mark = " †" if r["type"] in REFERENCE_TYPES else (" **[locked]**" if r["arm"] == locked_arm else "")
        lines.append(f"| {r['arm']}{mark} | {r['seed']} | " +
                     " | ".join(f"{v:.4f}" for v in vals) + f" | {np.nanmean(vals):.4f} |")
    if any(mats[s]["run"]["type"] in REFERENCE_TYPES for s in mats):
        lines.append("\n† prompt-assisted DDIM inversion — **violates the receiver protocol**. "
                     "Diagnostic only; never a baseline, never in a gate.")

    # ---------------- Gate 3A on the locked method ------------------------
    provisional = args.tag != "full"
    lines.append(f"\n## Gate 3A — locked method vs Haar (k = {locked_k})\n")
    conditions = {}
    for g in groups:
        haar_draws = [mats[s]["per_group"][g] for s in haar_stems]
        locked_seeds = [mats[s]["per_group"][g] for s in locked_stems]
        reference = np.mean(haar_draws, axis=0)
        treatment = np.mean(locked_seeds, axis=0)
        cond = paired_bootstrap(reference, treatment,
                                resamples=int(cfg["eval"]["bootstrap_resamples"]), seed=0)
        hier = hierarchical_paired_bootstrap(haar_draws, locked_seeds,
                                             resamples=2000, seed=0)
        control_means = defaultdict(list)
        for s in control_stems:
            control_means[mats[s]["run"]["arm"]].append(float(mats[s]["per_group"][g].mean()))
        controls = {a: float(np.mean(v)) for a, v in control_means.items()}
        beaten_by = {a: m for a, m in controls.items() if m < float(treatment.mean())}
        conditions[g] = {
            **gate3a_condition(cond, cfg["gate3a"]),
            "hierarchical_ci": [hier.ci_low, hier.ci_high],
            "hierarchical_delta": hier.mean_delta,
            "haar_spread": float(np.std([d.mean() for d in haar_draws])),
            "controls": controls, "controls_beating_locked": beaten_by,
        }
    verdict = gate3a_verdict(conditions, cfg["gate3a"], provisional=provisional)
    report["gate3a"] = {**verdict, "conditions": conditions}

    lines.append(f"**Verdict: {verdict['verdict']}** "
                 f"({verdict['n_passed']}/{verdict['n_required']} channel groups"
                 + (", provisional: a pilot may not close the question)" if provisional else ")") + "\n")
    lines.append("| group | Haar E_Q[BER] | spread | locked BER | ΔBER | CI95 (paired) | "
                 "CI95 (hierarchical) | rel | pass |")
    lines.append("|" + "---|" * 9)
    for g, c in conditions.items():
        lines.append(
            f"| {g} | {c['baseline']:.4f} | ±{c['haar_spread']:.4f} | {c['treatment']:.4f} | "
            f"{c['mean_delta']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
            f"[{c['hierarchical_ci'][0]:+.4f}, {c['hierarchical_ci'][1]:+.4f}] | "
            f"{c['relative_reduction']*100:.1f}% | {'yes' if c['passed'] else 'no'} |")
    lines.append("\nThe paired interval is conditional on the Haar draws actually made; the "
                 "hierarchical one resamples draws and seeds too, which is the interval that "
                 "matches a claim about `E_{Q~Haar}[BER]`. Gate thresholds use the paired "
                 "interval and the hierarchical one is reported beside it.\n")
    flagged = {g: c["controls_beating_locked"] for g, c in conditions.items()
               if c["controls_beating_locked"]}
    if flagged:
        lines.append("\n> **A control beats the locked arm.** If a structured random family "
                     "is stronger than the data-derived one, the Haar comparison alone "
                     "overstates the result.\n")
        for g, ctrls in flagged.items():
            lines.append(f">   - {g}: " + ", ".join(f"{a} {m:.4f}" for a, m in
                                                    sorted(ctrls.items(), key=lambda x: x[1])))
        lines.append("")
    else:
        lines.append("\nNo control family beat the locked arm in any group.\n")

    # ---------------- Gate 3B — framing only ------------------------------
    lines.append("\n## Gate 3B — learning vs characterisation (framing only)\n")
    spectral = stems_where(lambda r: r["type"] == "spectral_topk")
    learned = stems_where(lambda r: r["type"] == "householder")
    if spectral and learned:
        g3b = {}
        for g in groups:
            sp = np.mean([mats[s]["per_group"][g] for s in spectral], axis=0)
            le = np.mean([mats[s]["per_group"][g] for s in learned], axis=0)
            r = paired_bootstrap(sp, le, resamples=2000, seed=0)
            g3b[g] = r.as_dict()
            lines.append(f"- **{g}**: certified-spectral {r.baseline:.4f} vs learned "
                         f"{r.treatment:.4f} (Δ {r.mean_delta:+.4f}, "
                         f"CI95 [{r.ci_low:+.4f}, {r.ci_high:+.4f}])")
        wins = sum(1 for r in g3b.values() if r["ci_low"] > 0)
        losses = sum(1 for r in g3b.values() if r["ci_high"] < 0)
        story = ("direct optimisation improves discovery" if wins > losses else
                 "the observability geometry is discoverable in closed form"
                 if wins == losses else
                 "trust the certified spectral result; the optimisation is the weak part")
        lines.append(f"\nStory: _{story}_. Never a kill gate. Both families are compared as "
                     "seed averages — best-seed selection is not available here.\n")
        report["gate3b"] = {"per_group": g3b, "story": story}
    else:
        lines.append("_Needs both a certified-spectral and a learned arm._\n")

    # ---------------- P0-7 — subspace vs basis --------------------------
    # Everything here must live in ONE subspace. The rotation arms pin it with
    # `base_seed`, so the D1 comparator is D_spectral at exactly that seed -- averaging
    # D1 over all its seeds would compare "several different subspaces, averaged"
    # against "rotations of one subspace" while claiming to hold the subspace fixed.
    rot_runs = [m["run"] for m in mats.values() if m["run"]["type"] in BASIS_ONLY_TYPES
                and m["run"].get("analysis_scope", "p0_7_basis") == "p0_7_basis"]
    if rot_runs:
        base_seeds = {r.get("base_seed", 0) for r in rot_runs}
        if len(base_seeds) > 1:
            raise SystemExit(f"rotation arms pin different base_seeds {sorted(base_seeds)}: "
                             "they would not share a subspace")
        base_seed = base_seeds.pop()

        def in_p0_7(run) -> bool:
            if run.get("analysis_scope", "p0_7_basis") != "p0_7_basis":
                return False
            if run["type"] in BASIS_ONLY_TYPES:
                return True
            # the single spectral run whose subspace the rotations actually use
            return (run["type"] == "spectral_topk"
                    and run.get("basis_seed", run["seed"]) == base_seed)

        basis_stems = [s for s in mats if in_p0_7(mats[s]["run"])]
        lines.append("\n## P0-7 — same subspace, different coding basis\n")
        lines.append(f"All rows below span the **same** subspace (spectral `base_seed = "
                     f"{base_seed}`), so every certified observability quantity is "
                     "identical for them by construction (`span(AV) = span(V)`, asserted "
                     "to 1e-9). Only the sign coding can differ, so any gap here is a "
                     "**coding-basis** result and must not be reported as more observable "
                     "information.\n")
        lines.append("| basis | arm | basis draws | receiver seeds | mean sign BER | "
                     "basis spread |")
        lines.append("|" + "---|" * 6)
        label = {"spectral_topk": "D1 certified eigenbasis",
                 "rotated_random": "D2 random O(k) basis",
                 "rotated_learned": "D3 learned SO(k) basis"}

        # receiver randomness is marginalised WITHIN a basis, then the spread is taken
        # ACROSS bases: otherwise the spread mixes basis variability with extractor
        # training noise and cannot be attributed to the coding basis (P0-7.1).
        per_basis = defaultdict(lambda: defaultdict(list))
        receivers = defaultdict(set)
        for s in basis_stems:
            run = mats[s]["run"]
            score = float(np.mean([v.mean() for v in mats[s]["per_group"].values()]))
            per_basis[run["arm"]][run.get("basis_seed", run["seed"])].append(score)
            receivers[run["arm"]].add(run.get("receiver_seed", run["seed"]))

        basis_report = {}
        for arm in sorted(per_basis):
            rtype = next(mats[s]["run"]["type"] for s in basis_stems
                         if mats[s]["run"]["arm"] == arm)
            per = {bs: float(np.mean(v)) for bs, v in per_basis[arm].items()}
            vals = list(per.values())
            basis_report[arm] = {
                "type": rtype, "n_basis_draws": len(vals),
                "receiver_seeds": sorted(receivers[arm]),
                "mean_sign_ber": float(np.mean(vals)),
                "basis_spread": float(np.std(vals)) if len(vals) > 1 else None,
                "per_basis_seed": per,
            }
            spread = basis_report[arm]["basis_spread"]
            lines.append(f"| {label.get(rtype, rtype)} | {arm} | {len(vals)} | "
                         f"{sorted(receivers[arm])} | {np.mean(vals):.4f} | "
                         + (f"±{spread:.4f} |" if spread is not None else "n/a |"))
        lines.append("\nD2 is averaged over its draws; best-of-N would be the same free "
                     "win the Haar denominator forbids. The spread column is taken ACROSS "
                     "bases after marginalising the receiver seed within each basis, so it "
                     "is basis-to-basis variability rather than a mixture of that and "
                     "extractor training noise. It is reported only when more than one "
                     "basis draw exists.\n")
        if any(len(v["receiver_seeds"]) < 2 for v in basis_report.values()):
            lines.append("> Only one receiver seed per basis in this run, so the "
                         "marginalisation is nominal and the spread still carries "
                         "extractor noise. Treat it as descriptive.\n")
        report["p0_7_basis"] = {"base_seed": base_seed, "arms": basis_report}

    ident = [s for s in mats if mats[s]["run"]["type"] in SANITY_TYPES]
    if ident:
        m = float(np.mean([np.mean(list(mats[s]["per_group"].values())) for s in ident]))
        lines.append(f"\n> Identity arm mean sign BER: {m:.4f} — sanity only. Identity loses "
                     "to any global transform for a trivial locality reason (R1) and is "
                     "excluded from every gate.\n")

    out = Path(args.out or (Path(cfg["paths"]["reports_dir"]) /
                            f"gate3_{args.tag}_{args.split}.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    log.info("%s -> %s", verdict["verdict"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
