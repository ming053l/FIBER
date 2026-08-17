#!/usr/bin/env python
"""Choose the data-derived method on VAL, then lock it (P0-3).

Everything that could be chosen -- method family, k, hyperparameters, which seeds
count as replications -- is decided here, from the validation split only, and written
to `reports/selection_<tag>.json`. The test evaluation loads that artifact and is not
able to choose anything.

This module reads `results[VAL]` and the `val|...` arrays. It never opens a test key;
`_forbid_test()` enforces that at runtime rather than by convention.

    python scripts/select_method.py --tag pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fiber.channels import ChannelBank
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean

log = get_logger("select")

SELECTION_SPLIT = "val"
GATE_SCOPE = "gate"
# Gate 3A asks whether an observable SUBSPACE exists, so only subspace-discovery
# families are selectable: spectral_topk (closed-form) and householder (differentiable).
#
# `rotated_learned` (D3) is excluded on purpose. It only rotates the basis inside D1's
# subspace, so span and D_cert are identical to D1's by construction; letting it win
# Gate 3A would fold coding-basis optimisation into a claim about subspace anisotropy.
# `rotated_random` (D2) is excluded for the same reason plus the obvious one: selecting
# it would be picking the best of N random bases. Both live only in the P0-7 analysis.
DERIVED_TYPES = {"spectral_topk", "householder"}
BASIS_ONLY_TYPES = {"rotated_random", "rotated_learned"}
PRIMARY_RANDOM_TYPE = "haar"


def _forbid_test(key: str) -> str:
    if key.startswith("test"):
        raise RuntimeError(f"selection tried to read {key!r}: selection is VAL-only (P0-3)")
    return key


PROTOCOL_BLOCKS = ("model", "latent", "vae", "dataset", "fiber", "extractor", "training",
                   "spectrum", "gate3a", "attacks", "train_attacks", "eval_attacks")


def config_fingerprint(cfg) -> str:
    """Everything that defines the protocol. The test evaluator recomputes this and
    hard-fails on a mismatch: a lock that survives a protocol edit is not a lock."""
    keep = {k: cfg[k] for k in PROTOCOL_BLOCKS if k in cfg}
    return hashlib.blake2s(json.dumps(keep, sort_keys=True, default=str).encode(),
                           digest_size=8).hexdigest()


def registered_seeds(cfg, overrides: dict[str, list[int]] | None = None,
                     required: list[str] | None = None) -> dict:
    """The seeds each REQUIRED arm was pre-registered to run.

    "Required" is declared (`fiber.required_arms`, or --required-arms), never inferred
    from which arms happen to have produced files -- otherwise an arm whose every run
    failed would exempt itself. A reduced seed set is allowed, since a pilot legitimately
    runs fewer draws, but it too must be declared.
    """
    arms = cfg["fiber"]["arms"]
    names = required if required is not None else cfg["fiber"].get("required_arms",
                                                                   list(arms))
    unknown = [a for a in names if a not in arms]
    if unknown:
        raise SystemExit(f"required arms name unknown arm(s) {unknown}")
    out = {arm: sorted(arms[arm].get("seeds", [])) for arm in names}
    for arm, seeds in (overrides or {}).items():
        if arm not in arms:
            raise SystemExit(f"--registered-seeds names an unknown arm {arm!r}")
        out[arm] = sorted(seeds)
    return out


def check_completeness(runs: list[dict], registered: dict, k: int) -> dict:
    """Every registered seed of every competing arm must be present.

    Averaging over whatever files exist is survivorship bias with no warning: a seed
    that crashed, or one that was deleted, simply stops counting, and an arm whose worst
    draw failed to write looks better than it is.
    """
    present: dict[str, set] = defaultdict(set)
    for r in runs:
        if r["k"] == k:
            present[r["arm"]].add(r.get("structure_seed", r["seed"]))
    report, missing = {}, {}
    for arm, seeds in registered.items():
        if not seeds:
            continue                      # nothing was registered for this arm
        # An arm with NO runs at all is the worst case, not an exemption. Skipping it
        # made whole-arm failure invisible -- the same survivorship bias this check
        # exists for, one level larger. Exercised for real: run_pilot.sh trained a
        # misspelled `C3_rand_hh` while the config registered `C3_frozen_hh`, so every
        # run of that arm failed and completeness stayed silent.
        have = present.get(arm, set())
        gap = sorted(set(seeds) - have)
        report[arm] = {"registered": seeds, "present": sorted(have), "missing": gap}
        if gap:
            missing[arm] = gap
    return {"per_arm": report, "missing": missing, "complete": not missing}


def run_stem(r: dict) -> str:
    """Identify a run without requiring the loader to have attached a path, so select()
    stays unit-testable on plain summaries."""
    npz = r.get("_npz")
    return Path(npz).stem if npz else f"{r['arm']}_k{r['k']}_s{r['seed']}"


def file_digest(path: Path) -> str:
    return hashlib.blake2s(Path(path).read_bytes(), digest_size=16).hexdigest()


def run_manifest(runs: list[dict]) -> list[dict]:
    """Name the EXACT runs, with content hashes. The test evaluator loads these and
    nothing else, so a run added to the results directory after the lock cannot change
    the reported result."""
    out = []
    for r in sorted(runs, key=lambda r: (r["arm"], r["k"], r["seed"])):
        jf = Path(r["_npz"]).with_suffix(".json")
        entry = {
            "stem": run_stem(r), "arm": r["arm"], "k": r["k"], "seed": r["seed"],
            "hyperparameters_fingerprint": r.get("hyperparameters_fingerprint", "legacy"),
            "json_sha": file_digest(jf), "npz_sha": file_digest(r["_npz"]),
        }
        # The CHECKPOINTS are what test evaluation will load, so they are what has to be
        # locked. Hashing only the result files would let the frame or the receiver be
        # swapped after selection while evaluate_locked still accepted them (B1).
        for kind in ("frame", "extractor"):
            ck = jf.with_name(f"{jf.stem}_{kind}.pt")
            entry[f"{kind}_sha"] = file_digest(ck) if ck.exists() else None
        out.append(entry)
    return out


def load_val_runs(out_dir: Path) -> list[dict]:
    runs = []
    for jf in sorted(out_dir.glob("*.json")):
        meta = json.loads(jf.read_text())
        if meta.get("analysis_scope", GATE_SCOPE) != GATE_SCOPE:
            # receiver-architecture controls and P0-7 basis runs are separate analyses;
            # letting them into the candidate pool would mix a spatial-receiver result
            # into a ResNet-receiver method, or overweight one structural seed
            continue
        res = meta.get("results", {})
        if SELECTION_SPLIT not in res:
            log.warning("%s has no %s results, skipping (rerun the arm so selection has "
                        "something to choose on)", jf.name, SELECTION_SPLIT)
            continue
        npz = jf.with_suffix(".npz")
        if not npz.exists():
            continue
        meta["_npz"] = npz
        runs.append(meta)
    return runs


def val_mean_ber(run: dict, attacks: list[str]) -> float:
    res = run["results"][_forbid_test(SELECTION_SPLIT)]
    vals = [res[a]["sign_ber"] for a in attacks if a in res]
    return float(np.mean(vals)) if vals else float("nan")


def select(runs: list[dict], attacks: list[str]) -> dict:
    """Lowest seed-averaged val sign BER among the data-derived families."""
    # Same arm and same k but different hyperparameters are DIFFERENT candidates;
    # averaging them together would hide which configuration was actually chosen.
    by_candidate: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        by_candidate[(r["arm"], r["k"], r.get("hyperparameters_fingerprint", "legacy"))].append(r)

    scored = []
    for (arm, k, hp), rs in sorted(by_candidate.items()):
        rtype = rs[0]["type"]
        # Hierarchical: average receiver replications WITHIN a structural seed, then
        # average structural seeds. A flat mean over files would weight a structural
        # seed by how many receiver replications it happens to have.
        by_structure: dict[int, list[float]] = defaultdict(list)
        for r in rs:
            by_structure[r.get("structure_seed", r["seed"])].append(val_mean_ber(r, attacks))
        per_seed = {s_: float(np.mean(v)) for s_, v in by_structure.items()}
        scored.append({
            "arm": arm, "k": k, "type": rtype, "hyperparameters_fingerprint": hp,
            "runs": [run_stem(r) for r in rs],
            "seeds": sorted(per_seed),
            "receiver_replications": {str(s_): len(v) for s_, v in by_structure.items()},
            "val_sign_ber_per_structure_seed": per_seed,
            "val_sign_ber": float(np.mean(list(per_seed.values()))),
            "is_derived": rtype in DERIVED_TYPES,
        })
    derived = [c for c in scored if c["is_derived"]]
    if not derived:
        raise SystemExit("no data-derived runs with val results; nothing to select")
    winner = min(derived, key=lambda c: c["val_sign_ber"])
    return {"winner": winner, "candidates": scored}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--registered-seeds", action="append", default=[],
                    metavar="ARM=0,1,2",
                    help="declare a reduced seed set for an arm; recorded in the lock")
    ap.add_argument("--required-arms", default=None,
                    help="comma-separated arms the gate requires; defaults to "
                         "fiber.required_arms in the config")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="proceed with missing registered seeds; recorded in the lock "
                         "and flagged in the gate report")
    args = ap.parse_args()

    overrides = {}
    for spec in args.registered_seeds:
        arm, _, seeds = spec.partition("=")
        overrides[arm] = [int(x) for x in seeds.split(",") if x != ""]

    cfg = load_config(args.config)
    prov = require_clean("a selection lock", allow_dirty=args.allow_dirty)
    bank = ChannelBank(cfg)
    out_dir = Path(args.results_dir or (Path(cfg["paths"]["data_root"]) / "results" / args.tag))
    runs = load_val_runs(out_dir)
    if not runs:
        raise SystemExit(f"no runs with {SELECTION_SPLIT} results in {out_dir}")

    required = ([a for a in args.required_arms.split(",") if a]
                if args.required_arms else None)
    registered = registered_seeds(cfg, overrides, required)
    chosen = select(runs, bank.eval)
    winner = chosen["winner"]
    completeness = check_completeness(runs, registered, winner["k"])
    if not completeness["complete"]:
        detail = "; ".join(f"{a} missing {s}" for a, s in completeness["missing"].items())
        if not args.allow_incomplete:
            raise SystemExit(
                f"registered seeds are missing at k={winner['k']}: {detail}.\n"
                "Averaging over the survivors is survivorship bias -- a crashed or "
                "deleted seed silently stops counting. Re-run them, declare a smaller "
                "set with --registered-seeds ARM=0,1,..., or pass --allow-incomplete, "
                "which is recorded in the lock and flagged in the gate report.")
        log.warning("proceeding with missing registered seeds: %s", detail)
    winner_runs = [r for r in runs if run_stem(r) in set(winner["runs"])]
    ref_runs = [r for r in runs if r["type"] == PRIMARY_RANDOM_TYPE and r["k"] == winner["k"]]
    locked_stems = {run_stem(r) for r in winner_runs} | {run_stem(r) for r in ref_runs}
    context_runs = [r for r in runs if r["k"] == winner["k"] and run_stem(r) not in locked_stems]
    if not ref_runs:
        raise SystemExit(f"no {PRIMARY_RANDOM_TYPE} runs at k={winner['k']}: the gate "
                         "denominator is a uniformly random subspace (P0-2)")

    selection = {
        "tag": args.tag,
        "split_used": SELECTION_SPLIT,
        "selection_rule": ("lowest seed-averaged val sign BER over the eval attacks, "
                           "among data-derived families only"),
        "selected": {
            "arm": winner["arm"], "family": winner["type"], "k": winner["k"],
            "seeds": winner["seeds"],
            "hyperparameters": winner_runs[0].get("arm_spec",
                                                   cfg["fiber"]["arms"].get(winner["arm"], {})),
            "hyperparameters_fingerprint": winner["hyperparameters_fingerprint"],
            "val_sign_ber": winner["val_sign_ber"],
        },
        # THE lock: exact runs with content hashes, not a family plus a seed count.
        # `context_runs` covers everything else the report displays -- controls, the
        # identity sanity arm, the DDIM reference -- because the invariant is that
        # NOTHING added after the lock changes the test output, not merely that the
        # gate statistic is unchanged.
        "selected_runs": run_manifest(winner_runs),
        "reference_runs": run_manifest(ref_runs),
        "context_runs": run_manifest(context_runs),
        "random_reference": {
            "arm": ref_runs[0]["arm"], "type": PRIMARY_RANDOM_TYPE,
            "seeds": sorted(r["seed"] for r in ref_runs),
            "val_sign_ber": float(np.mean([val_mean_ber(r, bank.eval) for r in ref_runs])),
        },
        # The TEST EXECUTION PROTOCOL is locked here too. Locking the method while
        # leaving the evaluated population choosable at test time would still allow a
        # post-hoc decision -- `--limit 100` and `--limit 0` are different experiments.
        "test_protocol": {"splits": ["test", "test_heldout_prompts"], "limit": 0},
        "seed_completeness": {**completeness, "registered": registered,
                              "required_arms": sorted(registered),
                              "enforced": not args.allow_incomplete,
                              "source": "config+cli" if overrides else "config"},
        "candidates": chosen["candidates"],
        # FULL sha: the locked evaluator refuses to run unless HEAD matches, so an
        # abbreviation is not enough to pin the code the lock was taken under.
        **prov,
        "commit": prov["git_commit_short"],
        "config_fingerprint": config_fingerprint(cfg),
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "locked": True,
    }
    out = Path(args.out or (Path(cfg["paths"]["reports_dir"]) / f"selection_{args.tag}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2))

    for c in sorted(chosen["candidates"], key=lambda c: c["val_sign_ber"]):
        log.info("%-14s k=%-4d %-18s val BER %.4f %s", c["arm"], c["k"], c["type"],
                 c["val_sign_ber"], "<- LOCKED" if c is winner else "")
    log.info("locked %s (k=%d, seeds %s) against %s; wrote %s",
             winner["arm"], winner["k"], winner["seeds"],
             selection["random_reference"]["arm"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
