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

log = get_logger("select")

SELECTION_SPLIT = "val"
DERIVED_TYPES = {"spectral_topk", "householder"}
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
        out.append({
            "stem": run_stem(r), "arm": r["arm"], "k": r["k"], "seed": r["seed"],
            "hyperparameters_fingerprint": r.get("hyperparameters_fingerprint", "legacy"),
            "json_sha": file_digest(jf), "npz_sha": file_digest(r["_npz"]),
        })
    return out


def load_val_runs(out_dir: Path) -> list[dict]:
    runs = []
    for jf in sorted(out_dir.glob("*.json")):
        meta = json.loads(jf.read_text())
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
        scored.append({
            "arm": arm, "k": k, "type": rtype, "hyperparameters_fingerprint": hp,
            "runs": [run_stem(r) for r in rs],
            "seeds": sorted(r["seed"] for r in rs),
            "val_sign_ber": float(np.mean([val_mean_ber(r, attacks) for r in rs])),
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    bank = ChannelBank(cfg)
    out_dir = Path(args.results_dir or (Path(cfg["paths"]["data_root"]) / "results" / args.tag))
    runs = load_val_runs(out_dir)
    if not runs:
        raise SystemExit(f"no runs with {SELECTION_SPLIT} results in {out_dir}")

    chosen = select(runs, bank.eval)
    winner = chosen["winner"]
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
        "candidates": chosen["candidates"],
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or "uncommitted",
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
