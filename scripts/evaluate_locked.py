#!/usr/bin/env python
"""B1 — compute the test splits for the FIRST time, after selection is frozen.

Pre-lock training runs are forbidden from touching test at all (`train_coordinates.py`
hard-fails on a test eval split), so before this script runs there is no test
prediction, no test corruption and no test metric anywhere on disk. That is what makes
the isolation physical rather than a property of which file the selector opens.

This script may only EXECUTE. It cannot choose a family, a seed, a k, a reference or a
threshold, and it never scans the results directory: it reads
`reports/selection_<tag>.json` and touches exactly the runs named there.

Everything it depends on is verified before anything is computed:

  * the working tree is clean and HEAD equals the commit the lock was taken under, so a
    one-line edit to the evaluator after the lock is a hard failure rather than a silent
    protocol change;
  * every locked run's .json, .npz, _frame.pt and _extractor.pt still hash to what the
    lock recorded -- the checkpoints matter most, since those are what gets loaded;
  * the protocol config fingerprint still matches.

    python scripts/evaluate_locked.py --tag pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.models import build_extractor
from fiber.training import evaluate
from fiber.transforms.spectral import SpectralFrame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean

import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_method import config_fingerprint, file_digest  # noqa: E402

log = get_logger("locked-eval")


def resolve_cache_tag(cli_tag: str | None, protocol: dict, run_tag: str,
                      debug: bool = False) -> str:
    """Which image cache this evaluation reads.

    The cache namespace is part of the lock: swapping to a different cache afterwards
    would evaluate the frozen checkpoints on different data. Resolved in one place, and
    unit-tested, because the alternative -- an assignment somewhere in the middle of a
    long main() -- is how it came to be used a dozen lines before it was defined.
    """
    locked = protocol.get("cache_tag")
    tag = cli_tag or locked or run_tag
    if locked and tag != locked and not debug:
        raise SystemExit(
            f"the lock names cache namespace {locked!r}, not {tag!r}: evaluating the "
            "frozen checkpoints on a different image cache is a different experiment.")
    return tag


def verify(entry: dict, out_dir: Path) -> dict:
    """Every artifact the lock named must still be byte-identical."""
    stem = entry["stem"]
    paths = {"json": out_dir / f"{stem}.json", "npz": out_dir / f"{stem}.npz",
             "frame": out_dir / f"{stem}_frame.pt",
             "extractor": out_dir / f"{stem}_extractor.pt"}
    for kind, path in paths.items():
        recorded = entry.get(f"{kind}_sha")
        if recorded is None:
            if kind in ("frame", "extractor"):
                raise SystemExit(
                    f"{stem}: the lock recorded no {kind} checkpoint. Test evaluation "
                    "loads the pre-lock checkpoints, so a run without them cannot be "
                    "evaluated; re-run training and re-select.")
            continue
        if not path.exists():
            raise SystemExit(f"{stem}: locked {kind} artifact is missing ({path})")
        if file_digest(path) != recorded:
            raise SystemExit(
                f"{stem}: {kind} changed since the lock. Test evaluation loads exactly "
                "the artifacts selection froze; a changed one is a different experiment.")
    return paths


def load_locked_run(paths: dict, device) -> tuple:
    frame_blob = torch.load(paths["frame"], map_location="cpu", weights_only=True)
    rows = frame_blob["rows"]
    # Any arm is a k-frame, and evaluation only projects, so carrying the frozen rows is
    # exactly equivalent to rebuilding the arm class -- and cannot silently rebuild it
    # differently.
    frame = SpectralFrame(d=rows.shape[1], k=rows.shape[0], rows=rows,
                          reorthonormalise=False).to(device)
    ck = torch.load(paths["extractor"], map_location="cpu", weights_only=True)
    model = build_extractor(ck["arch"], k=int(ck["k"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return frame, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--selection", default=None)
    ap.add_argument("--cache-tag", default=None,
                    help="image cache namespace; defaults to --tag. A triage writes its "
                         "artifacts under a fresh --tag while reusing an existing cache.")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    # Debug mode is a separate mode, not a flag that relaxes the official one. It cannot
    # write the official manifest, and the gate refuses anything it produces.
    ap.add_argument("--debug", action="store_true",
                    help="throwaway evaluation from a dirty tree or a different commit; "
                         "requires --out-dir and is rejected by the gate")
    ap.add_argument("--splits", nargs="*", default=None, help="debug mode only")
    ap.add_argument("--limit", type=int, default=None, help="debug mode only")
    args = ap.parse_args()

    if not args.debug and (args.splits is not None or args.limit is not None):
        raise SystemExit(
            "--splits/--limit are locked by the selection artifact (test_protocol). "
            "Choosing the evaluated population after the lock is a post-hoc decision; "
            "use --debug with --out-dir if you need a throwaway run.")

    cfg = load_config(args.config)
    prov = require_clean("a locked test evaluation", allow_dirty=args.debug)
    sel_path = Path(args.selection or (Path(cfg["paths"]["reports_dir"]) /
                                       f"selection_{args.tag}.json"))
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found: test evaluation requires a val lock (B1)")
    sel = json.loads(sel_path.read_text())

    # ---- the code itself is locked -------------------------------------
    locked_commit = sel.get("git_commit")
    if not locked_commit:
        raise SystemExit("the selection artifact has no full commit sha; re-select")
    code_matches = (prov["git_commit"] == locked_commit) and not prov["git_dirty"]
    official = code_matches and not args.debug
    if not official and not args.debug:
        raise SystemExit(
            f"HEAD {prov['git_commit_short']} != the commit the lock was taken under "
            f"({locked_commit[:7]})" + (" and the tree is dirty" if prov["git_dirty"] else "")
            + ". Changing the evaluator after the lock is a protocol revision, not a "
              "rerun: check out that commit, or re-select. --debug --out-dir gives a "
              "throwaway evaluation the gate will not accept.")
    if args.debug and not args.out_dir:
        raise SystemExit("--debug requires --out-dir: a throwaway evaluation must not "
                         "land where the official one lives.")

    protocol = sel.get("test_protocol", {"splits": ["test", "test_heldout_prompts"],
                                         "limit": 0})
    locked_attacks = protocol.get("attacks")
    cache_tag = resolve_cache_tag(args.cache_tag, protocol, args.tag, args.debug)
    splits = args.splits if (args.debug and args.splits) else protocol["splits"]
    limit = args.limit if (args.debug and args.limit is not None) else int(protocol["limit"])
    now_fp = config_fingerprint(cfg)
    if sel.get("config_fingerprint") and sel["config_fingerprint"] != now_fp:
        raise SystemExit(f"protocol config changed since selection "
                         f"({sel['config_fingerprint']} -> {now_fp}); re-select")

    out_dir = Path(args.results_dir or (Path(cfg["paths"]["data_root"]) / "results" / args.tag))
    test_dir = Path(args.out_dir or (out_dir / "locked_test"))
    test_dir.mkdir(parents=True, exist_ok=True)
    mpath = Path(cfg["paths"]["reports_dir"]) / f"test_eval_{args.tag}.json"
    if official and mpath.exists():
        # WRITE-ONCE. Re-running the official evaluation after seeing its result would
        # let a second "official" number replace the first.
        raise SystemExit(
            f"{mpath} already exists: the official test evaluation is write-once. Having "
            "seen it, producing another would make the first one revisable. Delete it "
            "deliberately as a recorded protocol revision, or use --debug --out-dir.")

    # Whether the test PIXELS were generated after the lock, or merely never read
    # before it. Both are defensible; only the first supports "no test sample existed".
    cache_manifest = Path(cfg["paths"]["cache_dir"]) / cache_tag / "test_cache_manifest.json"
    test_cache_post_lock = False
    if cache_manifest.exists():
        tc = json.loads(cache_manifest.read_text())
        # Every shard must carry the binding, not just the manifest: a manifest can be
        # rewritten over skipped pixels, a per-shard marker written at generation cannot.
        # Fail closed on a MISMATCH, rather than downgrading the claim to False. The
        # test pixels carry a cryptographic binding to the lock bytes they were
        # generated under; if the lock no longer hashes to that, the lock changed after
        # the test set was materialised. Recording False would let an edited lock
        # through with nothing but a quieter boolean to show for it -- measured: editing
        # selection_<tag>.json before this step used to exit 0.
        # Having NO manifest at all stays legitimate: that is test pixels generated
        # pre-lock, the weaker "never accessed" claim, and it is reported as such.
        if tc.get("selection_sha") != file_digest(sel_path):
            raise SystemExit(
                f"{cache_manifest} binds the test pixels to selection_sha "
                f"{tc.get('selection_sha')}, but {sel_path.name} now hashes to "
                f"{file_digest(sel_path)}. The lock changed after the test set was "
                "generated.\nRegenerate the test pixels under the current lock "
                "(scripts/cache_native_dataset.py --post-lock), or delete this manifest "
                "deliberately to fall back to the weaker 'never accessed' claim.")
        test_cache_post_lock = (tc.get("generated_after_lock") is True
                                and not tc.get("shards_not_bound_to_this_lock"))
    bank = ChannelBank(cfg)
    if locked_attacks:
        bank.eval = list(locked_attacks)
    root = Path(cfg["paths"]["cache_dir"]) / cache_tag

    entries = (sel["selected_runs"] + sel["reference_runs"] + sel.get("context_runs", []))
    log.info("verifying %d locked runs against %s", len(entries), sel_path.name)
    verified = {e["stem"]: verify(e, out_dir) for e in entries}

    manifest = {"tag": args.tag, "selection_sha": file_digest(sel_path),
                "selection_commit": locked_commit, **prov,
                "official": official, "config_fingerprint": now_fp,
                "test_protocol": {"splits": list(splits), "limit": limit,
                                  "cache_tag": cache_tag},
                "test_cache_post_lock": test_cache_post_lock,
                "claim": ("no test image was materialised or accessed, and no test "
                          "corruption, prediction or metric computed, before the lock"
                          if test_cache_post_lock else
                          "no test image was ACCESSED and no test corruption, prediction "
                          "or metric computed, before the lock; the test images "
                          "themselves predate it"),
                "runs": []}
    for entry in entries:
        stem = entry["stem"]
        paths = verified[stem]
        summary = json.loads(paths["json"].read_text())
        if any(k.startswith("test") for k in summary.get("results", {})):
            raise SystemExit(
                f"{stem}: a PRE-LOCK artifact already contains test results. Physical "
                "test isolation means no test metric existed before the lock.")
        frame, model = load_locked_run(paths, args.device)

        arrays, results = {}, {}
        for split in splits:
            ev = evaluate(model, frame, root, bank, split, bank.eval,
                          device=args.device, limit=limit)
            results[split] = {a: {kk: v for kk, v in r.items()
                                  if kk not in ("per_sample", "sample_ids",
                                                "pearson_per_coord")}
                              for a, r in ev.items()}
            for a, r in ev.items():
                arrays[f"{split}|{a}|per_sample"] = r["per_sample"]
                arrays[f"{split}|{a}|sample_ids"] = np.array(r["sample_ids"])
        np.savez_compressed(test_dir / f"{stem}.npz", **arrays)
        merged = {**summary, "results": {**summary.get("results", {}), **results},
                  "locked_test_eval": True, "selection_commit": locked_commit}
        (test_dir / f"{stem}.json").write_text(json.dumps(merged, indent=2, default=float))
        manifest["runs"].append({
            "stem": stem,
            "json_sha": file_digest(test_dir / f"{stem}.json"),
            "npz_sha": file_digest(test_dir / f"{stem}.npz")})
        log.info("%-42s %s", stem, " ".join(
            f"{s}:{np.mean([v['sign_ber'] for v in results[s].values()]):.4f}"
            for s in splits))

    if not official:
        mpath = test_dir / f"test_eval_{args.tag}_DEBUG.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    log.info("wrote %d locked test evaluations -> %s (manifest %s)",
             len(entries), test_dir, mpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
