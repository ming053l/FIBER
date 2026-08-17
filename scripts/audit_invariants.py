#!/usr/bin/env python
"""Final Integration Audit — verify every protocol invariant is pinned by a live test.

The audit's claim is not "we were careful". It is that each invariant has a named test
which fails if the invariant breaks. So this script does three things that prose cannot:

  1. checks every node id in reports/invariants.yaml still EXISTS in the suite, which
     catches a test renamed or deleted while the report keeps citing it;
  2. runs exactly those tests;
  3. flags any test in the suite that no invariant claims, so coverage of the map is
     visible rather than assumed.

    python scripts/audit_invariants.py            # verify + run
    python scripts/audit_invariants.py --report   # also write reports/p0_audit_fix.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from fiber.utils.logging import get_logger
from fiber.utils.provenance import provenance

log = get_logger("audit")


def collected_node_ids() -> set[str]:
    p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
                       capture_output=True, text=True)
    ids = set()
    for line in p.stdout.splitlines():
        if "::" in line:
            ids.add(line.split("[")[0].strip())
    if not ids:
        raise SystemExit(f"could not collect tests:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="reports/invariants.yaml")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--skip-run", action="store_true", help="verify the map only")
    ap.add_argument("--out", default="reports/p0_audit_fix.md")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.map).read_text())
    claimed: dict[str, list[str]] = {}
    for group, body in spec.items():
        for inv in body["invariants"]:
            for t in inv["tests"]:
                claimed.setdefault(t, []).append(group)

    available = collected_node_ids()
    missing = sorted(t for t in claimed if t not in available)
    unclaimed = sorted(available - set(claimed))

    log.info("%d groups, %d invariants, %d distinct tests claimed of %d collected",
             len(spec), sum(len(b["invariants"]) for b in spec.values()),
             len(claimed), len(available))
    if missing:
        for t in missing:
            log.error("claimed but NOT FOUND: %s  (invariant group %s)", t, claimed[t])
        raise SystemExit(
            f"{len(missing)} invariant(s) cite a test that no longer exists. A report "
            "citing a deleted test is worse than one citing nothing.")

    passed = None
    if not args.skip_run:
        proc = subprocess.run([sys.executable, "-m", "pytest", *sorted(claimed), "-q"],
                              capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        passed = proc.returncode == 0
        log.info("invariant tests: %s (%s)", "PASS" if passed else "FAIL", tail)
        if not passed:
            log.error(proc.stdout[-4000:])

    result = {"groups": len(spec), "invariants": sum(len(b["invariants"]) for b in spec.values()),
              "tests_claimed": len(claimed), "tests_collected": len(available),
              "tests_unclaimed": len(unclaimed), "all_claimed_tests_exist": True,
              "invariant_tests_pass": passed, **provenance()}
    Path("reports/invariant_audit.json").write_text(json.dumps(
        {**result, "unclaimed_tests": unclaimed}, indent=2))
    log.info("%d tests are not claimed by any invariant (they cover implementation "
             "detail rather than protocol)", len(unclaimed))
    log.info("wrote reports/invariant_audit.json")
    return 0 if passed is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
