"""Git provenance for anything that becomes evidence.

A result file that says which commit produced it, and whether the tree was clean, is
reproducible. One that does not is "some commit plus an unknown diff" -- which is how a
result stops being checkable without anybody noticing.

`require_clean()` makes that a precondition rather than a habit, so an audit or a gate
artifact cannot be generated from an edited working tree by accident.
"""
from __future__ import annotations

import subprocess


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def provenance() -> dict:
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_commit_short": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(status),
        "git_dirty_files": [line[3:] for line in status.splitlines()] if status else [],
    }


def require_clean(what: str = "this artifact", allow_dirty: bool = False) -> dict:
    p = provenance()
    if p["git_dirty"] and not allow_dirty:
        files = ", ".join(p["git_dirty_files"][:5])
        raise SystemExit(
            f"refusing to produce {what} from a dirty working tree: {files}"
            + (" ..." if len(p["git_dirty_files"]) > 5 else "")
            + "\nCommit first, or pass --allow-dirty for a throwaway run. An artifact "
              "from an uncommitted tree is 'some commit plus an unknown diff'.")
    return p
