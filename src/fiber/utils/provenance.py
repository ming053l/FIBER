"""Git provenance for anything that becomes evidence.

A result file that says which commit produced it, and whether the tree was clean, is
reproducible. One that does not is "some commit plus an unknown diff" -- which is how a
result stops being checkable without anybody noticing.

`require_clean()` makes that a precondition rather than a habit, so an audit or a gate
artifact cannot be generated from an edited working tree by accident.
"""
from __future__ import annotations

import subprocess


def _git(*args: str) -> str | None:
    """None on failure, never a silent empty string: outside a repository `git
    rev-parse` exits non-zero, and treating that as "unknown but fine" would make the
    provenance check fail OPEN -- the one direction a guarantee must not fail."""
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def provenance() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_available": commit is not None and status is not None,
        "git_dirty": bool(status) if status is not None else None,
        "git_dirty_files": [line[3:] for line in status.splitlines()] if status else [],
    }


def require_clean(what: str = "this artifact", allow_dirty: bool = False) -> dict:
    p = provenance()
    if not p["git_available"] and not allow_dirty:
        raise SystemExit(
            f"refusing to produce {what}: not a git repository, so the result could not "
            "name the code that produced it. Pass --allow-dirty for a throwaway run.")
    if p["git_dirty"] and not allow_dirty:
        files = ", ".join(p["git_dirty_files"][:5])
        raise SystemExit(
            f"refusing to produce {what} from a dirty working tree: {files}"
            + (" ..." if len(p["git_dirty_files"]) > 5 else "")
            + "\nCommit first, or pass --allow-dirty for a throwaway run. An artifact "
              "from an uncommitted tree is 'some commit plus an unknown diff'.")
    return p
