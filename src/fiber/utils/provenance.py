"""Git provenance for anything that becomes evidence.

A result file that says which commit produced it, and whether the tree was clean, is
reproducible. One that does not is "some commit plus an unknown diff" -- which is how a
result stops being checkable without anybody noticing.

`require_clean()` makes that a precondition rather than a habit, so an audit or a gate
artifact cannot be generated from an edited working tree by accident.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def _git(*args: str, strip: bool = True) -> str | None:
    """None on failure, never a silent empty string: outside a repository `git
    rev-parse` exits non-zero, and treating that as "unknown but fine" would make the
    provenance check fail OPEN -- the one direction a guarantee must not fail.

    `strip=False` for `status --porcelain`, whose format is `XY<space>PATH`: an
    unstaged modification starts with a SPACE, and stripping the whole output eats the
    first line's leading space, shifting that path by one character. Cosmetic while the
    list was only displayed; not cosmetic once entries are matched against a prefix.
    """
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    return p.stdout.strip() if strip else p.stdout.rstrip("\n")


# Paths that are OUTPUTS, not code. A pipeline writes its own artifacts into the
# repository, so counting them as "dirty" makes every step after the first refuse to
# run -- observed: the first spectrum wrote reports/spectrum_triage1_seed0.json and the
# whole triage then aborted on its own provenance check. What has to be committed is the
# code that produced the artifact, not the artifact.
ARTIFACT_PREFIXES = ("reports/",)


def provenance(artifact_prefixes: tuple[str, ...] = ARTIFACT_PREFIXES) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", strip=False)
    files = [line[3:] for line in status.splitlines()] if status else []
    ignored = [f for f in files if f.startswith(artifact_prefixes)]
    dirty = [f for f in files if f not in ignored]
    return {
        "git_commit": commit,
        "git_commit_short": _git("rev-parse", "--short", "HEAD"),
        "git_available": commit is not None and status is not None,
        "git_dirty": bool(dirty) if status is not None else None,
        "git_dirty_files": dirty,
        # listed rather than dropped: "the tree was clean apart from these outputs" is
        # the honest statement, and it stays checkable
        "git_dirty_artifacts": ignored,
    }


def require_clean(what: str = "this artifact", allow_dirty: bool = False,
                  artifact_prefixes: tuple[str, ...] = ARTIFACT_PREFIXES) -> dict:
    p = provenance(artifact_prefixes)
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


def write_once(path, text: str, what: str = "this artifact"):
    """Create `path` exclusively. Refuse, loudly, if it is already there.

    A selection lock whose file can be overwritten is not a lock: re-running selection
    after seeing a test number would silently replace the artifact the test evaluation
    claims to be bound to, and nothing downstream could tell. `require_clean()` does not
    catch this -- `reports/` is an artifact prefix, so a rewritten lock leaves the tree
    just as "clean" as a first one.

    tmp + `os.link` rather than a plain O_EXCL open, so the artifact is never partially
    visible: `link` is atomic and fails if the destination exists, so an interrupted
    write leaves a temp file rather than a truncated lock that still parses.
    """
    path = Path(path)
    if path.exists():
        raise SystemExit(
            f"refusing to overwrite {path}: {what} is write-once. It already exists, "
            "which means either this selection has already been locked, or an earlier "
            "attempt left it behind.\nA lock that can be rewritten cannot support "
            "'the method was chosen before the test set was touched'. Use a new --tag, "
            "or delete the file deliberately if you are certain no test evaluation has "
            "been run against it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            # lost the race against a concurrent selection; the first writer wins
            raise SystemExit(
                f"refusing to overwrite {path}: {what} is write-once and was created "
                "concurrently by another process.")
    finally:
        os.unlink(tmp)
    return path
