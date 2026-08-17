"""A whole experiment, start to finish, on synthetic pixels.

Every other test in this suite checks one mechanism against a hand-built fixture. This
one runs the actual chain --

    cache -> spectrum -> train -> select -> LOCK -> materialise test -> locked eval -> gate

-- as the real scripts, in a real git repository, with real on-disk state, and then
attacks the finished experiment. Unit fixtures cannot catch the failures this is for:
the three runtime defects that only appeared when the triage was executed (a KeyError
after all outputs were written, a variable used before assignment on the post-lock path,
and provenance deadlocking the pipeline against its own artifacts) were all invisible to
tests that never ran the chain.

It costs about 40 seconds because it is a real run: ~250 images at 64x64, d=256, k=8,
2 epochs, on the CPU. The pixels come from cache_native_dataset.py --synthetic and carry
partial information about z, so the numbers are non-degenerate -- but they are a plumbing
fixture and nothing here is evidence about the diffusion channel.

The repository under test is a SNAPSHOT of the working tree, committed in a throwaway
repo. Not a git worktree at HEAD: that would test the last commit rather than the code
being edited, and this file's whole purpose is to run what is actually there.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TAG = "dry"


def _sh(*args, cwd, check=True, env=None):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if check and p.returncode != 0:
        raise AssertionError(f"{' '.join(args[:3])} failed ({p.returncode})\n"
                             f"--- stdout ---\n{p.stdout[-3000:]}\n"
                             f"--- stderr ---\n{p.stderr[-3000:]}")
    return p


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Snapshot -> commit -> run the whole chain. Returns (repo, experiment dir)."""
    base = tmp_path_factory.mktemp("e2e")
    repo, exp = base / "repo", base / "exp"
    repo.mkdir()
    for d in ("src", "scripts", "configs"):
        shutil.copytree(REPO / d, repo / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    _sh("git", "init", "-q", cwd=repo)
    _sh("git", "add", "-A", cwd=repo)
    _sh("git", "-c", "user.email=e2e@test", "-c", "user.name=e2e",
        "commit", "-qm", "dry-run snapshot", cwd=repo)

    env = {"PYTHONPATH": str(repo / "src")}
    def run(script, *a, check=True):
        return _sh(sys.executable, "-W", "ignore", f"scripts/{script}", *a,
                   cwd=repo, check=check, env=env)

    cfg = str(exp / "dryrun.yaml")
    run("make_dryrun_config.py", "--root", str(exp))
    run("cache_native_dataset.py", "--config", cfg, "--pilot", "--synthetic",
        "--splits", "train", "val", "--shard-size", "64")
    run("fit_observability_spectrum.py", "--config", cfg, "--tag", TAG,
        "--cache-tag", "pilot", "--seed", "0", "--device", "cpu", "--k", "8")
    for arm, seed in (("C2_haar", "0"), ("C2_haar", "1"), ("D_spectral", "0")):
        run("train_coordinates.py", "--config", cfg, "--tag", TAG, "--cache-tag", "pilot",
            "--arm", arm, "--seed", seed, "--k", "8", "--device", "cpu")
    run("select_method.py", "--config", cfg, "--tag", TAG)
    run("cache_native_dataset.py", "--config", cfg, "--pilot", "--synthetic",
        "--splits", "test", "test_heldout_prompts", "--shard-size", "64",
        "--post-lock", str(exp / "reports" / f"selection_{TAG}.json"))
    run("evaluate_locked.py", "--config", cfg, "--tag", TAG, "--device", "cpu")
    run("eval_coordinates.py", "--config", cfg, "--tag", TAG, "--split", "test")
    return repo, exp, env


def _run(chain, script, *a, check=False):
    repo, exp, env = chain
    return _sh(sys.executable, "-W", "ignore", f"scripts/{script}",
               "--config", str(exp / "dryrun.yaml"), *a, cwd=repo, check=check, env=env)


# --- the chain itself ------------------------------------------------------------

def test_the_whole_chain_produces_a_gate_report(chain):
    _, exp, _ = chain
    report = json.loads((exp / "reports" / f"gate3_{TAG}_test.json").read_text())
    assert report["selection"]["arm"] == "D_spectral"
    g = report["gate3a"]
    assert g["verdict"] in {"PASS", "PROVISIONAL_FAIL", "FAIL", "KILL"}
    assert g["n_required"] == 3 and 0 <= g["n_passed"] <= 5
    assert set(g["conditions"]) == {"identity", "jpeg", "resize", "noise", "blur"}
    # a pilot-sized run can never be a final kill, only provisional
    assert g["provisional"] is True


def test_test_pixels_did_not_exist_before_the_lock(chain):
    """The strongest form of B1: not 'the test set was never read' but 'no test sample
    existed'. Every shard carries the lock's hash, written when the pixels were made."""
    _, exp, _ = chain
    tman = json.loads((exp / "reports" / f"test_eval_{TAG}.json").read_text())
    assert tman["test_cache_post_lock"] is True
    cache = json.loads((exp / "cache" / "pilot" / "test_cache_manifest.json").read_text())
    assert cache["generated_after_lock"] is True
    assert cache["shards_not_bound_to_this_lock"] == []


def test_synthetic_pixels_are_stamped_everywhere(chain):
    """A dry-run artifact must never be readable as evidence."""
    _, exp, _ = chain
    man = json.loads((exp / "cache" / "pilot" / "manifest.json").read_text())
    assert man["synthetic_pixels"] is True
    for marker in (exp / "cache" / "pilot").rglob("*.done"):
        assert json.loads(marker.read_text())["synthetic_pixels"] is True


# --- attacks on the finished experiment ------------------------------------------

def test_attack_second_selection_on_a_locked_tag(chain):
    _, exp, _ = chain
    lock = exp / "reports" / f"selection_{TAG}.json"
    before = lock.read_bytes()
    p = _run(chain, "select_method.py", "--tag", TAG)
    assert p.returncode != 0
    assert lock.read_bytes() == before


def test_attack_selection_from_a_dirty_tree(chain):
    repo, _, _ = chain
    victim = repo / "scripts" / "train_coordinates.py"
    victim.write_text(victim.read_text() + "\n# edit\n")
    try:
        p = _run(chain, "select_method.py", "--tag", "dry_dirty")
        assert p.returncode != 0
        assert "dirty working tree" in p.stdout + p.stderr
    finally:
        _sh("git", "checkout", "--", "scripts/train_coordinates.py", cwd=repo)


def test_attack_locked_evaluation_from_a_dirty_tree(chain):
    repo, exp, _ = chain
    victim = repo / "scripts" / "train_coordinates.py"
    victim.write_text(victim.read_text() + "\n# edit\n")
    try:
        p = _run(chain, "evaluate_locked.py", "--tag", TAG, "--device", "cpu")
        assert p.returncode != 0
        assert "dirty working tree" in p.stdout + p.stderr
    finally:
        _sh("git", "checkout", "--", "scripts/train_coordinates.py", cwd=repo)


def test_attack_swapped_cache_namespace(chain):
    p = _run(chain, "evaluate_locked.py", "--tag", TAG, "--cache-tag", "other",
             "--device", "cpu")
    assert p.returncode != 0
    assert "different image cache" in p.stdout + p.stderr


def test_the_official_test_evaluation_is_also_write_once(chain):
    p = _run(chain, "evaluate_locked.py", "--tag", TAG, "--device", "cpu")
    assert p.returncode != 0
    assert "write-once" in p.stdout + p.stderr


def test_attack_missing_checkpoint_is_not_silently_skipped(chain):
    """The dangerous failure is not a crash, it is evaluating two runs where the lock
    names three and reporting the result as complete."""
    _, exp, _ = chain
    tman = exp / "reports" / f"test_eval_{TAG}.json"
    ckpt = exp / "results" / TAG / "D_spectral_k8_s0_rxresnet18_r0_scgate_extractor.pt"
    saved, hidden = tman.read_bytes(), ckpt.read_bytes()
    tman.unlink(); ckpt.unlink()
    try:
        p = _run(chain, "evaluate_locked.py", "--tag", TAG, "--device", "cpu")
        assert p.returncode != 0
        assert "is missing" in p.stdout + p.stderr
        assert not tman.exists(), "a partial evaluation was still written"
    finally:
        ckpt.write_bytes(hidden); tman.write_bytes(saved)


def test_attack_lock_edited_before_the_locked_evaluation(chain):
    """The one this dry run actually found. The test pixels are bound to the lock's
    bytes; if the lock no longer hashes to that, it changed after the test set was
    materialised. That used to exit 0 and merely record test_cache_post_lock = False."""
    _, exp, _ = chain
    lock = exp / "reports" / f"selection_{TAG}.json"
    tman = exp / "reports" / f"test_eval_{TAG}.json"
    saved_lock, saved_tman = lock.read_bytes(), tman.read_bytes()
    blob = json.loads(saved_lock)
    blob["note_added_by_attacker"] = "tamper"
    lock.write_text(json.dumps(blob, indent=2))
    tman.unlink()
    try:
        p = _run(chain, "evaluate_locked.py", "--tag", TAG, "--device", "cpu")
        assert p.returncode != 0
        assert "changed after the test set was generated" in p.stdout + p.stderr
    finally:
        lock.write_bytes(saved_lock); tman.write_bytes(saved_tman)


def test_attack_lock_edited_after_the_locked_evaluation(chain):
    """Same edit, later: the gate compares the evaluation's recorded selection_sha
    against the lock it is handed."""
    _, exp, _ = chain
    lock = exp / "reports" / f"selection_{TAG}.json"
    saved = lock.read_bytes()
    blob = json.loads(saved)
    blob["selected"]["arm"] = "C2_haar"
    lock.write_text(json.dumps(blob, indent=2))
    try:
        p = _run(chain, "eval_coordinates.py", "--tag", TAG, "--split", "test")
        assert p.returncode != 0
        assert "different selection artifact" in p.stdout + p.stderr
    finally:
        lock.write_bytes(saved)
