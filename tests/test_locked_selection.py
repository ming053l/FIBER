"""P0-3: the test split must not be able to choose anything.

The pipeline has to be VAL -> selection.json -> LOCK -> TEST, with the test script
structurally unable to pick the family, the seed, k, the hyperparameters or the
reference. These tests build artifacts where the val winner and the test winner are
DIFFERENT arms and check that the locked one is what gets reported.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fiber.channels import ChannelBank
from fiber.utils.config import load_config

sys.path.insert(0, "scripts")
from select_method import _forbid_test, select  # noqa: E402

CFG = load_config("configs/linear_fiber.yaml")
BANK = ChannelBank(CFG)
N = 64
K = 16


def _write_run(dirpath: Path, arm: str, rtype: str, seed: int, val_ber: float, test_ber: float):
    """One arm run: per-sample BER arrays plus the json summary the scripts read."""
    rng = np.random.default_rng(abs(hash((arm, seed))) % 2**31)
    arrays, results = {}, {}
    for split, level in (("val", val_ber), ("test", test_ber)):
        results[split] = {}
        for a in BANK.eval:
            per = np.clip(level + rng.normal(0, 0.01, N), 0, 1)
            arrays[f"{split}|{a}|per_sample"] = per
            arrays[f"{split}|{a}|sample_ids"] = np.array([f"s{i:04d}" for i in range(N)])
            results[split][a] = {"sign_ber": float(per.mean()), "n": N}
    np.savez_compressed(dirpath / f"{arm}_k{K}_s{seed}.npz", **arrays)
    (dirpath / f"{arm}_k{K}_s{seed}.json").write_text(json.dumps({
        "arm": arm, "type": rtype, "k": K, "seed": seed, "tag": "unit",
        "results": results}))


@pytest.fixture
def crossed_runs(tmp_path):
    """D_spectral wins on VAL; E_learned wins on TEST. The gate must report D."""
    d = tmp_path / "results"
    d.mkdir()
    for s in (0, 1):
        _write_run(d, "C2_haar", "haar", s, val_ber=0.50, test_ber=0.50)
        _write_run(d, "D_spectral", "spectral_topk", s, val_ber=0.40, test_ber=0.46)
        _write_run(d, "E_learned", "householder", s, val_ber=0.45, test_ber=0.30)
    return d


def _run(script, *args):
    return subprocess.run([sys.executable, f"scripts/{script}", *args],
                          capture_output=True, text=True)


def test_selection_picks_the_val_winner(crossed_runs, tmp_path):
    sel = tmp_path / "selection.json"
    p = _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
             "--out", str(sel))
    assert p.returncode == 0, p.stderr
    blob = json.loads(sel.read_text())
    assert blob["selected"]["arm"] == "D_spectral"
    assert blob["split_used"] == "val"


def test_test_evaluation_reports_the_val_winner_not_the_test_winner(crossed_runs, tmp_path):
    """The whole point of P0-3. E_learned is much better on test (0.30 vs 0.46) and
    must still not be the reported method."""
    sel = tmp_path / "selection.json"
    assert _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
                "--out", str(sel)).returncode == 0
    out = tmp_path / "gate.md"
    p = _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
             "--selection", str(sel), "--split", "test", "--out", str(out))
    assert p.returncode == 0, p.stderr
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["selection"]["arm"] == "D_spectral"
    for cond in report["gate3a"]["conditions"].values():
        assert abs(cond["treatment"] - 0.46) < 0.02, "gate evaluated the test winner"


def test_test_evaluation_refuses_without_a_selection_artifact(crossed_runs, tmp_path):
    p = _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
             "--selection", str(tmp_path / "missing.json"), "--split", "test",
             "--out", str(tmp_path / "g.md"))
    assert p.returncode != 0
    assert "locked" in (p.stdout + p.stderr).lower()


def test_test_evaluation_refuses_to_run_on_val(crossed_runs, tmp_path):
    p = _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
             "--split", "val", "--out", str(tmp_path / "g.md"))
    assert p.returncode != 0
    assert "SELECTION split" in (p.stdout + p.stderr)


def test_selection_cannot_read_a_test_key():
    assert _forbid_test("val") == "val"
    for bad in ("test", "test_heldout_prompts"):
        with pytest.raises(RuntimeError, match="VAL-only"):
            _forbid_test(bad)


def test_selection_records_provenance(crossed_runs, tmp_path):
    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--out", str(sel))
    blob = json.loads(sel.read_text())
    for key in ("commit", "config_fingerprint", "written_at", "selection_rule", "locked"):
        assert key in blob
    assert blob["selected"]["k"] == K
    assert blob["random_reference"]["type"] == "haar"


def test_seeds_are_averaged_never_minimised(tmp_path):
    """A good seed and a bad seed must report their mean, not the good one."""
    d = tmp_path / "results"
    d.mkdir()
    _write_run(d, "C2_haar", "haar", 0, 0.50, 0.50)
    _write_run(d, "D_spectral", "spectral_topk", 0, 0.30, 0.30)
    _write_run(d, "D_spectral", "spectral_topk", 1, 0.50, 0.50)
    runs = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    chosen = select(runs, BANK.eval)
    scored = {c["arm"]: c["val_sign_ber"] for c in chosen["candidates"]}
    assert abs(scored["D_spectral"] - 0.40) < 0.01, "selection used the best seed"


def test_gate_denominator_is_the_haar_family_not_its_best_draw(crossed_runs, tmp_path):
    """Point 6 of the audit: no picking a lucky or unlucky Haar draw on test."""
    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--out", str(sel))
    out = tmp_path / "gate.md"
    _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--selection", str(sel), "--split", "test", "--out", str(out))
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["n_haar_draws"] == 2
    for cond in report["gate3a"]["conditions"].values():
        assert abs(cond["baseline"] - 0.50) < 0.02


def test_pilot_verdict_cannot_be_a_final_kill(crossed_runs, tmp_path):
    """Secondary cleanup C: a rehearsal may not close the scientific question."""
    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--out", str(sel))
    out = tmp_path / "gate.md"
    _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--selection", str(sel), "--split", "test", "--out", str(out))
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["gate3a"]["provisional"] is True
    assert report["gate3a"]["verdict"] in {"PROVISIONAL_PASS", "PROVISIONAL_FAIL", "INCONCLUSIVE"}


def test_hierarchical_interval_is_reported_beside_the_paired_one(crossed_runs, tmp_path):
    """Haar draws are themselves a sample; the claim is about E_Q[BER]."""
    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--out", str(sel))
    out = tmp_path / "gate.md"
    _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(crossed_runs),
         "--selection", str(sel), "--split", "test", "--out", str(out))
    report = json.loads(out.with_suffix(".json").read_text())
    for cond in report["gate3a"]["conditions"].values():
        assert len(cond["hierarchical_ci"]) == 2
        assert cond["hierarchical_ci"][0] <= cond["hierarchical_ci"][1]
