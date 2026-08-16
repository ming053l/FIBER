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


# --------------------------------------------------------------------------
# P0-3.1: the lock must name EXACT runs. Identifying only (arm, k) means a run
# dropped into the results directory after selection silently joins the average.
# --------------------------------------------------------------------------
def _lock(runs_dir, tmp_path):
    sel = tmp_path / "selection.json"
    p = _run("select_method.py", "--tag", "unit", "--results-dir", str(runs_dir),
             "--out", str(sel))
    assert p.returncode == 0, p.stderr
    return sel


def _gate(runs_dir, sel, tmp_path, name="gate.md", config=None):
    out = tmp_path / name
    args = ["--tag", "unit", "--results-dir", str(runs_dir), "--selection", str(sel),
            "--split", "test", "--out", str(out)]
    if config:
        args += ["--config", str(config)]
    return _run("eval_coordinates.py", *args), out


def test_a_seed_added_after_the_lock_cannot_change_the_result(crossed_runs, tmp_path):
    """The concrete hole: lock on seeds [0,1], then drop in a spectacular seed 2."""
    sel = _lock(crossed_runs, tmp_path)
    p, out = _gate(crossed_runs, sel, tmp_path, "before.md")
    assert p.returncode == 0, p.stderr
    before = out.with_suffix(".json").read_text()

    _write_run(crossed_runs, "D_spectral", "spectral_topk", 2, val_ber=0.10, test_ber=0.05)
    _write_run(crossed_runs, "C2_haar", "haar", 2, val_ber=0.90, test_ber=0.90)
    p2, out2 = _gate(crossed_runs, sel, tmp_path, "after.md")
    assert p2.returncode == 0, p2.stderr
    assert json.loads(before) == json.loads(out2.with_suffix(".json").read_text()), \
        "a post-lock run changed the locked result"


def test_locked_run_content_change_is_detected(crossed_runs, tmp_path):
    sel = _lock(crossed_runs, tmp_path)
    blob = json.loads(sel.read_text())
    stem = blob["selected_runs"][0]["stem"]
    _write_run(crossed_runs, "D_spectral", "spectral_topk",
               int(stem.rsplit("_s", 1)[1]), val_ber=0.11, test_ber=0.11)
    p, _ = _gate(crossed_runs, sel, tmp_path, "tampered.md")
    assert p.returncode != 0
    assert "changed since the lock" in (p.stdout + p.stderr)


def test_protocol_config_change_after_the_lock_hard_fails(crossed_runs, tmp_path):
    """A lock that survives a protocol edit is not a lock."""
    sel = _lock(crossed_runs, tmp_path)
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    for name in ("linear_fiber.yaml", "sd15.yaml", "channels.yaml"):
        (cfgdir / name).write_text(Path("configs") / name and (Path("configs") / name).read_text())
    leaf = cfgdir / "linear_fiber.yaml"
    leaf.write_text(leaf.read_text().replace("robust_dims: 64", "robust_dims: 128"))
    p, _ = _gate(crossed_runs, sel, tmp_path, "cfg.md", config=leaf)
    assert p.returncode != 0
    assert "protocol config changed" in (p.stdout + p.stderr)


def test_same_arm_and_k_with_different_hyperparameters_are_separate_candidates(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    _write_run(d, "C2_haar", "haar", 0, 0.50, 0.50)
    for seed, (nref, ber) in enumerate([(128, 0.42), (256, 0.30)]):
        _write_run(d, "E_learned", "householder", seed, val_ber=ber, test_ber=ber)
        jf = d / f"E_learned_k{K}_s{seed}.json"
        blob = json.loads(jf.read_text())
        blob["arm_spec"] = {"type": "householder", "num_reflectors": nref}
        blob["hyperparameters_fingerprint"] = f"hp{nref}"
        jf.write_text(json.dumps(blob))

    sel = _lock(d, tmp_path)
    blob = json.loads(sel.read_text())
    hp_seen = {c["hyperparameters_fingerprint"] for c in blob["candidates"]
               if c["arm"] == "E_learned"}
    assert hp_seen == {"hp128", "hp256"}, "hyperparameter variants were averaged together"
    assert blob["selected"]["hyperparameters_fingerprint"] == "hp256"
    assert len(blob["selected_runs"]) == 1
    assert blob["selected_runs"][0]["stem"] == f"E_learned_k{K}_s1"


# --------------------------------------------------------------------------
# P0-7.1: the basis arms must not leak into Gate 3A, and the P0-7 comparator
# must be the single spectral subspace the rotations actually sit in.
# --------------------------------------------------------------------------
from select_method import BASIS_ONLY_TYPES, DERIVED_TYPES  # noqa: E402


def _write_rot_run(dirpath, arm, rtype, seed, val_ber, test_ber, base_seed=0,
                   receiver_seed=None):
    _write_run(dirpath, arm, rtype, seed, val_ber, test_ber)
    jf = dirpath / f"{arm}_k{K}_s{seed}.json"
    blob = json.loads(jf.read_text())
    blob["base_seed"] = base_seed
    blob["basis_seed"] = seed
    blob["receiver_seed"] = seed if receiver_seed is None else receiver_seed
    jf.write_text(json.dumps(blob))


def test_a_basis_arm_can_never_win_gate_3a(tmp_path):
    """D3 only rotates inside D1's subspace, so span and D_cert are identical to D1's.
    Letting it win Gate 3A would fold coding-basis optimisation into a claim about
    subspace anisotropy."""
    assert "rotated_learned" not in DERIVED_TYPES
    assert BASIS_ONLY_TYPES == {"rotated_random", "rotated_learned"}

    d = tmp_path / "results"
    d.mkdir()
    _write_run(d, "C2_haar", "haar", 0, 0.50, 0.50)
    _write_run(d, "D_spectral", "spectral_topk", 0, 0.45, 0.45)
    _write_rot_run(d, "D3_rot_learn", "rotated_learned", 0, val_ber=0.05, test_ber=0.05)

    sel = tmp_path / "selection.json"
    assert _run("select_method.py", "--tag", "unit", "--results-dir", str(d),
                "--out", str(sel)).returncode == 0
    blob = json.loads(sel.read_text())
    assert blob["selected"]["arm"] == "D_spectral", "a basis arm won the scientific gate"


def test_p0_7_comparator_is_the_pinned_spectral_seed_not_an_average(tmp_path):
    """Adversarial: spectral seed 1 is far better than seed 0, but the rotations sit in
    seed 0's subspace, so the basis table must report seed 0's BER."""
    d = tmp_path / "results"
    d.mkdir()
    _write_run(d, "C2_haar", "haar", 0, 0.50, 0.50)
    _write_run(d, "D_spectral", "spectral_topk", 0, 0.40, 0.40)
    _write_run(d, "D_spectral", "spectral_topk", 1, 0.10, 0.10)
    for s in (0, 1):
        _write_rot_run(d, "D2_rot_rand", "rotated_random", s, 0.52, 0.52, base_seed=0)

    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(d), "--out", str(sel))
    out = tmp_path / "gate.md"
    p = _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(d),
             "--selection", str(sel), "--split", "test", "--out", str(out))
    assert p.returncode == 0, p.stderr
    basis = json.loads(out.with_suffix(".json").read_text())["p0_7_basis"]
    assert basis["base_seed"] == 0
    d1 = basis["arms"]["D_spectral"]
    assert abs(d1["mean_sign_ber"] - 0.40) < 0.02, \
        "D1 averaged several different subspaces while claiming one was held fixed"
    assert d1["n_basis_draws"] == 1


def test_basis_spread_marginalises_the_receiver_seed(tmp_path):
    """Two basis draws, each with two receiver seeds. The reported spread must be
    across BASES after averaging receivers, not the pooled standard deviation."""
    d = tmp_path / "results"
    d.mkdir()
    _write_run(d, "C2_haar", "haar", 0, 0.50, 0.50)
    _write_run(d, "D_spectral", "spectral_topk", 0, 0.40, 0.40)
    for s, level in ((0, 0.40), (1, 0.50)):
        for r, jitter in ((0, -0.05), (1, +0.05)):
            _write_run(d, f"D2_rot_rand_b{s}", "rotated_random", r,
                       level + jitter, level + jitter)
            jf = d / f"D2_rot_rand_b{s}_k{K}_s{r}.json"
            blob = json.loads(jf.read_text())
            blob.update({"arm": "D2_rot_rand", "base_seed": 0,
                         "basis_seed": s, "receiver_seed": r})
            jf.write_text(json.dumps(blob))

    sel = tmp_path / "selection.json"
    _run("select_method.py", "--tag", "unit", "--results-dir", str(d), "--out", str(sel))
    out = tmp_path / "gate.md"
    p = _run("eval_coordinates.py", "--tag", "unit", "--results-dir", str(d),
             "--selection", str(sel), "--split", "test", "--out", str(out))
    assert p.returncode == 0, p.stderr
    d2 = json.loads(out.with_suffix(".json").read_text())["p0_7_basis"]["arms"]["D2_rot_rand"]
    assert d2["n_basis_draws"] == 2 and d2["receiver_seeds"] == [0, 1]
    # receiver jitter cancels within a basis: 0.40 and 0.50 -> spread 0.05, not 0.0707
    assert abs(d2["basis_spread"] - 0.05) < 0.005, \
        "the spread still contains extractor training noise"
