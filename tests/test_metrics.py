import numpy as np
import torch

from fiber.metrics import (gate3a_condition, gate3a_verdict, paired_bootstrap,
                           sign_ber, coord_pearson)
from fiber.utils.config import load_config

CFG = load_config("configs/linear_fiber.yaml")


def test_sign_ber_chance_level_is_one_half():
    torch.manual_seed(0)
    w = torch.randn(4000, 64)
    logits = torch.randn(4000, 64)          # an extractor that knows nothing
    assert abs(sign_ber(logits, w) - 0.5) < 0.02


def test_sign_ber_is_zero_for_a_perfect_extractor():
    w = torch.randn(100, 16)
    assert sign_ber(w * 10, w) == 0.0


def test_per_sample_reduction_is_the_bootstrap_unit():
    w = torch.randn(50, 8)
    per = sign_ber(torch.randn(50, 8), w, reduce="per_sample")
    assert per.shape == (50,)


def test_paired_bootstrap_detects_a_real_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.5, 1000)
    treat = base - 0.08 + rng.normal(0, 0.01, 1000)      # paired, consistent gain
    r = paired_bootstrap(base, treat, resamples=2000, seed=0)
    assert r.ci_low > 0 and abs(r.mean_delta - 0.08) < 0.01


def test_paired_bootstrap_rejects_noise():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.5, 1000)
    treat = rng.uniform(0.3, 0.5, 1000)
    r = paired_bootstrap(base, treat, resamples=2000, seed=0)
    assert r.ci_low < 0 < r.ci_high


def test_pairing_matters():
    """If the arms were compared unpaired the shared per-sample difficulty would
    swamp the effect. This is why Gate 3A is a paired statistic."""
    rng = np.random.default_rng(2)
    difficulty = rng.uniform(0.0, 0.5, 2000)
    base = difficulty + 0.02
    treat = difficulty
    paired = paired_bootstrap(base, treat, resamples=2000, seed=0)
    unpaired_gap = base.mean() - rng.permutation(treat).mean()
    assert paired.ci_low > 0
    assert paired.ci_high - paired.ci_low < 0.01      # pairing kills the shared variance
    assert abs(unpaired_gap - paired.mean_delta) < 1e-9


def test_small_absolute_gain_fails_gate_despite_large_relative_gain():
    """0.005 -> 0.004 is a 20% relative reduction and must NOT pass."""
    rng = np.random.default_rng(3)
    base = np.full(2000, 0.005) + rng.normal(0, 1e-4, 2000)
    treat = np.full(2000, 0.004) + rng.normal(0, 1e-4, 2000)
    cond = gate3a_condition(paired_bootstrap(base, treat, resamples=1000, seed=0), CFG["gate3a"])
    assert cond["relative_ok"] and cond["ci_ok"]
    assert not cond["absolute_ok"] and not cond["passed"]


def test_gate_verdict_requires_three_of_five_groups():
    def cond(passed, rel):
        return {"passed": passed, "relative_reduction": rel}
    groups = {f"g{i}": cond(i < 2, 0.3) for i in range(5)}
    assert gate3a_verdict(groups, CFG["gate3a"])["verdict"] == "INCONCLUSIVE"
    groups = {f"g{i}": cond(i < 3, 0.3) for i in range(5)}
    assert gate3a_verdict(groups, CFG["gate3a"])["verdict"] == "PASS"


def test_kill_verdict_only_when_derived_matches_random():
    groups = {f"g{i}": {"passed": False, "relative_reduction": 0.002} for i in range(5)}
    assert gate3a_verdict(groups, CFG["gate3a"])["verdict"] == "KILL"


def test_coord_pearson_is_one_for_a_perfect_readout():
    w = torch.randn(200, 8)
    assert torch.allclose(coord_pearson(w, w), torch.ones(8), atol=1e-5)
