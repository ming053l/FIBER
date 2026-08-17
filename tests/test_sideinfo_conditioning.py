"""S_null and the different-conditioning derangement, as preregistered.

Phase A machinery. The derangement tests concentrate on the case with REPEATED
conditionings, because that is the only case where the guard does anything -- and today's
pilot index has 2000 distinct prompts for 2000 samples, so a test that only used real
data would pass with the guard deleted.
"""
import numpy as np
import pytest

from fiber.sideinfo import different_conditioning_derangement, train_mean_conditioning
from fiber.sideinfo.conditioning import derangement_manifest


def _prompts(groups):
    """groups={"a":3,"b":2} -> ['a','a','a','b','b'] shuffled deterministically."""
    out = [k for k, n in groups.items() for _ in range(n)]
    np.random.default_rng(0).shuffle(out)
    return out


def test_no_sample_receives_its_own_conditioning():
    pi = different_conditioning_derangement([f"p{i}" for i in range(50)], seed=0)
    assert (pi != np.arange(50)).all()


def test_no_sample_receives_an_IDENTICAL_conditioning_from_another_sample():
    """The reason the guard exists. pi(i) != i is satisfied by a plain derangement while
    S_pi(i) == S_i for every repeated prompt, so the placebo would be handed correct side
    information for exactly those samples."""
    p = _prompts({"a": 8, "b": 7, "c": 9, "d": 6})
    pi = different_conditioning_derangement(p, seed=0)
    assert all(p[pi[i]] != p[i] for i in range(len(p)))


def test_a_plain_derangement_would_NOT_satisfy_that():
    """Demonstrates the failure the guard prevents, so the test above is not vacuous."""
    rng = np.random.default_rng(0)
    p = _prompts({"a": 8, "b": 7, "c": 9, "d": 6})
    n = len(p)
    for _ in range(200):
        pi = rng.permutation(n)
        if (pi != np.arange(n)).all():
            break
    leaks = sum(p[pi[i]] == p[i] for i in range(n))
    assert leaks > 0, "a plain derangement happened not to leak; the point stands"


def test_it_refuses_when_no_valid_derangement_exists():
    """One conditioning covering more than half the split. A silent fallback to an
    ordinary shuffle here would leak on the majority of samples."""
    with pytest.raises(ValueError, match="no different-conditioning derangement"):
        different_conditioning_derangement(_prompts({"a": 11, "b": 4, "c": 5}), seed=0)


def test_the_boundary_case_of_exactly_half_is_allowed():
    p = _prompts({"a": 10, "b": 6, "c": 4})
    pi = different_conditioning_derangement(p, seed=0)
    assert all(p[pi[i]] != p[i] for i in range(len(p)))


def test_it_is_a_permutation_and_deterministic_given_the_seed():
    p = _prompts({"a": 5, "b": 6, "c": 7})
    a = different_conditioning_derangement(p, seed=3)
    b = different_conditioning_derangement(p, seed=3)
    c = different_conditioning_derangement(p, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert sorted(a.tolist()) == list(range(len(p)))


def test_the_manifest_records_what_an_auditor_needs():
    p = _prompts({"a": 5, "b": 6, "c": 7})
    pi = different_conditioning_derangement(p, seed=1)
    m = derangement_manifest(p, pi, 1)
    assert m["fixed_points"] == 0 and m["same_conditioning_pairs"] == 0
    assert m["distinct_conditionings"] == 3 and m["largest_group"] == 7
    assert len(m["permutation_digest"]) == 32


def test_S_null_keeps_the_tensor_shape(tmp_path):
    """Per token position and per feature -- never pooled, or the side encoder could tell
    S_null from a real S structurally and the control would stop being capacity-matched."""
    d = tmp_path / "conditioning" / "train"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((7, 5, 4)).astype(np.float16) for _ in range(3)]
    for i, b in enumerate(blocks):
        np.save(d / f"cond_{i:05d}.npy", b)
    got = train_mean_conditioning(tmp_path)
    assert got.shape == (5, 4) and got.dtype == np.float32
    want = np.concatenate(blocks).astype(np.float64).mean(0)
    assert np.abs(got - want).max() < 1e-5


def test_S_null_is_accumulated_in_float64(tmp_path):
    """A float16 running sum over thousands of tensors loses the mean badly, and S_null
    is a constant reused for every sample, so its error is systematic."""
    d = tmp_path / "conditioning" / "train"
    d.mkdir(parents=True)
    block = np.full((4000, 2, 2), 0.30, dtype=np.float16)
    np.save(d / "cond_00000.npy", block)
    got = train_mean_conditioning(tmp_path)
    assert np.abs(got - np.float16(0.30).astype(np.float64)).max() < 1e-6
    naive = block.sum(axis=0, dtype=np.float16) / block.shape[0]
    assert np.abs(naive - 0.30).max() > 1e-3, "float16 accumulation should visibly drift"
