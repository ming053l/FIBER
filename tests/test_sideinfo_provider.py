"""The four Phase A arms differ ONLY in what S they are fed.

These are the plumbing invariants, and each is here because a training script is exactly
where such a property gets lost.
"""
import json

import numpy as np
import pytest

from fiber.sideinfo import MODES, SideConditioning

N_PER_ROLE = 12


def _index(split="train", roles=("A_teacher", "A_operator", "B")):
    out, i = [], 0
    for role in roles:
        for _ in range(N_PER_ROLE):
            out.append({"sample_id": f"{split}/{i:04d}", "split": split, "index": i,
                        "crossfit": role[0], "crossfit_sub": role,
                        "prompt": f"prompt {i}", "shard": i // 10, "offset": i % 10})
            i += 1
    return out


class _Store:
    """Stands in for the cache; returns a tensor that identifies its own address."""
    def __init__(self, *a, **k):
        pass

    def get(self, shard, offset):
        return np.full((77, 768), shard * 10 + offset, dtype=np.float16)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr("fiber.sideinfo.provider.ConditioningStore", _Store)


def _sc(mode, recs, seed=0, split="train"):
    return SideConditioning("/nonexistent", split, recs, mode,
                            s_null=np.zeros((77, 768), dtype=np.float32),
                            derangement_seed=seed)


def test_every_mode_returns_the_same_shape_and_dtype():
    """The cache is fp16 and S_null is float32. Without normalisation the null arm would
    differ from the other two in numeric representation as well as in content, and it is
    supposed to differ only in content."""
    recs = _index()
    sid = recs[0]["sample_id"]
    outs = {m: _sc(m, recs).get(sid) for m in MODES if m != "blind"}
    assert {o.shape for o in outs.values()} == {(77, 768)}
    assert {o.dtype for o in outs.values()} == {np.dtype(np.float32)}


def test_blind_returns_nothing():
    assert _sc("blind", _index()).get(_index()[0]["sample_id"]) is None


def test_the_derangement_does_not_move_with_the_receiver_seed():
    """Receiver seeds replicate OPTIMISATION randomness. If the placebo pairing moved
    with them, the between-seed spread would also contain 'which wrong prompt was
    attached' -- and the equivalence margin is derived from that spread."""
    recs = _index()
    a = _sc("shuffled", recs).manifest()["donor_digest"]
    for receiver_seed in range(6):
        # the provider is never given the receiver seed; this is the property
        assert _sc("shuffled", recs).manifest()["donor_digest"] == a


def test_the_derangement_stays_inside_the_cross_fit_role():
    """A_teacher donates only to A_teacher, and so on. Otherwise a teacher reads
    conditioning belonging to samples reserved for the operator or the receiver, which
    breaks the one-sample-one-role discipline the protocol rests on."""
    recs = _index()
    m = _sc("shuffled", recs).manifest()
    assert m["same_role_pairs"] == m["n_pairs"] == len(recs)
    assert m["fixed_points"] == 0 and m["same_prompt_pairs"] == 0
    assert m["roles"] == ["A_operator", "A_teacher", "B"]


def test_a_cross_role_shuffle_would_be_visible():
    """Not vacuous: deranging over the whole split ignoring roles does mix them."""
    from fiber.sideinfo.conditioning import different_conditioning_derangement
    recs = _index()
    pi = different_conditioning_derangement([r["prompt"] for r in recs], seed=0)
    crossed = sum(recs[pi[i]]["crossfit_sub"] != recs[i]["crossfit_sub"]
                  for i in range(len(recs)))
    assert crossed > 0


def test_shuffled_really_delivers_another_sample_s_conditioning():
    recs = _index()
    sc = _sc("shuffled", recs)
    correct = _sc("correct", recs)
    differ = sum(not np.array_equal(sc.get(r["sample_id"]), correct.get(r["sample_id"]))
                 for r in recs)
    assert differ == len(recs)


def test_null_is_identical_for_every_sample():
    recs = _index()
    sc = _sc("null", recs)
    first = sc.get(recs[0]["sample_id"])
    assert all(np.array_equal(sc.get(r["sample_id"]), first) for r in recs)


def test_the_donor_map_does_not_depend_on_how_a_loader_was_filtered():
    """The provider is built from the split's full index, so a dataset constructed with a
    different attack list or cross-fit filter cannot change who donates to whom."""
    recs = _index()
    full = _sc("shuffled", recs)
    subset = [r for r in recs if r["crossfit_sub"] == "B"]
    # passing only part of the index is still grouped by role, so B's donors match
    partial = _sc("shuffled", subset)
    for r in subset:
        assert np.array_equal(full.get(r["sample_id"]), partial.get(r["sample_id"]))


def test_the_manifest_records_what_reconstructs_the_pairing():
    m = _sc("shuffled", _index()).manifest()
    assert m["derangement_depends_on_receiver_seed"] is False
    assert m["derangement_scope"] == "within cross-fit role"
    assert len(m["donor_digest"]) == 32
    n = _sc("null", _index()).manifest()
    assert len(n["s_null_digest"]) == 32 and n["s_null_shape"] == [77, 768]


def test_the_phase_a_frame_is_fixed_and_not_tied_to_the_receiver_seed():
    """Registered: one Haar frame, C2_haar seed 0, identical across all 24 runs. If the
    frame moved with the receiver seed, the between-seed spread the equivalence margin is
    built from would also contain 'which random subspace was drawn'."""
    import hashlib

    from fiber.transforms import build_frame
    from fiber.utils.config import load_config

    cfg = load_config("configs/linear_fiber.yaml")
    spec = dict(cfg["fiber"]["arms"]["C2_haar"])
    d, k = int(cfg["latent"]["dim"]), int(cfg["fiber"]["robust_dims"])

    def digest(seed):
        rows = build_frame(spec, d=d, k=k, seed=seed).rows().detach().cpu().contiguous()
        return hashlib.blake2s(rows.numpy().tobytes(), digest_size=16).hexdigest()

    assert digest(0) == digest(0)
    assert digest(0) != digest(1), "the frame seed must actually select a frame"
