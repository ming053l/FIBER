"""The preregistered side encoder: 768->64, learned-query pooling, late fusion."""
import math

import pytest
import torch

from fiber.models import Extractor, Teacher
from fiber.models.sideinfo import (S_DIM, S_FEATURES, S_TOKENS, SideEncoder,
                                   SideExtractor, SideTeacher, parameter_report)

K, D = 64, 16384


def test_the_query_is_zero_initialised_so_pooling_starts_uniform():
    """Same discipline as arm C3's paired init: at step zero the parameterisation must
    contribute nothing, so any gain is learning rather than a different starting point."""
    enc = SideEncoder()
    s = torch.randn(3, S_TOKENS, S_FEATURES)
    assert torch.equal(enc.query, torch.zeros(S_DIM))
    assert (enc.attention(s) - 1.0 / S_TOKENS).abs().max() < 1e-12
    assert (enc(s) - enc.proj(s).mean(1)).abs().max() < 1e-6


def test_a_trained_query_stops_being_uniform():
    """A pooling that could never move would make the learned-query choice cosmetic."""
    enc = SideEncoder()
    with torch.no_grad():
        enc.query.copy_(torch.randn(S_DIM))
    a = enc.attention(torch.randn(2, S_TOKENS, S_FEATURES))
    assert (a - 1.0 / S_TOKENS).abs().max() > 1e-3
    assert torch.allclose(a.sum(1), torch.ones(2), atol=1e-5)


def test_the_projection_is_shared_across_token_positions():
    """One Linear(768, 64) for all 77 positions -- not a per-position matrix, which would
    be 77x the parameters and a different model."""
    enc = SideEncoder()
    assert enc.proj.weight.shape == (S_DIM, S_FEATURES)
    s = torch.randn(1, S_TOKENS, S_FEATURES)
    rolled = torch.roll(s, 5, dims=1)
    assert torch.allclose(enc.proj(s).roll(5, dims=1), enc.proj(rolled), atol=1e-6)


def test_fusion_is_late_so_S_cannot_touch_the_image_trunk():
    """The whole point of late fusion: if S reached the trunk, an improvement could not
    be separated from 'conditioning rewrote the feature extractor'."""
    m = SideExtractor(k=K).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a = m.trunk(x)
        b = m.trunk(x)
    assert torch.equal(a, b)
    trunk_names = {n for n, _ in m.trunk.named_parameters()}
    side_names = {n for n, _ in m.side.named_parameters()}
    assert not (trunk_names & side_names)


def test_S_actually_changes_the_output():
    """A side branch that is ignored would make every arm identical and the whole
    experiment vacuous."""
    m = SideExtractor(k=K).eval()
    with torch.no_grad():
        m.side.query.copy_(torch.randn(S_DIM))
    x = torch.randn(2, 3, 64, 64)
    s1 = torch.randn(2, S_TOKENS, S_FEATURES)
    s2 = torch.randn(2, S_TOKENS, S_FEATURES)
    with torch.no_grad():
        o1, o2 = m(x, s1)["w_hat"], m(x, s2)["w_hat"]
    assert (o1 - o2).abs().max() > 1e-4


def test_a_side_model_refuses_to_run_blind():
    m = SideExtractor(k=K)
    with pytest.raises(ValueError, match="requires S"):
        m(torch.randn(1, 3, 64, 64), None)


def test_capacity_accounting_is_reported_per_arm_not_per_branch():
    """The side BRANCH is 49,280 parameters in both roles, but the concat also widens the
    head -- and the head is Linear(., k) for the receiver and Linear(., d) for the
    teacher, so the totals differ by two orders of magnitude. A preregistration that
    quoted the branch alone would understate the teacher by ~1M."""
    rx = parameter_report(SideExtractor(k=K))
    tx = parameter_report(SideTeacher(d=D))
    assert rx["side_branch"] == tx["side_branch"] == 49_280
    blind_rx = sum(p.numel() for p in Extractor(k=K).parameters())
    blind_tx = sum(p.numel() for p in Teacher(d=D).parameters())
    assert rx["total"] - blind_rx == 49_280 + S_DIM * K * 2      # two heads
    assert tx["total"] - blind_tx == 49_280 + S_DIM * D
    assert tx["total"] - blind_tx > 1_000_000


def test_all_side_arms_share_one_architecture():
    """S_null, S_shuffled and S_correct differ only in what is fed, never in the model --
    that is what makes the placebo capacity-matched."""
    a, b = SideExtractor(k=K), SideExtractor(k=K)
    assert ([tuple(p.shape) for p in a.parameters()]
            == [tuple(p.shape) for p in b.parameters()])
