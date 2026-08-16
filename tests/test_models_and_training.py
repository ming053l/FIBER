"""Model / loop contracts that do not need a GPU or a cache."""
import pytest
import torch
import torch.nn.functional as F

from fiber.models import Extractor, Teacher
from fiber.training.loops import TrainConfig, _lr_at, head_losses
from fiber.utils.config import load_config

CFG = load_config("configs/linear_fiber.yaml")


def test_extractor_refuses_imagenet_weights():
    """A pretrained trunk is a semantic shortcut: it would let the extractor read
    content instead of coordinates (linear_fiber.yaml)."""
    with pytest.raises(ValueError, match="forbidden"):
        Extractor(k=8, pretrained=True)
    assert CFG["extractor"]["pretrained"] is False


def test_extractor_emits_both_heads_at_the_right_width():
    m = Extractor(k=16)
    out = m(torch.randn(2, 3, 64, 64))
    assert out["w_hat"].shape == (2, 16) and out["sign_logits"].shape == (2, 16)


def test_level1_uses_gaussian_heads_not_circular():
    """PLAN.md §5.2: W = QZ ~ N(0,I) is not a torus variable; a circular head
    would alias W and W+1. Circular heads are Phase 4+."""
    heads = CFG["extractor"]["heads"]
    assert heads["circular"]["enabled"] is False
    assert heads["sign"]["enabled"] and heads["sign"]["loss"] == "bce"
    assert heads["regression"]["loss"] == "mse"


def test_teacher_loss_is_mse():
    a, b = torch.randn(4, 32), torch.randn(4, 32)
    assert torch.allclose(Teacher.loss(a, b), F.mse_loss(a, b))
    assert CFG["spectrum"]["teacher_loss"] == "mse"


def test_teacher_predicts_the_full_latent():
    m = Teacher(d=1024)
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 1024)


def test_head_losses_vanish_for_a_perfect_readout():
    cfg = TrainConfig()
    w = torch.randn(8, 16)
    out = {"w_hat": w.clone(), "sign_logits": torch.sign(w) * 20}
    loss, parts = head_losses(out, w, cfg)
    assert parts["mse"] == 0 and float(parts["bce"]) < 1e-3


def test_head_weights_come_from_the_config():
    cfg = TrainConfig.from_config(CFG)
    assert cfg.w_regression == 1.0 and cfg.w_sign == 1.0
    assert cfg.batch_size == CFG["training"]["batch_size"]
    assert cfg.epochs == CFG["training"]["epochs"]


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainConfig(lr=1e-3, warmup_steps=100)
    assert _lr_at(0, 1000, cfg) < _lr_at(50, 1000, cfg) < _lr_at(99, 1000, cfg)
    assert abs(_lr_at(99, 1000, cfg) - cfg.lr) < 1e-4
    assert _lr_at(999, 1000, cfg) < 1e-5


def test_config_pins_the_receiver_protocol():
    """Rule D: the receiver observes Y only. DDIM inversion is a prompt-assisted
    REFERENCE, not a baseline (PLAN.md §5.3)."""
    tm = CFG["threat_model"]
    assert tm["receiver_sees"] == ["received_image"]
    assert tm["prompt_is_public"] is False
    assert tm["ddim_inversion_reference"]["role"] == "capacity_diagnostic_only"
