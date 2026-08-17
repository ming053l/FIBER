"""Model / loop contracts that do not need a GPU or a cache."""
import pytest
import torch
import torch.nn.functional as F

from fiber.models import Extractor, Teacher
from fiber.training.loops import TrainConfig, _lr_at, head_losses
from fiber.utils.config import load_config

import sys
from pathlib import Path

from fiber.channels import ChannelBank

sys.path.insert(0, "scripts")

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


# --------------------------------------------------------------------------
# P0-4: the learned frame is trained by MSE, and the code says so.
# --------------------------------------------------------------------------
def test_hard_sign_target_severs_the_gradient_to_the_frame():
    """`(w > 0)` returns a bool tensor, so the autograd graph ends there: the frame's
    grad is None, not small. Any claim that the learned arm optimises sign BER would
    therefore be false."""
    from fiber.transforms import HouseholderFrame

    for objective in ("bce", "mse"):
        frame = HouseholderFrame(256, 16, num_reflectors=8, seed=0)
        w = frame.project(torch.randn(32, 256))
        pred = torch.randn(32, 16, requires_grad=True)
        loss = (F.binary_cross_entropy_with_logits(pred, (w > 0).float())
                if objective == "bce" else F.mse_loss(pred, w))
        loss.backward()
        if objective == "bce":
            assert frame.V.grad is None
        else:
            assert frame.V.grad is not None and float(frame.V.grad.abs().max()) > 0


def test_discovery_refuses_a_sign_weight():
    """Guards the interpretation against regression: joint discovery with w_sign > 0
    would look like sign-BER optimisation while being MSE-only."""
    from fiber.channels import ChannelBank
    from fiber.training.loops import TrainConfig, train_extractor
    from fiber.transforms import HouseholderFrame

    cfg = TrainConfig(w_sign=1.0, epochs=1)
    with pytest.raises(ValueError, match="no gradient"):
        train_extractor(HouseholderFrame(256, 8, num_reflectors=4), "/nonexistent",
                        ChannelBank(CFG), cfg, learn_frame=True, device="cpu")


def test_config_records_the_discovery_objective():
    assert CFG["training"]["discovery_objective"] == "mse"
    assert CFG["training"]["frame_weight_decay"] == 0.0
    assert CFG["fiber"]["arms"]["E_learned"]["base"] == "haar"
    assert CFG["fiber"]["arms"]["E_learned"]["paired_init"] is True


# --------------------------------------------------------------------------
# P0-5: the decoder architecture must not be what defines the geometry.
# --------------------------------------------------------------------------
def _top_512_energy(model, image_size, n=700):
    """Fraction of the centered output variance carried by the first 512 principal
    directions. Stated as energy rather than as `matrix_rank`, because the forward pass
    runs in float32: the mathematically-zero singular values come back around 1e-6 and
    a rank count then reports 700 instead of 512."""
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(n, 3, *image_size))
    s = torch.linalg.svdvals((out - out.mean(0)).double())
    return float(((s**2).cumsum(0) / (s**2).sum())[511])


def test_global_teacher_output_lives_in_a_512_dimensional_subspace():
    """ResNet18 -> GAP -> Linear(512, d): centered outputs are W(h - h_bar) with h in
    R^512, so they cannot leave a 512-dimensional subspace whatever the channel
    carries. A spectrum measured only through it is the geometry of that bottleneck."""
    from fiber.models import Teacher

    assert _top_512_energy(Teacher(d=1024), (64, 64)) > 1 - 1e-9


def test_spatial_teacher_is_not_confined_to_that_subspace():
    from fiber.models import SpatialTeacher

    e = _top_512_energy(SpatialTeacher(latent_shape=(4, 16, 16), width=16, blocks=1),
                        (128, 128))
    assert e < 0.99, f"top-512 energy {e:.6f}: the spatial teacher is bottlenecked too"


def test_spatial_models_contain_no_global_pooling():
    from fiber.models import SpatialTeacher

    m = SpatialTeacher(latent_shape=(4, 8, 8), width=8, blocks=1)
    assert not any(isinstance(mod, (torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d))
                   for mod in m.modules())


def test_spatial_extractor_keeps_position_and_refuses_imagenet_weights():
    from fiber.models import SpatialExtractor, build_extractor

    with pytest.raises(ValueError, match="forbidden"):
        SpatialExtractor(k=4, pretrained=True)
    e = build_extractor("spatial", k=8, reduce_channels=8, spatial=4)
    out = e(torch.randn(2, 3, 128, 128))
    assert out["w_hat"].shape == (2, 8) and out["sign_logits"].shape == (2, 8)
    # a moved feature must change the readout; under GAP it would not
    x = torch.zeros(1, 3, 128, 128); x[:, :, 10:20, 10:20] = 1
    y = torch.zeros(1, 3, 128, 128); y[:, :, 90:100, 90:100] = 1
    e.eval()
    with torch.no_grad():
        assert (e(x)["w_hat"] - e(y)["w_hat"]).abs().max() > 1e-4


def test_config_registers_the_architecture_controls():
    """The primary teacher control shares the trunk, so the isolated variable is the
    pooling rather than the whole network."""
    assert CFG["spectrum"]["teacher_arch_control"] == "spatial_sharedtrunk"
    assert CFG["spectrum"]["teacher_arch_control2"] == "spatial"
    assert CFG["extractor"]["arch_control"] == "spatial"
    assert CFG["spectrum"]["alignment_tol"] == 0.5


def test_shared_trunk_teacher_really_shares_the_trunk():
    from fiber.models import SharedTrunkSpatialTeacher, Teacher

    g, sh = Teacher(d=1024), SharedTrunkSpatialTeacher(latent_shape=(4, 16, 16))
    # compare SHAPES in order: the modules are the same objects but nn.Sequential
    # renames them ("0.weight" vs "conv1.weight")
    gt = [tuple(p.shape) for p in g.trunk.parameters()]
    st = [tuple(p.shape) for p in sh.trunk.parameters()]
    assert gt and gt == st, "the control changes more than the head"
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in sh.modules())


def test_spatial_extractor_is_capacity_comparable_to_the_global_one():
    """A spatial receiver with far more parameters would confound 'keeps position'
    with 'has more capacity'. Measured ratio 1.09 at k=64."""
    from fiber.models import build_extractor

    n = {a: sum(p.numel() for p in build_extractor(a, k=64).parameters())
         for a in ("resnet18", "spatial")}
    ratio = n["spatial"] / n["resnet18"]
    assert 0.8 < ratio < 1.25, f"capacity ratio {ratio:.2f} confounds the control"


def test_teacher_and_extractor_factories_reject_unknown_architectures():
    from fiber.models import build_extractor, build_teacher

    for build in (build_teacher, build_extractor):
        with pytest.raises(KeyError, match="unknown"):
            build("resnet50", k=4, d=16)


def test_factories_accept_the_union_of_architecture_kwargs():
    """Callers pass d and latent_shape without knowing which teacher they will get;
    the factory must not hand an unexpected keyword to either class."""
    from fiber.models import build_teacher

    for arch in ("resnet18", "spatial"):
        m = build_teacher(arch, d=256, latent_shape=(4, 4, 4))
        out = m(torch.randn(2, 3, 64, 64))
        assert out.shape[0] == 2


# --- nested random subsets for sample-size experiments -----------------------------

def test_subsets_are_nested_so_the_curve_varies_only_size():
    """N=100 must be a SUBSET of N=200 at the same subset seed. Otherwise each point of
    a learning curve is a different draw and size is confounded with composition."""
    from fiber.training.loops import make_loader
    root = Path("/ssd2/ming/FIBER/cache/pilot")
    if not (root / "index.jsonl").exists():
        pytest.skip("no pilot cache")
    bank = ChannelBank(load_config("configs/linear_fiber.yaml"))
    ids = {}
    for n in (100, 200, 400):
        ds = make_loader(root, "train", bank, crossfit="B", workers=0,
                         subset_size=n, subset_seed=3).dataset
        ids[n] = [r["sample_id"] for r in ds.records]
        assert len(ids[n]) == n
    assert set(ids[100]) < set(ids[200]) < set(ids[400])


def test_a_different_subset_seed_gives_a_different_subset():
    from fiber.training.loops import make_loader
    root = Path("/ssd2/ming/FIBER/cache/pilot")
    if not (root / "index.jsonl").exists():
        pytest.skip("no pilot cache")
    bank = ChannelBank(load_config("configs/linear_fiber.yaml"))
    a, b = (set(r["sample_id"] for r in make_loader(
        root, "train", bank, crossfit="B", workers=0, subset_size=100,
        subset_seed=s).dataset.records) for s in (0, 1))
    assert a != b and 0 < len(a & b) < 100


def test_a_prefix_subset_would_not_be_a_random_one():
    """Why subset_size exists at all rather than reusing --limit: the records are stored
    in sample-index order, and the index determines the latent seed and the prompt draw.
    A prefix therefore selects a structured subset, not a random one."""
    from fiber.training.loops import make_loader
    root = Path("/ssd2/ming/FIBER/cache/pilot")
    if not (root / "index.jsonl").exists():
        pytest.skip("no pilot cache")
    bank = ChannelBank(load_config("configs/linear_fiber.yaml"))
    prefix = [r["index"] for r in make_loader(root, "train", bank, crossfit="B",
                                              workers=0, limit=100).dataset.records]
    random = [r["index"] for r in make_loader(root, "train", bank, crossfit="B",
                                              workers=0, subset_size=100,
                                              subset_seed=0).dataset.records]
    assert max(prefix) < max(random), "the prefix is not concentrated at low indices"
    assert prefix == sorted(prefix)


def test_subset_runs_get_their_own_stem_and_cannot_be_gate_scoped():
    from train_coordinates import run_stem
    plain = run_stem("C2_haar", 64, 0, "resnet18", 0, "powercurve")
    a = run_stem("C2_haar", 64, 0, "resnet18", 0, "powercurve", 100, 0)
    b = run_stem("C2_haar", 64, 0, "resnet18", 0, "powercurve", 200, 0)
    c = run_stem("C2_haar", 64, 0, "resnet18", 0, "powercurve", 100, 1)
    assert len({plain, a, b, c}) == 4, "sample-size points would overwrite each other"
