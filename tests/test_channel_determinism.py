"""Gate 1: 'attacks byte-reproducible'. If this test ever fails, every BER
number in the project became incomparable across runs."""
import numpy as np
import pytest

from fiber.channels import ChannelBank
from fiber.utils.config import load_config

CFG = load_config("configs/channels.yaml")
BANK = ChannelBank(CFG)
ALL = list(BANK.attacks)


def make_image(seed=0):
    rng = np.random.default_rng(seed)
    # structured, not pure noise: JPEG/blur on white noise is a degenerate case
    y, x = np.mgrid[0:512, 0:512]
    base = (128 + 100 * np.sin(x / 17.0) * np.cos(y / 23.0)).astype(np.float32)
    img = np.stack([base, np.roll(base, 40, 0), np.roll(base, 80, 1)], -1)
    img = img + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", ALL)
def test_attack_is_byte_identical_across_calls(name):
    img = make_image()
    a = BANK.apply(img, name, sample_id=7, split_salt="train")
    b = BANK.apply(img, name, sample_id=7, split_salt="train")
    assert np.array_equal(a, b), name


@pytest.mark.parametrize("name", ALL)
def test_attack_preserves_shape_and_dtype(name):
    img = make_image()
    out = BANK.apply(img, name, sample_id=1)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_noise_differs_across_samples_and_salts():
    img = make_image()
    a = BANK.apply(img, "noise005", sample_id=1, split_salt="train")
    b = BANK.apply(img, "noise005", sample_id=2, split_salt="train")
    c = BANK.apply(img, "noise005", sample_id=1, split_salt="test")
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_severity_changes_the_draw():
    """Same sample, same attack name, different severity -> different noise.
    Guards against reusing a stale draw after retuning an attack."""
    from fiber.channels.registry import attack_seed
    s1 = attack_seed(1, "noise", {"type": "gaussian_noise", "sigma": 0.05})
    s2 = attack_seed(1, "noise", {"type": "gaussian_noise", "sigma": 0.10})
    assert s1 != s2


@pytest.mark.parametrize("name", [a for a in ALL if a != "clean"])
def test_attack_actually_degrades(name):
    """A misconfigured attack that silently returns the input would inflate
    every BER comparison. Require a visible change."""
    img = make_image()
    out = BANK.apply(img, name, sample_id=3)
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 0.05, name


def test_clean_is_a_true_identity():
    img = make_image()
    assert np.array_equal(BANK.apply(img, "clean"), img)


def test_holdout_attacks_are_disjoint_from_training():
    assert not (set(BANK.holdout) & set(BANK.train))
    assert not (set(BANK.holdout) & set(BANK.eval))


def test_channel_groups_cover_five_families():
    groups = {BANK.group_of(a) for a in BANK.train}
    assert groups == {"identity", "jpeg", "resize", "noise", "blur"}


# ==========================================================================
# P0-6: stochastic attacks must be FRESH every training epoch and FIXED across
# arms at evaluation.
# ==========================================================================
from fiber.channels.registry import attack_seed  # noqa: E402


def test_training_epochs_get_different_noise_realisations():
    """Without this the stochastic channel is a finite deterministic corruption
    table and the extractor can memorise it."""
    img = make_image()
    a = BANK.apply(img, "noise005", sample_id=7, split_salt="train", draw_salt="e1")
    b = BANK.apply(img, "noise005", sample_id=7, split_salt="train", draw_salt="e2")
    assert not np.array_equal(a, b)


def test_evaluation_draw_is_fixed_and_reproducible():
    img = make_image()
    a = BANK.apply(img, "noise010", sample_id=3, split_salt="test", draw_salt="eval-v1")
    b = BANK.apply(img, "noise010", sample_id=3, split_salt="test", draw_salt="eval-v1")
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["clean", "jpeg50", "resize050", "blur20", "webp75"])
def test_deterministic_attacks_ignore_the_draw_salt(name):
    """Only stochastic attacks may respond to the salt; JPEG or resize changing with
    the epoch would silently alter the channel severity."""
    img = make_image()
    a = BANK.apply(img, name, sample_id=5, draw_salt="e1")
    b = BANK.apply(img, name, sample_id=5, draw_salt="e2")
    assert np.array_equal(a, b), name


def test_every_arm_sees_a_bit_identical_evaluation_corruption():
    """The paired bootstrap differences arms on the SAME (z_i, prompt_i, attack_i). If
    two arms saw different noise realisations they would be compared across different
    channels and the pairing would be meaningless."""
    img = make_image()
    outs = [BANK.apply(img, "noise005", sample_id=11, split_salt="test",
                       draw_salt="eval-v1") for _ in range(5)]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


def test_evaluation_corruption_does_not_depend_on_loader_order():
    """The draw is keyed by sample_id, not by position in a batch, so shuffling,
    batch size and worker count cannot change it."""
    img = make_image()
    ids = [3, 17, 42]
    forward = {i: BANK.apply(img, "noise010", sample_id=i, split_salt="test",
                             draw_salt="eval-v1") for i in ids}
    backward = {i: BANK.apply(img, "noise010", sample_id=i, split_salt="test",
                              draw_salt="eval-v1") for i in reversed(ids)}
    assert all(np.array_equal(forward[i], backward[i]) for i in ids)


def test_draw_salt_enters_the_seed_derivation():
    base = attack_seed(1, "noise005", {"type": "gaussian_noise", "sigma": 0.05})
    assert attack_seed(1, "noise005", {"type": "gaussian_noise", "sigma": 0.05},
                       draw_salt="e1") != base
