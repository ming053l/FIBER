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
