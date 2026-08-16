"""The dataloader contract: cached X is read losslessly, the channel is applied
on the fly, and the target z is exactly what the generator consumed.

Skipped unless a cache exists (these tests are cheap but need Phase 1 output).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fiber.channels import ChannelBank
from fiber.diffusion.cache_dataset import FiberDataset
from fiber.utils.config import load_config

CFG = load_config("configs/linear_fiber.yaml")
BANK = ChannelBank(CFG)
ROOT = Path(CFG["paths"]["cache_dir"]) / "pilot"

pytestmark = pytest.mark.skipif(
    not (ROOT / "index.jsonl").exists(), reason="no pilot cache yet")


def _split_with_data():
    for split in ("val", "test", "train"):
        if (ROOT / split).exists() and any((ROOT / split).glob("*.done")):
            return split
    pytest.skip("no cached shard")


def test_dataset_returns_aligned_image_and_latent():
    split = _split_with_data()
    ds = FiberDataset(ROOT, split, BANK, attacks=["clean"], mode="fixed")
    item = ds[0]
    assert item["image"].shape == (3, 512, 512)
    assert item["z"].shape == (16384,)
    assert item["attack"] == "clean"


def test_latent_is_standard_normal():
    """z must arrive as N(0,I): a stray 0.18215 (the VAE factor) would show up
    here as std ~= 0.18 (PLAN.md §1.3)."""
    split = _split_with_data()
    ds = FiberDataset(ROOT, split, BANK, attacks=["clean"], mode="fixed")
    z = torch.stack([ds[i]["z"] for i in range(min(16, len(ds)))])
    assert abs(float(z.mean())) < 0.05
    assert abs(float(z.std()) - 1.0) < 0.05


def test_same_index_gives_byte_identical_output():
    split = _split_with_data()
    ds = FiberDataset(ROOT, split, BANK, attacks=["jpeg50"], mode="fixed")
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"])


def test_attack_actually_changes_the_image():
    split = _split_with_data()
    clean = FiberDataset(ROOT, split, BANK, attacks=["clean"], mode="fixed")[0]["image"]
    hit = FiberDataset(ROOT, split, BANK, attacks=["jpeg50"], mode="fixed")[0]["image"]
    assert not torch.equal(clean, hit)
    assert (clean - hit).abs().mean() > 1e-4


def test_sampled_mode_covers_the_attack_list_deterministically():
    split = _split_with_data()
    ds = FiberDataset(ROOT, split, BANK, mode="sampled", epoch_salt="e0")
    n = min(64, len(ds))
    names = [ds.attack_for(ds.records[i]) for i in range(n)]
    again = [ds.attack_for(ds.records[i]) for i in range(n)]
    assert names == again
    assert len(set(names)) > 1


def test_crossfit_filter_splits_train_disjointly():
    if not (ROOT / "train").exists():
        pytest.skip("train not cached")
    a = FiberDataset(ROOT, "train", BANK, crossfit="A")
    b = FiberDataset(ROOT, "train", BANK, crossfit="B")
    ids_a = {r["sample_id"] for r in a.records}
    ids_b = {r["sample_id"] for r in b.records}
    assert ids_a and ids_b and not (ids_a & ids_b)


def test_dataset_filters_by_teacher_operator_subsplit():
    """P0-1: the loader must be able to serve A_teacher and A_operator separately,
    with no sample in both."""
    if not (ROOT / "train").exists():
        pytest.skip("train not cached")
    t = FiberDataset(ROOT, "train", BANK, crossfit_sub="A_teacher")
    o = FiberDataset(ROOT, "train", BANK, crossfit_sub="A_operator")
    ids_t = {r["sample_id"] for r in t.records}
    ids_o = {r["sample_id"] for r in o.records}
    assert ids_t and ids_o and not (ids_t & ids_o)
    a = FiberDataset(ROOT, "train", BANK, crossfit="A")
    assert ids_t | ids_o == {r["sample_id"] for r in a.records}
