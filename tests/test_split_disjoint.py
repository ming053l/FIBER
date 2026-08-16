"""Gate 1: latents AND prompts disjoint across splits, and split A / split B
disjoint within train (the cross-fit protocol, PLAN.md §4).

If any of these leak, every 'held-out' number in the project is contaminated and
the cross-fit control silently becomes a co-adaptation study.
"""
import pytest

from fiber.diffusion.cache_dataset import build_index, split_sizes, verify_index
from fiber.diffusion.prompts import load_captions, split_prompts
from fiber.utils.config import load_config
from fiber.utils.seeding import derive_seed

CFG = load_config("configs/linear_fiber.yaml")
INDEX = build_index(CFG, pilot=True)
REPORT = verify_index(INDEX)


def test_index_passes_gate1_disjointness():
    assert REPORT["pass"], REPORT


def test_no_prompt_is_shared_between_splits():
    assert REPORT["prompt_overlaps"] == {}


def test_no_latent_seed_is_shared_between_splits():
    assert REPORT["latent_seed_overlaps"] == {}
    assert REPORT["duplicate_seeds_global"] == 0


def test_crossfit_halves_are_disjoint_and_balanced():
    cf = REPORT["crossfit"]
    assert cf["overlap"] == 0
    assert cf["A"] + cf["B"] == REPORT["counts"]["train"]
    assert abs(cf["A"] - cf["B"]) < 0.1 * REPORT["counts"]["train"]


def test_prompts_are_unique_within_each_split():
    for split, n_unique in REPORT["unique_prompts_per_split"].items():
        assert n_unique == REPORT["counts"][split], split


def test_heldout_prompt_split_comes_from_a_different_caption_corpus():
    """Gate 3A requires held-out PROMPTS, not just held-out latents (R6)."""
    heldout = {r.prompt for r in INDEX if r.split == "test_heldout_prompts"}
    in_domain = {r.prompt for r in INDEX if r.split != "test_heldout_prompts"}
    assert heldout and not (heldout & in_domain)
    val_pool = set(load_captions(CFG["dataset"]["prompts"]["heldout_source"]))
    assert heldout <= val_pool


def test_index_is_a_pure_function_of_the_config():
    """A rebuild must reproduce the same assignment, or cached shards and the
    index can silently disagree."""
    again = build_index(CFG, pilot=True)
    assert [(r.sample_id, r.prompt, r.crossfit) for r in INDEX] == \
           [(r.sample_id, r.prompt, r.crossfit) for r in again]


def test_pilot_and_full_are_different_draws():
    """The pilot is a rehearsal, not a subset masquerading as one: its sample_ids
    are namespaced so pilot images can never be mistaken for full-run images."""
    full_ids = {r.sample_id for r in build_index(CFG, pilot=False)}
    pilot_ids = {r.sample_id for r in INDEX}
    assert not (full_ids & pilot_ids)


def test_prompt_pool_is_large_enough_to_prevent_memorisation():
    """R6: 100 prompts would let the extractor infer z through the prompt."""
    pool = load_captions(CFG["dataset"]["prompts"]["source"])
    assert len(pool) > 100_000
    sizes = split_sizes(CFG, pilot=False)
    assert sum(v for k, v in sizes.items() if k != "test_heldout_prompts") <= len(pool)


def test_split_prompts_helper_is_disjoint():
    pool = [f"caption {i}" for i in range(1000)]
    out = split_prompts(pool, {"a": 100, "b": 100, "c": 50})
    assert len(set(out["a"]) | set(out["b"]) | set(out["c"])) == 250


def test_latent_seed_derivation_is_stable():
    r = INDEX[0]
    assert derive_seed(*r.latent_seed_parts) == derive_seed(*r.latent_seed_parts)
