"""Gate 1: latents AND prompts disjoint across splits, split A / split B disjoint
within train, and (P0-1) A_teacher / A_operator disjoint within A.

If any of these leak, every 'held-out' number in the project is contaminated and
the cross-fit control silently becomes a co-adaptation study.
"""
import hashlib
import json
from pathlib import Path

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


# --------------------------------------------------------------------------
# P0-1: A is split again, into teacher-fitting and operator-estimation halves.
# --------------------------------------------------------------------------
def test_teacher_and_operator_subsplits_are_disjoint():
    """Fitting the teacher and estimating C_cert on the same samples makes the
    operator read back the teacher's own fitting noise as observability."""
    sub = REPORT["crossfit_sub"]
    assert sub["teacher_operator_overlap"] == 0
    assert sub["A_teacher"] > 0 and sub["A_operator"] > 0


def test_subsplit_partitions_A_exactly_and_never_touches_B():
    sub = REPORT["crossfit_sub"]
    assert sub["sub_covers_A"] and sub["sub_leaks_into_B"] == 0
    assert sub["A_teacher"] + sub["A_operator"] == REPORT["crossfit"]["A"]


def test_every_sample_has_exactly_one_role():
    roles = {}
    for r in INDEX:
        key = (r.split, r.crossfit_sub)
        roles.setdefault(r.sample_id, set()).add(key)
    assert all(len(v) == 1 for v in roles.values())
    train_subs = {r.crossfit_sub for r in INDEX if r.split == "train"}
    assert train_subs == {"A_teacher", "A_operator", "B"}


def _fingerprint(pilot):
    rows = [(r.sample_id, r.split, r.index, r.shard, r.offset,
             derive_seed(*r.latent_seed_parts), r.prompt) for r in build_index(CFG, pilot=pilot)]
    blob = "\n".join("|".join(map(str, row)) for row in rows).encode()
    return len(rows), hashlib.blake2s(blob, digest_size=16).hexdigest(), rows


@pytest.mark.parametrize("tag,pilot", [("pilot", True), ("full", False)])
def test_index_addressing_is_unchanged(tag, pilot):
    """The cached shards are addressed by (split, shard, offset) and their contents
    by the latent seed and prompt. This golden fingerprint is what lets a protocol
    change be made WITHOUT regenerating 2768 GPU-hours' worth of images: if it still
    matches, every cached image is still the image this index claims it is.
    """
    golden = json.loads(Path("tests/data/index_fingerprint.json").read_text())[tag]
    n, digest, rows = _fingerprint(pilot)
    assert n == golden["n"]
    if digest != golden["digest"]:
        # narrow the failure to a field so the message is actionable
        for i, expected in golden["spot"].items():
            got = list(rows[int(i)])
            assert got == expected, f"record {i}: {got} != {expected}"
        pytest.fail(f"index digest changed ({digest} != {golden['digest']}) with all "
                    "spot checks matching: a record outside the spot set moved")


def test_cache_fingerprint_tracks_pixels_not_protocol():
    """The resume guard must invalidate a cache when the IMAGES would change and
    never when only the downstream protocol changed -- a false invalidation costs a
    full regeneration (~1 GPU-hour for the pilot, ~4.4 h for the full run)."""
    import copy
    import sys
    sys.path.insert(0, "scripts")
    from cache_native_dataset import config_fingerprint

    base = config_fingerprint(CFG)

    protocol = copy.deepcopy(dict(CFG))
    protocol["dataset"]["crossfit"] = {"discovery_split": "A", "extractor_split": "B",
                                       "teacher_subsplit": "changed"}
    protocol["training"]["epochs"] = 999
    protocol["extractor"]["arch"] = "resnet50"
    assert config_fingerprint(protocol) == base, "protocol change invalidated the cache"

    for block, field, value in [("model", "guidance_scale", 9.9),
                                ("model", "num_inference_steps", 50),
                                ("model", "checkpoint", "/elsewhere"),
                                ("latent", "height", 32)]:
        pixels = copy.deepcopy(dict(CFG))
        pixels[block][field] = value
        assert config_fingerprint(pixels) != base, f"{block}.{field} did not invalidate"

    prompts = copy.deepcopy(dict(CFG))
    prompts["dataset"]["prompts"]["num_train_prompts"] = 123
    assert config_fingerprint(prompts) != base


def test_caching_a_test_split_requires_a_method_lock():
    """B1 strongest form: test pixels generated before the lock only support 'the test
    set was never accessed'. Generating them after selection supports 'no test sample
    existed', which is a statement about the filesystem rather than about control flow."""
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "scripts/cache_native_dataset.py", "--pilot",
                        "--splits", "test", "--limit", "1"],
                       capture_output=True, text=True)
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert "before a method lock" in out and "--post-lock" in out


def test_pre_lock_caching_of_train_and_val_is_allowed():
    src = Path("scripts/cache_native_dataset.py").read_text()
    assert 'ap.add_argument("--post-lock"' in src
    assert "generated_after_lock" in src
