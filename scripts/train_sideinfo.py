#!/usr/bin/env python
"""Phase A: one arm of the blind-vs-side-information ladder.

    python scripts/train_sideinfo.py --side-mode correct --receiver-seed 0

Four arms, one architecture, one frame. Only S varies:

    blind      the legacy Extractor, no side branch at all
    null       S_null, the training-split mean
    shuffled   S from another sample, deranged WITHIN cross-fit role
    correct    the sample's own S

Preregistered in reports/blind_vs_sideinfo.md. This script does not choose anything
scientific; every knob it exposes is already fixed there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.models import build_extractor
from fiber.models.sideinfo import SideExtractor, parameter_report
from fiber.sideinfo import MODES, SideConditioning
from fiber.sideinfo.conditioning import train_mean_conditioning
from fiber.training import TrainConfig, evaluate, train_extractor
from fiber.transforms import build_frame
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean
from fiber.utils.seeding import derive_seed, set_determinism

log = get_logger("sideinfo")

SCOPE = "sideinfo"


def init_digest(model: torch.nn.Module) -> str:
    """Hash of the model's parameters as initialised.

    The preregistration says the sole intervention between S_correct and S_shuffled is
    the pairing. That is a claim about the TRAINING RANDOMNESS too: same receiver seed
    must mean the same initial weights, or the contrast carries SGD noise as well. This
    is recorded so the claim is checkable from the artifacts rather than asserted.
    """
    h = hashlib.blake2s(digest_size=16)
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_receiver(mode: str, k: int, seed: int, device):
    """Init is seeded on the RECEIVER seed alone, never on the side mode, so the three
    side arms start from identical weights."""
    torch.manual_seed(derive_seed("extractor", seed) % (2**31))
    model = (build_extractor("resnet18", k=k) if mode == "blind"
             else SideExtractor(k=k))
    return model.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="sideinfo1")
    ap.add_argument("--cache-tag", default="pilot")
    ap.add_argument("--side-mode", required=True, choices=list(MODES))
    ap.add_argument("--receiver-seed", type=int, default=0)
    ap.add_argument("--arm", default="C2_haar",
                    help="the FRAME. Phase A varies the receiver's information set, not "
                         "the coordinate system, so this is held fixed across all arms.")
    ap.add_argument("--frame-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--eval-splits", nargs="*", default=["val"])
    ap.add_argument("--derangement-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_determinism(cfg)
    prov = require_clean("a side-information run", allow_dirty=args.allow_dirty)
    for split in args.eval_splits:
        if split.startswith("test"):
            raise SystemExit("Phase A is a val-only stage; test is evaluated only after "
                             "the side-info method is locked (B1).")

    k = int(args.k or cfg["fiber"]["robust_dims"])
    d = int(cfg["latent"]["dim"])
    root = Path(cfg["paths"]["cache_dir"]) / args.cache_tag
    bank = ChannelBank(cfg)
    tcfg = TrainConfig.from_config(cfg)
    if args.epochs:
        tcfg.epochs = int(args.epochs)

    index = [json.loads(l) for l in open(root / "index.jsonl")]
    s_null = (train_mean_conditioning(root) if args.side_mode == "null" else None)
    xfit = cfg["dataset"]["crossfit"]["extractor_split"]

    def provider(split):
        return SideConditioning(root, split, index, args.side_mode, s_null=s_null,
                                derangement_seed=args.derangement_seed)

    frame = build_frame(dict(cfg["fiber"]["arms"][args.arm]), d=d, k=k,
                        seed=args.frame_seed)
    model = build_receiver(args.side_mode, k, args.receiver_seed, args.device)
    init_sha = init_digest(model)
    params = (parameter_report(model) if args.side_mode != "blind"
              else {"total": sum(p.numel() for p in model.parameters()),
                    "side_branch": 0})

    t0 = time.time()
    model, frame, hist = train_extractor(
        frame, root, bank, tcfg, split="train", crossfit=xfit, device=args.device,
        seed=args.receiver_seed, learn_frame=False, attacks=bank.train,
        side=provider("train"), model=model)

    results, arrays = {}, {}
    for split in args.eval_splits:
        ev = evaluate(model, frame, root, bank, split, bank.eval, device=args.device,
                      side=provider(split))
        results[split] = {a: {kk: v for kk, v in r.items()
                              if kk not in ("per_sample", "sample_ids",
                                            "pearson_per_coord")}
                          for a, r in ev.items()}
        for a, r in ev.items():
            arrays[f"{split}|{a}|per_sample"] = r["per_sample"]
            arrays[f"{split}|{a}|sample_ids"] = np.array(r["sample_ids"])
            if "pearson_per_coord" in r:
                arrays[f"{split}|{a}|pearson"] = r["pearson_per_coord"]

    out_dir = Path(cfg["paths"]["data_root"]) / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"side{args.side_mode}_{args.arm}_k{k}_fs{args.frame_seed}_r{args.receiver_seed}_sc{SCOPE}"
    cond_manifest = json.loads((root / "conditioning" / "manifest.json").read_text())
    summary = {
        "analysis_scope": SCOPE, "side_mode": args.side_mode,
        "tag": args.tag, "cache_tag": args.cache_tag,
        "arm": args.arm, "frame_seed": args.frame_seed, "k": k,
        "receiver_seed": args.receiver_seed,
        # invariant: same receiver seed -> identical initial weights across side arms
        "init_state_sha": init_sha,
        "init_seed_rule": "derive_seed('extractor', receiver_seed); side mode NOT mixed in",
        "attack_seed_rule": "blake2s(sample_id|attack|severity|split_salt)",
        "parameters": params,
        "side_provider": {sp: provider(sp).manifest() for sp in ["train", *args.eval_splits]},
        "conditioning_manifest_digest": hashlib.blake2s(
            json.dumps(cond_manifest, sort_keys=True).encode(),
            digest_size=16).hexdigest(),
        "conditioning_protocol": cond_manifest.get("protocol"),
        "epochs": tcfg.epochs, "final_train_loss": hist[-1]["loss"] if hist else None,
        "results": results, "seconds": round(time.time() - t0, 1), **prov,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    torch.save({"state_dict": {kk: v.cpu() for kk, v in model.state_dict().items()},
                "side_mode": args.side_mode, "k": k,
                "receiver_seed": args.receiver_seed},
               out_dir / f"{stem}_extractor.pt")
    for split, r in results.items():
        log.info("[%s] %s mean sign BER %.4f", args.side_mode, split,
                 sum(x["sign_ber"] for x in r.values()) / len(r))
    log.info("wrote %s", out_dir / f"{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
