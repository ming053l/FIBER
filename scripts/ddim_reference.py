#!/usr/bin/env python
"""Prompt-assisted reference: sign BER of W = Q z_hat with z_hat from DDIM inversion.

It is never a baseline. It differs from the learned receiver in TWO ways at once --
it is given the prompt, and it runs the frozen generator backwards -- so a gap
against a prompt-free CNN cannot be attributed to the extra information alone.
Under the receiver-conditional reading it is simply a different information set
AND a different estimator class; the honest comparison is with a side-informed
learned receiver, which isolates the estimator class.

Its role is the capacity diagnostic that separates 'weak channel' from 'weak
extractor' (PLAN.md §5.3). It produces a number the paper reports, so it carries the
same provenance as every other evidence-producing script.

    python scripts/ddim_reference.py --tag pilot --arm C_hadamard --k 64 --limit 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fiber.channels import ChannelBank
from fiber.diffusion import FrozenGenerator, GeneratorSpec
from fiber.diffusion.cache_dataset import FiberDataset
from fiber.diffusion.inversion import ddim_invert
from fiber.metrics import sign_ber
from fiber.utils.config import load_config
from fiber.utils.provenance import require_clean
from fiber.utils.logging import get_logger
from fiber.utils.seeding import set_determinism

from train_coordinates import build_arm_frame  # noqa: E402

log = get_logger("ddim-ref")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--arm", default="C_hadamard")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=128, help="images per attack (this is slow)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="throwaway run; the result must not be reported")
    args = ap.parse_args()

    prov = require_clean("a prompt-assisted reference", allow_dirty=args.allow_dirty)
    cfg = load_config(args.config)
    if not cfg["threat_model"]["ddim_inversion_reference"]["enabled"]:
        raise SystemExit("ddim_inversion_reference disabled in the config")
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / args.tag
    d = int(cfg["latent"]["dim"])
    k = args.k or int(cfg["fiber"]["robust_dims"])
    frame, _ = build_arm_frame(cfg, args.arm, k, args.seed, args.tag, d,
                               cache_tag=args.tag, scope="reference", prov=prov)
    frame = frame.to(args.device)
    rows = frame.rows().detach().cpu().contiguous()
    frame_digest = hashlib.blake2s(rows.numpy().tobytes(), digest_size=16).hexdigest()
    orth = float((rows @ rows.T - torch.eye(k)).abs().max())
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])

    results, arrays = {}, {}
    for attack in bank.eval:
        ds = FiberDataset(root, args.split, bank, attacks=[attack], mode="fixed", normalise=False)
        ds.records = ds.records[: args.limit or len(ds.records)]
        per_sample, ids = [], []
        for lo in range(0, len(ds), args.batch):
            hi = min(lo + args.batch, len(ds))
            recs = ds.records[lo:hi]
            imgs, zs, prompts = [], [], []
            for j in range(lo, hi):
                item = ds[j]
                imgs.append((item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                zs.append(item["z"])
                prompts.append(ds.records[j]["prompt"])
            z_hat = ddim_invert(gen, np.stack(imgs), prompts).reshape(len(recs), -1)
            w_true = frame.project(torch.stack(zs).to(args.device))
            w_hat = frame.project(z_hat.to(args.device))
            per_sample.append(sign_ber(w_hat, w_true, reduce="per_sample").cpu())
            ids.extend(r["sample_id"] for r in recs)
        per_sample = torch.cat(per_sample)
        results[attack] = {"sign_ber": float(per_sample.mean()), "n": len(ids)}
        arrays[f"{args.split}|{attack}|per_sample"] = per_sample.numpy()
        arrays[f"{args.split}|{attack}|sample_ids"] = np.array(ids)
        log.info("%-10s prompt-assisted sign BER %.4f", attack, results[attack]["sign_ber"])

    out_dir = Path(cfg["paths"]["data_root"]) / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"REF_ddim_{args.arm}_k{k}_s{args.seed}"
    np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
    ids_digest = hashlib.blake2s(
        "|".join(sorted(str(x) for a in arrays for x in arrays[a]
                        if a.endswith("sample_ids"))).encode(), digest_size=16).hexdigest()
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "arm": f"REF_ddim({args.arm})", "type": "ddim_inversion_reference", "k": k,
        "seed": args.seed, "tag": args.tag, "split": args.split, "limit": args.limit,
        "results": {args.split: results},
        "frame_rows_digest": frame_digest, "frame_orthonormality_error": orth,
        "sample_ids_digest": ids_digest,
        "config_fingerprint": hashlib.blake2s(
            json.dumps({k2: cfg[k2] for k2 in ("model", "latent", "vae")},
                       sort_keys=True, default=str).encode(), digest_size=8).hexdigest(),
        "attacks": list(bank.eval), "n_per_attack": {a: r["n"] for a, r in results.items()},
        "protocol_note": ("uses the prompt AND runs the frozen generator backwards, so it "
                          "differs from a learned receiver in information set and in "
                          "estimator class at once"),
        "role": "capacity_diagnostic_only", **prov,
    }, indent=2))
    log.info("wrote %s", out_dir / f"{stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
