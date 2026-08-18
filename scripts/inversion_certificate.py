#!/usr/bin/env python
"""P1: does the inversion receiver induce a certified geometry, and is it anisotropic?

Preregistered in reports/practical_p1.md. One intervention against Phase A -- the
receiver -- on the same parent frame, the same cache and the same attacks.

    zhat = f_inv(Y, S; G)        DDIM inversion, no training
    c    = R z,   chat = R zhat  R = C2_haar seed 0, fixed a priori
    C_cert^inv = E[c_c chat_c' + chat_c c_c' - chat_c chat_c']

Two questions, deliberately separated:

  P1-A  can it certify at all?  Per-coordinate bounds in the FIXED frame R. Nothing is
        selected here, so no inner cross-fit is needed and none is used -- the certificate
        is a sample mean and is bootstrapped directly, Bonferroni over the k coordinates.

  P1-B  is there anything to select?  Top and bottom COORDINATES of the fixed frame are
        chosen on the DISCOVERY split and measured on the CERTIFICATION split. Choosing
        and measuring on the same samples is exactly the rectification the inner cross-fit
        exists for.

NOTE ON WHAT P1-B CAN AND CANNOT SHOW. It reads the DIAGONAL of the restricted operator in
a fixed frame. A flat diagonal in a random basis does not imply an isotropic operator:
r_j' C r_j = sum_m lambda_m (u_m . r_j)^2, and in high dimension (u_m . r_j)^2 concentrates
near 1/d, so any spectrum is flattened by the basis. FIBER is an eigenspace framework, so
ruling out coordinate ranking does not rule out a reproducible eigen-direction. That is a
separate question (P1-C) on the same saved coordinates.

A large BER gap says the receiver recovers something; it says nothing about where. A
near-isotropic certificate would mean FIBER has no coordinate to select, and that is a
reportable result rather than a reason to try another receiver.
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
from fiber.diffusion import FrozenGenerator, GeneratorSpec
from fiber.diffusion.cache_dataset import FiberDataset
from fiber.diffusion.inversion import ddim_invert
from fiber.spectrum.certified import _per_sample_contributions, project_operator
from fiber.utils.config import load_config
from fiber.utils.logging import get_logger
from fiber.utils.provenance import require_clean
from fiber.utils.seeding import set_determinism

import sys
sys.path.insert(0, "scripts")
from train_coordinates import build_arm_frame  # noqa: E402

log = get_logger("p1")


def project_split(gen, frame, cfg, bank, root, split, attack, crossfit_sub, limit, batch,
                  device):
    """c and chat for one split under one channel condition."""
    ds = FiberDataset(root, split, bank, attacks=[attack], mode="fixed", normalise=False,
                      crossfit_sub=crossfit_sub)
    ds.records = ds.records[: limit or len(ds.records)]
    C, CH, ids = [], [], []
    for lo in range(0, len(ds), batch):
        hi = min(lo + batch, len(ds))
        imgs, zs = [], []
        for j in range(lo, hi):
            item = ds[j]
            imgs.append((item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            zs.append(item["z"])
        zhat = ddim_invert(gen, np.stack(imgs), [ds.records[j]["prompt"]
                                                 for j in range(lo, hi)]).reshape(hi - lo, -1)
        C.append(frame.project(torch.stack(zs).to(device)).cpu().numpy())
        CH.append(frame.project(zhat.to(device)).cpu().numpy())
        ids.extend(ds.records[j]["sample_id"] for j in range(lo, hi))
        if (lo // batch) % 5 == 0:
            log.info("%s/%s %s: %d/%d", split, crossfit_sub or "-", attack, hi, len(ds))
    return np.concatenate(C), np.concatenate(CH), ids


def bounds(C, CH, k, boot, alpha, rng):
    """Per-coordinate certificate and its one-sided bound in the fixed frame."""
    V = np.eye(k)
    lam = np.diag(project_operator(C, CH, V, center=True))
    terms = _per_sample_contributions(C, CH, V, center=True)
    idx = rng.integers(0, terms.shape[0], size=(boot, terms.shape[0]))
    draws = terms[idx].mean(axis=1)
    return lam, np.quantile(draws, alpha / k, axis=0)      # Bonferroni over k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/linear_fiber.yaml")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--arm", default="C2_haar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--attack", default="clean",
                    help="P1-A runs on clean first: the receiver's most favourable "
                         "condition, so a failure there is decisive")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-top", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = require_clean("a P1 certificate", allow_dirty=args.allow_dirty)
    cfg = load_config(args.config)
    set_determinism(cfg)
    bank = ChannelBank(cfg)
    root = Path(cfg["paths"]["cache_dir"]) / args.tag
    d = int(cfg["latent"]["dim"])
    k = args.k or int(cfg["fiber"]["robust_dims"])
    frame, _ = build_arm_frame(cfg, args.arm, k, args.seed, args.tag, d,
                               cache_tag=args.tag, scope="p1", prov=prov)
    frame = frame.to(args.device)
    rows = frame.rows().detach().cpu().contiguous()
    gen = FrozenGenerator(GeneratorSpec.from_config(cfg), cfg["latent"])
    rng = np.random.default_rng(0)
    t0 = time.time()

    # discovery: train/A_operator      certification: val
    Cd, CHd, ids_d = project_split(gen, frame, cfg, bank, root, "train", args.attack,
                                   "A_operator", args.limit, args.batch, args.device)
    Cc, CHc, ids_c = project_split(gen, frame, cfg, bank, root, "val", args.attack,
                                   None, args.limit, args.batch, args.device)
    assert not (set(ids_d) & set(ids_c)), "discovery and certification splits overlap"

    # ---- P1-A: nothing is selected, so bootstrap the fixed-frame coordinates directly
    lam_c, lcb_c = bounds(Cc, CHc, k, args.bootstrap, args.alpha, rng)
    certified = int((lcb_c > 0).sum())

    # ---- P1-B: choose on discovery, measure on certification
    lam_d, _ = bounds(Cd, CHd, k, args.bootstrap, args.alpha, rng)
    order = np.argsort(lam_d)[::-1]
    top, bottom = order[: args.n_top], order[-args.n_top:]
    haar = rng.choice(k, size=args.n_top, replace=False)
    D = lambda idx: float(np.clip(lam_c[idx], 0, None).sum())
    Dl = lambda idx: float(np.clip(lcb_c[idx], 0, None).sum())

    # Saved so that operator-level questions can be asked without paying for inversion
    # again: 793 inversions is 17 minutes, and c / chat are a few hundred kilobytes.
    coord_path = (Path(cfg["paths"]["data_root"]) / "results" / args.tag
                  / f"p1_coords_{args.attack}.npz")
    coord_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coord_path, c_disc=Cd, chat_disc=CHd, c_cert=Cc, chat_cert=CHc,
                        ids_disc=np.array(ids_d), ids_cert=np.array(ids_c))

    summary = {
        "tag": args.tag, "arm": args.arm, "seed": args.seed, "k": k,
        "attack": args.attack, "receiver": "ddim_inversion_prompt_assisted",
        "discovery": {"split": "train/A_operator", "n": len(ids_d)},
        "certification": {"split": "val", "n": len(ids_c)},
        "frame_rows_digest": hashlib.blake2s(rows.numpy().tobytes(),
                                             digest_size=16).hexdigest(),
        "p1b_reads": "the DIAGONAL of the restricted operator in the fixed frame",
        "p1a_certified_positive_coordinates": certified, "p1a_of": k,
        "lambda_certification": lam_c.tolist(), "lcb_certification": lcb_c.tolist(),
        "lambda_discovery": lam_d.tolist(),
        "p1b": {"n_top": args.n_top,
                "D_top": D(top), "D_bottom": D(bottom), "D_haar_subset": D(haar),
                "D_top_lcb": Dl(top), "D_bottom_lcb": Dl(bottom),
                "top_idx": top.tolist(), "bottom_idx": bottom.tolist()},
        "anisotropy": {"lambda_max": float(lam_c.max()), "lambda_min": float(lam_c.min()),
                       "lambda_mean": float(lam_c.mean()), "lambda_sd": float(lam_c.std(ddof=1)),
                       "cv": float(lam_c.std(ddof=1) / abs(lam_c.mean()))
                       if lam_c.mean() else None},
        "bootstrap": args.bootstrap, "alpha": args.alpha,
        "bound_rule": "one-sided bootstrap percentile, Bonferroni over k, fixed frame",
        "coordinates": str(coord_path),
        "seconds": round(time.time() - t0, 1), **prov,
    }
    out = Path(cfg["paths"]["reports_dir"]) / f"p1_inversion_{args.tag}_{args.attack}.json"
    out.write_text(json.dumps(summary, indent=2))
    log.info("P1-A: %d/%d coordinates certified positive (Bonferroni %.0f%%)",
             certified, k, 100 * args.alpha)
    log.info("P1-B: D_top %.4f  D_bottom %.4f  D_haar %.4f   (lambda sd %.4f, mean %.4f)",
             summary["p1b"]["D_top"], summary["p1b"]["D_bottom"],
             summary["p1b"]["D_haar_subset"], lam_c.std(ddof=1), lam_c.mean())
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
