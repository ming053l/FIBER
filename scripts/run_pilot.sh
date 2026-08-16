#!/usr/bin/env bash
# Phase 2 + Phase 3 on the pilot cache, in the order PLAN.md §8 requires.
# Roughly 3-4 h on a single TITAN RTX. Every step is resumable: rerunning skips
# nothing, but each run writes its own {arm}_k{k}_s{seed}.{json,npz}.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-pilot}
K=${K:-64}
EPOCHS=${EPOCHS:-40}
export HF_HOME=${HF_HOME:-/ssd2/ming/STEGO/hf_cache}

echo "== Phase 2a: the observability spectrum (arm D's frame comes from here) =="
python scripts/fit_observability_spectrum.py --tag "$TAG" --seed 0 --per-attack
for s in 1 2; do
  python scripts/fit_observability_spectrum.py --tag "$TAG" --seed "$s"
done

echo "== Phase 2b: denominators, k sweep, identity + random only =="
for k in 16 32 64 128 256; do
  python scripts/train_coordinates.py --tag "$TAG" --arm A_identity  --k "$k" --seed 0 --epochs "$EPOCHS"
  for s in 0 1 2; do
    python scripts/train_coordinates.py --tag "$TAG" --arm B_signperm --k "$k" --seed "$s" --epochs "$EPOCHS"
    python scripts/train_coordinates.py --tag "$TAG" --arm C_hadamard --k "$k" --seed "$s" --epochs "$EPOCHS"
  done
done

echo "== Phase 3: all arms at k=$K, >=5 random draws, cross-fit =="
for s in 3 4 5 6 7; do
  python scripts/train_coordinates.py --tag "$TAG" --arm B_signperm --k "$K" --seed "$s" --epochs "$EPOCHS"
  python scripts/train_coordinates.py --tag "$TAG" --arm C_hadamard --k "$K" --seed "$s" --epochs "$EPOCHS"
done
for s in 0 1 2; do
  python scripts/train_coordinates.py --tag "$TAG" --arm D_spectral --k "$K" --seed "$s" --epochs "$EPOCHS"
  python scripts/train_coordinates.py --tag "$TAG" --arm E_learned  --k "$K" --seed "$s" --epochs "$EPOCHS"
done

echo "== Prompt-assisted reference (diagnostic only, violates the receiver protocol) =="
python scripts/ddim_reference.py --tag "$TAG" --arm C_hadamard --k "$K" --limit 128

echo "== Gates =="
python scripts/eval_coordinates.py --tag "$TAG" --k "$K" --split test
python scripts/eval_coordinates.py --tag "$TAG" --k "$K" --split test_heldout_prompts
