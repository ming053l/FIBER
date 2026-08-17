#!/usr/bin/env bash
# EXPLORATORY. Pre-registered in reports/powercurve.md. Not a gate, not a selection.
#
# The learning curve ruled out data volume. This asks whether 64 simultaneous targets is
# itself the limiter, by shrinking the task to the 8 and 16 most observable directions of
# the SAME spectral fit (SpectralFrame takes rows[:k], so the subspaces are nested).
#
# Full split B throughout and a fixed 40 epochs, so k is the only variable.
set -u
TAG=${TAG:-triage1}
CACHE=${CACHE:-pilot}
DEV=${DEV:-cuda:0}
cd "$(dirname "$0")/.."
export PYTHONPATH=src

for K in 8 16 64; do
  for ARM in C2_haar D_spectral; do
    for S in 0 1 2; do
      echo "=== k=$K $ARM receiver_seed=$S ==="
      python -W ignore scripts/train_coordinates.py \
        --tag "$TAG" --cache-tag "$CACHE" --arm "$ARM" --seed 0 --k "$K" \
        --receiver-seed "$S" --scope powercurve --epochs 40 \
        --device "$DEV" || echo "FAILED k=$K $ARM s=$S"
    done
  done
done
echo "done"
