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

# Fail on the first line rather than 18 times: a missing environment otherwise produces
# a full sweep of instant failures that the loop below would report one by one and then
# still finish with "done".
python -c "import numpy, torch, fiber" 2>/dev/null || {
  echo "environment not ready (numpy/torch/fiber not importable). Activate the conda"
  echo "environment first; PYTHONPATH=src alone is not enough."; exit 2; }
FAILURES=0

for K in 8 16 64; do
  for ARM in C2_haar D_spectral; do
    for S in 0 1 2; do
      echo "=== k=$K $ARM receiver_seed=$S ==="
      python -W ignore scripts/train_coordinates.py \
        --tag "$TAG" --cache-tag "$CACHE" --arm "$ARM" --seed 0 --k "$K" \
        --receiver-seed "$S" --scope powercurve --epochs 40 \
        --device "$DEV" || { FAILURES=$((FAILURES+1)); echo "FAILED k=$K $ARM s=$S"; }
    done
  done
done
if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES runs FAILED -- the sweep is incomplete and must not be read as one"
  exit 1
fi
echo "done"
