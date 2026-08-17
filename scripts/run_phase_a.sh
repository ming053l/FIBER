#!/usr/bin/env bash
# Phase A: blind vs side information. 4 arms x 6 receiver seeds = 24 runs.
#
# Everything here is preregistered in reports/blind_vs_sideinfo.md and nothing in this
# script chooses anything scientific:
#   frame        C2_haar seed 0, k=64, identical in every run, never tied to receiver seed
#   arms         blind, null, shuffled, correct -- one architecture, only S varies
#   derangement  fixed, within cross-fit role, independent of the receiver seed
#   primary      correct vs shuffled, paired across receiver seeds
#   equivalence  BER only, eps = 0.007, on f(Y)<->f(Y,S_null) and f(Y,S_null)<->f(Y,S_shuffled)
set -u
TAG=${TAG:-sideinfo1}
CACHE=${CACHE:-pilot}
K=${K:-64}
DEV=${DEV:-cuda:0}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
cd "$(dirname "$0")/.."
export PYTHONPATH=src

python -c "import numpy, torch, fiber" 2>/dev/null || {
  echo "environment not ready (numpy/torch/fiber not importable). Activate the conda"
  echo "environment first; PYTHONPATH=src alone is not enough."; exit 2; }
FAILURES=0

for MODE in blind null shuffled correct; do
  for S in $SEEDS; do
    echo "=== $MODE receiver_seed=$S ==="
    python -W ignore scripts/train_sideinfo.py \
      --tag "$TAG" --cache-tag "$CACHE" --side-mode "$MODE" \
      --receiver-seed "$S" --arm C2_haar --frame-seed 0 --k "$K" \
      --device "$DEV" || { FAILURES=$((FAILURES+1)); echo "FAILED $MODE s=$S"; }
  done
done
if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES runs FAILED -- the sweep is incomplete and must not be read as one"
  exit 1
fi
echo "done"
