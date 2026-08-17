#!/usr/bin/env bash
# P0 TRIAGE — decide whether the ~31 GPU-hour full protocol is worth starting.
#
# It has NO PASS/KILL authority. Its verdict is PROVISIONAL by construction
# (gate3a_verdict downgrades anything not tagged `full`), and its purpose is to answer
# one question: is there a reproducible derived-versus-Haar signal, and do the two
# teacher architectures agree well enough that the geometry is not obviously an
# artefact of the decoder?
#
# Scope, fixed by the audit before any of it ran:
#   k = 64
#   five representative channels, one per family: clean jpeg50 resize050 noise010 blur20
#   Haar 3 draws, Hadamard 1, frozen-Householder 1
#   certified-spectrum 2 seeds, learned-Q 2 seeds
#   both teacher architectures
#
# Protocol order is enforced by the scripts, not by this file: training refuses a test
# split, caching test pixels refuses to run without the lock, and the locked evaluation
# refuses unless HEAD matches the commit the lock was taken under.
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-pilot}
K=${K:-64}
EPOCHS=${EPOCHS:-40}
ATTACKS="clean jpeg50 resize050 noise010 blur20"
HAAR_SEEDS="0 1 2"
DERIVED_SEEDS="0 1"
export HF_HOME=${HF_HOME:-/ssd2/ming/STEGO/hf_cache}

failed=()
run() { echo "--- $*"; if ! "$@"; then echo "!!! FAILED: $*"; failed+=("$*"); fi; }
arm() {
  run python scripts/train_coordinates.py --tag "$TAG" --arm "$1" --k "$K" --seed "$2" \
      --epochs "$EPOCHS" --eval-attacks $ATTACKS "${@:3}"
}

echo "===== teachers: the certified operator under both decoder architectures ====="
for s in 0 1; do
  run python scripts/fit_observability_spectrum.py --tag "$TAG" --seed "$s" --k "$K"
done
run python scripts/compare_teachers.py --tag "$TAG" --seed 0 --k "$K" --skip-fit

echo "===== arms (VAL ONLY -- training refuses a test split) ====="
for s in $HAAR_SEEDS; do arm C2_haar "$s"; done
arm C_hadamard 0
arm C3_frozen_hh 0
for s in $DERIVED_SEEDS; do
  arm D_spectral "$s"
  arm E_learned  "$s"
done

echo "===== selection on VAL, with the reduced seed sets declared ====="
run python scripts/select_method.py --tag "$TAG" \
    --required-arms C2_haar,C_hadamard,C3_frozen_hh,D_spectral,E_learned \
    --registered-seeds "C2_haar=$(echo $HAAR_SEEDS | tr ' ' ',')" \
    --registered-seeds C_hadamard=0 --registered-seeds C3_frozen_hh=0 \
    --registered-seeds "D_spectral=$(echo $DERIVED_SEEDS | tr ' ' ',')" \
    --registered-seeds "E_learned=$(echo $DERIVED_SEEDS | tr ' ' ',')"

echo "===== test pixels generated for the FIRST time, bound to the lock ====="
run python scripts/cache_native_dataset.py --pilot --batch 12 \
    --splits test test_heldout_prompts --post-lock "reports/selection_${TAG}.json"

echo "===== test evaluated for the FIRST time, from the frozen checkpoints ====="
run python scripts/evaluate_locked.py --tag "$TAG"

echo "===== provisional gate ====="
run python scripts/eval_coordinates.py --tag "$TAG" --split test
run python scripts/eval_coordinates.py --tag "$TAG" --split test_heldout_prompts

if [ ${#failed[@]} -gt 0 ]; then
  echo "===== ${#failed[@]} step(s) failed ====="; printf '  %s\n' "${failed[@]}"; exit 1
fi
echo "===== triage complete (PROVISIONAL -- no PASS/KILL authority) ====="
