#!/usr/bin/env bash
# Phase 2 + Phase 3 on the PILOT cache, in the order PLAN.md §8 requires.
#
# The pilot is a rehearsal, so its coverage is deliberately smaller than the
# protocol in linear_fiber.yaml. NO SILENT CAPS -- what is reduced, and why:
#
#   NOTE: P0-2 made Haar the gate denominator, and the review defines a smaller
#   P0_TRIAGE (k=64, five representative channels) that must run BEFORE any full
#   sweep. This script is the pilot sweep; scripts/run_triage.sh supersedes it as
#   the gate on whether the full protocol is worth its ~31 GPU-hours.
#
#   random draws   8 -> 3 per family   (config asks 8; 3 still gives a spread)
#   derived seeds  3 -> 2
#   k sweep        {16,32,64,128,256} -> {16,64,256}
#   spectrum seeds 3 -> 2
#
# The epoch budget is NOT reduced and is identical across every arm, because
# that is the one thing a cross-fit comparison cannot tolerate being uneven.
# Full coverage runs on the 10k cache (TAG=full).
#
# ~19 arm runs + 2 teacher fits, roughly 3.5 h on one TITAN RTX.
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=${TAG:-pilot}
K=${K:-64}
EPOCHS=${EPOCHS:-40}
KSWEEP=${KSWEEP:-"16 64 256"}
RANDOM_SEEDS=${RANDOM_SEEDS:-"0 1 2"}
DERIVED_SEEDS=${DERIVED_SEEDS:-"0 1"}
SPECTRUM_SEEDS=${SPECTRUM_SEEDS:-"0 1"}
export HF_HOME=${HF_HOME:-/ssd2/ming/STEGO/hf_cache}

failed=()
run() {   # never let one arm abort the sweep; the gates run on whatever exists
  echo "--- $* "
  if ! "$@"; then
    echo "!!! FAILED: $*"
    failed+=("$*")
  fi
}

echo "===== Phase 2a: the Generative Observability Spectrum ====="
# Arm D's frame comes from here, and Tr(C_obs) is the number that decides whether
# Phase 3 is worth running at all: a flat spectrum kills the premise (PLAN.md §3.2).
for s in $SPECTRUM_SEEDS; do
  extra=""
  [ "$s" = "0" ] && extra="--per-attack"
  run python scripts/fit_observability_spectrum.py --tag "$TAG" --seed "$s" $extra
done

echo "===== Phase 2b: denominators, k sweep, identity + random only ====="
for k in $KSWEEP; do
  run python scripts/train_coordinates.py --tag "$TAG" --arm A_identity --k "$k" --seed 0 --epochs "$EPOCHS"
  run python scripts/train_coordinates.py --tag "$TAG" --arm C2_haar    --k "$k" --seed 0 --epochs "$EPOCHS"
  run python scripts/train_coordinates.py --tag "$TAG" --arm B_signperm --k "$k" --seed 0 --epochs "$EPOCHS"
  run python scripts/train_coordinates.py --tag "$TAG" --arm C_hadamard --k "$k" --seed 0 --epochs "$EPOCHS"
done

echo "===== Phase 3: all arms at k=$K, cross-fit ====="
for s in $RANDOM_SEEDS; do
  for arm in C2_haar B_signperm C_hadamard C3_rand_hh; do
    # k=64 seed 0 is already done by the sweep above
    [ "$K" = "64" ] && [ "$s" = "0" ] && continue
    run python scripts/train_coordinates.py --tag "$TAG" --arm "$arm" --k "$K" --seed "$s" --epochs "$EPOCHS"
  done
done
for s in $DERIVED_SEEDS; do
  run python scripts/train_coordinates.py --tag "$TAG" --arm D_spectral   --k "$K" --seed "$s" --epochs "$EPOCHS"
  run python scripts/train_coordinates.py --tag "$TAG" --arm E_learned    --k "$K" --seed "$s" --epochs "$EPOCHS"
done

echo "===== P0-7: same subspace, different coding basis ====="
# `--seed` is the BASIS; `--receiver-seed` is the extractor. Marginalising the receiver
# within each basis is what makes the reported spread attributable to the basis.
RECEIVER_SEEDS=${RECEIVER_SEEDS:-"0 1"}
for r in $RECEIVER_SEEDS; do
  run python scripts/train_coordinates.py --tag "$TAG" --arm D_spectral --k "$K" \
      --seed 0 --receiver-seed "$r" --epochs "$EPOCHS"
  for s in 0 1 2 3; do
    run python scripts/train_coordinates.py --tag "$TAG" --arm D2_rot_rand --k "$K" \
        --seed "$s" --receiver-seed "$r" --epochs "$EPOCHS"
  done
  for s in 0 1; do
    run python scripts/train_coordinates.py --tag "$TAG" --arm D3_rot_learn --k "$K" \
        --seed "$s" --receiver-seed "$r" --epochs "$EPOCHS"
  done
done

echo "===== P0-5: same operator, second decoder architecture ====="
# The certified operator is certified BY a decoder class. If the two teachers disagree
# on the subspace, the geometry is architecture-dependent and must be reported as such.
run python scripts/compare_teachers.py --tag "$TAG" --seed 0 --k "$K" --skip-fit

echo "===== P0-5: receiver-architecture control (no global pooling) ====="
# GAP discards where a feature occurred, which is what a LOCAL frame needs, so a global
# frame could win for a receiver-side reason. Locked method + Haar under both receivers.
for arm in C2_haar D_spectral E_learned; do
  run python scripts/train_coordinates.py --tag "$TAG" --arm "$arm" --k "$K" --seed 0 \
      --epochs "$EPOCHS" --extractor-arch spatial
done

echo "===== Prompt-assisted reference (diagnostic only; violates the protocol) ====="
run python scripts/ddim_reference.py --tag "$TAG" --arm C_hadamard --k "$K" --limit 128

echo "===== Selection (VAL only) — everything choosable is chosen here ====="
run python scripts/select_method.py --tag "$TAG"

echo "===== Gates (TEST, locked) ====="
# eval_coordinates cannot pick the family, the seed, k or the reference: it reads
# reports/selection_$TAG.json and refuses to run without it (P0-3).
run python scripts/eval_coordinates.py --tag "$TAG" --split test
run python scripts/eval_coordinates.py --tag "$TAG" --split test_heldout_prompts

if [ ${#failed[@]} -gt 0 ]; then
  echo "===== ${#failed[@]} step(s) failed ====="
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "===== pilot complete ====="
