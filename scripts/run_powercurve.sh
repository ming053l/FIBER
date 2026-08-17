#!/usr/bin/env bash
# EXPLORATORY POWER DIAGNOSTIC. Not a gate, not a selection, not evidence for any claim
# about the channel. Registered in reports/powercurve.md BEFORE it was run.
#
# Question: at pilot scale, is the experiment sample-limited or is there nothing to find?
# The triage put every arm at BER 0.5 with per-coordinate correlations whose entire
# distribution matches pure noise (variance ratio 0.968, max |rho| 0.2234 against an
# expected noise maximum of 0.2249). Before paying ~4.4 GPU-hours to cache 10k images,
# measure whether ANY of the readouts move with the training-set size.
#
# Design:
#   * NESTED random subsets: N=100 is a subset of N=300 is a subset of all of split B,
#     at each subset seed. So the curve varies size and nothing else. A prefix would
#     confound size with composition -- the record order carries the sample index, and
#     therefore the latent seed and the prompt draw.
#   * CONSTANT OPTIMISATION STEPS, not constant epochs. At fixed epochs a small-N run
#     gets proportionally fewer gradient steps, so a flat curve could mean "not enough
#     steps" rather than "not enough data". Holding steps fixed makes data quantity the
#     isolated variable. Target 2320 steps = 40 epochs x ceil(928/16), the gate budget.
#   * Two arms only: C2_haar is the null the claim is against, D_spectral the derived
#     one. The question is whether EITHER moves, and whether they separate.
#   * --scope powercurve, so select_method.py cannot see any of it (it reads
#     analysis_scope == "gate" only), and the stem carries n/ss so the points cannot
#     overwrite each other.
#   * tag triage1: same cache, same frozen spectrum, same everything but N. That lock
#     is already written and is write-once, so nothing here can reach it.
#
# Read the result from BER, coord_mse, pearson_mean AND per-coordinate rho -- sign BER
# is saturated at 0.5 and has no resolving power at this effect size.
set -u
TAG=${TAG:-triage1}
CACHE=${CACHE:-pilot}
K=${K:-64}
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

# N -> epochs, chosen so epochs * ceil(N/16) ~= 2320
declare -A EPOCHS=( [100]=331 [300]=122 [928]=40 )

for N in 100 300 928; do
  for ARM in C2_haar D_spectral; do
    for S in 0 1 2; do
      echo "=== N=$N $ARM subset_seed=$S epochs=${EPOCHS[$N]} ==="
      python -W ignore scripts/train_coordinates.py \
        --tag "$TAG" --cache-tag "$CACHE" --arm "$ARM" --seed 0 --k "$K" \
        --subset-size "$N" --subset-seed "$S" --receiver-seed "$S" \
        --scope powercurve --epochs "${EPOCHS[$N]}" \
        --device "$DEV" || { FAILURES=$((FAILURES+1)); echo "FAILED N=$N $ARM s=$S"; }
    done
  done
done
if [ "$FAILURES" -ne 0 ]; then
  echo "$FAILURES runs FAILED -- the sweep is incomplete and must not be read as one"
  exit 1
fi
echo "done"
