#!/usr/bin/env bash
set -euo pipefail

# Offline counterfactual mode screening.
#
# Optional environment variables:
#   TRACE          50 s NPZ to sample (default: evidence-first, position 1)
#   OUTPUT_DIR     result directory
#   SAMPLES        number of usable LLM decisions (default: 12)
#   SHORT_SECONDS  first-stage horizon (default: 2)
#   LONG_SECONDS   ambiguous-sample horizon (default: 5)
#   PROGRESS_RATIO closeness to the best change in lead over the leading opponent (default: 0.80)
#   SAMPLING_STRATEGY stratified (curves/traffic/rare modes) or uniform (default: stratified)
#   MAX_GIFS       maximum temporary local-scene GIFs (default: 4; 0 disables)
#
# Example:
#   SAMPLES=4 SHORT_SECONDS=2 LONG_SECONDS=5 ./run_counterfactual_mode_rollouts.sh

TRACE="${TRACE:-result/llm_basis_ab_berlin_rho3e4_reset_50s_evidence_first/summary_traces/trial_001.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-result/counterfactual_mode_rollout_evidence_first_p1}"
SAMPLES="${SAMPLES:-12}"
SHORT_SECONDS="${SHORT_SECONDS:-2}"
LONG_SECONDS="${LONG_SECONDS:-5}"
PROGRESS_RATIO="${PROGRESS_RATIO:-0.80}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-stratified}"
MAX_GIFS="${MAX_GIFS:-4}"

PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/racer_llm_pycache" \
MPLCONFIGDIR="${TMPDIR:-/tmp}/racer_llm_mpl" \
.venv/bin/python evaluation/counterfactual_mode_rollout.py \
  --trace "$TRACE" \
  --output-dir "$OUTPUT_DIR" \
  --samples "$SAMPLES" \
  --short-seconds "$SHORT_SECONDS" \
  --long-seconds "$LONG_SECONDS" \
  --progress-ratio "$PROGRESS_RATIO" \
  --sampling-strategy "$SAMPLING_STRATEGY" \
  --max-gifs "$MAX_GIFS"
