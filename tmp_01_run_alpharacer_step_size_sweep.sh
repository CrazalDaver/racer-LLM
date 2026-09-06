#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# TEMPORARY diagnostic; safe to delete together with
# evaluation/tmp_01_alpharacer_step_size_sweep.py after choosing a step size.
#
# Usage:
#   ./tmp_01_run_alpharacer_step_size_sweep.sh
#   TOTAL_TIME=15 ./tmp_01_run_alpharacer_step_size_sweep.sh  # quick check
#
# Optional: TRACK, TOTAL_TIME, OUTPUT_DIR, VELOCITY_TRACK_WEIGHT,
# TRACK_SLACK_WEIGHT, MPC_HORIZON, and space-separated RATES.

TRACK="${TRACK:-berlin}"
TOTAL_TIME="${TOTAL_TIME:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-result/tmp_01_alpharacer_step_size_sweep}"
VELOCITY_TRACK_WEIGHT="${VELOCITY_TRACK_WEIGHT:-10}"
TRACK_SLACK_WEIGHT="${TRACK_SLACK_WEIGHT:-30000}"
MPC_HORIZON="${MPC_HORIZON:-10}"
RATES_STRING="${RATES:-1e-5 3e-5 1e-4}"
read -r -a RATE_ARGS <<< "$RATES_STRING"

mkdir -p "$OUTPUT_DIR"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/racer_llm_pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/racer_llm_matplotlib}"

.venv/bin/python evaluation/tmp_01_alpharacer_step_size_sweep.py \
  --rates "${RATE_ARGS[@]}" \
  --total-time "$TOTAL_TIME" \
  --track "$TRACK" \
  --output-dir "$OUTPUT_DIR" \
  --velocity-track-weight "$VELOCITY_TRACK_WEIGHT" \
  --track-slack-weight "$TRACK_SLACK_WEIGHT" \
  --mpc-horizon "$MPC_HORIZON" \
  2>&1 | tee "$OUTPUT_DIR/runner.log"
