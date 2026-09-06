#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# TEMPORARY interaction-rich alpha-potential parameter trace.
#
# One locally trained alpha-RACER starts third against two trajectory-IBR
# opponents.  The normal NPZ trace contains all three cars' progress, speed,
# lateral position, rank-relevant longitudinal progress, safety/reset events,
# and the alpha car's five online parameters.
#
# Usage:
#   ./tmp_02_run_alpha_vs_traj_ibr_parameter_trace.sh
#
# Optional environment variables:
#   TOTAL_TIME=50
#   TRACK=berlin
#   OUTPUT_NAME=alpha_local_p3_vs_traj_ibr_parameter_trace_50s
#   ALPHA_WEIGHT=train/weights/alpha_racer/potential_local.pth
#   VELOCITY_TRACK_WEIGHT=10
#   TRACK_SLACK_WEIGHT=30000
#   DEADLOCK_RESET=1          1 enables the shared sustained-stall reset

TOTAL_TIME="${TOTAL_TIME:-50}"
TRACK="${TRACK:-berlin}"
OUTPUT_NAME="${OUTPUT_NAME:-alpha_local_p3_vs_traj_ibr_parameter_trace_50s}"
ALPHA_WEIGHT="${ALPHA_WEIGHT:-train/weights/alpha_racer/potential_local.pth}"
VELOCITY_TRACK_WEIGHT="${VELOCITY_TRACK_WEIGHT:-10}"
TRACK_SLACK_WEIGHT="${TRACK_SLACK_WEIGHT:-30000}"
DEADLOCK_RESET="${DEADLOCK_RESET:-1}"

RESET_ARG="--no-deadlock-reset"
if [[ "$DEADLOCK_RESET" == "1" ]]; then
  RESET_ARG="--deadlock-reset"
fi

RESULT_DIR="result/${OUTPUT_NAME}"
mkdir -p "$RESULT_DIR"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/racer_llm_pycache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/racer_llm_matplotlib}"

.venv/bin/python main.py \
  --name "$OUTPUT_NAME" \
  --policies alpharacer ibr ibr \
  --ego-position 3 \
  --track-name "$TRACK" \
  --total-time "$TOTAL_TIME" \
  --alpharacer-weight "$ALPHA_WEIGHT" \
  --ibr-implementation local_wang \
  --mpc-formulation constrained_local \
  --mpc-velocity-track-weight "$VELOCITY_TRACK_WEIGHT" \
  --mpc-track-slack-weight "$TRACK_SLACK_WEIGHT" \
  --no-low-speed-boundary-recovery \
  "$RESET_ARG" \
  --no-animation \
  2>&1 | tee "$RESULT_DIR/runner.log"

echo "Summary: $RESULT_DIR/summary.json"
echo "Trace:   $RESULT_DIR/summary_traces/trial_001.npz"
echo "Report:  $RESULT_DIR/report.html"
