#!/usr/bin/env bash
# TEMPORARY balanced+two-step screen followed by stratified counterfactual
# rollouts. The basis stage is unchanged; only the mode stage receives the
# same no-default-mode sentence used by llm_raw_balanced.
#
# Usage:
#   ./tmp_11_run_balanced_two_step_screen.sh
# Optional: TOTAL_TIME=20 EGO_POSITIONS="2 3" SAMPLES=12 MAX_GIFS=4

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-20}"
EGO_POSITIONS="${EGO_POSITIONS:-2 3}"
SAMPLES="${SAMPLES:-12}"
MAX_GIFS="${MAX_GIFS:-4}"
ROOT_NAME="${ROOT_NAME:-tmp_11_balanced_two_step_20s}"
RESULT_ROOT="result/$ROOT_NAME"
LOG_PATH="$RESULT_ROOT/runner.log"
SUMMARY_PATHS=()

mkdir -p "$RESULT_ROOT"
export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

for position in $EGO_POSITIONS; do
  name="${ROOT_NAME}_p${position}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] balanced two-step P$position" | tee -a "$LOG_PATH"
  .venv/bin/python main.py \
    --name "$name" \
    --policies llm_balanced_two_step_evidence alpharacer alpharacer \
    --ego-position "$position" \
    --track-name berlin \
    --total-time "$TOTAL_TIME" \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --mpc-formulation constrained_local \
    --mpc-velocity-track-weight 10 \
    --mpc-track-slack-weight 30000 \
    --deadlock-reset \
    --no-low-speed-boundary-recovery \
    --no-animation 2>&1 | tee -a "$LOG_PATH"
  SUMMARY_PATHS+=("result/$name/summary.json")

  TRACE="result/$name/summary_traces/trial_001.npz" \
  OUTPUT_DIR="$RESULT_ROOT/counterfactual_p${position}" \
  SAMPLES="$SAMPLES" \
  SAMPLING_STRATEGY=stratified \
  MAX_GIFS="$MAX_GIFS" \
  SHORT_SECONDS=2 \
  LONG_SECONDS=5 \
  PROGRESS_RATIO=0.80 \
  ./run_counterfactual_mode_rollouts.sh 2>&1 | tee -a "$LOG_PATH"
done

.venv/bin/python generate_html_report.py \
  "${SUMMARY_PATHS[@]}" \
  --output "$RESULT_ROOT/race_report.html" --no-animation 2>&1 | tee -a "$LOG_PATH"

echo "Race report: $RESULT_ROOT/race_report.html" | tee -a "$LOG_PATH"
find "$RESULT_ROOT" -name temporary_counterfactual_report.html -print | sort | tee -a "$LOG_PATH"
