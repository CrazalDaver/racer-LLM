#!/usr/bin/env bash
# TEMPORARY clean prompt A/B: original raw LLM versus the same raw prompt with
# one neutral no-default-mode sentence. Both race the same two alpha-RACER cars
# from all three starting positions.
#
# Usage:
#   TOTAL_TIME=15 ./tmp_09_run_raw_no_default_mode_ab.sh
#   TOTAL_TIME=50 EGO_POSITIONS="2 3" ./tmp_09_run_raw_no_default_mode_ab.sh
# Optional: TRACK_NAME, LLM_MODEL, ROOT_NAME, EGO_POSITIONS (space-separated).

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-15}"
TRACK_NAME="${TRACK_NAME:-berlin}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
ROOT_NAME="${ROOT_NAME:-tmp_09_raw_no_default_mode_${TRACK_NAME}}"
EGO_POSITIONS="${EGO_POSITIONS:-1 2 3}"
RESULT_ROOT="result/${ROOT_NAME}"
RAW_NAME="${ROOT_NAME}_raw"
BALANCED_NAME="${ROOT_NAME}_balanced"
LOG_PATH="$RESULT_ROOT/runner.log"
SUMMARY_PATHS=()

mkdir -p "$RESULT_ROOT"
export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

run_case() {
  local base_name="$1"
  local policy="$2"
  local position="$3"
  local name="${base_name}_p${position}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $policy at P$position" | tee -a "$LOG_PATH"
  .venv/bin/python main.py \
    --name "$name" \
    --policies "$policy" alpharacer alpharacer \
    --ego-position "$position" \
    --track-name "$TRACK_NAME" \
    --total-time "$TOTAL_TIME" \
    --llm-model "$LLM_MODEL" \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --mpc-formulation constrained_local \
    --mpc-velocity-track-weight 10 \
    --mpc-track-slack-weight 30000 \
    --deadlock-reset \
    --no-low-speed-boundary-recovery \
    --no-animation 2>&1 | tee -a "$LOG_PATH"
  SUMMARY_PATHS+=("result/$name/summary.json")
}

for position in $EGO_POSITIONS; do
  run_case "$RAW_NAME" llm_alpharacer "$position"
done
for position in $EGO_POSITIONS; do
  run_case "$BALANCED_NAME" llm_raw_balanced "$position"
done

.venv/bin/python generate_html_report.py \
  "${SUMMARY_PATHS[@]}" \
  --output "$RESULT_ROOT/race_report.html" --no-animation 2>&1 | tee -a "$LOG_PATH"

echo "Temporary A/B report: $RESULT_ROOT/race_report.html" | tee -a "$LOG_PATH"
