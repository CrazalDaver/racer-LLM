#!/usr/bin/env bash
# Run raw, structured action-first, and evidence-first LLMs against two local
# alpha-RACER cars from all three starting positions (nine races total), then
# create race and basis-grounding HTML reports.
#
# Usage:
#   ./run_llm_basis_ab.sh
#   TOTAL_TIME=15 ./run_llm_basis_ab.sh
#   TRACK_NAME=modena TOTAL_TIME=15 ./run_llm_basis_ab.sh
#   ROOT_NAME=llm_basis_ab_berlin_50s TOTAL_TIME=50 ./run_llm_basis_ab.sh
# TRACK_NAME: berlin (default), modena, or rounded_rectangle.
# ROOT_NAME: optional result-directory name, useful for preserving earlier runs.
# TRACK_SLACK_WEIGHT: shared MPC track-slack penalty (default 100000).
# DEADLOCK_RESET=1: enable the shared sustained zero-progress reset.

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-15}"
TRACK_NAME="${TRACK_NAME:-berlin}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
VELOCITY_TRACK_WEIGHT="${VELOCITY_TRACK_WEIGHT:-10}"
TRACK_SLACK_WEIGHT="${TRACK_SLACK_WEIGHT:-100000}"
DEADLOCK_RESET="${DEADLOCK_RESET:-0}"
ROOT_NAME="${ROOT_NAME:-llm_basis_track_ab_${TRACK_NAME}}"
RESULT_ROOT="result/${ROOT_NAME}"
RAW_NAME="${ROOT_NAME}_raw"
EVIDENCE_NAME="${ROOT_NAME}_evidence_first"
ACTION_NAME="${ROOT_NAME}_structured_action_first"
LOG_PATH="$RESULT_ROOT/runner.log"
DEADLOCK_RESET_ARG="--no-deadlock-reset"
if [[ "$DEADLOCK_RESET" == "1" ]]; then
  DEADLOCK_RESET_ARG="--deadlock-reset"
fi

mkdir -p "$RESULT_ROOT"
export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

run_case() {
  local name="$1"
  local candidate="$2"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $name" | tee -a "$LOG_PATH"
  .venv/bin/python main.py \
    --name "$name" \
    --policies "$candidate" alpharacer alpharacer \
    --track-name "$TRACK_NAME" \
    --total-time "$TOTAL_TIME" \
    --llm-model "$LLM_MODEL" \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --mpc-formulation constrained_local \
    --mpc-velocity-track-weight "$VELOCITY_TRACK_WEIGHT" \
    --mpc-track-slack-weight "$TRACK_SLACK_WEIGHT" \
    --no-low-speed-boundary-recovery \
    "$DEADLOCK_RESET_ARG" \
    --no-animation \
    2>&1 | tee -a "$LOG_PATH"
}

run_case "$RAW_NAME" "llm_alpharacer"
run_case "$ACTION_NAME" "llm_structured_action_first"
run_case "$EVIDENCE_NAME" "llm_evidence_first"

.venv/bin/python generate_html_report.py \
  "result/$RAW_NAME/summary.json" \
  "result/$ACTION_NAME/summary.json" \
  "result/$EVIDENCE_NAME/summary.json" \
  --output "$RESULT_ROOT/race_report.html" --no-animation \
  2>&1 | tee -a "$LOG_PATH"

.venv/bin/python evaluation/llm_scene_understanding_audit.py \
  "result/$RAW_NAME" "result/$ACTION_NAME" "result/$EVIDENCE_NAME" \
  --output-dir "$RESULT_ROOT/audit" \
  2>&1 | tee -a "$LOG_PATH"

echo "Race report: $RESULT_ROOT/race_report.html" | tee -a "$LOG_PATH"
echo "Basis audit: $RESULT_ROOT/audit/report.html" | tee -a "$LOG_PATH"
