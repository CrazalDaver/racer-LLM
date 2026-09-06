#!/usr/bin/env bash
# TEMPORARY A/B screen: one-call evidence-first versus the detachable two-call
# variant. Both use identical observations, presets, MPC, model, and temperature.
# Usage: TOTAL_TIME=15 ./tmp_08_run_one_vs_two_step_evidence.sh

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-15}"
TRACK_NAME="${TRACK_NAME:-berlin}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
ROOT_NAME="${ROOT_NAME:-tmp_08_one_vs_two_step_${TRACK_NAME}}"
RESULT_ROOT="result/${ROOT_NAME}"
ONE_NAME="${ROOT_NAME}_one_call"
TWO_NAME="${ROOT_NAME}_two_call"
LOG_PATH="$RESULT_ROOT/runner.log"

mkdir -p "$RESULT_ROOT"
export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

run_case() {
  local name="$1"
  local policy="$2"
  .venv/bin/python main.py \
    --name "$name" \
    --policies "$policy" alpharacer alpharacer \
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
}

run_case "$ONE_NAME" llm_evidence_first
run_case "$TWO_NAME" llm_two_step_evidence

.venv/bin/python generate_html_report.py \
  "result/$ONE_NAME/summary.json" \
  "result/$TWO_NAME/summary.json" \
  --output "$RESULT_ROOT/race_report.html" --no-animation 2>&1 | tee -a "$LOG_PATH"

echo "Temporary A/B report: $RESULT_ROOT/race_report.html" | tee -a "$LOG_PATH"
