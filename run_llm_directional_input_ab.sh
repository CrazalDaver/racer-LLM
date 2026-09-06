#!/usr/bin/env bash
# Compare the raw LLM and minimal traffic-only evidence-first pipeline after
# replacing signed observation fields with explicit direction/trend plus
# non-negative magnitude.
# Each main.py call runs the candidate from all three starting positions. An
# existing directional-input raw run is reused because only the pipeline was
# changed; raw is run automatically only when that result is missing.
#
# Usage:
#   ./run_llm_directional_input_ab.sh
#   TOTAL_TIME=15 ./run_llm_directional_input_ab.sh
#   TRACK_NAME=modena TOTAL_TIME=15 ./run_llm_directional_input_ab.sh
#
# Options:
#   TOTAL_TIME: simulated seconds per race (default: 15)
#   TRACK_NAME: berlin, modena, or rounded_rectangle (default: berlin)
#   LLM_MODEL: local Ollama model name (default: qwen2.5:7b)
#   VELOCITY_TRACK_WEIGHT: shared MPC velocity tracking weight (default: 10)

set -euo pipefail
cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-15}"
TRACK_NAME="${TRACK_NAME:-berlin}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
VELOCITY_TRACK_WEIGHT="${VELOCITY_TRACK_WEIGHT:-10}"
ROOT_NAME="llm_directional_minimal_ab_${TRACK_NAME}"
RESULT_ROOT="result/${ROOT_NAME}"
RAW_NAME="llm_directional_input_ab_${TRACK_NAME}_raw"
PIPELINE_NAME="${ROOT_NAME}_evidence_first"
LOG_PATH="$RESULT_ROOT/runner.log"

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
    --no-low-speed-boundary-recovery \
    --no-animation \
    2>&1 | tee -a "$LOG_PATH"
}

if [[ ! -f "result/$RAW_NAME/summary.json" ]]; then
  run_case "$RAW_NAME" "llm_alpharacer"
else
  echo "Reusing unchanged raw result: result/$RAW_NAME/summary.json" | tee -a "$LOG_PATH"
fi
run_case "$PIPELINE_NAME" "llm_evidence_first"

.venv/bin/python generate_html_report.py \
  "result/$RAW_NAME/summary.json" \
  "result/$PIPELINE_NAME/summary.json" \
  --output "$RESULT_ROOT/race_report.html" --no-animation \
  2>&1 | tee -a "$LOG_PATH"

.venv/bin/python evaluation/llm_scene_understanding_audit.py \
  "result/$RAW_NAME" "result/$PIPELINE_NAME" \
  --output-dir "$RESULT_ROOT/audit" \
  2>&1 | tee -a "$LOG_PATH"

echo "Race report: $RESULT_ROOT/race_report.html" | tee -a "$LOG_PATH"
echo "Basis audit: $RESULT_ROOT/audit/report.html" | tee -a "$LOG_PATH"
