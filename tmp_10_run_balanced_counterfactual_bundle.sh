#!/usr/bin/env bash
# TEMPORARY counterfactual bundle for the 50 s P2/P3 raw-vs-balanced traces.
#
# Primary test:
#   identical stratified sampling settings for raw and balanced traces.
# Sampling check:
#   smaller uniform-time audit of balanced traces (no GIFs).
#
# Usage:
#   ./tmp_10_run_balanced_counterfactual_bundle.sh
# Optional: SAMPLES=16 UNIFORM_SAMPLES=8 MAX_GIFS=4 INCLUDE_UNIFORM=1

set -euo pipefail
cd "$(dirname "$0")"

SAMPLES="${SAMPLES:-16}"
UNIFORM_SAMPLES="${UNIFORM_SAMPLES:-8}"
MAX_GIFS="${MAX_GIFS:-4}"
INCLUDE_UNIFORM="${INCLUDE_UNIFORM:-1}"
SOURCE_PREFIX="result/tmp_09_raw_no_default_mode_berlin_50s_p23"
OUTPUT_ROOT="result/tmp_10_balanced_counterfactual_bundle"
LOG_PATH="$OUTPUT_ROOT/runner.log"

mkdir -p "$OUTPUT_ROOT"
export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

run_audit() {
  local variant="$1"
  local position="$2"
  local strategy="$3"
  local samples="$4"
  local gifs="$5"
  local trace="${SOURCE_PREFIX}_${variant}_p${position}/summary_traces/trial_001.npz"
  local output="${OUTPUT_ROOT}/${variant}_p${position}_${strategy}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $variant P$position $strategy ($samples samples)" | tee -a "$LOG_PATH"
  TRACE="$trace" \
  OUTPUT_DIR="$output" \
  SAMPLES="$samples" \
  SAMPLING_STRATEGY="$strategy" \
  MAX_GIFS="$gifs" \
  SHORT_SECONDS=2 \
  LONG_SECONDS=5 \
  PROGRESS_RATIO=0.80 \
  ./run_counterfactual_mode_rollouts.sh 2>&1 | tee -a "$LOG_PATH"
}

for variant in raw balanced; do
  for position in 2 3; do
    run_audit "$variant" "$position" stratified "$SAMPLES" "$MAX_GIFS"
  done
done

if [[ "$INCLUDE_UNIFORM" == "1" ]]; then
  for position in 2 3; do
    run_audit balanced "$position" uniform "$UNIFORM_SAMPLES" 0
  done
fi

echo "Temporary reports:" | tee -a "$LOG_PATH"
find "$OUTPUT_ROOT" -name temporary_counterfactual_report.html -print | sort | tee -a "$LOG_PATH"
