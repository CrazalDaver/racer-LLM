#!/usr/bin/env bash
# Nine-race regression screen on the shared Frenet MPC:
#   raw mode LLM, explicit-rule-prompt LLM, and deterministic rule policy
#   each race two locally trained alpha-RACER opponents from P1, P2, and P3.
#
# Usage:
#   ./run_llm_alpha_frenet_screen.sh
#   TOTAL_TIME=40 ./run_llm_alpha_frenet_screen.sh
#
# Each main.py call automatically runs all three ego starting positions. Raw
# JSON/NPZ files are retained, and the final command combines all nine races
# into one static HTML report.

set -euo pipefail

cd "$(dirname "$0")"

TOTAL_TIME="${TOTAL_TIME:-25}"
RESULT_ROOT="result/frenet_llm_alpha_screen"
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
    --total-time "$TOTAL_TIME" \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --mpc-formulation constrained_local \
    --mpc-velocity-track-weight 0 \
    --no-low-speed-boundary-recovery \
    --no-animation \
    2>&1 | tee -a "$LOG_PATH"
}

run_case "frenet_llm_alpha_screen_raw_llm" "llm_alpharacer"
run_case "frenet_llm_alpha_screen_explicit_rule_llm" "llm_explicit_rules"
run_case "frenet_llm_alpha_screen_rule_only" "rule_based"

.venv/bin/python generate_html_report.py \
  result/frenet_llm_alpha_screen_raw_llm/summary.json \
  result/frenet_llm_alpha_screen_explicit_rule_llm/summary.json \
  result/frenet_llm_alpha_screen_rule_only/summary.json \
  --output "$RESULT_ROOT/report.html" \
  --no-animation \
  2>&1 | tee -a "$LOG_PATH"

echo "Complete report: $RESULT_ROOT/report.html" | tee -a "$LOG_PATH"
