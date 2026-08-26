#!/usr/bin/env bash
# Minimal P3 hairpin horizon screen. N=10 is already covered by the saved
# 50-second failures in result/frenet_llm_alpha_screen_*.
#
# Stage 1 (recommended first):
#   ./run_mpc_horizon_hairpin_screen.sh rule
#
# Stage 2, only if N=15 or N=20 lets rule-only pass the hairpin:
#   ./run_mpc_horizon_hairpin_screen.sh confirm 15
#   ./run_mpc_horizon_hairpin_screen.sh confirm 20
#
# Each run lasts 25 simulated seconds, enough to reach and leave s ~= 64 m.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

MODE="${1:-rule}"

run_case() {
  local name="$1"
  local candidate="$2"
  local horizon="$3"

  .venv/bin/python main.py \
    --name "$name" \
    --policies "$candidate" alpharacer alpharacer \
    --ego-position 3 \
    --total-time 25 \
    --mpc-horizon "$horizon" \
    --mpc-formulation constrained_local \
    --mpc-velocity-track-weight 0 \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --no-low-speed-boundary-recovery \
    --no-animation
}

if [[ "$MODE" == "rule" ]]; then
  run_case "hairpin_rule_only_h15" "rule_based" 15
  run_case "hairpin_rule_only_h20" "rule_based" 20
  exit 0
fi

if [[ "$MODE" == "confirm" ]]; then
  HORIZON="${2:-}"
  if [[ "$HORIZON" != "15" && "$HORIZON" != "20" ]]; then
    echo "Usage: $0 confirm 15|20" >&2
    exit 2
  fi
  run_case "hairpin_raw_llm_h${HORIZON}" "llm_alpharacer" "$HORIZON"
  run_case "hairpin_alpha_control_h${HORIZON}" "alpharacer" "$HORIZON"
  exit 0
fi

echo "Usage: $0 rule | $0 confirm 15|20" >&2
exit 2
