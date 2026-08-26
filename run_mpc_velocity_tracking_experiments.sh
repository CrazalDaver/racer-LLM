#!/usr/bin/env bash
# Reproducible velocity-tracking experiments for the shared MPC.
#
# Usage (run from anywhere):
#   ./run_mpc_velocity_tracking_experiments.sh screen
#       P3 hairpin screen at weights 1, 3, and 10 (3 races).
#
#   ./run_mpc_velocity_tracking_experiments.sh baseline [WEIGHT]
#       Three-position Alpha self-play, rule-vs-Alpha, and
#       Parameter-IBR-vs-Alpha regression (9 races; default WEIGHT=10).
#
#   ./run_mpc_velocity_tracking_experiments.sh llm [WEIGHT]
#       Three-position raw-LLM and explicit-rule-LLM comparisons against
#       two Alpha cars (6 races and therefore substantially slower).

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

GROUP="${1:-}"
WEIGHT="${2:-10}"

run_case() {
  local name="$1"
  local ego_policy="$2"
  local weight="$3"
  shift 3

  .venv/bin/python main.py \
    --name "$name" \
    --policies "$ego_policy" alpharacer alpharacer \
    --total-time 25 \
    --mpc-horizon 10 \
    --mpc-track-slack-weight 10000 \
    --mpc-velocity-track-weight "$weight" \
    --mpc-terminal-viability-weight 0 \
    --mpc-stall-debt-max-weight 0 \
    --no-low-speed-boundary-recovery \
    --no-animation \
    "$@"
}

if [[ "$GROUP" == "screen" ]]; then
  for weight in 1 3 10; do
    run_case \
      "velocity_screen_w${weight}_p3" \
      rule_based \
      "$weight" \
      --ego-position 3
  done
  exit 0
fi

if [[ "$GROUP" == "baseline" ]]; then
  run_case "velocity_w${WEIGHT}_alpha_selfplay" alpharacer "$WEIGHT"
  run_case "velocity_w${WEIGHT}_rule_vs_alpha" rule_based "$WEIGHT"
  run_case \
    "velocity_w${WEIGHT}_parameter_ibr_vs_alpha" \
    parameter_ibr \
    "$WEIGHT" \
    --parameter-ibr-rounds 1
  exit 0
fi

if [[ "$GROUP" == "llm" ]]; then
  run_case \
    "velocity_w${WEIGHT}_raw_llm_vs_alpha" \
    llm_alpharacer \
    "$WEIGHT" \
    --llm-output-mode mode \
    --llm-temperature 0
  run_case \
    "velocity_w${WEIGHT}_rule_prompt_llm_vs_alpha" \
    llm_explicit_rules \
    "$WEIGHT" \
    --llm-output-mode mode \
    --llm-temperature 0
  exit 0
fi

echo "Usage: $0 screen | baseline [WEIGHT] | llm [WEIGHT]" >&2
exit 2
