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
#
#   ./run_mpc_velocity_tracking_experiments.sh calibration [WEIGHT]
#       Position-3 Alpha, fixed-push, and Parameter-IBR cases (3 races).
#
# Optional environment variables:
#   TOTAL_TIME=50  Override the default 25 s race duration.
#   RUN_TAG=rho1e4_50s  Append a tag to result names to preserve earlier runs.
#   TRACK_SLACK_WEIGHT=30000  Override the default 10000 track-slack penalty.
#   DEADLOCK_RESET=1  Enable the shared sustained zero-progress reset.

set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

GROUP="${1:-}"
WEIGHT="${2:-10}"
TOTAL_TIME="${TOTAL_TIME:-25}"
RUN_TAG="${RUN_TAG:-}"
TRACK_SLACK_WEIGHT="${TRACK_SLACK_WEIGHT:-10000}"
DEADLOCK_RESET="${DEADLOCK_RESET:-0}"
DEADLOCK_RESET_ARG="--no-deadlock-reset"
if [[ "$DEADLOCK_RESET" == "1" ]]; then
  DEADLOCK_RESET_ARG="--deadlock-reset"
fi

run_case() {
  local name="$1"
  local ego_policy="$2"
  local weight="$3"
  shift 3

  if [[ -n "$RUN_TAG" ]]; then
    name="${name}_${RUN_TAG}"
  fi

  .venv/bin/python main.py \
    --name "$name" \
    --policies "$ego_policy" alpharacer alpharacer \
    --total-time "$TOTAL_TIME" \
    --mpc-horizon 10 \
    --mpc-track-slack-weight "$TRACK_SLACK_WEIGHT" \
    --mpc-velocity-track-weight "$weight" \
    --no-low-speed-boundary-recovery \
    "$DEADLOCK_RESET_ARG" \
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

if [[ "$GROUP" == "calibration" ]]; then
  run_case \
    "slack_calibration_alpha_p3" \
    alpharacer \
    "$WEIGHT" \
    --ego-position 3
  run_case \
    "slack_calibration_fixed_push_p3" \
    static \
    "$WEIGHT" \
    --ego-position 3 \
    --static-params 1.1764705882352942 1.20 0.10 0.50 0.00
  run_case \
    "slack_calibration_parameter_ibr_p3" \
    parameter_ibr \
    "$WEIGHT" \
    --ego-position 3 \
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

echo "Usage: $0 screen | baseline [WEIGHT] | calibration [WEIGHT] | llm [WEIGHT]" >&2
exit 2
