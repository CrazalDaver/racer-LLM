#!/usr/bin/env bash
set -euo pipefail

# Detachable diagnostic for the four current LLM mode presets.
#
# Usage:
#   ./run_mode_preset_audit.sh          One 24-rollout audit
#   ./run_mode_preset_audit.sh gym      Four audits / 96 rollouts (~30 min)
#
# Optional environment variables:
#   TRACK=berlin                  berlin | modena | rounded_rectangle
#   ROLLOUT_SECONDS=3             Duration of each of the 24 branches
#   MPC_HORIZON=10                Horizon in 0.1 s MPC steps
#   VELOCITY_TRACK_WEIGHT=10      Shared low-level MPC weight
#   TRACK_SLACK_WEIGHT=30000      Shared low-level MPC slack weight
#   OUTPUT_DIR=result/...         Result directory in the single-track mode
#   ROOT_DIR=result/...           Root directory in gym mode
#   RUN_TAG=...                   Optional suffix for gym-mode output
#
# Example:
#   TRACK=berlin ROLLOUT_SECONDS=3 ./run_mode_preset_audit.sh

TRACK="${TRACK:-berlin}"
ROLLOUT_SECONDS="${ROLLOUT_SECONDS:-3}"
MPC_HORIZON="${MPC_HORIZON:-10}"
VELOCITY_TRACK_WEIGHT="${VELOCITY_TRACK_WEIGHT:-10}"
TRACK_SLACK_WEIGHT="${TRACK_SLACK_WEIGHT:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-result/mode_preset_audit_${TRACK}_${ROLLOUT_SECONDS}s}"
ROOT_DIR="${ROOT_DIR:-result/mode_preset_audit_gym${RUN_TAG:+_${RUN_TAG}}}"
MODE="${1:-single}"

run_audit() {
  local audit_track="$1"
  local audit_seconds="$2"
  local audit_output="$3"
  echo "==> ${audit_track}, ${audit_seconds}s branches -> ${audit_output}"
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/racer_llm_pycache}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/racer_llm_matplotlib}" \
  .venv/bin/python evaluation/mode_preset_audit.py \
    --track "${audit_track}" \
    --rollout-seconds "${audit_seconds}" \
    --mpc-horizon "${MPC_HORIZON}" \
    --velocity-track-weight "${VELOCITY_TRACK_WEIGHT}" \
    --track-slack-weight "${TRACK_SLACK_WEIGHT}" \
    --output-dir "${audit_output}"
}

case "${MODE}" in
  single)
    run_audit "${TRACK}" "${ROLLOUT_SECONDS}" "${OUTPUT_DIR}"
    ;;
  gym)
    mkdir -p "${ROOT_DIR}"
    {
      run_audit berlin 3 "${ROOT_DIR}/berlin_3s"
      run_audit modena 3 "${ROOT_DIR}/modena_3s"
      run_audit rounded_rectangle 3 "${ROOT_DIR}/rounded_rectangle_3s"
      run_audit berlin 5 "${ROOT_DIR}/berlin_5s"
    } 2>&1 | tee "${ROOT_DIR}/runner.log"
    echo "Completed preset-audit gym suite: ${ROOT_DIR}"
    ;;
  *)
    echo "Usage: $0 [single|gym]" >&2
    exit 2
    ;;
esac
