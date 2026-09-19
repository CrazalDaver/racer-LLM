#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")"

PROFILE="${1:-screen}"
case "$PROFILE" in
  screen) DEFAULT_DURATION=20 ;;
  full) DEFAULT_DURATION=50 ;;
  *) echo "Usage: $0 [screen|full]" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-qwen2.5:7b}"
TRACK="${TRACK:-berlin}"
DURATION="${DURATION:-$DEFAULT_DURATION}"
POSITIONS="${POSITIONS:-1 2 3}"
VARIANTS="${VARIANTS:-baseline all_opponents no_lap_progress raceline_speed decision_feedback}"
OPPONENT_POLICIES="${OPPONENT_POLICIES:-alpharacer}"
RUN_REFERENCE_BASELINES="${RUN_REFERENCE_BASELINES:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
START_OLLAMA="${START_OLLAMA:-1}"
RUN_PREFIX="${RUN_PREFIX:-observation_ablation_${PROFILE}_${TRACK}}"
LOG_ROOT="result/${RUN_PREFIX}_logs"
SUMMARY_ROOT="result/${RUN_PREFIX}_summary"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/racer-mpl}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/racer-pycache}"
mkdir -p "$LOG_ROOT" "$SUMMARY_ROOT"

OLLAMA_PID=""
cleanup() {
  if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ollama_ready() {
  curl --max-time 2 --silent --fail http://localhost:11434/api/tags >/dev/null 2>&1
}

if ! ollama_ready; then
  if [[ "$START_OLLAMA" != "1" ]] || ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is not reachable at http://localhost:11434." >&2
    echo "Start it with 'ollama serve', then rerun this script." >&2
    exit 1
  fi
  echo "Starting a temporary Ollama server..."
  ollama serve >"$LOG_ROOT/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 60); do
    ollama_ready && break
    sleep 1
  done
  if ! ollama_ready; then
    echo "Ollama did not become ready; inspect $LOG_ROOT/ollama.log" >&2
    exit 1
  fi
fi

if ! ollama show "$MODEL" >/dev/null 2>&1; then
  echo "Model '$MODEL' is not installed. Run: ollama pull $MODEL" >&2
  exit 1
fi

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  echo "[preflight] observation units, golden snapshot, and variant isolation"
  if ! "$PYTHON" - <<'PY'
from evaluation.local_preflight import (
    run_observation_checks,
    run_observation_snapshot_check,
    run_observation_variant_checks,
)
run_observation_checks()
run_observation_snapshot_check()
run_observation_variant_checks()
PY
  then
    echo "Preflight failed; no experiment was started." >&2
    exit 1
  fi
fi

failures=0
result_dirs=()

run_case() {
  local name="$1"
  shift
  local output_dir="result/$name"
  local log_path="$LOG_ROOT/$name.log"
  result_dirs+=("$output_dir")
  if [[ -f "$output_dir/summary.json" ]]; then
    echo "[skip] $name (summary already exists)"
    return
  fi
  echo "[run] $name"
  "$PYTHON" main.py \
    --name "$name" \
    --track-name "$TRACK" \
    --total-time "$DURATION" \
    --llm-model "$MODEL" \
    --llm-output-mode mode \
    --llm-temperature 0 \
    --no-animation \
    "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[failed:$status] $name" >&2
    failures=$((failures + 1))
  fi
}

for opponent in $OPPONENT_POLICIES; do
  for variant in $VARIANTS; do
    for position in $POSITIONS; do
      case_name="${RUN_PREFIX}_${opponent}_${variant}_p${position}"
      run_case "$case_name" \
        --policies llm_alpharacer "$opponent" "$opponent" \
        --ego-position "$position" \
        --llm-observation-variant "$variant"
    done
  done

  if [[ "$RUN_REFERENCE_BASELINES" == "1" ]]; then
    for ego_policy in alpharacer ibr rule_based; do
      for position in $POSITIONS; do
        case_name="${RUN_PREFIX}_${opponent}_reference_${ego_policy}_p${position}"
        run_case "$case_name" \
          --policies "$ego_policy" "$opponent" "$opponent" \
          --ego-position "$position" \
          --llm-observation-variant baseline
      done
    done
  fi
done

existing_dirs=()
for directory in "${result_dirs[@]}"; do
  [[ -d "$directory" ]] && existing_dirs+=("$directory")
done
if [[ ${#existing_dirs[@]} -gt 0 ]]; then
  "$PYTHON" -m evaluation.summarize_observation_ablation \
    "${existing_dirs[@]}" \
    --output-dir "$SUMMARY_ROOT"
fi

echo
echo "Observation ablation run complete."
echo "Profile: $PROFILE, duration: ${DURATION}s, track: $TRACK, model: $MODEL"
echo "Aggregate summary: $SUMMARY_ROOT/aggregate.csv"
echo "Per-race summary:  $SUMMARY_ROOT/races.csv"
echo "Failures: $failures"
exit "$failures"
