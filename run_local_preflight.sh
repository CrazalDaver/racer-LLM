#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_DIR="${1:-result/local_preflight}"
TRACK_SECONDS="${TRACK_SECONDS:-3}"
MODE_SECONDS="${MODE_SECONDS:-2}"
INTERACTIVE_SECONDS="${INTERACTIVE_SECONDS:-3}"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/racer-mpl}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/racer-pycache}"

mkdir -p "$OUTPUT_DIR/logs"

echo "[1/5] Compiling Python sources"
find . -name '*.py' \
  -not -path './.git/*' \
  -not -path './.venv/*' \
  -not -path './result/*' \
  -not -path './unity/*' \
  -print0 \
  | xargs -0 "$PYTHON" -m py_compile

echo "[2/5] Checking the main CLI"
"$PYTHON" main.py --help > "$OUTPUT_DIR/logs/main_help.txt"

echo "[3/5] Running interface and three-track checks"
"$PYTHON" -m evaluation.local_preflight \
  --output-dir "$OUTPUT_DIR/core" \
  --track-seconds "$TRACK_SECONDS" \
  2>&1 | tee "$OUTPUT_DIR/logs/core.log"

echo "[4/5] Running fixed-mode short scenarios"
"$PYTHON" -m evaluation.mode_preset_audit \
  --track berlin \
  --rollout-seconds "$MODE_SECONDS" \
  --scenario clear_straight \
  --scenario clear_curve_approach \
  --scenario closing_on_front \
  --scenario rear_vehicle_closing \
  --output-dir "$OUTPUT_DIR/mode_presets" \
  2>&1 | tee "$OUTPUT_DIR/logs/mode_presets.log"

echo "[5/5] Running responsive overtake/block scenarios"
"$PYTHON" -m evaluation.interactive_mode_preset_audit \
  --total-time "$INTERACTIVE_SECONDS" \
  --output-dir "$OUTPUT_DIR/interactive_modes" \
  2>&1 | tee "$OUTPUT_DIR/logs/interactive_modes.log"

echo
echo "All local preflight checks completed."
echo "Results: $OUTPUT_DIR"
echo "Send me the terminal output or the three summary JSON files:"
echo "  $OUTPUT_DIR/core/preflight_summary.json"
echo "  $OUTPUT_DIR/mode_presets/summary.json"
echo "  $OUTPUT_DIR/interactive_modes/summary.json"
