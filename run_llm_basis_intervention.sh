#!/usr/bin/env bash
# Offline two-stage probe using observations from the completed directional
# minimal-basis experiment. No simulator races are run.
#
# Usage:
#   ./run_llm_basis_intervention.sh
#   SAMPLES_PER_RELATION=2 ./run_llm_basis_intervention.sh
#
# Options:
#   INPUT_DIR: saved evidence-first result directory
#   OUTPUT_DIR: probe output directory
#   SAMPLES_PER_RELATION: samples for each of four dynamic relations (default 3)
#   LLM_MODEL: local Ollama model (default qwen2.5:7b)

set -euo pipefail
cd "$(dirname "$0")"

INPUT_DIR="${INPUT_DIR:-result/llm_directional_minimal_ab_berlin_evidence_first}"
OUTPUT_DIR="${OUTPUT_DIR:-result/llm_basis_intervention_berlin}"
SAMPLES_PER_RELATION="${SAMPLES_PER_RELATION:-3}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"

export PYTHONPYCACHEPREFIX="/tmp/racer_llm_pycache"
export MPLCONFIGDIR="/tmp/racer_llm_matplotlib"

.venv/bin/python evaluation/llm_basis_intervention_probe.py \
  "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --samples-per-relation "$SAMPLES_PER_RELATION" \
  --model "$LLM_MODEL"
