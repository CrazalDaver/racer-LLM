#!/usr/bin/env bash
set -euo pipefail

experiment_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${experiment_dir}/../.." && pwd)"
stage="${1:-smoke}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/racer-llm-matplotlib"

cd "${repository_root}"

case "${stage}" in
  preflight)
    .venv/bin/python protocols/validate_k_constrained_v1.py
    .venv/bin/python experiments/llm_param_minimal_v1/preflight.py
    ;;
  smoke|stage1|stage2)
    .venv/bin/python protocols/validate_k_constrained_v1.py
    .venv/bin/python experiments/llm_param_minimal_v1/preflight.py
    .venv/bin/python experiments/llm_param_minimal_v1/run.py \
      --config experiments/llm_param_minimal_v1/experiment.json \
      --stage "${stage}"
    ;;
  *)
    echo "usage: $0 {preflight|smoke|stage1|stage2}" >&2
    exit 2
    ;;
esac
