#!/usr/bin/env bash
# Shared local/server entry point; keep execution logic in one place.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/policy/alpharacer/train/train_remote.sh" "$@"
