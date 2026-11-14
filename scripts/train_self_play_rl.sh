#!/usr/bin/env bash
# Local/server entry point for the native Torch self-play baseline.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'HELP'
Usage: bash scripts/train_self_play_rl.sh [options]
  --work-dir DIR    Stable resume directory (default outputs/self_play_rl/berlin-dbm)
  --output PATH     Runtime checkpoint (default policy/self_play_rl/weights/self_play_rl_policy.pth)
  --track NAME      Registered track (default berlin)
  --workers N       CPU rollout workers/environments (default 8; paper uses 8)
  --device DEVICE   auto | cpu | cuda | cuda:N (one training GPU; rollout stays on CPU)
  --seed N          Training seed
  --smoke           Run the short shared-policy three-phase pipeline check
  --check-only      Validate imports, track and selected device
  --dry-run         Print the command only

Training resumes automatically from WORK_DIR/trainer.pth. Algorithm and
self-play hyperparameters are fixed in policy.self_play_rl.train.
HELP
}

CHECK_ONLY=0
DRY_RUN=0
TRAIN_ARGS=()
while (($#)); do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done

TRAIN_PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
COMMAND=("$TRAIN_PYTHON" -u -m policy.self_play_rl.train)
if ((${#TRAIN_ARGS[@]})); then COMMAND+=("${TRAIN_ARGS[@]}"); fi
printf 'Repository: %s\nCommand:' "$REPO_ROOT"
printf ' %q' "${COMMAND[@]}"
printf '\n'
if ((DRY_RUN)); then exit 0; fi
[[ -x "$TRAIN_PYTHON" ]] || { echo "Missing Python: $TRAIN_PYTHON" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if ((CHECK_ONLY)); then
    "$TRAIN_PYTHON" - "${TRAIN_ARGS[@]}" <<'PY'
import sys
import torch
from config import params
from policy.self_play_rl.network import PPOActorCritic
from policy.self_play_rl.train import parse_args
from track import configure_track, load_centerline_csv

args = parse_args(sys.argv[1:])
configure_track(params, args.track)
load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE)
device = torch.device(
    'cuda' if args.device == 'auto' and torch.cuda.is_available()
    else 'cpu' if args.device == 'auto' else args.device
)
PPOActorCritic().to(device).actor(torch.zeros(1, 15, device=device))
print('Environment check passed. Device:', device, 'Workers:', args.workers)
PY
    exit 0
fi

WORK_DIR="$($TRAIN_PYTHON -c 'from policy.self_play_rl.train import parse_args; print(parse_args().work_dir)' "${TRAIN_ARGS[@]}")"
mkdir -p "$WORK_DIR/logs"
"${COMMAND[@]}" 2>&1 | tee -a "$WORK_DIR/logs/train.log"
