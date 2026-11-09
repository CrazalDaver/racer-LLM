#!/usr/bin/env bash
# Stable experiment paths; each Python stage decides whether to reuse or resume.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'HELP'
Usage: bash train_alpharacer.sh [options]
  --work-dir DIR       Fixed experiment directory (default outputs/alpharacer/berlin-dbm)
  --stage STAGE        status | collect | train | own-values | relative-values |
                       potential | evaluate | all (default all)
  --workers N|auto     CPU collection workers; auto uses SLURM_CPUS_PER_TASK
  --device DEVICE      auto | cpu | cuda
  --check-only         Validate the environment without running stages
  --dry-run            Print the command without creating files or installing packages
  --early-stopping-patience N   Epochs without improvement (default 1000; 0 disables)

Completed models are reused; incomplete checkpoints resume automatically.
The same command can be submitted again after interruption. Job IDs do not set paths.
Other training options are forwarded to policy.alpharacer.train.training.
Environment: PYTHON_BIN selects Python; INSTALL_DEPS=0 prevents dependency installation.
HELP
}

WORKERS=1
CHECK_ONLY=0
DRY_RUN=0
TRAIN_ARGS=()
while (($#)); do
    case "$1" in
        --workers)
            [[ $# -ge 2 ]] || { echo "--workers needs a positive integer or auto" >&2; exit 2; }
            WORKERS="$2"; shift 2 ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --check-only) CHECK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --output-dir|--resume)
            echo "Use --work-dir DIR; reuse and resume are automatic." >&2; exit 2 ;;
        --n-workers|--n-workers=*) echo "Use --workers to set CPU collection workers." >&2; exit 2 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done
[[ "$WORKERS" != auto ]] || WORKERS="${SLURM_CPUS_PER_TASK:-1}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --workers value" >&2; exit 2; }
TRAIN_PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TRAIN_COMMAND=("$TRAIN_PYTHON" -u -m policy.alpharacer.train.training)
if ((${#TRAIN_ARGS[@]})); then TRAIN_COMMAND+=("${TRAIN_ARGS[@]}"); fi
TRAIN_COMMAND+=(--n-workers "$WORKERS")
printf 'Repository: %s\nCommand:' "$REPO_ROOT"
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
if ((DRY_RUN)); then exit 0; fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
if [[ -z "${PYTHON_BIN:-}" && ! -x "$TRAIN_PYTHON" ]]; then python3 -m venv "$REPO_ROOT/.venv"; fi
if ! "$TRAIN_PYTHON" - "$REPO_ROOT/requirements.txt" <<'PY'
import importlib.metadata
import sys
problems = []
for line in open(sys.argv[1]):
    requirement = line.partition('#')[0].strip()
    if not requirement:
        continue
    name, version = requirement.split('==')
    try:
        installed = importlib.metadata.version(name).split('+')[0]
    except importlib.metadata.PackageNotFoundError:
        installed = 'missing'
    if installed != version:
        problems.append(f'{name}: need {version}, found {installed}')
print('\n'.join(problems) if problems else 'Pinned dependencies found.')
sys.exit(bool(problems))
PY
then
    [[ "${INSTALL_DEPS:-1}" == 1 ]] || { echo "Dependencies do not match; INSTALL_DEPS=0" >&2; exit 2; }
    "$TRAIN_PYTHON" -m pip install -r requirements.txt
fi

if ((CHECK_ONLY)); then
    "$TRAIN_PYTHON" - "${TRAIN_COMMAND[@]:4}" <<'PY'
import sys
import casadi as ca
import torch
from policy.alpharacer.train.training import parse_args, training_device
from config import params
from track import configure_track, load_centerline_csv, load_raceline_csv
sys.argv = ['training', *sys.argv[1:]]
args = parse_args()
configure_track(params, args.track)
load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE)
load_raceline_csv(params.RACELINE_CSV, scale=params.TRACK_SCALE)
device = training_device(args)
torch.ones(1, device=device).sum().item()
x = ca.SX.sym('x')
solver = ca.nlpsol('preflight', 'ipopt', {'x': x, 'f': x*x}, {'ipopt.print_level': 0, 'print_time': 0})
solver(x0=1.)
assert solver.stats()['success']
print('Environment check passed. Device:', device, 'Work directory:', args.work_dir)
PY
    exit 0
fi

DETAILS="$("$TRAIN_PYTHON" -c 'from policy.alpharacer.train.training import parse_args; a=parse_args(); print(a.stage+"\t"+str(a.work_dir))' "${TRAIN_COMMAND[@]:4}")"
STAGE="${DETAILS%%$'\t'*}"
WORK_DIR="${DETAILS#*$'\t'}"
if [[ "$STAGE" == status ]]; then exec "${TRAIN_COMMAND[@]}"; fi
mkdir -p "$WORK_DIR/logs"
export MPLCONFIGDIR="$WORK_DIR/logs/matplotlib"
"$TRAIN_PYTHON" -m pip freeze > "$WORK_DIR/logs/pip-freeze.txt"
printf 'Work directory: %s\n' "$WORK_DIR"
"${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$WORK_DIR/logs/$STAGE.log"
