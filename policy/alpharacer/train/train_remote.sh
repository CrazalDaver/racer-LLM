#!/usr/bin/env bash
# Run from a transferred checkout; all paths are resolved from this script.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'HELP'
Usage: bash policy/alpharacer/train/train_remote.sh [wrapper options] [training options]

Wrapper options:
  --output-dir DIR  New, empty run directory (default: outputs/alpharacer/<UTC>-<PID>)
  --resume DIR      Continue a previously started run directory
  --workers N       Rollout worker processes, or auto to use SLURM_CPUS_PER_TASK (default: 1)
  --check-only      Check dependencies, CUDA, assets and IPOPT; do not train
  --dry-run         Print the command; do not install, create files or train
  -h, --help        Show this help

All other arguments are forwarded to policy.alpharacer.train.training.
Defaults: DBM, Berlin, 4000 valid training races + 500 validation + 500 test,
50s races, gamma=1, 350-step returns, 50000 value / 100000 potential epochs.
Example: bash train_alpharacer.sh --workers auto --track berlin --seed 7

Environment:
  PYTHON_BIN    Existing Python interpreter; otherwise use/create .venv/bin/python
  INSTALL_DEPS  1 (default): install pinned requirements if missing/mismatched;
                0: report dependency problems without installing

The process stays in the foreground. Use nohup or a persistent server session.
Collection and training state are saved for continuation with --resume.
HELP
}

RUN_DIR=""
RESUME_RUN=0
CHECK_ONLY=0
DRY_RUN=0
WORKERS=1
TRAIN_ARGS=()
while (($#)); do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 && -n "$2" && "$RESUME_RUN" == 0 ]] || { echo "--output-dir requires a directory and cannot be combined with --resume" >&2; exit 2; }
            RUN_DIR="$2"; shift 2 ;;
        --resume)
            [[ $# -ge 2 && -n "$2" && -z "$RUN_DIR" ]] || { echo "--resume requires a run directory and cannot be combined with --output-dir" >&2; exit 2; }
            RUN_DIR="$2"; RESUME_RUN=1; shift 2 ;;
        --workers)
            [[ $# -ge 2 && -n "$2" ]] || { echo "--workers requires a positive integer or auto" >&2; exit 2; }
            WORKERS="$2"; shift 2 ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --n-workers|--n-workers=*)
            echo "Use --workers to select the number of rollout workers." >&2; exit 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --data|--data=*|--value-dir|--value-dir=*|--potential-output|--potential-output=*)
            echo "Use --output-dir to select artifact paths; $1 is managed by this script." >&2
            exit 2 ;;
        --) shift ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done
if [[ "$WORKERS" == auto ]]; then
    WORKERS="${SLURM_CPUS_PER_TASK:-1}"
fi
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--workers requires a positive integer or auto" >&2; exit 2;
}

RUN_DIR="${RUN_DIR:-outputs/alpharacer/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ "$RUN_DIR" = /* ]] || RUN_DIR="$REPO_ROOT/$RUN_DIR"
TRAIN_PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
TRAIN_COMMAND=("$TRAIN_PYTHON" -u -m policy.alpharacer.train.training
    --vehicle-model dbm --track berlin
    --data "$RUN_DIR/rollouts.npz"
    --value-dir "$RUN_DIR/values"
    --potential-output "$RUN_DIR/potential.pth" --n-workers "$WORKERS" --resumable)
if ((${#TRAIN_ARGS[@]})); then
    TRAIN_COMMAND+=("${TRAIN_ARGS[@]}")
fi

printf 'Repository: %s\nRun directory: %s\nCommand:' "$REPO_ROOT" "$RUN_DIR"
printf ' %q' "${TRAIN_COMMAND[@]}"
printf '\n'
if ((DRY_RUN)); then
    exit 0
fi

command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
git rev-parse --verify HEAD >/dev/null 2>&1 || {
    echo "This checkout needs its .git history for training provenance; transfer .git or use git clone." >&2
    exit 2
}
if ((RESUME_RUN)); then
    [[ -f "$RUN_DIR/run.json" ]] || { echo "No resumable run.json in $RUN_DIR" >&2; exit 2; }
elif [[ -e "$RUN_DIR" ]]; then
    [[ -d "$RUN_DIR" && -z "$(ls -A -- "$RUN_DIR")" ]] || {
        echo "Run directory must be new or empty: $RUN_DIR" >&2; exit 2;
    }
fi
mkdir -p -- "$RUN_DIR"
if command -v flock >/dev/null; then
    exec 9>"$RUN_DIR/.lock"
    flock -n 9 || { echo "Run directory is already in use" >&2; exit 2; }
else
    mkdir -- "$RUN_DIR/.running" || { echo "Run directory is already in use" >&2; exit 2; }
fi
export ALPHARACER_RESUME_RUN="$RESUME_RUN"

# Limit numerical-library oversubscription in each CPU rollout worker.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
export TMPDIR="$RUN_DIR/tmp"
export PIP_CACHE_DIR="$RUN_DIR/.pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p -- "$MPLCONFIGDIR" "$TMPDIR" "$PIP_CACHE_DIR"

finish() {
    local result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$RUN_DIR/exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/finished_utc.txt"
    rmdir -- "$RUN_DIR/.running" 2>/dev/null || true
    if ((result)); then
        printf 'Run failed (exit %s). Inspect %s/train.log and any failure episode JSON files.\n' "$result" "$RUN_DIR"
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if ((!RESUME_RUN)); then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_utc.txt"
fi

run_work() {
if [[ -z "${PYTHON_BIN:-}" && ! -x "$TRAIN_PYTHON" ]]; then
    command -v python3 >/dev/null || { echo "Install Python 3 with venv support first."; exit 2; }
    python3 -m venv "$REPO_ROOT/.venv"
fi
"$TRAIN_PYTHON" --version

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
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        installed = 'missing'
    if installed.split('+')[0] != version:
        problems.append(f'{name}: need {version}, found {installed}')
print('\n'.join(problems) if problems else 'Pinned dependencies found.')
sys.exit(bool(problems))
PY
then
    if [[ "${INSTALL_DEPS:-1}" != 1 ]]; then
        echo "Dependencies do not match requirements.txt; INSTALL_DEPS=0 prevents installation."
        exit 2
    fi
    "$TRAIN_PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt"
fi

if ((!RESUME_RUN)); then
    "$TRAIN_PYTHON" -m pip freeze > "$RUN_DIR/pip-freeze.txt"
    git status --short > "$RUN_DIR/git-status.txt"
    git diff --binary HEAD > "$RUN_DIR/git-working-tree.patch"
fi
"$TRAIN_PYTHON" - "$RUN_DIR" "${TRAIN_COMMAND[@]:4}" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

import casadi as ca
import torch
from config import params
from policy.alpharacer.train import training
from track import Track, configure_track, load_centerline_csv, load_raceline_csv

output = Path(sys.argv[1])
sys.argv = ['training', *sys.argv[2:]]
args = training.parse_args()
configure_track(params, args.track)
track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
raceline = load_raceline_csv(params.RACELINE_CSV, scale=params.TRACK_SCALE)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.ones(1, device=device).sum().item()
if device == 'cpu':
    print('CUDA unavailable: rollout and network fitting will use CPU.')
else:
    print('Network device:', torch.cuda.get_device_name(0))
print(f'Rollout collection: {args.n_workers} CPU worker process(es).')
x = ca.SX.sym('x')
solver = ca.nlpsol('alpha_preflight', 'ipopt', {'x': x, 'f': x*x},
                  {'ipopt.print_level': 0, 'print_time': 0})
solution = solver(x0=1.)
if not solver.stats()['success'] or abs(float(solution['x'])) > 1e-5:
    raise RuntimeError('IPOPT preflight did not solve the scalar test problem')

tracked = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
sources = sorted({Path(os.fsdecode(name)) for name in tracked.split(b'\0') if name})
sources = [path for path in sources if path.is_file() and
           (path.suffix in {'.py', '.sh', '.csv', '.pth'} or path.name == 'requirements.txt')]
hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
metadata = dict(
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    python=sys.version, interpreter=sys.executable, platform=platform.platform(),
    torch=torch.__version__, torch_cuda=torch.version.cuda, device=device,
    settings={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    split_counts={name: len(ids) for name, ids in training.split_races(
        args.races, args.validation_fraction, args.test_fraction).items()},
    track_length_m=track.total_length, raceline_length_m=raceline.total_length,
    source_asset_sha256=hashes,
    environment={key: os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
        'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
)
if os.environ['ALPHARACER_RESUME_RUN'] == '1':
    previous = json.loads((output/'run.json').read_text())
    old_settings, new_settings = previous['settings'].copy(), metadata['settings'].copy()
    old_settings.pop('n_workers', None)
    new_settings.pop('n_workers', None)
    if (old_settings != new_settings or previous['source_asset_sha256'] != hashes
            or previous['torch'] != metadata['torch'] or previous['device'] != device):
        raise ValueError('Cannot resume: training settings, source/assets, PyTorch or device changed')
    print('Resume provenance verified.')
else:
    with tarfile.open(output/'source-snapshot.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(path, arcname=str(path), recursive=False)
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
print('Preflight passed. Effective configuration:')
print(json.dumps(metadata['settings'], indent=2))
PY

if ((CHECK_ONLY)); then
    echo "Environment check completed; no collection or training started. See $RUN_DIR/run.json"
    exit 0
fi

"${TRAIN_COMMAND[@]}"
"$TRAIN_PYTHON" - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
report = json.loads((output/'potential.test.json').read_text())
lines = [f'Training completed: {output}',
         f"Accepted races: {report['split_counts']}",
         f"Collection attempts: {report['collection_attempts']}; rejected: {report['rejected_episodes']}",
         f"Return: gamma={report['gamma']}, window={report['future_window_steps']} steps",
         f"Value test errors: {report['value_vs_recorded_returns']}",
         f"Potential difference test errors: {report['potential_vs_value_differences']}",
         f'Potential weights: {output / "potential.pth"}',
         f'Full test report: {output / "potential.test.json"}']
summary = '\n'.join(lines)+'\n'
(output/'summary.txt').write_text(summary)
print(summary)
PY
}

# A regular pipe also works where /dev/fd process substitution is unavailable.
set +e
(trap - EXIT INT TERM; set -e; run_work) 2>&1 | tee -a "$RUN_DIR/train.log"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
if ((PIPE_RESULTS[0])); then
    exit "${PIPE_RESULTS[0]}"
fi
exit "${PIPE_RESULTS[1]}"
