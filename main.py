#!/Users/lee/racer-LLM/.venv/bin/python
# PYTHON_ARGCOMPLETE_OK
import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

from simulation_core.config import LLM_OBSERVATION_VARIANTS

try:
    import argcomplete
except ImportError:  # Tab completion is optional; normal execution still works without it.
    argcomplete = None

_COMPLETING = bool(os.environ.get("_ARGCOMPLETE"))
if not _COMPLETING:
    import numpy as np

    from config import params
    from visualization.generate_html_report import generate_report
    from simulation import run_simulation
    from simulation_core import SimulationConfig

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None


def _jsonable(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no portable NaN/Infinity representation.
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _npz_contents(data):
    """Convert every array in an open NpzFile to JSON-compatible values."""
    # Full MPC horizons are retained in the accompanying NPZ.  Duplicating
    # them into summary.json (and then report.html) makes even a short sweep
    # unnecessarily large; scalar objective histories remain in JSON.
    npz_only = {
        "car_mpc_predicted_state_plan",
        "car_mpc_predicted_control_plan",
        "car_mpc_reference_plan",
        "car_mpc_track_slack_plan",
        "car_mpc_collision_slack_plan",
    }
    return {
        name: _jsonable(data[name])
        for name in data.files
        if name not in npz_only
    }


def _trace_safety_summary(data):
    """Compact activation metrics for comparing otherwise large trace JSONs."""
    sim_dt = float(np.asarray(data["simulation_dt"]))
    car_count = int(np.asarray(data["car_px"]).shape[1])

    def time_by_car(key):
        if key not in data.files:
            return [None] * car_count
        active = np.asarray(data[key], dtype=bool)
        return _jsonable(np.sum(active, axis=0) * sim_dt)

    def trigger_count_by_car(key):
        if key not in data.files:
            return [None] * car_count
        active = np.asarray(data[key], dtype=bool)
        if active.shape[0] == 0:
            return [0] * car_count
        starts = active.copy()
        starts[1:] &= ~active[:-1]
        return _jsonable(np.sum(starts, axis=0))

    def activation_fraction(key, threshold=0.0):
        if key not in data.files:
            return [None] * car_count
        values = np.asarray(data[key], dtype=float)
        valid = np.isfinite(values)
        active = valid & (values > threshold)
        counts = np.sum(valid, axis=0)
        fractions = np.divide(
            np.sum(active, axis=0),
            counts,
            out=np.full(car_count, np.nan),
            where=counts > 0,
        )
        return _jsonable(fractions)

    def max_by_car(key):
        if key not in data.files:
            return [None] * car_count
        values = np.asarray(data[key], dtype=float)
        result = []
        for column in values.T:
            finite = column[np.isfinite(column)]
            result.append(float(np.max(finite)) if finite.size else None)
        return result

    if "offroad_recovery_threshold" in data.files:
        recovery_boundary = float(np.asarray(data["offroad_recovery_threshold"]))
    else:  # compatibility with traces produced before the field was renamed
        recovery_boundary = float(np.asarray(data["offroad_half_width"]))

    return {
        "nominal_track_boundary": float(np.asarray(data["track_half_width"])),
        "recovery_boundary": recovery_boundary,
        "physical_collision_seconds": (
            float(np.sum(np.asarray(data["collision"], dtype=bool)) * sim_dt)
            if "collision" in data.files
            else None
        ),
        "track_boundary_violation_seconds_by_car": time_by_car(
            "car_track_boundary_violation"
        ),
        "offroad_recovery_seconds_by_car": time_by_car(
            "car_offroad_recovery_active"
        ),
        "low_speed_boundary_recovery_seconds_by_car": time_by_car(
            "car_low_speed_boundary_recovery_active"
        ),
        "low_speed_boundary_recovery_triggers_by_car": trigger_count_by_car(
            "car_low_speed_boundary_recovery_active"
        ),
        "deadlock_reset_count_by_car": (
            _jsonable(
                np.sum(
                    np.asarray(data["car_deadlock_reset_event"], dtype=bool),
                    axis=0,
                )
            )
            if "car_deadlock_reset_event" in data.files
            else [None] * car_count
        ),
        "deadlock_reset_blocked_seconds_by_car": time_by_car(
            "car_deadlock_reset_blocked"
        ),
        "mpc_recovery_seconds_by_car": time_by_car("car_mpc_recovery_active"),
        "max_track_slack_by_car": max_by_car("car_mpc_max_track_slack"),
        "max_collision_slack_by_car": max_by_car(
            "car_mpc_max_collision_slack"
        ),
        "track_slack_active_fraction_by_car": activation_fraction(
            "car_mpc_max_track_slack", threshold=1.0e-6
        ),
        "collision_slack_active_fraction_by_car": activation_fraction(
            "car_mpc_max_collision_slack", threshold=1.0e-6
        ),
        "max_ibr_collision_slack_by_car": max_by_car(
            "car_ibr_max_collision_slack"
        ),
        "max_ibr_track_slack_by_car": max_by_car("car_ibr_max_track_slack"),
        "ibr_collision_slack_active_fraction_by_car": activation_fraction(
            "car_ibr_max_collision_slack", threshold=1.0e-6
        ),
        "ibr_track_slack_active_fraction_by_car": activation_fraction(
            "car_ibr_max_track_slack", threshold=1.0e-6
        ),
    }


def _trace_llm_summary(data):
    """Per-car LLM/rule decision and timing summary from a saved trace."""
    car_count = int(np.asarray(data["car_px"]).shape[1])
    modes = np.asarray(data["car_policy_mode"], dtype=str)
    rule_modes = (
        np.asarray(data["car_explicit_rule_mode"], dtype=str)
        if "car_explicit_rule_mode" in data.files
        else np.full_like(modes, "")
    )
    rule_consistent = (
        np.asarray(data["car_explicit_rule_consistent"], dtype=bool)
        if "car_explicit_rule_consistent" in data.files
        else np.ones_like(modes, dtype=bool)
    )
    control_update = np.asarray(data["control_update"], dtype=bool)
    call_indices = np.asarray(data.get("llm_raw_call_indices", np.empty((car_count, 0))))
    errors = np.asarray(data.get("llm_raw_errors", np.empty((car_count, 0))), dtype=str)
    latency = np.asarray(
        data.get("llm_raw_latency_seconds", np.empty((car_count, 0))), dtype=float
    )
    output_tokens = np.asarray(
        data.get("llm_raw_eval_counts", np.empty((car_count, 0))), dtype=float
    )
    prompt_tokens = np.asarray(
        data.get("llm_raw_prompt_eval_counts", np.empty((car_count, 0))), dtype=float
    )

    summaries = []
    decision_counts = (
        np.asarray(data["car_policy_decision_counts"], dtype=int)
        if "car_policy_decision_counts" in data.files
        else np.zeros(car_count, dtype=int)
    )
    decision_deadline = (
        float(np.asarray(data["llm_update_interval_seconds"]))
        if "llm_update_interval_seconds" in data.files
        else np.inf
    )
    for car_idx in range(car_count):
        valid_calls = call_indices[car_idx] >= 0 if call_indices.ndim == 2 else np.zeros(0, dtype=bool)
        valid_latency = valid_calls & np.isfinite(latency[car_idx]) if latency.ndim == 2 else valid_calls
        latency_values = latency[car_idx][valid_latency] if latency.ndim == 2 else np.empty(0)
        update_modes = modes[control_update, car_idx]
        comparable_rules = control_update & (rule_modes[:, car_idx] != "")
        mode_switches = int(np.sum(update_modes[1:] != update_modes[:-1])) if update_modes.size > 1 else 0

        def token_sum(values):
            if values.ndim != 2:
                return 0
            selected = values[car_idx][valid_calls]
            return int(np.sum(selected[selected >= 0]))

        input_schema_version = (
            str(np.asarray(data["car_llm_input_schema_versions"])[car_idx])
            if "car_llm_input_schema_versions" in data.files
            else "unknown"
        )
        summaries.append({
            "input_schema_version": input_schema_version,
            "input_modules": {
                "base": input_schema_version == "direct_params_modular_v1",
                "raceline_speed": (
                    bool(np.asarray(data["car_llm_include_raceline_speed"])[car_idx])
                    if "car_llm_include_raceline_speed" in data.files
                    else False
                ),
                "current_parameters": (
                    bool(np.asarray(data["car_llm_include_current_parameters"])[car_idx])
                    if "car_llm_include_current_parameters" in data.files
                    else False
                ),
                "decision_feedback": (
                    bool(np.asarray(data["car_llm_include_decision_feedback"])[car_idx])
                    if "car_llm_include_decision_feedback" in data.files
                    else False
                ),
            },
            "prompt_variant": (
                str(np.asarray(data["car_llm_prompt_variants"])[car_idx])
                if "car_llm_prompt_variants" in data.files
                else "unknown"
            ),
            "observation_variant": (
                str(np.asarray(data["car_llm_observation_variants"])[car_idx])
                if "car_llm_observation_variants" in data.files
                else "unknown"
            ),
            "call_count": int(np.sum(valid_calls)),
            "decision_count": int(decision_counts[car_idx]),
            "error_count": int(np.sum(errors[car_idx][valid_calls] != "")) if errors.ndim == 2 else 0,
            "mean_latency_seconds": float(np.mean(latency_values)) if latency_values.size else None,
            "p95_latency_seconds": float(np.percentile(latency_values, 95)) if latency_values.size else None,
            "deadline_miss_count": int(np.sum(latency_values > decision_deadline)),
            "prompt_tokens": token_sum(prompt_tokens),
            "output_tokens": token_sum(output_tokens),
            "mode_switches": mode_switches,
            "explicit_rule_agreement_fraction": (
                float(np.mean(rule_consistent[comparable_rules, car_idx]))
                if np.any(comparable_rules)
                else None
            ),
        })
    return summaries


def params_snapshot():
    snapshot = {}
    for name in dir(params):
        if name.startswith("_"):
            continue
        value = getattr(params, name)
        if callable(value):
            continue
        snapshot[name] = _jsonable(value)
    return snapshot


def _slug(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-").lower()


def save_experiment_config(
    output_dir,
    policy_names,
    snapshot,
    *,
    runner="single_experiment",
    run_simulation_kwargs=None,
    filename="experiment_config.json",
):
    """Write the config defaults and explicit runtime overrides for one artifact set."""
    config_path = output_dir / filename
    contents = {
        "runner": runner,
        "policy_names": policy_names,
        "params": snapshot,
    }
    if run_simulation_kwargs is not None:
        # Keep this separate from ``params``: these arguments take precedence
        # for a run, but do not mutate the global config object.
        contents["run_simulation_overrides"] = _jsonable(run_simulation_kwargs)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(contents, file, indent=2, sort_keys=True)
    return config_path


def _run_position_trial(
    policy_names,
    ego_position,
    trace_path,
    simulation_overrides,
    *,
    suppress_output,
):
    """Run one race and reduce its saved trace to sweep-level results."""
    config_values = dict(simulation_overrides)
    config_values.pop("ego_position", None)
    config_values.pop("save_trace", None)
    config_values.pop("trace_output", None)
    config_values.setdefault("show_plot", False)
    config = SimulationConfig(
        policies=tuple(policy_names),
        ego_position=int(ego_position),
        save_trace=True,
        trace_output=trace_path,
        **config_values,
    )

    if suppress_output:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                run_simulation(config)
    else:
        run_simulation(config)

    with np.load(trace_path, allow_pickle=True) as data:
        final_progress = np.asarray(data["car_px"], dtype=float)[-1].copy()
        winner_index = int(np.nanargmax(final_progress))
        start_ranks = [
            int(value) for value in np.asarray(data["start_ranks"]).tolist()
        ]
        result = {
            "winner_index": winner_index,
            "final_progress": _jsonable(final_progress),
            "start_ranks": start_ranks,
            "safety_summary": _trace_safety_summary(data),
            "decision_summary_by_car": _trace_llm_summary(data),
            "npz": _npz_contents(data),
        }

    if len(start_ranks) >= 3 and start_ranks[1] > start_ranks[2]:
        raise RuntimeError(f"oppo1 should start before oppo2, got {start_ranks}")
    return result


def _run_sweep_trials(
    policy_names,
    ego_positions,
    trials_per_position,
    trace_dir,
    *,
    keep_npz,
    show_progress,
    suppress_output,
    simulation_overrides,
):
    """Execute the trial matrix; every iteration calls one simulation."""
    roles = ["ego", "oppo1", "oppo2"]
    winner_counts = {role: 0 for role in roles}
    winner_counts_by_position = {
        str(position): {role: 0 for role in roles}
        for position in ego_positions
    }
    records = []
    total_trials = int(trials_per_position) * len(ego_positions)
    progress_bar = (
        tqdm(
            total=total_trials,
            desc="Position sweep",
            unit="sim",
            dynamic_ncols=True,
            leave=True,
        )
        if show_progress and tqdm is not None
        else None
    )

    try:
        trial_id = 0
        for ego_position in ego_positions:
            for repeat in range(int(trials_per_position)):
                trial_id += 1
                if progress_bar is None:
                    print(
                        f"[{trial_id}/{total_trials}] "
                        f"ego_position={ego_position}, trial={repeat + 1}"
                    )
                else:
                    progress_bar.set_postfix(
                        ego_position=ego_position,
                        trial=repeat + 1,
                        refresh=True,
                    )

                trace_path = trace_dir / f"trial_{trial_id:03d}.npz"
                result = _run_position_trial(
                    policy_names,
                    ego_position,
                    trace_path,
                    simulation_overrides,
                    suppress_output=suppress_output,
                )
                winner_index = result["winner_index"]
                winner_role = roles[winner_index]
                if not keep_npz:
                    trace_path.unlink(missing_ok=True)

                winner_counts[winner_role] += 1
                winner_counts_by_position[str(ego_position)][winner_role] += 1
                records.append({
                    "trial": trial_id,
                    "ego_position": ego_position,
                    "repeat": repeat + 1,
                    "winner": winner_role,
                    "winner_index": winner_index,
                    "winner_policy": policy_names[winner_index],
                    "final_progress": result["final_progress"],
                    "start_ranks": result["start_ranks"],
                    "safety_summary": result["safety_summary"],
                    "decision_summary_by_car": result["decision_summary_by_car"],
                    "npz": result["npz"],
                })
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return records, winner_counts, winner_counts_by_position


def run_position_sweep(
    policy_names=("llm_alpharacer", "ibr", "ibr"),
    trials_per_position=1,
    ego_positions=(1, 2, 3),
    output_path="result/position_sweep_summary.json",
    keep_npz=False,
    show_progress=True,
    suppress_simulation_output=True,
    **simulation_overrides,
):
    """Run a matrix of single-race simulations and save its summary."""
    policy_names = list(policy_names)
    if len(policy_names) != 3:
        raise ValueError(f"policy_names must have length 3, got {len(policy_names)}")
    ego_positions = [int(position) for position in ego_positions]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_name = (
        "experiment_config.json"
        if output_path.stem == "summary"
        else f"{output_path.stem}_experiment_config.json"
    )
    config_path = save_experiment_config(
        output_path.parent,
        policy_names,
        params_snapshot(),
        runner="position_sweep",
        run_simulation_kwargs=simulation_overrides,
        filename=manifest_name,
    )
    trace_dir = output_path.parent / (
        f"{output_path.stem}_traces" if keep_npz else f"_{output_path.stem}_tmp"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)

    records, winner_counts, counts_by_position = _run_sweep_trials(
        policy_names,
        ego_positions,
        trials_per_position,
        trace_dir,
        keep_npz=keep_npz,
        show_progress=show_progress,
        suppress_output=suppress_simulation_output,
        simulation_overrides=simulation_overrides,
    )
    summary = {
        "policy_names": policy_names,
        "trials_per_position": int(trials_per_position),
        "ego_positions": ego_positions,
        "winner_counts": winner_counts,
        "winner_counts_by_position": counts_by_position,
        "trials": records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True, allow_nan=False)

    print(f"Winner counts: {winner_counts}")
    print(f"Saved summary to {output_path}")
    print(f"Saved config to {config_path}")
    return summary


def _add_case_args(parser):
    group = parser.add_argument_group("experiment cases")
    group.add_argument("--policies", nargs="+", default=None)
    group.add_argument("--ego-position", type=int, choices=[1, 2, 3], default=None)
    group.add_argument("--name", default=None)
    group.add_argument(
        "--suite",
        choices=["llm-params", "ibr-speed-cap"],
        default=None,
        help=(
            "llm-params runs a direct five-parameter LLM against "
            "both opponent types. Every candidate is tested at all three positions."
        ),
    )


def _add_sim_args(parser):
    group = parser.add_argument_group("simulation")
    group.add_argument(
        "--track-name",
        choices=["berlin", "modena", "rounded_rectangle"],
        default=None,
        help="Use a registered centerline, raceline, scale, and track width together.",
    )
    group.add_argument("--vehicle-model", choices=["kinematic", "kinetic", "kinetics", "dbm"], default=None)
    group.add_argument("--total-time", type=float, default=None, metavar="SECONDS")
    group.add_argument(
        "--mpc-velocity-track-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the shared MPC reference-speed tracking weight.",
    )
    group.add_argument(
        "--mpc-horizon",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override the prediction horizon in 0.1 s MPC steps; "
            "N=10, 15, and 20 correspond to 1.0, 1.5, and 2.0 s."
        ),
    )
    group.add_argument(
        "--mpc-track-slack-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the quadratic MPC track-boundary slack weight.",
    )
    group.add_argument(
        "--mpc-collision-slack-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the quadratic MPC collision-avoidance slack weight.",
    )
    group.add_argument(
        "--mpc-formulation",
        choices=["constrained_local", "upstream_simple"],
        default=None,
        help="Select the local constrained or simplified upstream-style MPC.",
    )
    group.add_argument(
        "--raceline-csv",
        type=Path,
        default=None,
        help="Override the track's registered raceline CSV for this experiment.",
    )
    group.add_argument("--start-gap", type=float, default=None, metavar="METERS")
    group.add_argument(
        "--initial-speeds",
        nargs="+",
        type=float,
        default=None,
        metavar="MPS",
        help="Initial speed for each car, in ego/opponent policy order.",
    )
    group.add_argument("--static-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--alpha-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument(
        "--alpharacer-weight",
        type=Path,
        default=None,
        help="Potential checkpoint for every alpha-RACER policy (official raw or local metadata-bearing .pth).",
    )
    group.add_argument("--llm-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--llm-output-mode", choices=["auto", "mode", "params"], default=None)
    group.add_argument("--llm-model", default=None)
    group.add_argument("--llm-models", nargs="+", default=None)
    group.add_argument("--llm-temperature", type=float, default=None)
    group.add_argument("--llm-temperatures", nargs="+", type=float, default=None)
    group.add_argument(
        "--llm-update-interval-seconds",
        type=float,
        default=None,
        help="Simulated-time interval between LLM decisions (default: 1.0 s).",
    )
    group.add_argument(
        "--llm-observation-variant",
        choices=LLM_OBSERVATION_VARIANTS,
        default="baseline",
        help="Select one independent LLM observation ablation.",
    )
    group.add_argument(
        "--llm-include-raceline-speed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle the direct-parameter raceline_speed input module.",
    )
    group.add_argument(
        "--llm-include-current-parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle the direct-parameter current_parameters input module.",
    )
    group.add_argument(
        "--llm-include-decision-feedback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle the direct-parameter decision_feedback input module.",
    )
    group.add_argument(
        "--ibr-implementation",
        choices=["local_wang", "historical_strict"],
        default="local_wang",
        help=(
            "IBR solver: local_wang uses local dynamics; "
            "historical_strict is the locally adapted alpha-RACER IBR "
            "(default: local_wang)."
        ),
    )
    group.add_argument(
        "--ibr-speed-cap",
        choices=["global", "raceline"],
        default=None,
        help=(
            "IBR speed bound: global 5 m/s or a physically reachable bound "
            "that tightens toward 1.04 times local raceline speed."
        ),
    )
    group.add_argument(
        "--parameter-ibr-rounds",
        type=int,
        default=None,
        metavar="N",
        help="Complete joint five-parameter best-response rounds per update.",
    )
    group.add_argument(
        "--parameter-ibr-search-points",
        type=int,
        default=None,
        metavar="N",
        help="Local candidates per parameter in coarse-to-fine search (minimum 3).",
    )
    group.add_argument(
        "--parameter-ibr-update-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Simulation-time interval between five-parameter IBR updates.",
    )
    group.add_argument("--rl-model-path", default=None)
    group.add_argument("--rl-backend", choices=["auto", "npz", "json", "npy", "sb3", "ppo"], default=None)
    group.add_argument("--rl-strict-load", action="store_true")
    group.add_argument("--rl-action-order", choices=["throttle_steer", "steer_throttle"], default=None)


def _add_output_args(parser):
    group = parser.add_argument_group("output")
    group.add_argument("--no-animation", action="store_true")
    group.add_argument("--show-plot", action="store_true")
    group.add_argument("--show-ref-debug", action="store_true")
    group.add_argument("--stride", type=int, default=2)
    group.add_argument("--interval", type=int, default=30)
    group.add_argument("--dpi", type=int, default=100)


def parse_args():
    parser = argparse.ArgumentParser(description="Run racer experiments.")
    _add_case_args(parser)
    _add_sim_args(parser)
    _add_output_args(parser)
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    return parser.parse_args()

def _apply_cli_overrides(args):
    """Validate CLI values that still configure experimental policy internals."""
    params.IBR_IMPLEMENTATION = args.ibr_implementation
    if args.ibr_speed_cap is not None:
        params.IBR_SPEED_CAP_MODE = args.ibr_speed_cap
    if args.parameter_ibr_rounds is not None:
        if args.parameter_ibr_rounds < 1:
            raise ValueError("--parameter-ibr-rounds must be at least 1")
        params.PARAMETER_IBR_ROUNDS = args.parameter_ibr_rounds
    if args.parameter_ibr_search_points is not None:
        if args.parameter_ibr_search_points < 3:
            raise ValueError("--parameter-ibr-search-points must be at least 3")
        params.PARAMETER_IBR_SEARCH_POINTS = args.parameter_ibr_search_points
    if args.parameter_ibr_update_interval is not None:
        if args.parameter_ibr_update_interval < 0.0:
            raise ValueError("--parameter-ibr-update-interval must be non-negative")
        params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS = (
            args.parameter_ibr_update_interval
        )
    if args.mpc_horizon is not None and args.mpc_horizon < 2:
        raise ValueError("--mpc-horizon must be at least 2")
    if (
        args.mpc_track_slack_weight is not None
        and args.mpc_track_slack_weight < 0.0
    ):
        raise ValueError("--mpc-track-slack-weight must be non-negative")
    if (
        args.mpc_collision_slack_weight is not None
        and args.mpc_collision_slack_weight < 0.0
    ):
        raise ValueError("--mpc-collision-slack-weight must be non-negative")


def _sweep_kwargs_from_args(args):
    """Translate CLI names once into sweep and SimulationConfig field names."""
    return {
        "trials_per_position": 1,
        "ego_positions": (
            [args.ego_position] if args.ego_position is not None else [1, 2, 3]
        ),
        "track_name": args.track_name,
        "vehicle_model": args.vehicle_model or "kinematic",
        "duration": args.total_time,
        "raceline": args.raceline_csv,
        "start_gap": args.start_gap,
        "start_speeds": args.initial_speeds,
        "mpc_velocity_weight": args.mpc_velocity_track_weight,
        "mpc_track_slack": args.mpc_track_slack_weight,
        "mpc_collision_slack": args.mpc_collision_slack_weight,
        "mpc_horizon": args.mpc_horizon,
        "mpc_formulation": args.mpc_formulation,
        "static_params": args.static_params,
        "alpha_params": args.alpha_params,
        "alpha_weight_path": args.alpharacer_weight,
        "ibr_implementation": args.ibr_implementation,
        "llm_params": args.llm_params,
        "llm_output_mode": args.llm_output_mode,
        "llm_model": args.llm_model,
        "llm_models": args.llm_models,
        "llm_update_interval": args.llm_update_interval_seconds,
        "llm_observation_variant": args.llm_observation_variant,
        "llm_include_raceline_speed": args.llm_include_raceline_speed,
        "llm_include_current_parameters": args.llm_include_current_parameters,
        "llm_include_decision_feedback": args.llm_include_decision_feedback,
        "llm_temperature": args.llm_temperature,
        "llm_temperatures": args.llm_temperatures,
        "show_plot": args.show_plot,
        "rl_model_path": args.rl_model_path,
        "rl_backend": args.rl_backend,
        "rl_strict_load": args.rl_strict_load or None,
        "rl_action_order": args.rl_action_order,
    }


def _suite_definition(name, sweep_kwargs):
    """Return the output directory and policy rosters for one named suite."""
    if name == "llm-params":
        sweep_kwargs["llm_output_mode"] = "params"
        return Path("result/direct_params_modular_v1_suite"), {
            "param_llm_vs_alpharacer": [
                "llm_alpharacer", "alpharacer", "alpharacer"
            ],
            "param_llm_vs_ibr": ["llm_alpharacer", "ibr", "ibr"],
        }
    if name == "ibr-speed-cap":
        params.IBR_SPEED_CAP_MODE = "raceline"
        return Path("result/ibr_speed_cap_suite"), {
            "raceline_capped_ibr": ["ibr", "alpharacer", "alpharacer"],
        }
    raise ValueError(f"Unknown experiment suite: {name}")


def _run_requested_sweeps(args, sweep_kwargs):
    """Expand the requested case(s), run each sweep, and return report inputs."""
    if args.suite is not None:
        suite_dir, candidates = _suite_definition(args.suite, sweep_kwargs)
        suite_dir.mkdir(parents=True, exist_ok=True)
        sweep_paths = []
        for label, policy_names in candidates.items():
            sweep_path = suite_dir / f"{label}.json"
            run_position_sweep(
                policy_names=policy_names,
                output_path=sweep_path,
                keep_npz=True,
                **sweep_kwargs,
            )
            sweep_paths.append(sweep_path)
        return sweep_paths, suite_dir / "report.html"

    if args.name:
        output_dir = Path("result") / _slug(args.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        sweep_path = output_dir / "summary.json"
        report_path = output_dir / "report.html"
    else:
        sweep_path = Path("result/position_sweep_summary.json")
        report_path = Path("result/position_sweep_report.html")

    run_position_sweep(
        policy_names=args.policies or ["llm_alpharacer", "ibr", "ibr"],
        output_path=sweep_path,
        keep_npz=bool(args.name),
        **sweep_kwargs,
    )
    return [sweep_path], report_path


def main():
    """Parse one experiment request, run its races, then build the report."""
    args = parse_args()
    _apply_cli_overrides(args)

    sweep_paths, report_path = _run_requested_sweeps(
        args, _sweep_kwargs_from_args(args)
    )

    generate_report(
        sweep_paths,
        output=report_path,
        animation_stride=args.stride,
        animation_interval=args.interval,
        animation_dpi=args.dpi,
        include_animation=not args.no_animation,
    )


if __name__ == "__main__":
    main()
