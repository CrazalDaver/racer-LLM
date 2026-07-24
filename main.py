#!/Users/lee/racer-LLM/.venv/bin/python
# PYTHON_ARGCOMPLETE_OK
import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

try:
    import argcomplete
except ImportError:  # Tab completion is optional; normal execution still works without it.
    argcomplete = None

_COMPLETING = bool(os.environ.get("_ARGCOMPLETE"))
if not _COMPLETING:
    import numpy as np

    from animate_trace import make_animation
    from config import params
    from simulation import run_simulation

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None


def _jsonable(value):
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
    return {name: _jsonable(data[name]) for name in data.files}


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


def _policy_params_for_name(policy_names, snapshot):
    first_policy = _slug(policy_names[0]) if policy_names else "static"
    if first_policy in {
        "llm",
        "llm-alpha",
        "llm-alpharacer",
        "llm_alpharacer",
        "llm_parameter_policy",
        "rule",
        "rules",
        "rule-based",
        "rule_based_safeguard",
        "safeguard_baseline",
        "decision_tree",
    }:
        return snapshot["LLM_POLICY_PARAMS"]
    return snapshot["STATIC_POLICY_PARAMS"]


def _format_param(value):
    return f"{float(value):.1f}"


def _policy_param_slug(values):
    q, zeta, s1, s2, s3 = values
    return "-".join(
        [
            f"q{_format_param(q)}",
            f"zeta{_format_param(zeta)}",
            f"s1-{_format_param(s1)}",
            f"s2-{_format_param(s2)}",
            f"s3-{_format_param(s3)}",
        ]
    )


def experiment_name(policy_names, snapshot, custom_name=None):
    if custom_name:
        return _slug(custom_name)

    policy_slug = "-".join(_slug(name) for name in policy_names)
    policy_params = _policy_params_for_name(policy_names, snapshot)
    ego_suffix = ""
    ego_position = int(snapshot.get("EGO_POSITION", 1))
    if ego_position != 1:
        ego_suffix = f"-ego{ego_position}"
    model_suffix = ""
    if snapshot.get("LLM_MODEL_NAMES"):
        model_slug = "-".join(_slug(model or "default") for model in snapshot["LLM_MODEL_NAMES"])
        model_suffix = f"-models-{model_slug}"
    return f"policies-{policy_slug}{ego_suffix}-{_policy_param_slug(policy_params)}{model_suffix}"


def save_experiment_config(output_dir, policy_names, snapshot):
    config_path = output_dir / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "policy_names": policy_names,
                "params": snapshot,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    return config_path


def simulation_kwargs(
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_temperature=None,
    llm_temperatures=None,
    llm_update_interval_seconds=None,
    llm_safeguard_enabled=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
):
    return {
        "vehicle_model_type": vehicle_model_type,
        "static_policy_params": static_policy_params,
        "alpharacer_params": alpharacer_params,
        "llm_policy_params": llm_policy_params,
        "llm_output_mode": llm_output_mode,
        "llm_model_name": llm_model_name,
        "llm_model_names": llm_model_names,
        "llm_temperature": llm_temperature,
        "llm_temperatures": llm_temperatures,
        "llm_update_interval_seconds": llm_update_interval_seconds,
        "llm_safeguard_enabled": llm_safeguard_enabled,
        "self_play_rl_model_path": self_play_rl_model_path,
        "self_play_rl_backend": self_play_rl_backend,
        "self_play_rl_strict_load": self_play_rl_strict_load,
        "self_play_rl_action_order": self_play_rl_action_order,
    }


def _override_snapshot(
    snapshot,
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_temperature=None,
    llm_temperatures=None,
    llm_update_interval_seconds=None,
    llm_safeguard_enabled=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
    ego_position=None,
):
    snapshot = dict(snapshot)
    overrides = simulation_kwargs(
        vehicle_model_type=vehicle_model_type,
        static_policy_params=static_policy_params,
        alpharacer_params=alpharacer_params,
        llm_policy_params=llm_policy_params,
        llm_output_mode=llm_output_mode,
        llm_model_name=llm_model_name,
        llm_model_names=llm_model_names,
        llm_temperature=llm_temperature,
        llm_temperatures=llm_temperatures,
        llm_update_interval_seconds=llm_update_interval_seconds,
        llm_safeguard_enabled=llm_safeguard_enabled,
        self_play_rl_model_path=self_play_rl_model_path,
        self_play_rl_backend=self_play_rl_backend,
        self_play_rl_strict_load=self_play_rl_strict_load,
        self_play_rl_action_order=self_play_rl_action_order,
    )
    if overrides["vehicle_model_type"] is not None:
        snapshot["VEHICLE_MODEL"] = overrides["vehicle_model_type"]
    fixed_params = overrides["static_policy_params"]
    if fixed_params is None:
        fixed_params = overrides["alpharacer_params"]
    if fixed_params is not None:
        fixed_params = [float(value) for value in fixed_params]
        snapshot["STATIC_POLICY_PARAMS"] = fixed_params
    if overrides["llm_policy_params"] is not None:
        snapshot["LLM_POLICY_PARAMS"] = [float(value) for value in overrides["llm_policy_params"]]
    if overrides["llm_output_mode"] is not None:
        snapshot["LLM_OUTPUT_MODE"] = overrides["llm_output_mode"]
    if overrides["llm_model_name"] is not None:
        snapshot["LLM_MODEL"] = overrides["llm_model_name"]
    if overrides["llm_model_names"] is not None:
        snapshot["LLM_MODEL_NAMES"] = list(overrides["llm_model_names"])
    if overrides["llm_temperature"] is not None:
        snapshot["LLM_TEMPERATURE"] = float(overrides["llm_temperature"])
    if overrides["llm_temperatures"] is not None:
        snapshot["LLM_TEMPERATURES"] = [float(value) for value in overrides["llm_temperatures"]]
    if overrides["llm_update_interval_seconds"] is not None:
        snapshot["LLM_UPDATE_INTERVAL_SECONDS"] = float(
            overrides["llm_update_interval_seconds"]
        )
    if overrides["llm_safeguard_enabled"] is not None:
        snapshot["LLM_SAFEGUARD_ENABLED"] = bool(overrides["llm_safeguard_enabled"])
    if overrides["self_play_rl_model_path"] is not None:
        snapshot["SELF_PLAY_RL_MODEL_PATH"] = overrides["self_play_rl_model_path"]
    if overrides["self_play_rl_backend"] is not None:
        snapshot["SELF_PLAY_RL_BACKEND"] = overrides["self_play_rl_backend"]
    if overrides["self_play_rl_strict_load"] is not None:
        snapshot["SELF_PLAY_RL_STRICT_LOAD"] = bool(overrides["self_play_rl_strict_load"])
    if overrides["self_play_rl_action_order"] is not None:
        snapshot["SELF_PLAY_RL_ACTION_ORDER"] = overrides["self_play_rl_action_order"]
    if ego_position is not None:
        snapshot["EGO_POSITION"] = int(ego_position)
    return snapshot


def run_experiment(
    policy_names,
    name=None,
    save_animation=True,
    stride=2,
    interval=30,
    dpi=100,
    show_plot=False,
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_temperature=None,
    llm_temperatures=None,
    llm_update_interval_seconds=None,
    llm_safeguard_enabled=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
    ego_position=1,
    show_ref_debug=False,
):
    """
    Run a single experiment with specified policies and config overrides,
    save trace and animation under result/.
    """
    sim_kwargs = simulation_kwargs(
        vehicle_model_type=vehicle_model_type,
        static_policy_params=static_policy_params,
        alpharacer_params=alpharacer_params,
        llm_policy_params=llm_policy_params,
        llm_output_mode=llm_output_mode,
        llm_model_name=llm_model_name,
        llm_model_names=llm_model_names,
        llm_temperature=llm_temperature,
        llm_temperatures=llm_temperatures,
        llm_update_interval_seconds=llm_update_interval_seconds,
        llm_safeguard_enabled=llm_safeguard_enabled,
        self_play_rl_model_path=self_play_rl_model_path,
        self_play_rl_backend=self_play_rl_backend,
        self_play_rl_strict_load=self_play_rl_strict_load,
        self_play_rl_action_order=self_play_rl_action_order,
    )
    snapshot = params_snapshot()
    snapshot = _override_snapshot(snapshot, **sim_kwargs, ego_position=ego_position)
    exp_name = experiment_name(policy_names, snapshot, custom_name=name)
    output_dir = Path("result") / f"experiment_{exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "simulation_trace.npz"
    animation_path = output_dir / "simulation_trace.gif"
    old_trace_output = params.TRACE_OUTPUT
    params.TRACE_OUTPUT = str(trace_path)

    try:
        config_path = save_experiment_config(output_dir, policy_names, snapshot)
        print(f"Experiment: {exp_name}")
        print(f"Output dir: {output_dir}")
        print(f"Config: {config_path}")
        run_simulation(
            policy_names=policy_names,
            save_trace=True,
            show_plot=show_plot,
            ego_position=ego_position,
            **sim_kwargs,
        )

        if save_animation:
            make_animation(
                trace_path,
                stride=stride,
                interval=interval,
                save_path=animation_path,
                dpi=dpi,
                show_ref_debug=show_ref_debug,
            )
    finally:
        params.TRACE_OUTPUT = old_trace_output

    return output_dir


def _ego_first_from_start_order(values, ego_position):
    ego_index = int(ego_position) - 1
    if ego_index < 0 or ego_index >= len(values):
        raise ValueError(f"ego_position must be between 1 and {len(values)}, got {ego_position}")
    return [values[ego_index], *[value for idx, value in enumerate(values) if idx != ego_index]]


def _as_case_list(cases):
    return [(list(start_order), int(ego_position)) for start_order, ego_position in cases]


def _llm_model_case_for_index(llm_model_cases, case_idx, car_count):
    if llm_model_cases is None:
        return None

    if not llm_model_cases:
        raise ValueError("llm_model_cases cannot be empty.")

    first_item = llm_model_cases[0]
    if isinstance(first_item, str) or first_item is None:
        model_case = llm_model_cases
    else:
        if case_idx >= len(llm_model_cases):
            raise ValueError(
                f"Missing llm_model_cases entry for case {case_idx}; "
                f"got {len(llm_model_cases)} model cases."
            )
        model_case = llm_model_cases[case_idx]

    if len(model_case) != car_count:
        raise ValueError(f"LLM model case length must be {car_count}, got {len(model_case)}.")
    return list(model_case)


def _case_name(name_prefix, start_order, ego_position, start_order_models=None):
    policy_slug = "-".join(_slug(policy) for policy in start_order)
    name = f"{name_prefix}-" if name_prefix else ""
    name = f"{name}startorder-{policy_slug}-ego{ego_position}"
    if start_order_models is not None:
        model_slug = "-".join(_slug(model or "default") for model in start_order_models)
        name = f"{name}-models-{model_slug}"
    return name


def run_experiment_suite(
    start_order_cases,
    llm_model_cases=None,
    name_prefix="suite",
    save_animation=True,
    stride=2,
    interval=30,
    dpi=100,
    show_plot=False,
    show_ref_debug=False,
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_temperature=None,
    llm_temperatures=None,
    llm_update_interval_seconds=None,
    llm_safeguard_enabled=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
):
    """
    Run experiments described in start-order coordinates.

    start_order_cases is a list of (policies_in_start_order, ego_position).
    llm_model_cases is either one model list reused for every case, or a list
    of model lists aligned with start_order_cases. Model lists are also in
    start-order coordinates. Internally both are converted to ego-first order
    because run_simulation treats car 0 as ego.
    """
    output_dirs = []
    for case_idx, (start_order, ego_position) in enumerate(_as_case_list(start_order_cases)):
        start_order_models = _llm_model_case_for_index(llm_model_cases, case_idx, len(start_order))
        policy_names = _ego_first_from_start_order(start_order, ego_position)
        llm_model_names = (
            _ego_first_from_start_order(start_order_models, ego_position)
            if start_order_models is not None
            else None
        )

        output_dirs.append(
            run_experiment(
                policy_names=policy_names,
                name=_case_name(name_prefix, start_order, ego_position, start_order_models),
                save_animation=save_animation,
                stride=stride,
                interval=interval,
                dpi=dpi,
                show_plot=show_plot,
                show_ref_debug=show_ref_debug,
                vehicle_model_type=vehicle_model_type,
                static_policy_params=static_policy_params,
                alpharacer_params=alpharacer_params,
                llm_policy_params=llm_policy_params,
                llm_output_mode=llm_output_mode,
                llm_model_name=llm_model_name,
                llm_model_names=llm_model_names,
                llm_temperature=llm_temperature,
                llm_temperatures=llm_temperatures,
                llm_update_interval_seconds=llm_update_interval_seconds,
                llm_safeguard_enabled=llm_safeguard_enabled,
                self_play_rl_model_path=self_play_rl_model_path,
                self_play_rl_backend=self_play_rl_backend,
                self_play_rl_strict_load=self_play_rl_strict_load,
                self_play_rl_action_order=self_play_rl_action_order,
                ego_position=ego_position,
            )
        )
    return output_dirs


def run_position_sweep(
    policy_names=("LLM", "IBR", "IBR"),
    trials_per_position=1,
    ego_positions=(1, 2, 3),
    output_path="result/position_sweep_summary.json",
    show_progress=True,
    suppress_simulation_output=True,
    **run_simulation_kwargs,  # kwargs passed to run_simulation, except ego_position and save_trace
):
    """Run each ego starting position repeatedly and count wins.

    This intentionally keeps all simulation logic inside run_simulation.
    Temporary traces are used only to read the final car progress, then deleted.
    Car order is fixed as car0=ego, car1=oppo1, car2=oppo2.
    """
    policy_names = list(policy_names)
    if len(policy_names) != 3:
        raise ValueError(f"policy_names must have length 3, got {len(policy_names)}")

    ego_positions = [int(position) for position in ego_positions]
    roles = ["ego", "oppo1", "oppo2"]
    winner_counts = {role: 0 for role in roles}
    winner_counts_by_position = {
        str(int(position)): {role: 0 for role in roles}
        for position in ego_positions
    }
    trial_records = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_tmp_dir = output_path.parent / f"_{output_path.stem}_tmp"
    trace_tmp_dir.mkdir(parents=True, exist_ok=True)
    old_trace_output = params.TRACE_OUTPUT
    total_trials = int(trials_per_position) * len(ego_positions)
    trial_id = 0

    try:
        progress_bar = None
        if show_progress and tqdm is not None:
            progress_bar = tqdm(
                total=total_trials,
                desc="Position sweep",
                unit="sim",
                dynamic_ncols=True,
                leave=True,
            )
        for ego_position in ego_positions:
            ego_position = int(ego_position)
            for repeat in range(int(trials_per_position)):
                trial_id += 1
                if progress_bar is None:
                    print(f"[{trial_id}/{total_trials}] ego_position={ego_position}, trial={repeat + 1}")
                else:
                    progress_bar.set_postfix(
                        ego_position=ego_position,
                        trial=repeat + 1,
                        refresh=True,
                    )
                trace_path = trace_tmp_dir / f"trial_{trial_id:03d}.npz"
                params.TRACE_OUTPUT = str(trace_path)
                kwargs = dict(run_simulation_kwargs)
                kwargs.pop("ego_position", None)
                kwargs.pop("save_trace", None)
                kwargs.setdefault("show_plot", False)
                if suppress_simulation_output:
                    with open(os.devnull, "w", encoding="utf-8") as sink:
                        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                            run_simulation(
                                policy_names=policy_names,
                                ego_position=ego_position,
                                save_trace=True,
                                **kwargs,
                            )
                else:
                    run_simulation(
                        policy_names=policy_names,
                        ego_position=ego_position,
                        save_trace=True,
                        **kwargs,
                    )

                with np.load(trace_path, allow_pickle=True) as data:
                    trace_contents = _npz_contents(data)
                    car_px = np.asarray(data["car_px"], dtype=float)
                    final_progress = car_px[-1].copy()
                    winner_index = int(np.nanargmax(final_progress))
                    winner_role = roles[winner_index]
                    start_ranks = [int(value) for value in np.asarray(data["start_ranks"]).tolist()]
                trace_path.unlink(missing_ok=True)
                if start_ranks[1] > start_ranks[2]:
                    raise RuntimeError(f"oppo1 should start before oppo2, got {start_ranks}")

                winner_counts[winner_role] += 1
                winner_counts_by_position[str(ego_position)][winner_role] += 1
                trial_records.append(
                    {
                        "trial": trial_id,
                        "ego_position": ego_position,
                        "repeat": repeat + 1,
                        "winner": winner_role,
                        "winner_index": winner_index,
                        "winner_policy": policy_names[winner_index],
                        "final_progress": _jsonable(final_progress),
                        "start_ranks": start_ranks,
                        "npz": trace_contents,
                    }
                )
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if "progress_bar" in locals() and progress_bar is not None:
            progress_bar.close()
        params.TRACE_OUTPUT = old_trace_output

    summary = {
        "policy_names": policy_names,
        "trials_per_position": int(trials_per_position),
        "ego_positions": ego_positions,
        "winner_counts": winner_counts,
        "winner_counts_by_position": winner_counts_by_position,
        "trials": trial_records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True, allow_nan=False)

    print(f"Winner counts: {winner_counts}")
    print(f"Saved summary to {output_path}")
    return summary


def _parse_start_order_cases(start_order_args):
    if not start_order_args:
        return None

    cases = []
    for text in start_order_args:
        parts = [part.strip() for part in str(text).replace(":", ",").split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError("Use --start-order policy1,policy2,...,ego_position")
        cases.append((parts[:-1], int(parts[-1])))
    return cases


def _add_case_args(parser):
    group = parser.add_argument_group("experiment cases")
    group.add_argument("--policies", nargs="+", default=None)
    group.add_argument("--ego-position", type=int, choices=[1, 2, 3], default=None)
    group.add_argument("--start-order", action="append", default=None)
    group.add_argument("--name", default=None)


def _add_sim_args(parser):
    group = parser.add_argument_group("simulation")
    group.add_argument("--vehicle-model", choices=["kinematic", "kinetic", "kinetics", "dbm"], default=None)
    group.add_argument("--static-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--alpha-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
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
        help="Wall-clock interval between LLM decisions (default: 1.0 s).",
    )
    group.add_argument(
        "--llm-safeguard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable validation/correction of LLM policy modes.",
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

if __name__ == "__main__":
    args = parse_args()
    run_position_sweep(
        policy_names=args.policies or ["LLM", "alpharacer", "alpharacer"],
        trials_per_position=1,  # total race = trials_per_position * len(ego_positions)
        ego_positions=[args.ego_position] if args.ego_position is not None else [1, 2, 3],
        output_path="result/position_sweep_summary.json",
        vehicle_model_type=args.vehicle_model or "kinematic",
        static_policy_params=args.static_params,
        alpharacer_params=args.alpha_params,
        llm_policy_params=args.llm_params,
        llm_output_mode=args.llm_output_mode or "mode",
        llm_model_name=args.llm_model,
        llm_model_names=args.llm_models,
        llm_update_interval_seconds=(
            args.llm_update_interval_seconds
            if args.llm_update_interval_seconds is not None
            else 1.0
        ),
        llm_temperature=args.llm_temperature if args.llm_temperature is not None else 1.0,
        llm_temperatures=args.llm_temperatures,
        llm_safeguard_enabled=args.llm_safeguard if args.llm_safeguard is not None else False,
        self_play_rl_model_path=args.rl_model_path,
        self_play_rl_backend=args.rl_backend,
        self_play_rl_strict_load=args.rl_strict_load or None,
        self_play_rl_action_order=args.rl_action_order,
    )
