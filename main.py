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
    from generate_html_report import generate_report
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


def _trace_safety_summary(data):
    """Compact activation metrics for comparing otherwise large trace JSONs."""
    sim_dt = float(np.asarray(data["simulation_dt"]))
    car_count = int(np.asarray(data["car_px"]).shape[1])

    def time_by_car(key):
        if key not in data.files:
            return [None] * car_count
        active = np.asarray(data[key], dtype=bool)
        return _jsonable(np.sum(active, axis=0) * sim_dt)

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

        summaries.append({
            "prompt_variant": (
                str(np.asarray(data["car_llm_prompt_variants"])[car_idx])
                if "car_llm_prompt_variants" in data.files
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


def _policy_params_for_name(policy_names, snapshot):
    first_policy = _slug(policy_names[0]) if policy_names else "static"
    if first_policy in {
        "llm",
        "llm-alpha",
        "llm-alpharacer",
        "llm_alpharacer",
        "llm_parameter_policy",
        "llm_rules",
        "llm_with_rules",
        "llm_explicit_rules",
        "rule",
        "rules",
        "rule-based",
        "rule_based_safeguard",
        "safeguard_baseline",
        "decision_tree",
        "rule_based",
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
    alpharacer_weight_path=None,
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
        "alpharacer_weight_path": alpharacer_weight_path,
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
    alpharacer_weight_path=None,
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
        alpharacer_weight_path=alpharacer_weight_path,
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
    if overrides["alpharacer_weight_path"] is not None:
        snapshot["ALPHARACER_WEIGHT_PATH"] = str(overrides["alpharacer_weight_path"])
        snapshot["POTENTIAL_FUNCTION_WEIGHT_PATH"] = str(overrides["alpharacer_weight_path"])
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
    alpharacer_weight_path=None,
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
        alpharacer_weight_path=alpharacer_weight_path,
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
    alpharacer_weight_path=None,
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
                alpharacer_weight_path=alpharacer_weight_path,
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
    keep_npz=False,
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
    trace_tmp_dir = output_path.parent / (
        f"{output_path.stem}_traces" if keep_npz else f"_{output_path.stem}_tmp"
    )
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
                    safety_summary = _trace_safety_summary(data)
                    llm_summary = _trace_llm_summary(data)
                    car_px = np.asarray(data["car_px"], dtype=float)
                    final_progress = car_px[-1].copy()
                    winner_index = int(np.nanargmax(final_progress))
                    winner_role = roles[winner_index]
                    start_ranks = [int(value) for value in np.asarray(data["start_ranks"]).tolist()]
                if not keep_npz:
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
                        "safety_summary": safety_summary,
                        "decision_summary_by_car": llm_summary,
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
    group.add_argument(
        "--suite",
        choices=["llm-baselines", "llm-vs-ibr", "ibr-speed-cap"],
        default=None,
        help=(
            "llm-baselines runs raw LLM, explicit-rule LLM, rule-only, "
            "alpha-RACER, and IBR against two alpha-RACER opponents; "
            "llm-vs-ibr runs the three LLM/rule candidates against two IBR "
            "opponents. Every candidate is tested at all three positions."
        ),
    )


def _add_sim_args(parser):
    group = parser.add_argument_group("simulation")
    group.add_argument("--vehicle-model", choices=["kinematic", "kinetic", "kinetics", "dbm"], default=None)
    group.add_argument("--total-time", type=float, default=None, metavar="SECONDS")
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
        help="Wall-clock interval between LLM decisions (default: 1.0 s).",
    )
    group.add_argument(
        "--llm-safeguard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable validation/correction of LLM policy modes.",
    )
    group.add_argument(
        "--ibr-implementation",
        choices=["local_wang", "historical_strict"],
        default="local_wang",
        help=(
            "IBR solver: local_wang uses local dynamics and supports SE; "
            "historical_strict is the locally adapted alpha-RACER IBR "
            "(default: local_wang)."
        ),
    )
    group.add_argument(
        "--ibr-se",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable SE-IBR using config.IBR_SE_DEFAULT_WEIGHT (default: ordinary IBR).",
    )
    group.add_argument(
        "--ibr-sensitivity-weight",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Set Wang SE-IBR alpha explicitly; 0 selects ordinary IBR.",
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
    params.IBR_IMPLEMENTATION = args.ibr_implementation
    if args.ibr_sensitivity_weight is not None:
        if args.ibr_sensitivity_weight < 0.0:
            raise ValueError("--ibr-sensitivity-weight must be non-negative")
        params.IBR_SENSITIVITY_WEIGHT = args.ibr_sensitivity_weight
    elif args.ibr_se is True:
        params.IBR_SENSITIVITY_WEIGHT = params.IBR_SE_DEFAULT_WEIGHT
    elif args.ibr_se is False:
        params.IBR_SENSITIVITY_WEIGHT = 0.0
    if args.ibr_speed_cap is not None:
        params.IBR_SPEED_CAP_MODE = args.ibr_speed_cap
    shared_sweep_kwargs = dict(
        trials_per_position=1,
        ego_positions=[args.ego_position] if args.ego_position is not None else [1, 2, 3],
        vehicle_model_type=args.vehicle_model or "kinematic",
        total_time=args.total_time,
        start_gap=args.start_gap,
        initial_speeds=args.initial_speeds,
        static_policy_params=args.static_params,
        alpharacer_params=args.alpha_params,
        alpharacer_weight_path=args.alpharacer_weight,
        ibr_implementation=params.IBR_IMPLEMENTATION,
        llm_policy_params=args.llm_params,
        llm_output_mode=args.llm_output_mode,
        llm_model_name=args.llm_model,
        llm_model_names=args.llm_models,
        llm_update_interval_seconds=args.llm_update_interval_seconds,
        llm_temperature=args.llm_temperature if args.llm_temperature is not None else 0.0,
        llm_temperatures=args.llm_temperatures,
        llm_safeguard_enabled=args.llm_safeguard if args.llm_safeguard is not None else False,
        self_play_rl_model_path=args.rl_model_path,
        self_play_rl_backend=args.rl_backend,
        self_play_rl_strict_load=args.rl_strict_load or None,
        self_play_rl_action_order=args.rl_action_order,
    )

    if args.suite in {"llm-baselines", "llm-vs-ibr", "ibr-speed-cap"}:
        if args.suite == "llm-baselines":
            suite_dir = Path("result/llm_baseline_suite")
            # All candidates face the same two alpha-RACER opponents. For the
            # alpha-RACER candidate this naturally becomes a homogeneous race.
            candidates = {
                "raw_llm": ["llm_alpharacer", "alpharacer", "alpharacer"],
                "explicit_rule_llm": ["llm_explicit_rules", "alpharacer", "alpharacer"],
                "rule_only": ["rule_based", "alpharacer", "alpharacer"],
                "local_alpharacer": ["alpharacer", "alpharacer", "alpharacer"],
                "local_wang_ibr": ["ibr", "alpharacer", "alpharacer"],
            }
        elif args.suite == "llm-vs-ibr":
            suite_dir = Path("result/llm_vs_ibr_suite")
            candidates = {
                "raw_llm_vs_ibr": ["llm_alpharacer", "ibr", "ibr"],
                "explicit_rule_llm_vs_ibr": ["llm_explicit_rules", "ibr", "ibr"],
                "rule_only_vs_ibr": ["rule_based", "ibr", "ibr"],
            }
        else:
            suite_dir = Path("result/ibr_speed_cap_suite")
            params.IBR_SPEED_CAP_MODE = "raceline"
            candidates = {
                "raceline_capped_ibr": ["ibr", "alpharacer", "alpharacer"],
            }
        suite_dir.mkdir(parents=True, exist_ok=True)
        sweep_paths = []
        for label, policy_names in candidates.items():
            sweep_path = suite_dir / f"{label}.json"
            run_position_sweep(
                policy_names=policy_names,
                output_path=sweep_path,
                keep_npz=True,
                **shared_sweep_kwargs,
            )
            sweep_paths.append(sweep_path)
        report_path = suite_dir / "report.html"
    else:
        sweep_output_path = Path("result/position_sweep_summary.json")
        run_position_sweep(
            policy_names=args.policies or ["LLM", "ibr", "ibr"],
            output_path=sweep_output_path,
            **shared_sweep_kwargs,
        )
        sweep_paths = [sweep_output_path]
        report_path = Path("result/position_sweep_report.html")

    generate_report(
        sweep_paths,
        output=report_path,
        animation_stride=args.stride,
        animation_interval=args.interval,
        animation_dpi=args.dpi,
        include_animation=not args.no_animation,
    )
