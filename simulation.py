import argparse

import numpy as np

from collision import oriented_rectangle_overlap_cost
from config import duration_to_steps, make_vehicle_params, params
from mpc_controller import MPCController
from plotting import plot_simulation_results
from policy import (
    AlphaRacerPolicy,
    CenterLinePolicy,
    ExplicitRuleLLMPolicy,
    IteratedBestResponsePolicy,
    LLMAlphaRacerPolicy,
    ParameterIBRPolicy,
    RuleBasedSafeguardPolicy,
    SelfPlayRLPolicy,
    StaticAlphaRacerPolicy,
    plan_alpharacer_policy_params,
    plan_ibr_joint_actions,
    progress_delta,
    track_heading_error,
)
from state import CarAction, CarState
from simulation_trace import (
    create_simulation_trace,
    record_simulation_step,
    save_simulation_trace,
)
from track import Track, configure_track, load_centerline_csv, load_raceline_csv
from vehicle_model import DynamicBicycleModel, KinematicBicycleModel, neutral_throttle

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def project_vehicle_state(track, state, s_hint, search_radius, max_heading_error):
    s_wrapped, e = track.project(state.x, state.y)
    if s_hint is None:
        return s_wrapped, e
    return s_hint + progress_delta(s_wrapped, s_hint, track.total_length), e


def _normalize_policy_name(policy_name):
    key = str(policy_name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "alpha": "alpharacer",
        "alpha_racer": "alpharacer",
        "alpharacer": "alpharacer",

        "weighted": "alpharacer",
        "weight": "alpharacer",
        "weights": "alpharacer",
        "weighted_alpha": "alpharacer",
        "weighted_alpha_racer": "alpharacer",
        "weighted_alpharacer": "alpharacer",
        "potential": "alpharacer",
        "potential_function": "alpharacer",
        "potential_function_policy": "alpharacer",
        "potential_function_min_policy": "alpharacer",
        "ours": "alpharacer",
        "ours_high_p": "alpharacer",

        "static": "static_alpharacer",
        "static_alpha": "static_alpharacer",
        "static_alpha_racer": "static_alpharacer",
        "static_alpharacer": "static_alpharacer",
        "fixed_alpha": "static_alpharacer",
        "fixed_alpharacer": "static_alpharacer",

        "llm": "llm_alpharacer",
        "llm_alpha": "llm_alpharacer",
        "llm_alpha_racer": "llm_alpharacer",
        "llm_alpharacer": "llm_alpharacer",
        "llmalpha": "llm_alpharacer",
        "llmalpharacer": "llm_alpharacer",

        "llm_rules": "llm_explicit_rules",
        "llm_rule": "llm_explicit_rules",
        "llm_with_rules": "llm_explicit_rules",
        "llm_explicit_rules": "llm_explicit_rules",
        "explicit_rule_llm": "llm_explicit_rules",

        "rule": "rule_based",
        "rules": "rule_based",
        "rule_based": "rule_based",
        "rule_based_safeguard": "rule_based",
        "safeguard_baseline": "rule_based",
        "decision_tree": "rule_based",

        "rl": "self_play_rl",
        "ppo": "self_play_rl",
        "ppo_rl": "self_play_rl",
        "selfplay": "self_play_rl",
        "self_play": "self_play_rl",
        "self_play_rl": "self_play_rl",
        "selfplay_rl": "self_play_rl",

        "ibr": "ibr",
        "parameter_ibr": "parameter_ibr",
        "param_ibr": "parameter_ibr",
        "ibr_parameters": "parameter_ibr",
        "ibr_params": "parameter_ibr",

        "center": "centerline",
        "center_line": "centerline",
        "centerline": "centerline",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unknown policy type: {policy_name}") from exc


def _resolve_policy_names(policy_type, policy_names, num_cars):
    if policy_names is not None:
        if len(policy_names) != num_cars:
            raise ValueError(f"policy_names length must equal num_cars ({num_cars}), got {len(policy_names)}")
        return [_normalize_policy_name(name) for name in policy_names]

    policy_type = _normalize_policy_name(policy_type)
    if policy_type == "llm_alpharacer" and num_cars > 1:
        return ["llm_alpharacer"] + ["ibr"] * (num_cars - 1)
    return [policy_type] * num_cars


def _per_car_option(value, car_idx, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if car_idx >= len(value):
            return default
        item = value[car_idx]
        if item is None:
            return default
        if isinstance(item, str) and item.strip().lower() in {"", "none", "default", "-"}:
            return default
        return item
    return value


def create_vehicle_model(model_type, dt, db_params, max_speed):
    model_type = model_type.lower()
    if model_type in {"kinematic", "kinetic", "kinetics"}:
        return KinematicBicycleModel(dt=dt, db_params=db_params, max_speed=max_speed)
    if model_type == "dbm":
        return DynamicBicycleModel(dt=dt, db_params=db_params)
    raise ValueError(f"Unknown vehicle model: {model_type}")


def create_policy(
    policy_type,
    track,
    mpc,
    raceline=None,
    car_idx=0,
    static_policy_params=None,
    alpharacer_params=None,
    alpharacer_weight_path=None,
    ibr_implementation=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_temperature=None,
    llm_update_interval_seconds=None,
    llm_safeguard_enabled=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
):
    policy_type = _normalize_policy_name(policy_type)
    if policy_type == "centerline":
        return CenterLinePolicy(
            track,
            mpc,
            step_distance=params.MPC_DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
        )
    if policy_type == "static_alpharacer":
        # alpharacer_params is accepted here only as a legacy name for fixed
        # executable q, zeta, s1, s2, s3 parameters.
        fixed_params = static_policy_params if static_policy_params is not None else alpharacer_params
        return StaticAlphaRacerPolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.STATIC_POLICY_PARAMS
                if fixed_params is None
                else fixed_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
        )
    if policy_type in {"llm_alpharacer", "llm_explicit_rules"}:
        policy_class = (
            ExplicitRuleLLMPolicy
            if policy_type == "llm_explicit_rules"
            else LLMAlphaRacerPolicy
        )
        return policy_class(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.LLM_POLICY_PARAMS if llm_policy_params is None else llm_policy_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            model_name=params.LLM_MODEL if llm_model_name is None else llm_model_name,
            base_url=params.LLM_BASE_URL,
            timeout=params.LLM_TIMEOUT,
            temperature=params.LLM_TEMPERATURE if llm_temperature is None else llm_temperature,
            update_interval_seconds=(
                params.LLM_UPDATE_INTERVAL_SECONDS
                if llm_update_interval_seconds is None
                else llm_update_interval_seconds
            ),
            decision_dt=params.CONTROL_DT,
            history_length=params.LLM_RECENT_MODES_MAX_LENGTH,
            output_mode=params.LLM_OUTPUT_MODE if llm_output_mode is None else llm_output_mode,
            safeguard_enabled=(
                params.LLM_SAFEGUARD_ENABLED
                if llm_safeguard_enabled is None
                else llm_safeguard_enabled
            ),
        )
    if policy_type == "rule_based":
        return RuleBasedSafeguardPolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.LLM_POLICY_PARAMS if llm_policy_params is None else llm_policy_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            update_interval_seconds=(
                params.LLM_UPDATE_INTERVAL_SECONDS
                if llm_update_interval_seconds is None
                else llm_update_interval_seconds
            ),
            decision_dt=params.CONTROL_DT,
            history_length=params.LLM_RECENT_MODES_MAX_LENGTH,
        )
    if policy_type == "alpharacer":
        return AlphaRacerPolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            weight_path=(
                params.ALPHARACER_WEIGHT_PATH
                if alpharacer_weight_path is None
                else alpharacer_weight_path
            ),
            search_points=params.ALPHARACER_WEIGHT_SEARCH_POINTS,
            search_iters=params.ALPHARACER_WEIGHT_SEARCH_ITERS,
            opt_style=params.POTENTIAL_FUNCTION_OPT_STYLE,
            gradient_rate=params.POTENTIAL_FUNCTION_GRADIENT_RATE,
            initial_params=params.ALPHARACER_INITIAL_PARAMS,
        )
    if policy_type == "self_play_rl":
        return SelfPlayRLPolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.STATIC_POLICY_PARAMS if alpharacer_params is None else alpharacer_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            model_path=params.SELF_PLAY_RL_MODEL_PATH if self_play_rl_model_path is None else self_play_rl_model_path,
            backend=params.SELF_PLAY_RL_BACKEND if self_play_rl_backend is None else self_play_rl_backend,
            strict_load=(
                params.SELF_PLAY_RL_STRICT_LOAD
                if self_play_rl_strict_load is None
                else self_play_rl_strict_load
            ),
            action_order=(
                params.SELF_PLAY_RL_ACTION_ORDER
                if self_play_rl_action_order is None
                else self_play_rl_action_order
            ),
            throttle_bounds=params.SELF_PLAY_RL_THROTTLE_BOUNDS,
            steer_bounds=params.SELF_PLAY_RL_STEER_BOUNDS,
        )
    if policy_type == "ibr":
        return IteratedBestResponsePolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            iterations=params.IBR_ITERATIONS,
            sensitivity_weight=params.IBR_SENSITIVITY_WEIGHT,
            implementation=(
                params.IBR_IMPLEMENTATION
                if ibr_implementation is None
                else ibr_implementation
            ),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
        )
    if policy_type == "parameter_ibr":
        return ParameterIBRPolicy(
            track,
            mpc,
            car_idx=car_idx,
            policy_name=policy_type,
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            initial_params=params.ALPHARACER_INITIAL_PARAMS,
            search_points=params.PARAMETER_IBR_SEARCH_POINTS,
            search_passes=params.PARAMETER_IBR_SEARCH_PASSES,
            update_interval_seconds=params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS,
            convergence_tolerance=params.PARAMETER_IBR_CONVERGENCE_TOLERANCE,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


def run_simulation(
    policy_type="centerline",
    track_name=None,
    track_scale=None,
    save_trace=True,
    show_plot=True,
    num_cars=None,
    policy_names=None,
    vehicle_model_type=None,
    static_policy_params=None,
    static_policy_params_per_car=None,
    alpharacer_params=None,
    alpharacer_weight_path=None,
    alpharacer_weight_paths=None,
    ibr_implementation=None,
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
    v0=params.v0 if hasattr(params, "v0") else 0.0,
    start_gap=None,
    initial_longitudinal_offsets=None,
    initial_lateral_offsets=None,
    initial_speeds=None,
    velocity_track_weight=None,
    horizon=None,
    ego_position=1,
    total_time=None,
    quiet=False,
):
    if track_name is not None:
        configure_track(params, track_name, scale=track_scale)
    elif track_scale is not None:
        params.TRACK_SCALE = float(track_scale)

    waypoints = load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE)
    raceline = load_raceline_csv(params.RACELINE_CSV, scale=params.TRACK_SCALE)
    track = Track(waypoints)

    if num_cars is None and policy_names is not None:
        num_cars = len(policy_names)
    elif num_cars is None:
        num_cars = params.NUM_CARS
    resolved_policy_names = _resolve_policy_names(policy_type, policy_names, num_cars)
    if alpharacer_weight_paths is not None and len(alpharacer_weight_paths) != num_cars:
        raise ValueError(
            f"alpharacer_weight_paths length must equal num_cars ({num_cars}), "
            f"got {len(alpharacer_weight_paths)}"
        )
    if llm_model_names is not None and len(llm_model_names) != num_cars:
        raise ValueError(f"llm_model_names length must equal num_cars ({num_cars}), got {len(llm_model_names)}")
    resolved_llm_model_names = [
        _per_car_option(llm_model_names, car_idx, llm_model_name)
        for car_idx in range(num_cars)
    ]
    if llm_temperatures is not None and len(llm_temperatures) != num_cars:
        raise ValueError(f"llm_temperatures length must equal num_cars ({num_cars}), got {len(llm_temperatures)}")
    resolved_llm_temperatures = [
        _per_car_option(llm_temperatures, car_idx, llm_temperature)
        for car_idx in range(num_cars)
    ]
    ego_position = int(ego_position)
    if ego_position < 1 or ego_position > num_cars:
        raise ValueError(f"ego_position must be between 1 and {num_cars}, got {ego_position}")
    ego_idx = 0
    ego_start_rank = ego_position - 1
    start_ranks = [None] * num_cars
    start_ranks[ego_idx] = ego_start_rank
    remaining_ranks = [rank for rank in range(num_cars) if rank != ego_start_rank]
    for car_idx in range(1, num_cars):
        start_ranks[car_idx] = remaining_ranks[car_idx - 1]
    start_gap = params.CAR_START_GAP if start_gap is None else float(start_gap)
    if start_gap <= 0.0:
        raise ValueError(f"start_gap must be positive, got {start_gap}")

    def initial_values(values, default, name):
        if values is None:
            return np.full(num_cars, default, dtype=float)
        array = np.asarray(values, dtype=float)
        if array.shape != (num_cars,):
            raise ValueError(f"{name} must contain {num_cars} values, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    initial_longitudinal_offsets = initial_values(
        initial_longitudinal_offsets, 0.0, "initial_longitudinal_offsets"
    )
    initial_lateral_offsets = initial_values(
        initial_lateral_offsets, 0.0, "initial_lateral_offsets"
    )
    initial_speeds = initial_values(initial_speeds, float(v0), "initial_speeds")
    if np.any(initial_speeds < 0.0):
        raise ValueError("initial_speeds must be non-negative")
    vehicle_model_type = (params.VEHICLE_MODEL if vehicle_model_type is None else vehicle_model_type).lower()
    vehicle_params = make_vehicle_params(vehicle_model_type)
    mpc_horizon = params.MPC_HORIZON if horizon is None else int(horizon)
    mpc_velocity_track_weight = (
        params.VELOCITY_TRACK_WEIGHT if velocity_track_weight is None else float(velocity_track_weight)
    )

    mpcs = []
    policies = []
    vehicle_models = []
    states = []
    for car_idx in range(num_cars):
        policy_name = resolved_policy_names[car_idx]
        car_mpc_horizon = (
            params.IBR_HORIZON
            if horizon is None and policy_name == "ibr"
            else mpc_horizon
        )
        mpc = MPCController(
            dt=params.MPC_DT,
            horizon=car_mpc_horizon,
            Q=params.Q,
            R=params.R,
            max_vel_rate=params.MAX_VEL_RATE,
            max_steer_delta=params.MPC_MAX_STEER_RATE * params.MPC_DT,
            db_params=vehicle_params,
            track_width=params.TRACK_WIDTH,
            track_constraint_half_width=params.MPC_CONSTRAINT_HALF_WIDTH,
            max_state_speed=params.MAX_STATE_SPEED,
            velocity_track_weight=mpc_velocity_track_weight,
            heading_track_weight=params.HEADING_TRACK_WEIGHT,
            track_length=track.total_length,
            min_longitudinal_separation=params.MPC_MIN_LONGITUDINAL_SEPARATION,
            min_lateral_separation=params.MPC_MIN_LATERAL_SEPARATION,
            max_constrained_opponents=params.MPC_MAX_CONSTRAINED_OPPONENTS,
            collision_superellipse_exponent=params.MPC_COLLISION_SUPERELLIPSE_EXPONENT,
            enforce_track_boundary=params.MPC_ENFORCE_TRACK_BOUNDARY,
            enforce_collision_avoidance=params.MPC_ENFORCE_COLLISION_AVOIDANCE,
            track_slack_weight=params.MPC_TRACK_SLACK_WEIGHT,
            collision_slack_weight=params.MPC_COLLISION_SLACK_WEIGHT,
            enforce_friction_speed_limit=params.MPC_ENFORCE_FRICTION_SPEED_LIMIT,
            friction_coefficient=params.MPC_FRICTION_COEFFICIENT,
        )
        mpcs.append(mpc)
        policies.append(
            create_policy(
                resolved_policy_names[car_idx],
                track,
                mpc,
                raceline=raceline,
                car_idx=car_idx,
                static_policy_params=(
                    static_policy_params
                    if static_policy_params_per_car is None
                    else np.asarray(static_policy_params_per_car, dtype=float)[car_idx]
                ),
                alpharacer_params=alpharacer_params,
                alpharacer_weight_path=(
                    alpharacer_weight_path
                    if alpharacer_weight_paths is None
                    else _per_car_option(alpharacer_weight_paths, car_idx, alpharacer_weight_path)
                ),
                ibr_implementation=ibr_implementation,
                llm_policy_params=llm_policy_params,
                llm_output_mode=llm_output_mode,
                llm_model_name=resolved_llm_model_names[car_idx],
                llm_temperature=resolved_llm_temperatures[car_idx],
                llm_update_interval_seconds=llm_update_interval_seconds,
                llm_safeguard_enabled=llm_safeguard_enabled,
                self_play_rl_model_path=self_play_rl_model_path,
                self_play_rl_backend=self_play_rl_backend,
                self_play_rl_strict_load=self_play_rl_strict_load,
                self_play_rl_action_order=self_play_rl_action_order,
            )
        )
        vehicle_models.append(
            create_vehicle_model(
                vehicle_model_type,
                dt=params.SIM_DT,
                db_params=vehicle_params,
                max_speed=params.MAX_STATE_SPEED,
            )
        )

        start_s = -start_ranks[car_idx] * start_gap + initial_longitudinal_offsets[car_idx]
        start_e = initial_lateral_offsets[car_idx]
        start_x, start_y = track.frenet_to_world(start_s, start_e)
        _, _, start_psi = track.interpolate(start_s)
        states.append(
            CarState(
                x=start_x,
                y=start_y,
                vx=initial_speeds[car_idx],
                vy=0.0,
                psi=start_psi,
                omega=0.0,
            )
        )

    control_steps = duration_to_steps(params.CONTROL_DT, params.SIM_DT)
    if not np.isclose(control_steps * params.SIM_DT, params.CONTROL_DT):
        raise ValueError("CONTROL_DT must be an integer multiple of SIM_DT")
    simulation_time = params.TOTAL_TIME if total_time is None else float(total_time)
    total_steps = int(simulation_time / params.SIM_DT)
    early_collision_window_steps = duration_to_steps(
        params.EARLY_COLLISION_WINDOW_SECONDS, params.SIM_DT
    )
    early_collision_consecutive_steps = duration_to_steps(
        params.EARLY_COLLISION_CONSECUTIVE_SECONDS, params.SIM_DT
    )
    ref_debug_sample_steps = duration_to_steps(
        params.REF_DEBUG_SAMPLE_INTERVAL_SECONDS, params.SIM_DT
    )
    trajectory_sample_steps = duration_to_steps(
        params.TRAJECTORY_SAMPLE_INTERVAL_SECONDS, params.SIM_DT
    )
    progress_report_steps = duration_to_steps(
        params.PROGRESS_REPORT_INTERVAL_SECONDS, params.SIM_DT
    )
    track_half_width = getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)
    offroad_recovery_threshold = params.OFFROAD_RECOVERY_THRESHOLD
    progress_s = [
        -start_ranks[car_idx] * start_gap + initial_longitudinal_offsets[car_idx]
        for car_idx in range(num_cars)
    ]

    traj_x, traj_y, velocities, lateral_errors = [], [], [], []
    all_traj_x = [[] for _ in range(num_cars)]
    all_traj_y = [[] for _ in range(num_cars)]
    longitudinal_positions = []
    mpc_losses, mpc_track_losses, mpc_action_losses = [], [], []
    mpc_velocity_losses, mpc_heading_losses = [], []
    throttle_history, steering_history = [], []
    frenet_ref_lines = []
    trace = create_simulation_trace(
        resolved_policy_names=resolved_policy_names,
        policies=policies,
        ego_idx=ego_idx,
        ibr_implementation=ibr_implementation,
        resolved_llm_model_names=resolved_llm_model_names,
        resolved_llm_temperatures=resolved_llm_temperatures,
        num_cars=num_cars,
        llm_update_interval_seconds=llm_update_interval_seconds,
        ego_position=ego_position,
        start_ranks=start_ranks,
        start_gap=start_gap,
        initial_longitudinal_offsets=initial_longitudinal_offsets,
        initial_lateral_offsets=initial_lateral_offsets,
        initial_speeds=initial_speeds,
        vehicle_model_type=vehicle_model_type,
        mpc_velocity_track_weight=mpc_velocity_track_weight,
    )
    ref_component_horizon = max(mpc.horizon for mpc in mpcs) + 1

    def ref_component_snapshot(policy, name):
        values = np.full(ref_component_horizon, np.nan, dtype=float)
        components = getattr(policy, "last_ref_components", None)
        if components is None or name not in components:
            return values
        source = np.asarray(components[name], dtype=float).reshape(-1)
        count = min(source.size, ref_component_horizon)
        values[:count] = source[:count]
        return values

    if not quiet:
        print(
            "Simulation setup: "
            f"track={params.TRACK_CSV}, scale={params.TRACK_SCALE:g}, "
            f"half_width={params.TRACK_HALF_WIDTH:g}, "
            f"policies={resolved_policy_names}, ego_start_position={ego_position}/{num_cars}, "
            f"vehicle_model={vehicle_model_type}, "
            f"timing=sim:{params.SIM_DT:g}s/control:{params.CONTROL_DT:g}s/"
            f"mpc:{params.MPC_DT:g}s"
        )
    if not quiet and any(name == "llm_alpharacer" for name in resolved_policy_names):
        print(f"LLM models by car: {trace['llm_model_names']}")
        print(f"LLM temperatures by car: {trace['llm_temperatures']}")
        print(f"LLM safeguards by car: {trace['car_llm_safeguard_enabled']}")
    progress_iter = range(total_steps)
    progress_bar = None
    if tqdm is not None and not quiet:
        progress_bar = tqdm(
            progress_iter,
            total=total_steps,
            desc="Simulation",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        progress_iter = progress_bar
        for policy in policies:
            policy.log_fn = progress_bar.write
    else:
        for policy in policies:
            policy.log_fn = print

    early_collision_count = 0
    early_stop = False
    early_stop_reason = ""
    last_step = -1
    held_actions = [CarAction(throttle=0.0, steer=0.0) for _ in range(num_cars)]
    for step in progress_iter:
        last_step = step
        car_s = []
        car_e = []
        for car_idx, state in enumerate(states):
            s, e_before = project_vehicle_state(
                track,
                state,
                progress_s[car_idx],
                params.PROJECTION_SEARCH_RADIUS,
                params.PROJECTION_MAX_HEADING_ERROR,
            )
            progress_s[car_idx] = s
            car_s.append(s)
            car_e.append(e_before)

        control_update = step % control_steps == 0
        if control_update:
            next_actions = []
            sim_time = step * params.SIM_DT
            plan_alpharacer_policy_params(policies, states, car_s, car_e)
            plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=sim_time)
            for car_idx, state in enumerate(states):
                opponents = [
                    {
                        "car_idx": j,
                        "px": car_s[j],
                        "py": car_e[j],
                        "vx": max(states[j].vx, 0.0),
                        "vy": states[j].vy,
                        "omega": states[j].omega,
                        "psi": states[j].psi,
                        "state": states[j],
                    }
                    for j in range(num_cars)
                    if j != car_idx
                ]
                next_actions.append(
                    policies[car_idx].get_action(
                        state,
                        car_s[car_idx],
                        opponents=opponents,
                        sim_time=sim_time,
                    )
                )
            held_actions = next_actions
        # Apply the released alpha-RACER simulator's off-track recovery outside
        # every high-level policy.  The upstream implementation runs at the
        # control period; exponent strengths are scaled here because vehicle
        # integration is finer than MPC/control updates.
        actions = [
            CarAction(throttle=action.throttle, steer=action.steer)
            for action in held_actions
        ]

        # Apply off-track recovery to all cars when happens
        offroad_recovery_active = []
        offroad_recovery_heading_error = []
        recovery_dt_ratio = params.SIM_DT / params.CONTROL_DT

        for car_idx, state in enumerate(states):
            e = car_e[car_idx]
            active = abs(e) > offroad_recovery_threshold
            heading_error = track_heading_error(track, state.psi, car_s[car_idx])
            offroad_recovery_active.append(active)
            offroad_recovery_heading_error.append(heading_error if active else 0.0)
            if not active:
                continue

            overshoot = abs(e) - offroad_recovery_threshold
            speed_factor = np.exp(
                -params.OFFROAD_STRENGTH * overshoot * recovery_dt_ratio
            )

            # off-road penalty on velocity
            state.vx *= speed_factor
            state.vy *= speed_factor

            # mandatory heading realignment when off-road
            heading_blend = 1.0 - np.exp(
                -params.OFFROAD_HEADING_REALIGN_STRENGTH
                * overshoot
                * recovery_dt_ratio
            )
            state.psi = np.arctan2(
                np.sin(state.psi + heading_blend * heading_error),
                np.cos(state.psi + heading_blend * heading_error),
            )

            # mandatory inward steering when off-road
            steer_blend = 1.0 - np.exp(
                -params.OFFROAD_STRENGTH * overshoot * recovery_dt_ratio
            )
            inward_steer = -np.sign(e) * params.OFFROAD_INWARD_STEER
            actions[car_idx].steer += (
                inward_steer - actions[car_idx].steer
            ) * steer_blend
            if abs(heading_error) > params.OFFROAD_LARGE_HEADING_ERROR:
                actions[car_idx].throttle += params.OFFROAD_THROTTLE_BOOST
            if state.vx < params.OFFROAD_LOW_SPEED_THRESHOLD:
                vehicle_params_for_car = vehicle_models[car_idx].p
                neutral_command = neutral_throttle(state.vx, vehicle_params_for_car)
                actions[car_idx].throttle = max(
                    actions[car_idx].throttle,
                    neutral_command + params.OFFROAD_THROTTLE_BOOST,
                )

            actions[car_idx].steer = float(np.clip(actions[car_idx].steer, -1.0, 1.0))
            actions[car_idx].throttle = float(
                np.clip(actions[car_idx].throttle, -1.0, 1.0)
            )

        ego_policy = policies[ego_idx]
        ego_mpc = mpcs[ego_idx]
        ego_state = states[ego_idx]
        ref_fx, ref_fy = record_simulation_step(
            trace,
            step=step,
            track=track,
            policies=policies,
            mpcs=mpcs,
            states=states,
            actions=actions,
            car_s=car_s,
            car_e=car_e,
            ego_idx=ego_idx,
            track_half_width=track_half_width,
            offroad_recovery_threshold=offroad_recovery_threshold,
            control_update=control_update,
            offroad_recovery_active=offroad_recovery_active,
            offroad_recovery_heading_error=offroad_recovery_heading_error,
            ref_component_snapshot=ref_component_snapshot,
        )

        if (step + 1) % ref_debug_sample_steps == 0 and ego_policy.last_ref_traj is not None:
            frenet_ref_lines.append((ego_state.x, ego_state.y, ref_fx, ref_fy))

        mpc_losses.append(ego_mpc.last_loss)
        mpc_track_losses.append(ego_mpc.last_track_loss)
        mpc_action_losses.append(ego_mpc.last_action_loss)
        mpc_velocity_losses.append(ego_mpc.last_velocity_loss)
        mpc_heading_losses.append(ego_mpc.last_heading_loss)
        throttle_history.append(actions[ego_idx].throttle)
        steering_history.append(actions[ego_idx].steer)

        for car_idx, state in enumerate(states):
            next_state_arr = vehicle_models[car_idx].step(
                state.to_array(),
                actions[car_idx].throttle,
                actions[car_idx].steer,
            )
            states[car_idx] = CarState.from_array(next_state_arr)

        # Project updated positions for progress ordering and trace output.
        # Physical contact itself follows the upstream simulator's oriented
        # vehicle rectangles; the smooth Frenet superellipse remains an MPC-
        # only constraint because IPOPT needs a differentiable expression.
        next_s = []
        next_e = []
        for car_idx, state in enumerate(states):
            s, e = project_vehicle_state(
                track,
                state,
                progress_s[car_idx],
                params.PROJECTION_SEARCH_RADIUS,
                params.PROJECTION_MAX_HEADING_ERROR,
            )
            progress_s[car_idx] = s
            next_s.append(s)
            next_e.append(e)

        collision_scales = np.ones(num_cars)
        max_collision_overlap = 0.0
        for i in range(num_cars):
            for j in range(i + 1, num_cars):
                overlap_cost = oriented_rectangle_overlap_cost(
                    states[i],
                    states[j],
                    params.VEHICLE_HALF_LENGTH,
                    params.VEHICLE_HALF_WIDTH,
                )
                if overlap_cost <= 0.0:
                    continue
                max_collision_overlap = max(max_collision_overlap, overlap_cost)

                delta_ij = progress_delta(next_s[i], next_s[j], track.total_length)
                if delta_ij > 0.0:
                    front_idx, rear_idx = i, j
                elif delta_ij < 0.0:
                    front_idx, rear_idx = j, i
                else:
                    continue

                collision_scales[front_idx] *= np.exp(
                    -params.COLLISION_FRONT_DECAY
                    * overlap_cost
                    * params.SIM_DT
                    / params.CONTROL_DT
                )
                collision_scales[rear_idx] *= np.exp(
                    -params.COLLISION_REAR_DECAY
                    * overlap_cost
                    * params.SIM_DT
                    / params.CONTROL_DT
                )

        collision_happened = bool(np.any(collision_scales < 1.0))
        trace["collision"][-1] = collision_happened
        trace["collision_overlap"][-1] = max_collision_overlap
        for car_idx, scale in enumerate(collision_scales):
            if scale < 1.0:
                states[car_idx].vx *= scale
                states[car_idx].vy *= scale

        if collision_happened and step < early_collision_window_steps:
            early_collision_count += 1
        elif step < early_collision_window_steps:
            early_collision_count = 0

        lateral_errors.append(next_e[ego_idx])
        longitudinal_positions.append(progress_s[ego_idx])

        if step % trajectory_sample_steps == 0:
            for car_idx, state in enumerate(states):
                all_traj_x[car_idx].append(state.x)
                all_traj_y[car_idx].append(state.y)
            traj_x.append(states[ego_idx].x)
            traj_y.append(states[ego_idx].y)
            velocities.append(states[ego_idx].vx)

        if not quiet and (step + 1) % progress_report_steps == 0:
            current_position = 1 + sum(
                progress_s[car_idx] > progress_s[ego_idx]
                for car_idx in range(num_cars)
                if car_idx != ego_idx
            )
            message = (
                f"Step {step+1}/{total_steps}, t={(step+1)*params.SIM_DT:.1f}s, "
                f"position={current_position}/{num_cars}, "
                f"vx={states[ego_idx].vx:.2f} m/s, e={next_e[ego_idx]:.3f}, "
                f"throttle={actions[ego_idx].throttle:.2f}, steer={actions[ego_idx].steer:.2f}"
            )
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

        if (
            params.EARLY_COLLISION_STOP
            and early_collision_count >= early_collision_consecutive_steps
        ):
            early_stop = True
            early_stop_reason = (
                f"continuous collision for at least "
                f"{params.EARLY_COLLISION_CONSECUTIVE_SECONDS:g}s "
                f"within the first {params.EARLY_COLLISION_WINDOW_SECONDS:g}s"
            )
            message = (
                f"Early stop at step {step+1}/{total_steps}, "
                f"t={(step+1)*params.SIM_DT:.2f}s: {early_stop_reason}"
            )
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)
            break

    completed_steps = last_step + 1 if last_step >= 0 else 0
    trace["early_stop"] = early_stop
    trace["early_stop_reason"] = early_stop_reason
    trace["completed_steps"] = completed_steps
    trace["planned_steps"] = total_steps

    final_time = last_step * params.SIM_DT if last_step >= 0 else 0.0
    for policy in policies:
        policy.flush_solver_warnings(sim_time=final_time)

    # Save every actual LLM call. Padding keeps per-car histories rectangular
    # when a race mixes LLM and non-LLM policies.
    raw_histories = [getattr(policy, "llm_raw_io_history", []) for policy in policies]
    max_raw_calls = max((len(history) for history in raw_histories), default=0)
    trace["llm_raw_call_indices"] = [
        [-1] * (max_raw_calls - len(history)) + [item["call"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_prompts"] = [
        [""] * (max_raw_calls - len(history)) + [item["prompt"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_responses"] = [
        [""] * (max_raw_calls - len(history)) + [item["response"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_errors"] = [
        [""] * (max_raw_calls - len(history)) + [item["error"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_done_reasons"] = [
        [""] * (max_raw_calls - len(history)) + [item.get("done_reason", "") for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_eval_counts"] = [
        [-1] * (max_raw_calls - len(history)) + [item.get("eval_count", -1) for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_prompt_eval_counts"] = [
        [-1] * (max_raw_calls - len(history))
        + [item.get("prompt_eval_count", -1) for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_latency_seconds"] = [
        [np.nan] * (max_raw_calls - len(history))
        + [item.get("latency_seconds", np.nan) for item in history]
        for history in raw_histories
    ]
    trace["car_policy_decision_counts"] = [
        int(getattr(policy, "policy_decision_count", 0)) for policy in policies
    ]

    if save_trace:
        save_simulation_trace(trace, track, params)
    elif not quiet:
        print("Skipped saving simulation trace")

    if show_plot:
        plot_simulation_results(
            waypoints,
            raceline,
            traj_x,
            traj_y,
            velocities,
            lateral_errors,
            mpc_losses,
            mpc_track_losses,
            mpc_action_losses,
            mpc_velocity_losses,
            mpc_heading_losses,
            longitudinal_positions,
            frenet_ref_lines,
            params,
            throttle_history=throttle_history,
            steering_history=steering_history,
            other_trajectories=[
                (all_traj_x[idx], all_traj_y[idx])
                for idx in range(num_cars)
                if idx != ego_idx
            ],
            ref_speed_history=trace["ref_speed"],
            ref_lateral_errors=trace["ref_py"],
            ref_longitudinal_positions=trace["ref_px"],
            policy_params_history=trace["policy_params"],
            policy_names=trace["policy_names"],
            policy_mode_history=trace["policy_mode"],
            track_length=track.total_length,
            velocity_track_weight=mpc_velocity_track_weight,
            heading_track_weight=params.HEADING_TRACK_WEIGHT,
        )

    # Returning the trace is useful for training/data analysis and does not
    # change existing CLI callers, which simply ignore the return value.
    return trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default racer simulation scenario.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["LLM", "ibr", "ibr"],
        help=(
            "One policy per car, e.g. --policies ibr ibr ibr or "
            "--policies parameter_ibr parameter_ibr parameter_ibr."
        ),
    )
    parser.add_argument(
        "--ibr-implementation",
        choices=["local_wang", "historical_strict"],
        default="local_wang",
        help="IBR solver implementation (default: local_wang).",
    )
    parser.add_argument(
        "--ibr-se",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable sensitivity enhancement with config.IBR_SE_DEFAULT_WEIGHT. "
            "Ordinary IBR without SE is the default."
        ),
    )
    parser.add_argument(
        "--ibr-sensitivity-weight",
        type=float,
        default=None,
        metavar="ALPHA",
        help="Explicit SE weight; 0 selects ordinary IBR and overrides --ibr-se.",
    )
    parser.add_argument("--parameter-ibr-rounds", type=int, default=None, metavar="N")
    parser.add_argument("--parameter-ibr-search-points", type=int, default=None, metavar="N")
    parser.add_argument(
        "--parameter-ibr-update-interval", type=float, default=None, metavar="SECONDS"
    )
    parser.add_argument("--ego-position", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--total-time", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--start-gap", type=float, default=None, metavar="METERS")
    parser.add_argument(
        "--initial-speeds",
        nargs="+",
        type=float,
        default=None,
        metavar="MPS",
        help="Initial speed for each car, in the same order as --policies.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not open the result plot.")
    parser.add_argument(
        "--llm-safeguard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable validation/correction of LLM policy modes.",
    )
    cli_args = parser.parse_args()

    if cli_args.ibr_sensitivity_weight is not None:
        if cli_args.ibr_sensitivity_weight < 0.0:
            parser.error("--ibr-sensitivity-weight must be non-negative")
        params.IBR_SENSITIVITY_WEIGHT = cli_args.ibr_sensitivity_weight
    elif cli_args.ibr_se:
        params.IBR_SENSITIVITY_WEIGHT = params.IBR_SE_DEFAULT_WEIGHT
    else:
        params.IBR_SENSITIVITY_WEIGHT = 0.0
    if cli_args.parameter_ibr_rounds is not None:
        if cli_args.parameter_ibr_rounds < 1:
            parser.error("--parameter-ibr-rounds must be at least 1")
        params.PARAMETER_IBR_ROUNDS = cli_args.parameter_ibr_rounds
    if cli_args.parameter_ibr_search_points is not None:
        if cli_args.parameter_ibr_search_points < 2:
            parser.error("--parameter-ibr-search-points must be at least 2")
        params.PARAMETER_IBR_SEARCH_POINTS = cli_args.parameter_ibr_search_points
    if cli_args.parameter_ibr_update_interval is not None:
        if cli_args.parameter_ibr_update_interval < 0.0:
            parser.error("--parameter-ibr-update-interval must be non-negative")
        params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS = cli_args.parameter_ibr_update_interval

    # Options: "centerline", "alpharacer" (trained potential), "static_alpharacer",
    # "ibr", "parameter_ibr", or "llm_alpharacer". "weighted_alpharacer"
    # aliases "alpharacer".
    # You can also pass per-car policies, e.g. policy_names=["alpha", "IBR", "IBR"].
    # Tracks: "berlin" (scale 2), "modena" (scale 2), or "rounded_rectangle" (scale 1).

    # Ordinary IBR (default, no SE):
    # .venv/bin/python simulation.py --policies ibr ibr ibr --no-plot
    # Optional SE-IBR:
    # .venv/bin/python simulation.py --policies ibr ibr ibr --ibr-se --no-plot
    # Five-parameter IBR using the same low-level MPC as alpha-RACER/LLM:
    # .venv/bin/python simulation.py --policies parameter_ibr parameter_ibr parameter_ibr --no-plot
    run_simulation(
        v0 = 0,
        # track_name="modena",  # Change this directly to "modena" or "rounded_rectangle".
        policy_names=cli_args.policies,
        # policy_names=["LLM", "alpharacer", "alpharacer"],  # ego, oppo1, oppo2
        llm_model_names=["qwen2.5:7b", None, None],
        vehicle_model_type="kinematic", ego_position=cli_args.ego_position,
        ibr_implementation=cli_args.ibr_implementation,
        total_time=cli_args.total_time,
        start_gap=cli_args.start_gap,
        initial_speeds=cli_args.initial_speeds,
        show_plot=not cli_args.no_plot,
        llm_temperature = 0.0,
        llm_safeguard_enabled=False,
    )
