import argparse
import json

import numpy as np

from collision import oriented_rectangle_overlap_cost
from config import GRAVITY, duration_to_steps, make_vehicle_params, params
from mpc_controller import MPCController
from plotting import plot_simulation_results
from policy import (
    AlphaRacerPolicy,
    CenterLinePolicy,
    IteratedBestResponsePolicy,
    LLMAlphaRacerPolicy,
    RuleBasedSafeguardPolicy,
    SelfPlayRLPolicy,
    StaticAlphaRacerPolicy,
    plan_alpharacer_policy_params,
    plan_ibr_joint_actions,
    progress_delta,
    track_heading_error,
)
from state import CarAction, CarState
from track import Track, configure_track, load_centerline_csv, load_raceline_csv
from vehicle_model import DynamicBicycleModel, KinematicBicycleModel

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

        "rule": "rule_based_safeguard",
        "rules": "rule_based_safeguard",
        "rule_based": "rule_based_safeguard",
        "rule_based_safeguard": "rule_based_safeguard",
        "safeguard_baseline": "rule_based_safeguard",
        "decision_tree": "rule_based_safeguard",

        "rl": "self_play_rl",
        "ppo": "self_play_rl",
        "ppo_rl": "self_play_rl",
        "selfplay": "self_play_rl",
        "self_play": "self_play_rl",
        "self_play_rl": "self_play_rl",
        "selfplay_rl": "self_play_rl",

        "ibr": "ibr",

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
    if policy_type == "llm_alpharacer":
        return LLMAlphaRacerPolicy(
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
    if policy_type == "rule_based_safeguard":
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
            weight_path=params.ALPHARACER_WEIGHT_PATH,
            search_points=params.ALPHARACER_WEIGHT_SEARCH_POINTS,
            search_iters=params.ALPHARACER_WEIGHT_SEARCH_ITERS,
            opt_style=params.POTENTIAL_FUNCTION_OPT_STYLE,
            gradient_rate=params.POTENTIAL_FUNCTION_GRADIENT_RATE,
            initial_original_params=params.ALPHARACER_INITIAL_ORIGINAL_PARAMS,
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
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


def save_simulation_trace(trace, track, params):
    if not trace["time"]:
        return

    np.savez(
        params.TRACE_OUTPUT,
        time=np.asarray(trace["time"]),
        x=np.asarray(trace["x"]),
        y=np.asarray(trace["y"]),
        psi=np.asarray(trace["psi"]),
        vx=np.asarray(trace["vx"]),
        vy=np.asarray(trace["vy"]),
        omega=np.asarray(trace["omega"]),
        car_x=np.asarray(trace["car_x"]),
        car_y=np.asarray(trace["car_y"]),
        car_psi=np.asarray(trace["car_psi"]),
        car_vx=np.asarray(trace["car_vx"]),
        car_vy=np.asarray(trace["car_vy"]),
        car_omega=np.asarray(trace["car_omega"]),
        car_px=np.asarray(trace["car_px"]),
        car_py=np.asarray(trace["car_py"]),
        car_throttle=np.asarray(trace["car_throttle"]),
        car_steer=np.asarray(trace["car_steer"]),
        car_mpc_loss=np.asarray(trace["car_mpc_loss"]),
        car_mpc_track_loss=np.asarray(trace["car_mpc_track_loss"]),
        car_mpc_action_loss=np.asarray(trace["car_mpc_action_loss"]),
        car_mpc_velocity_loss=np.asarray(trace["car_mpc_velocity_loss"]),
        car_mpc_heading_loss=np.asarray(trace["car_mpc_heading_loss"]),
        car_mpc_violation_loss=np.asarray(trace["car_mpc_violation_loss"]),
        car_mpc_max_track_slack=np.asarray(trace["car_mpc_max_track_slack"]),
        car_mpc_max_collision_slack=np.asarray(
            trace["car_mpc_max_collision_slack"]
        ),
        car_mpc_solver_success=np.asarray(trace["car_mpc_solver_success"]),
        car_mpc_solver_status=np.asarray(trace["car_mpc_solver_status"]),
        car_mpc_recovery_active=np.asarray(trace["car_mpc_recovery_active"]),
        car_mpc_recovery_reason=np.asarray(trace["car_mpc_recovery_reason"]),
        # TEMPORARY DIAGNOSTIC: delete after competitive-offset analysis.
        car_ref_raceline_e=np.asarray(trace["car_ref_raceline_e"]),
        car_ref_p_ot_front=np.asarray(trace["car_ref_p_ot_front"]),
        car_ref_p_ot_rear=np.asarray(trace["car_ref_p_ot_rear"]),
        car_ref_p_bl=np.asarray(trace["car_ref_p_bl"]),
        car_ref_p_ref_e=np.asarray(trace["car_ref_p_ref_e"]),
        collision=np.asarray(trace["collision"]),
        collision_overlap=np.asarray(trace.get("collision_overlap", []), dtype=float),
        px=np.asarray(trace["px"]),
        py=np.asarray(trace["py"]),
        ref_px=np.asarray(trace["ref_px"]),
        ref_py=np.asarray(trace["ref_py"]),
        ref_speed=np.asarray(trace["ref_speed"]),
        ref_x=np.asarray(trace["ref_x"]),
        ref_y=np.asarray(trace["ref_y"]),
        ref_raceline_s=np.asarray(trace["ref_raceline_s"]),
        raceline_horizon_x=np.asarray(trace["raceline_horizon_x"]),
        raceline_horizon_y=np.asarray(trace["raceline_horizon_y"]),
        ref_horizon_x=np.asarray(trace["ref_horizon_x"]),
        ref_horizon_y=np.asarray(trace["ref_horizon_y"]),
        offroad=np.asarray(trace["offroad"]),
        throttle=np.asarray(trace["throttle"]),
        steer=np.asarray(trace["steer"]),
        mpc_loss=np.asarray(trace["mpc_loss"]),
        mpc_track_loss=np.asarray(trace["mpc_track_loss"]),
        mpc_action_loss=np.asarray(trace["mpc_action_loss"]),
        mpc_velocity_loss=np.asarray(trace["mpc_velocity_loss"]),
        mpc_heading_loss=np.asarray(trace["mpc_heading_loss"]),
        mpc_violation_loss=np.asarray(trace["mpc_violation_loss"]),
        mpc_max_track_slack=np.asarray(trace["mpc_max_track_slack"]),
        mpc_max_collision_slack=np.asarray(trace["mpc_max_collision_slack"]),
        mpc_solver_success=np.asarray(trace["mpc_solver_success"]),
        mpc_solver_status=np.asarray(trace["mpc_solver_status"]),
        mpc_recovery_active=np.asarray(trace["mpc_recovery_active"]),
        mpc_recovery_reason=np.asarray(trace["mpc_recovery_reason"]),
        control_update=np.asarray(trace.get("control_update", []), dtype=bool),
        policy_params=np.asarray(trace["policy_params"]),
        policy_mode=np.asarray(trace["policy_mode"]),
        policy_reason=np.asarray(trace.get("policy_reason", []), dtype=str),
        policy_basis=np.asarray(trace.get("policy_basis", []), dtype=str),
        policy_assessment=np.asarray(trace.get("policy_assessment", []), dtype=str),
        policy_required_assessment=np.asarray(trace.get("policy_required_assessment", []), dtype=str),
        policy_assessment_consistent=np.asarray(trace.get("policy_assessment_consistent", []), dtype=bool),
        policy_names=np.asarray(trace.get("policy_names", [])),
        llm_model_names=np.asarray(trace.get("llm_model_names", []), dtype=str),
        llm_temperatures=np.asarray(trace.get("llm_temperatures", []), dtype=float),
        llm_safeguard_enabled=np.asarray(trace.get("llm_safeguard_enabled", False), dtype=bool),
        car_llm_safeguard_enabled=np.asarray(
            trace.get("car_llm_safeguard_enabled", []), dtype=bool
        ),
        llm_raw_call_indices=np.asarray(trace.get("llm_raw_call_indices", []), dtype=int),
        llm_raw_prompts=np.asarray(trace.get("llm_raw_prompts", []), dtype=str),
        llm_raw_responses=np.asarray(trace.get("llm_raw_responses", []), dtype=str),
        llm_raw_errors=np.asarray(trace.get("llm_raw_errors", []), dtype=str),
        llm_raw_done_reasons=np.asarray(trace.get("llm_raw_done_reasons", []), dtype=str),
        llm_raw_eval_counts=np.asarray(trace.get("llm_raw_eval_counts", []), dtype=int),
        car_policy_params=np.asarray(trace.get("car_policy_params", []), dtype=float),
        car_policy_mode=np.asarray(trace.get("car_policy_mode", []), dtype=str),
        car_policy_reason=np.asarray(trace.get("car_policy_reason", []), dtype=str),
        car_policy_basis=np.asarray(trace.get("car_policy_basis", []), dtype=str),
        car_policy_assessment=np.asarray(trace.get("car_policy_assessment", []), dtype=str),
        car_policy_required_assessment=np.asarray(
            trace.get("car_policy_required_assessment", []), dtype=str
        ),
        car_policy_assessment_consistent=np.asarray(
            trace.get("car_policy_assessment_consistent", []), dtype=bool
        ),
        car_policy_requested_mode=np.asarray(trace.get("car_policy_requested_mode", []), dtype=str),
        car_safeguard_override=np.asarray(trace.get("car_safeguard_override", []), dtype=str),
        car_offroad=np.asarray(trace.get("car_offroad", []), dtype=bool),
        car_offroad_recovery_active=np.asarray(
            trace.get("car_offroad_recovery_active", []), dtype=bool
        ),
        car_offroad_recovery_heading_error=np.asarray(
            trace.get("car_offroad_recovery_heading_error", []), dtype=float
        ),
        ego_index=np.asarray(trace.get("ego_index", 0)),
        ego_position=np.asarray(trace.get("ego_position", 1)),
        start_ranks=np.asarray(trace.get("start_ranks", [])),
        track_csv=np.asarray(params.TRACK_CSV),
        raceline_csv=np.asarray(params.RACELINE_CSV),
        track_scale=np.asarray(getattr(params, "TRACK_SCALE", 1.0)),
        track_width=np.asarray(params.TRACK_WIDTH),
        track_half_width=np.asarray(getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)),
        vehicle_half_length=np.asarray(params.VEHICLE_HALF_LENGTH),
        vehicle_half_width=np.asarray(params.VEHICLE_HALF_WIDTH),
        mpc_constraint_half_width=np.asarray(
            getattr(params, "MPC_CONSTRAINT_HALF_WIDTH", getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0))
        ),
        ref_lateral_clip=np.asarray(
            getattr(params, "REF_LATERAL_CLIP", getattr(params, "MPC_CONSTRAINT_HALF_WIDTH", params.TRACK_WIDTH / 2.0))
        ),
        offroad_tolerance=np.asarray(getattr(params, "OFFROAD_TOLERANCE", 0.0)),
        offroad_half_width=np.asarray(
            getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)
            + getattr(params, "OFFROAD_TOLERANCE", 0.0)
        ),
        offroad_strength=np.asarray(params.OFFROAD_STRENGTH),
        offroad_heading_realign_strength=np.asarray(
            params.OFFROAD_HEADING_REALIGN_STRENGTH
        ),
        offroad_inward_steer=np.asarray(params.OFFROAD_INWARD_STEER),
        offroad_large_heading_error=np.asarray(
            params.OFFROAD_LARGE_HEADING_ERROR
        ),
        offroad_throttle_boost=np.asarray(params.OFFROAD_THROTTLE_BOOST),
        offroad_low_speed_threshold=np.asarray(
            params.OFFROAD_LOW_SPEED_THRESHOLD
        ),
        velocity_track_weight=np.asarray(
            trace.get("velocity_track_weight", params.VELOCITY_TRACK_WEIGHT)
        ),
        heading_track_weight=np.asarray(
            trace.get("heading_track_weight", params.HEADING_TRACK_WEIGHT)
        ),
        track_length=np.asarray(track.total_length),
        vehicle_model=np.asarray(trace["vehicle_model"]),
        simulation_dt=np.asarray(params.SIM_DT),
        control_dt=np.asarray(params.CONTROL_DT),
        mpc_dt=np.asarray(params.MPC_DT),
        mpc_enforce_track_boundary=np.asarray(params.MPC_ENFORCE_TRACK_BOUNDARY),
        mpc_enforce_collision_avoidance=np.asarray(
            params.MPC_ENFORCE_COLLISION_AVOIDANCE
        ),
        mpc_track_slack_weight=np.asarray(params.MPC_TRACK_SLACK_WEIGHT),
        mpc_collision_slack_weight=np.asarray(params.MPC_COLLISION_SLACK_WEIGHT),
        mpc_enforce_friction_speed_limit=np.asarray(
            params.MPC_ENFORCE_FRICTION_SPEED_LIMIT
        ),
        mpc_friction_coefficient=np.asarray(params.MPC_FRICTION_COEFFICIENT),
        llm_update_interval_seconds=np.asarray(
            trace.get("llm_update_interval_seconds", params.LLM_UPDATE_INTERVAL_SECONDS)
        ),
        early_collision_window_seconds=np.asarray(params.EARLY_COLLISION_WINDOW_SECONDS),
        early_collision_consecutive_seconds=np.asarray(
            params.EARLY_COLLISION_CONSECUTIVE_SECONDS
        ),
        early_stop=np.asarray(trace.get("early_stop", False)),
        early_stop_reason=np.asarray(trace.get("early_stop_reason", "")),
        completed_steps=np.asarray(trace.get("completed_steps", len(trace["time"]))),
        planned_steps=np.asarray(trace.get("planned_steps", len(trace["time"]))),
    )
    print(f"Saved simulation trace to {params.TRACE_OUTPUT}")


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
    v0=params.v0 if hasattr(params, "v0") else 0.0,
    velocity_track_weight=None,
    horizon=None,
    ego_position=1,
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
                static_policy_params=static_policy_params,
                alpharacer_params=alpharacer_params,
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

        start_s = -start_ranks[car_idx] * params.CAR_START_GAP
        start_x, start_y = track.frenet_to_world(start_s, 0.0)
        _, _, start_psi = track.interpolate(start_s)
        states.append(CarState(x=start_x, y=start_y, vx=v0, vy=0.0, psi=start_psi, omega=0.0))

    control_steps = duration_to_steps(params.CONTROL_DT, params.SIM_DT)
    if not np.isclose(control_steps * params.SIM_DT, params.CONTROL_DT):
        raise ValueError("CONTROL_DT must be an integer multiple of SIM_DT")
    total_steps = int(params.TOTAL_TIME / params.SIM_DT)
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
    offroad_half_width = track_half_width + getattr(params, "OFFROAD_TOLERANCE", 0.0)
    progress_s = [-start_ranks[car_idx] * params.CAR_START_GAP for car_idx in range(num_cars)]

    traj_x, traj_y, velocities, lateral_errors = [], [], [], []
    all_traj_x = [[] for _ in range(num_cars)]
    all_traj_y = [[] for _ in range(num_cars)]
    longitudinal_positions = []
    mpc_losses, mpc_track_losses, mpc_action_losses = [], [], []
    mpc_velocity_losses, mpc_heading_losses = [], []
    throttle_history, steering_history = [], []
    frenet_ref_lines = []
    trace = {
        "time": [],
        "x": [],
        "y": [],
        "psi": [],
        "vx": [],
        "vy": [],
        "omega": [],
        "car_x": [],
        "car_y": [],
        "car_psi": [],
        "car_vx": [],
        "car_vy": [],
        "car_omega": [],
        "car_px": [],
        "car_py": [],
        "car_throttle": [],
        "car_steer": [],
        "car_mpc_loss": [],
        "car_mpc_track_loss": [],
        "car_mpc_action_loss": [],
        "car_mpc_velocity_loss": [],
        "car_mpc_heading_loss": [],
        "car_mpc_violation_loss": [],
        "car_mpc_max_track_slack": [],
        "car_mpc_max_collision_slack": [],
        "car_mpc_solver_success": [],
        "car_mpc_solver_status": [],
        "car_mpc_recovery_active": [],
        "car_mpc_recovery_reason": [],
        # TEMPORARY DIAGNOSTIC: delete after competitive-offset analysis.
        "car_ref_raceline_e": [],
        "car_ref_p_ot_front": [],
        "car_ref_p_ot_rear": [],
        "car_ref_p_bl": [],
        "car_ref_p_ref_e": [],
        "collision": [],
        "collision_overlap": [],
        "px": [],
        "py": [],
        "ref_px": [],
        "ref_py": [],
        "ref_speed": [],
        "ref_x": [],
        "ref_y": [],
        "ref_raceline_s": [],
        "raceline_horizon_x": [],
        "raceline_horizon_y": [],
        "ref_horizon_x": [],
        "ref_horizon_y": [],
        "offroad": [],
        "throttle": [],
        "steer": [],
        "mpc_loss": [],
        "mpc_track_loss": [],
        "mpc_action_loss": [],
        "mpc_velocity_loss": [],
        "mpc_heading_loss": [],
        "mpc_violation_loss": [],
        "mpc_max_track_slack": [],
        "mpc_max_collision_slack": [],
        "mpc_solver_success": [],
        "mpc_solver_status": [],
        "mpc_recovery_active": [],
        "mpc_recovery_reason": [],
        "control_update": [],
        "policy_params": [],
        "policy_mode": [],
        "policy_reason": [],
        "policy_basis": [],
        "policy_assessment": [],
        "policy_required_assessment": [],
        "policy_assessment_consistent": [],
        "car_policy_params": [],
        "car_policy_mode": [],
        "car_policy_reason": [],
        "car_policy_basis": [],
        "car_policy_assessment": [],
        "car_policy_required_assessment": [],
        "car_policy_assessment_consistent": [],
        "car_policy_requested_mode": [],
        "car_safeguard_override": [],
        "car_offroad": [],
        "car_offroad_recovery_active": [],
        "car_offroad_recovery_heading_error": [],
        "policy_names": resolved_policy_names,
        "llm_model_names": [
            (
                params.LLM_MODEL
                if resolved_policy_names[idx] == "llm_alpharacer" and resolved_llm_model_names[idx] is None
                else str(resolved_llm_model_names[idx])
            )
            if resolved_policy_names[idx] == "llm_alpharacer"
            else ""
            for idx in range(num_cars)
        ],
        "llm_temperatures": [
            (
                params.LLM_TEMPERATURE
                if resolved_llm_temperatures[idx] is None
                else float(resolved_llm_temperatures[idx])
            )
            if resolved_policy_names[idx] == "llm_alpharacer"
            else np.nan
            for idx in range(num_cars)
        ],
        "llm_safeguard_enabled": bool(getattr(policies[ego_idx], "safeguard_enabled", False)),
        "car_llm_safeguard_enabled": [
            bool(getattr(policy, "safeguard_enabled", False)) for policy in policies
        ],
        "llm_update_interval_seconds": (
            params.LLM_UPDATE_INTERVAL_SECONDS
            if llm_update_interval_seconds is None
            else float(llm_update_interval_seconds)
        ),
        "ego_index": ego_idx,
        "ego_position": ego_position,
        "start_ranks": start_ranks,
        "vehicle_model": vehicle_model_type,
        "velocity_track_weight": mpc_velocity_track_weight,
        "heading_track_weight": params.HEADING_TRACK_WEIGHT,
    }
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

    print(
        "Simulation setup: "
        f"track={params.TRACK_CSV}, scale={params.TRACK_SCALE:g}, "
        f"half_width={params.TRACK_HALF_WIDTH:g}, "
        f"policies={resolved_policy_names}, ego_start_position={ego_position}/{num_cars}, "
        f"vehicle_model={vehicle_model_type}, "
        f"timing=sim:{params.SIM_DT:g}s/control:{params.CONTROL_DT:g}s/"
        f"mpc:{params.MPC_DT:g}s"
    )
    if any(name == "llm_alpharacer" for name in resolved_policy_names):
        print(f"LLM models by car: {trace['llm_model_names']}")
        print(f"LLM temperatures by car: {trace['llm_temperatures']}")
        print(f"LLM safeguards by car: {trace['car_llm_safeguard_enabled']}")
    progress_iter = range(total_steps)
    progress_bar = None
    if tqdm is not None:
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
            active = abs(e) > offroad_half_width
            heading_error = track_heading_error(track, state.psi, car_s[car_idx])
            offroad_recovery_active.append(active)
            offroad_recovery_heading_error.append(heading_error if active else 0.0)
            if not active:
                continue

            overshoot = abs(e) - offroad_half_width
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
                if getattr(vehicle_params_for_car, "ros_kinematic", False):
                    neutral_throttle = (
                        vehicle_params_for_car.fr / vehicle_params_for_car.Ta
                    )
                else:
                    throttle_gain = (
                        vehicle_params_for_car.Ta
                        - vehicle_params_for_car.Tb * max(state.vx, 0.0)
                    )
                    neutral_throttle = (  # maintain velocity
                        vehicle_params_for_car.fr
                        * GRAVITY
                        * vehicle_params_for_car.MASS
                        / max(throttle_gain, 1.0e-6)
                    )
                actions[car_idx].throttle = max(
                    actions[car_idx].throttle,
                    neutral_throttle + params.OFFROAD_THROTTLE_BOOST,
                )

            actions[car_idx].steer = float(np.clip(actions[car_idx].steer, -1.0, 1.0))
            actions[car_idx].throttle = float(
                np.clip(actions[car_idx].throttle, -1.0, 1.0)
            )

        ego_policy = policies[ego_idx]
        ego_mpc = mpcs[ego_idx]
        ego_state = states[ego_idx]
        ref_traj = ego_policy.last_ref_traj
        ref_fx, ref_fy = np.nan, np.nan
        ref_horizon_x = np.full(ego_mpc.horizon + 1, np.nan)
        ref_horizon_y = np.full(ego_mpc.horizon + 1, np.nan)
        raceline_horizon_x = np.full(ego_mpc.horizon + 1, np.nan)
        raceline_horizon_y = np.full(ego_mpc.horizon + 1, np.nan)
        if ref_traj is not None:
            if ego_policy.last_ref_world_horizon is not None:
                ref_horizon_x = ego_policy.last_ref_world_horizon[0].copy()
                ref_horizon_y = ego_policy.last_ref_world_horizon[1].copy()
                ref_fx, ref_fy = ref_horizon_x[0], ref_horizon_y[0]
            else:
                ref_fx, ref_fy = track.frenet_to_world(ref_traj[0, 0], ref_traj[1, 0])
                for k in range(ego_mpc.horizon + 1):
                    ref_horizon_x[k], ref_horizon_y[k] = track.frenet_to_world(ref_traj[0, k], ref_traj[1, k])

            trace["ref_px"].append(ref_traj[0, 0])
            trace["ref_py"].append(ref_traj[1, 0])
            trace["ref_speed"].append(ref_traj[2, 0])
            if ego_policy.last_raceline_world_horizon is not None:
                raceline_horizon_x = ego_policy.last_raceline_world_horizon[0].copy()
                raceline_horizon_y = ego_policy.last_raceline_world_horizon[1].copy()
        else:
            trace["ref_px"].append(np.nan)
            trace["ref_py"].append(np.nan)
            trace["ref_speed"].append(np.nan)

        trace["time"].append(step * params.SIM_DT)
        trace["x"].append(ego_state.x)
        trace["y"].append(ego_state.y)
        trace["psi"].append(ego_state.psi)
        trace["vx"].append(ego_state.vx)
        trace["vy"].append(ego_state.vy)
        trace["omega"].append(ego_state.omega)
        trace["car_x"].append([state.x for state in states])
        trace["car_y"].append([state.y for state in states])
        trace["car_psi"].append([state.psi for state in states])
        trace["car_vx"].append([state.vx for state in states])
        trace["car_vy"].append([state.vy for state in states])
        trace["car_omega"].append([state.omega for state in states])
        trace["car_px"].append(car_s)
        trace["car_py"].append(car_e)
        trace["car_throttle"].append([action.throttle for action in actions])
        trace["car_steer"].append([action.steer for action in actions])
        trace["car_mpc_loss"].append([mpc.last_loss for mpc in mpcs])
        trace["car_mpc_track_loss"].append([mpc.last_track_loss for mpc in mpcs])
        trace["car_mpc_action_loss"].append([mpc.last_action_loss for mpc in mpcs])
        trace["car_mpc_velocity_loss"].append([mpc.last_velocity_loss for mpc in mpcs])
        trace["car_mpc_heading_loss"].append([mpc.last_heading_loss for mpc in mpcs])
        trace["car_mpc_violation_loss"].append(
            [mpc.last_violation_loss for mpc in mpcs]
        )
        trace["car_mpc_max_track_slack"].append(
            [
                float(np.max(mpc.last_track_slack_plan))
                if mpc.last_track_slack_plan is not None
                and np.size(mpc.last_track_slack_plan)
                else np.nan
                for mpc in mpcs
            ]
        )
        trace["car_mpc_max_collision_slack"].append(
            [
                float(np.max(mpc.last_collision_slack_plan))
                if mpc.last_collision_slack_plan is not None
                and np.size(mpc.last_collision_slack_plan)
                else np.nan
                for mpc in mpcs
            ]
        )
        trace["car_mpc_solver_success"].append([mpc.last_solver_success for mpc in mpcs])
        trace["car_mpc_solver_status"].append([mpc.last_solver_status for mpc in mpcs])
        trace["car_mpc_recovery_active"].append([mpc.last_recovery_active for mpc in mpcs])
        trace["car_mpc_recovery_reason"].append([mpc.last_recovery_reason for mpc in mpcs])
        # TEMPORARY DIAGNOSTIC: one row per car, one column per MPC step.
        for trace_key, component_name in (
            ("car_ref_raceline_e", "raceline_e"),
            ("car_ref_p_ot_front", "p_ot_front"),
            ("car_ref_p_ot_rear", "p_ot_rear"),
            ("car_ref_p_bl", "p_bl"),
            ("car_ref_p_ref_e", "p_ref_e"),
        ):
            trace[trace_key].append(
                [ref_component_snapshot(policy, component_name) for policy in policies]
            )
        trace["px"].append(car_s[ego_idx])
        trace["py"].append(car_e[ego_idx])
        trace["ref_x"].append(ref_fx)
        trace["ref_y"].append(ref_fy)
        trace["ref_raceline_s"].append(getattr(ego_policy, "last_raceline_s", np.nan))
        trace["raceline_horizon_x"].append(raceline_horizon_x)
        trace["raceline_horizon_y"].append(raceline_horizon_y)
        trace["ref_horizon_x"].append(ref_horizon_x)
        trace["ref_horizon_y"].append(ref_horizon_y)
        trace["offroad"].append(abs(car_e[ego_idx]) > offroad_half_width)
        trace["collision"].append(False)
        trace["collision_overlap"].append(0.0)
        trace["throttle"].append(actions[ego_idx].throttle)
        trace["steer"].append(actions[ego_idx].steer)
        trace["mpc_loss"].append(ego_mpc.last_loss)
        trace["mpc_track_loss"].append(ego_mpc.last_track_loss)
        trace["mpc_action_loss"].append(ego_mpc.last_action_loss)
        trace["mpc_velocity_loss"].append(ego_mpc.last_velocity_loss)
        trace["mpc_heading_loss"].append(ego_mpc.last_heading_loss)
        trace["mpc_violation_loss"].append(ego_mpc.last_violation_loss)
        trace["mpc_max_track_slack"].append(
            float(np.max(ego_mpc.last_track_slack_plan))
            if ego_mpc.last_track_slack_plan is not None
            and np.size(ego_mpc.last_track_slack_plan)
            else np.nan
        )
        trace["mpc_max_collision_slack"].append(
            float(np.max(ego_mpc.last_collision_slack_plan))
            if ego_mpc.last_collision_slack_plan is not None
            and np.size(ego_mpc.last_collision_slack_plan)
            else np.nan
        )
        trace["mpc_solver_success"].append(ego_mpc.last_solver_success)
        trace["mpc_solver_status"].append(ego_mpc.last_solver_status)
        trace["mpc_recovery_active"].append(ego_mpc.last_recovery_active)
        trace["mpc_recovery_reason"].append(ego_mpc.last_recovery_reason)
        trace["control_update"].append(control_update)
        trace["policy_params"].append(getattr(ego_policy, "params", np.full(5, np.nan)).copy())
        trace["policy_mode"].append(getattr(ego_policy, "current_mode", "fixed"))
        trace["policy_reason"].append(getattr(ego_policy, "last_llm_reason", ""))
        trace["policy_basis"].append(getattr(ego_policy, "last_llm_basis", ""))
        trace["policy_assessment"].append(
            json.dumps(getattr(ego_policy, "last_llm_assessment", {}), sort_keys=True)
        )
        trace["policy_required_assessment"].append(
            json.dumps(getattr(ego_policy, "last_required_assessment", {}), sort_keys=True)
        )
        trace["policy_assessment_consistent"].append(
            getattr(ego_policy, "last_assessment_consistent", True)
        )
        trace["car_policy_params"].append(
            [getattr(policy, "params", np.full(5, np.nan)).copy() for policy in policies]
        )
        trace["car_policy_mode"].append([getattr(policy, "current_mode", "fixed") for policy in policies])
        trace["car_policy_reason"].append([getattr(policy, "last_llm_reason", "") for policy in policies])
        trace["car_policy_basis"].append([getattr(policy, "last_llm_basis", "") for policy in policies])
        trace["car_policy_assessment"].append(
            [json.dumps(getattr(policy, "last_llm_assessment", {}), sort_keys=True) for policy in policies]
        )
        trace["car_policy_required_assessment"].append(
            [json.dumps(getattr(policy, "last_required_assessment", {}), sort_keys=True) for policy in policies]
        )
        trace["car_policy_assessment_consistent"].append(
            [getattr(policy, "last_assessment_consistent", True) for policy in policies]
        )
        trace["car_policy_requested_mode"].append(
            [getattr(policy, "last_llm_requested_mode", "") for policy in policies]
        )
        trace["car_safeguard_override"].append(
            [getattr(policy, "last_safeguard_override", "") for policy in policies]
        )
        trace["car_offroad"].append([abs(error) > offroad_half_width for error in car_e])
        trace["car_offroad_recovery_active"].append(offroad_recovery_active)
        trace["car_offroad_recovery_heading_error"].append(
            offroad_recovery_heading_error
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

        if (step + 1) % progress_report_steps == 0:
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

    # Save the latest five actual LLM calls per car. Padding keeps these as
    # regular string arrays that np.load can read without allow_pickle=True.
    raw_histories = [getattr(policy, "llm_raw_io_history", [])[-5:] for policy in policies]
    trace["llm_raw_call_indices"] = [
        [-1] * (5 - len(history)) + [item["call"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_prompts"] = [
        [""] * (5 - len(history)) + [item["prompt"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_responses"] = [
        [""] * (5 - len(history)) + [item["response"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_errors"] = [
        [""] * (5 - len(history)) + [item["error"] for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_done_reasons"] = [
        [""] * (5 - len(history)) + [item.get("done_reason", "") for item in history]
        for history in raw_histories
    ]
    trace["llm_raw_eval_counts"] = [
        [-1] * (5 - len(history)) + [item.get("eval_count", -1) for item in history]
        for history in raw_histories
    ]

    if save_trace:
        save_simulation_trace(trace, track, params)
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default racer simulation scenario.")
    parser.add_argument(
        "--llm-safeguard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable validation/correction of LLM policy modes.",
    )
    cli_args = parser.parse_args()

    # Options: "centerline", "alpharacer" (trained potential), "static_alpharacer",
    # "ibr", or "llm_alpharacer". "weighted_alpharacer" aliases "alpharacer".
    # You can also pass per-car policies, e.g. policy_names=["alpha", "IBR", "IBR"].
    # Tracks: "berlin" (scale 2), "modena" (scale 2), or "rounded_rectangle" (scale 1).

    run_simulation(
        v0 = 0,
        # track_name="modena",  # Change this directly to "modena" or "rounded_rectangle".
        policy_names=["LLM", "alpharacer", "alpharacer"],  # ego, oppo1, oppo2
        llm_model_names=["qwen2.5:7b", None, None],
        vehicle_model_type="kinematic", ego_position=1,
        llm_temperature = 0.0,
        llm_safeguard_enabled=False,
    )
