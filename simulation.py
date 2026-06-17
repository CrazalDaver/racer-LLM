import numpy as np

from config import make_vehicle_params, params
from mpc_controller import MPCController
from plotting import plot_simulation_results
from policies import (
    AlphaRacerPolicy,
    CenterLinePolicy,
    IBRBaselinePolicy,
    LLMAlphaRacerPolicy,
    SelfPlayRLPolicy,
    WeightedAlphaRacerPolicy,
    plan_ibr_joint_actions,
)
from state import CarState
from track import Track, load_centerline_csv, load_raceline_csv
from vehicle_model import DynamicBicycleModel, KinematicBicycleModel

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def progress_delta(s_a, s_b, total_length):
    return np.arctan2(
        np.sin((s_a - s_b) * 2.0 * np.pi / total_length),
        np.cos((s_a - s_b) * 2.0 * np.pi / total_length),
    ) * total_length / (2.0 * np.pi)


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
        "weighted": "weighted_alpharacer",
        "weight": "weighted_alpharacer",
        "weights": "weighted_alpharacer",
        "weighted_alpha": "weighted_alpharacer",
        "weighted_alpha_racer": "weighted_alpharacer",
        "weighted_alpharacer": "weighted_alpharacer",
        "ours": "weighted_alpharacer",
        "ours_high_p": "weighted_alpharacer",
        "llm": "llm_alpharacer",
        "llm_alpha": "llm_alpharacer",
        "llm_alpha_racer": "llm_alpharacer",
        "llm_alpharacer": "llm_alpharacer",
        "llmalpha": "llm_alpharacer",
        "llmalpharacer": "llm_alpharacer",
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
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_update_interval_steps=None,
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
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
        )
    if policy_type == "alpharacer":
        return AlphaRacerPolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.ALPHARACER_PARAMS if alpharacer_params is None else alpharacer_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
        )
    if policy_type == "llm_alpharacer":
        return LLMAlphaRacerPolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.LLM_POLICY_PARAMS if llm_policy_params is None else llm_policy_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            model_name=params.LLM_MODEL if llm_model_name is None else llm_model_name,
            base_url=params.LLM_BASE_URL,
            timeout=params.LLM_TIMEOUT,
            update_interval_steps=(
                params.LLM_UPDATE_INTERVAL_STEPS
                if llm_update_interval_steps is None
                else llm_update_interval_steps
            ),
            history_length=params.LLM_HISTORY_LENGTH,
            output_mode=params.LLM_OUTPUT_MODE if llm_output_mode is None else llm_output_mode,
        )
    if policy_type == "weighted_alpharacer":
        return WeightedAlphaRacerPolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.ALPHARACER_PARAMS if alpharacer_params is None else alpharacer_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            weight_path=params.ALPHARACER_WEIGHT_PATH,
            q_value=params.ALPHARACER_WEIGHT_Q,
            search_points=params.ALPHARACER_WEIGHT_SEARCH_POINTS,
            search_iters=params.ALPHARACER_WEIGHT_SEARCH_ITERS,
        )
    if policy_type == "self_play_rl":
        return SelfPlayRLPolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
            params=np.asarray(
                params.ALPHARACER_PARAMS if alpharacer_params is None else alpharacer_params,
                dtype=float,
            ).copy(),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
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
        return IBRBaselinePolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            car_idx=car_idx,
            policy_name=policy_type,
            q=params.IBR_PARAMS[0],
            zeta=params.IBR_PARAMS[1],
            s1=params.IBR_PARAMS[2],
            s2=params.IBR_PARAMS[3],
            s3=params.IBR_PARAMS[4],
            iterations=params.IBR_ITERATIONS,
            gamma=params.IBR_GAMMA,
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
        car_vx=np.asarray(trace["car_vx"]),
        car_px=np.asarray(trace["car_px"]),
        car_py=np.asarray(trace["car_py"]),
        collision=np.asarray(trace["collision"]),
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
        mpc_solver_success=np.asarray(trace["mpc_solver_success"]),
        mpc_solver_status=np.asarray(trace["mpc_solver_status"]),
        policy_params=np.asarray(trace["policy_params"]),
        policy_mode=np.asarray(trace["policy_mode"]),
        policy_names=np.asarray(trace.get("policy_names", [])),
        llm_model_names=np.asarray(trace.get("llm_model_names", []), dtype=str),
        car_policy_params=np.asarray(trace.get("car_policy_params", []), dtype=float),
        car_policy_mode=np.asarray(trace.get("car_policy_mode", []), dtype=str),
        car_offroad=np.asarray(trace.get("car_offroad", []), dtype=bool),
        ego_index=np.asarray(trace.get("ego_index", 0)),
        ego_position=np.asarray(trace.get("ego_position", 1)),
        start_ranks=np.asarray(trace.get("start_ranks", [])),
        track_csv=np.asarray(params.TRACK_CSV),
        raceline_csv=np.asarray(params.RACELINE_CSV),
        track_scale=np.asarray(getattr(params, "TRACK_SCALE", 1.0)),
        track_width=np.asarray(params.TRACK_WIDTH),
        track_half_width=np.asarray(getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)),
        mpc_boundary_slack=np.asarray(getattr(params, "MPC_BOUNDARY_SLACK", 0.0)),
        mpc_constraint_half_width=np.asarray(
            getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)
            + getattr(params, "MPC_BOUNDARY_SLACK", 0.0)
        ),
        offroad_tolerance=np.asarray(getattr(params, "OFFROAD_TOLERANCE", 0.0)),
        track_length=np.asarray(track.total_length),
        vehicle_model=np.asarray(trace["vehicle_model"]),
        early_stop=np.asarray(trace.get("early_stop", False)),
        early_stop_reason=np.asarray(trace.get("early_stop_reason", "")),
        completed_steps=np.asarray(trace.get("completed_steps", len(trace["time"]))),
        planned_steps=np.asarray(trace.get("planned_steps", len(trace["time"]))),
    )
    print(f"Saved simulation trace to {params.TRACE_OUTPUT}")


def run_simulation(
    policy_type="centerline",
    save_trace=True,
    show_plot=True,
    num_cars=None,
    policy_names=None,
    vehicle_model_type=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_update_interval_steps=None,
    self_play_rl_model_path=None,
    self_play_rl_backend=None,
    self_play_rl_strict_load=None,
    self_play_rl_action_order=None,
    v0=0.0,
    velocity_track_weight=None,
    horizon=None,
    ego_position=1,
):
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
            max_steer_rate=params.MAX_STEER_RATE,
            db_params=vehicle_params,
            track_width=params.TRACK_WIDTH,
            track_boundary_slack=params.MPC_BOUNDARY_SLACK,
            max_state_speed=params.MAX_STATE_SPEED,
            velocity_track_weight=mpc_velocity_track_weight,
            heading_track_weight=params.HEADING_TRACK_WEIGHT,
        )
        mpcs.append(mpc)
        policies.append(
            create_policy(
                resolved_policy_names[car_idx],
                track,
                mpc,
                raceline=raceline,
                car_idx=car_idx,
                alpharacer_params=alpharacer_params,
                llm_policy_params=llm_policy_params,
                llm_output_mode=llm_output_mode,
                llm_model_name=resolved_llm_model_names[car_idx],
                llm_update_interval_steps=llm_update_interval_steps,
                self_play_rl_model_path=self_play_rl_model_path,
                self_play_rl_backend=self_play_rl_backend,
                self_play_rl_strict_load=self_play_rl_strict_load,
                self_play_rl_action_order=self_play_rl_action_order,
            )
        )
        vehicle_models.append(
            create_vehicle_model(
                vehicle_model_type,
                dt=params.DT,
                db_params=vehicle_params,
                max_speed=params.MAX_STATE_SPEED,
            )
        )

        start_s = -start_ranks[car_idx] * params.CAR_START_GAP
        start_x, start_y = track.frenet_to_world(start_s, 0.0)
        _, _, start_psi = track.interpolate(start_s)
        states.append(CarState(x=start_x, y=start_y, vx=v0, vy=0.0, psi=start_psi, omega=0.0))

    total_steps = int(params.TOTAL_TIME / params.DT)
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
        "car_vx": [],
        "car_px": [],
        "car_py": [],
        "collision": [],
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
        "mpc_solver_success": [],
        "mpc_solver_status": [],
        "policy_params": [],
        "policy_mode": [],
        "car_policy_params": [],
        "car_policy_mode": [],
        "car_offroad": [],
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
        "ego_index": ego_idx,
        "ego_position": ego_position,
        "start_ranks": start_ranks,
        "vehicle_model": vehicle_model_type,
    }

    if policy_names is not None:
        print(
            f"Simulation with per-car policies {resolved_policy_names} "
            f"and {vehicle_model_type} vehicle model started for {num_cars} cars..."
        )
        if any(name == "llm_alpharacer" for name in resolved_policy_names):
            print(f"LLM models by car: {trace['llm_model_names']}")
    elif resolved_policy_names[0] == "llm_alpharacer" and any(name == "ibr" for name in resolved_policy_names[1:]):
        print(
            f"Simulation with ego LLM policy, IBR baseline opponents, "
            f"and {vehicle_model_type} vehicle model started for {num_cars} cars..."
        )
    else:
        print(
            f"Simulation with {resolved_policy_names[0]} policy and "
            f"{vehicle_model_type} vehicle model started for {num_cars} cars..."
        )
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

        actions = []
        plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=step * params.DT)
        for car_idx, state in enumerate(states):
            opponents = [
                {
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
            actions.append(
                policies[car_idx].get_action(
                    state,
                    car_s[car_idx],
                    opponents=opponents,
                    sim_time=step * params.DT,
                )
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

        trace["time"].append(step * params.DT)
        trace["x"].append(ego_state.x)
        trace["y"].append(ego_state.y)
        trace["psi"].append(ego_state.psi)
        trace["vx"].append(ego_state.vx)
        trace["vy"].append(ego_state.vy)
        trace["omega"].append(ego_state.omega)
        trace["car_x"].append([state.x for state in states])
        trace["car_y"].append([state.y for state in states])
        trace["car_vx"].append([state.vx for state in states])
        trace["car_px"].append(car_s)
        trace["car_py"].append(car_e)
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
        trace["throttle"].append(actions[ego_idx].throttle)
        trace["steer"].append(actions[ego_idx].steer)
        trace["mpc_loss"].append(ego_mpc.last_loss)
        trace["mpc_track_loss"].append(ego_mpc.last_track_loss)
        trace["mpc_action_loss"].append(ego_mpc.last_action_loss)
        trace["mpc_velocity_loss"].append(ego_mpc.last_velocity_loss)
        trace["mpc_heading_loss"].append(ego_mpc.last_heading_loss)
        trace["mpc_solver_success"].append(ego_mpc.last_solver_success)
        trace["mpc_solver_status"].append(ego_mpc.last_solver_status)
        trace["policy_params"].append(getattr(ego_policy, "params", np.full(5, np.nan)).copy())
        trace["policy_mode"].append(getattr(ego_policy, "current_mode", "fixed"))
        trace["car_policy_params"].append(
            [getattr(policy, "params", np.full(5, np.nan)).copy() for policy in policies]
        )
        trace["car_policy_mode"].append([getattr(policy, "current_mode", "fixed") for policy in policies])
        trace["car_offroad"].append([abs(error) > offroad_half_width for error in car_e])

        if (step + 1) % 100 == 0 and ego_policy.last_ref_traj is not None:
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

        collision_scales = np.ones(num_cars)
        for i in range(num_cars):
            for j in range(i + 1, num_cars):
                distance = np.hypot(states[i].x - states[j].x, states[i].y - states[j].y)
                if distance > params.UNSAFE_DISTANCE:
                    continue

                delta_ij = progress_delta(car_s[i], car_s[j], track.total_length)
                if delta_ij > 0.0:
                    front_idx, rear_idx = i, j
                elif delta_ij < 0.0:
                    front_idx, rear_idx = j, i
                else:
                    continue

                collision_scales[front_idx] = min(collision_scales[front_idx], 0.5)
                collision_scales[rear_idx] = min(collision_scales[rear_idx], 1.0 / 3.0)

        collision_happened = bool(np.any(collision_scales < 1.0))
        trace["collision"][-1] = collision_happened
        for car_idx, scale in enumerate(collision_scales):
            if scale < 1.0:
                states[car_idx].vx *= scale

        if collision_happened and step < params.EARLY_COLLISION_WINDOW_STEPS:
            early_collision_count += 1
        elif step < params.EARLY_COLLISION_WINDOW_STEPS:
            early_collision_count = 0

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

            if abs(e) > offroad_half_width:
                overshoot = abs(e) - offroad_half_width
                factor = np.exp(-params.OFFROAD_STRENGTH * overshoot)
                state.vx *= factor
                state.vy *= factor

        lateral_errors.append(next_e[ego_idx])
        longitudinal_positions.append(progress_s[ego_idx])

        if step % 2 == 0:
            for car_idx, state in enumerate(states):
                all_traj_x[car_idx].append(state.x)
                all_traj_y[car_idx].append(state.y)
            traj_x.append(states[ego_idx].x)
            traj_y.append(states[ego_idx].y)
            velocities.append(states[ego_idx].vx)

        if (step + 1) % 200 == 0:
            message = (
                f"Step {step+1}/{total_steps}, t={(step+1)*params.DT:.1f}s, "
                f"ego={ego_position}, vx={states[ego_idx].vx:.2f} m/s, e={next_e[ego_idx]:.3f}, "
                f"throttle={actions[ego_idx].throttle:.2f}, steer={actions[ego_idx].steer:.2f}"
            )
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

        if (
            params.EARLY_COLLISION_STOP
            and early_collision_count >= params.EARLY_COLLISION_CONSECUTIVE_STEPS
        ):
            early_stop = True
            early_stop_reason = (
                f"early collision for {early_collision_count} consecutive steps "
                f"within first {params.EARLY_COLLISION_WINDOW_STEPS} steps"
            )
            message = (
                f"Early stop at step {step+1}/{total_steps}, "
                f"t={(step+1)*params.DT:.2f}s: {early_stop_reason}"
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

    final_time = last_step * params.DT if last_step >= 0 else 0.0
    for policy in policies:
        policy.flush_mpc_warnings(sim_time=final_time)

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
        )


if __name__ == "__main__":
    # Options: "centerline", "alpharacer", "ibr", or "llm_alpharacer"
    # You can also pass per-car policies, e.g. policy_names=["alpha", "IBR", "IBR"].

    # baseline simulation with AlphaRacer policy and Dynamic Bicycle Model
    # run_simulation(
    #     policy_type="alpharacer", v0=5, vehicle_model_type="kinematic",
    #     alpharacer_params=[2.0, 1.0, 0.0, 0.0, 0.0], velocity_track_weight=2.0, horizon=8,
    #     ) # most basic settings.
    
    # run_simulation(
    #     policy_type="alpharacer",
    #     v0=0,
    #     vehicle_model_type="dbm",
    #     alpharacer_params=[1.0, 1.0, 0.0, 0.0, 0.0],
    #     velocity_track_weight=2.0,
    #     horizon=8,
    # ) # with dynamic bicycle model.

    # multiple cars with different policies
    # run_simulation(
    #     policy_names=["alpharacer", "centerline", "centerline"], vehicle_model_type="kinematic",
    #     alpharacer_params=[1.0, 1.0, 1.3, 0.3, 0.3], velocity_track_weight=2.0, horizon=8,
    #     ego_position=3,
    # )

    # simulation of multiple cars
    # run_simulation(
    #     # run_simulation treats policy_names[0] as the traced ego car.
    #     # ego_position only changes that ego car's starting rank.
    #     # ego, oppo 1, oppo 2
    #     policy_names=["LLM", "IBR", "IBR"], vehicle_model_type="kinematic",
    #     llm_policy_params=[1.0, 1.0, 1.5, 0.5, 0.5], velocity_track_weight=2.0, horizon=8,
    #     v0=0.0, llm_output_mode="mode", llm_update_interval_steps=10,
    #     ego_position=3,
    # )

    # run_simulation(
    #     policy_names=["LLM", "LLM", "IBR"],  # ego, oppo1, oppo2
    #     llm_model_names=["my-racing-ctrl-mode", "llama3.2:1b", None],
    #     vehicle_model_type="kinematic",
    # )

    run_simulation(
        policy_names=["LLM", "LLM", "LLM"],  # ego, oppo1, oppo2
        llm_model_names=["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"], # ego, oppo1, oppo2
        ego_position=2,
        vehicle_model_type="kinematic",
        llm_update_interval_steps=50,
    )

    # run_simulation(
    # policy_names=["weighted", "LLM", "IBR"],
    # vehicle_model_type="kinematic", ego_position=1,
    # )

    # run_simulation(
    #     policy_names=["self_play_rl", "IBR", "IBR"],
    #     vehicle_model_type="kinematic",
    #     ego_position=1,
    #     # Set self_play_rl_model_path to an exported .npz/.json policy, or to an
    #     # SB3 PPO .zip when stable-baselines3 is installed. None uses fallback.
    #     self_play_rl_model_path=None,
    #     self_play_rl_backend="auto",
    #     self_play_rl_strict_load=False,
    #     self_play_rl_action_order="throttle_steer",
    # )
