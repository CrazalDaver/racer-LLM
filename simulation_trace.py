"""Simulation trace collection and NPZ serialization helpers."""

import json

import numpy as np

from config import params


def create_simulation_trace(
    *,
    resolved_policy_names,
    policies,
    ego_idx,
    ibr_implementation,
    resolved_llm_model_names,
    resolved_llm_temperatures,
    num_cars,
    llm_update_interval_seconds,
    ego_position,
    start_ranks,
    start_gap,
    initial_longitudinal_offsets,
    initial_lateral_offsets,
    initial_speeds,
    vehicle_model_type,
    mpc_velocity_track_weight,
    mpc_track_slack_weight,
):
    llm_policy_names = {
        "llm_alpharacer",
        "llm_structured_action_first",
        "llm_evidence_first",
        "llm_explicit_rules",
    }
    trace = SimulationTrace({
        "policy_names": resolved_policy_names,
        "car_alpharacer_weight_paths": [
            str(getattr(policy, "weight_path", "")) for policy in policies
        ],
        "car_alpharacer_weight_parameterizations": [
            str(getattr(policy, "weight_parameterization", "")) for policy in policies
        ],
        "car_ibr_implementations": [
            str(getattr(policy, "implementation", "")) for policy in policies
        ],
        "ibr_implementation": (
            params.IBR_IMPLEMENTATION
            if ibr_implementation is None
            else str(ibr_implementation)
        ),
        "llm_model_names": [
            (
                params.LLM_MODEL
                if resolved_policy_names[idx] in llm_policy_names
                and resolved_llm_model_names[idx] is None
                else str(resolved_llm_model_names[idx])
            )
            if resolved_policy_names[idx] in llm_policy_names
            else ""
            for idx in range(num_cars)
        ],
        "llm_temperatures": [
            (
                params.LLM_TEMPERATURE
                if resolved_llm_temperatures[idx] is None
                else float(resolved_llm_temperatures[idx])
            )
            if resolved_policy_names[idx] in llm_policy_names
            else np.nan
            for idx in range(num_cars)
        ],
        "llm_safeguard_enabled": bool(getattr(policies[ego_idx], "safeguard_enabled", False)),
        "car_llm_safeguard_enabled": [
            bool(getattr(policy, "safeguard_enabled", False)) for policy in policies
        ],
        "car_llm_prompt_variants": [
            str(getattr(policy, "prompt_variant", "not_applicable")) for policy in policies
        ],
        "car_llm_system_prompt_paths": [
            str(getattr(policy, "system_prompt_path", "")) for policy in policies
        ],
        "car_llm_system_prompts": [
            str(getattr(policy, "system_prompt", "")) for policy in policies
        ],
        "llm_parameter_names": list(params.ALPHARACER_PARAMETER_NAMES),
        "llm_parameter_bounds": np.asarray(params.ALPHARACER_PARAMETER_BOUNDS).copy(),
        "llm_update_interval_seconds": (
            params.LLM_UPDATE_INTERVAL_SECONDS
            if llm_update_interval_seconds is None
            else float(llm_update_interval_seconds)
        ),
        "ego_index": ego_idx,
        "ego_position": ego_position,
        "start_ranks": start_ranks,
        "start_gap": start_gap,
        "initial_longitudinal_offsets": initial_longitudinal_offsets.copy(),
        "initial_lateral_offsets": initial_lateral_offsets.copy(),
        "initial_speeds": initial_speeds.copy(),
        "vehicle_model": vehicle_model_type,
        "mpc_state_coordinates": "frenet",
        "velocity_track_weight": mpc_velocity_track_weight,
        "heading_track_weight": params.HEADING_TRACK_WEIGHT,
        "track_slack_weight": mpc_track_slack_weight,
        "parameter_ibr_rounds": params.PARAMETER_IBR_ROUNDS,
        "parameter_ibr_search_points": params.PARAMETER_IBR_SEARCH_POINTS,
        "parameter_ibr_search_passes": params.PARAMETER_IBR_SEARCH_PASSES,
        "parameter_ibr_update_interval_seconds": params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS,
    })
    return trace


class SimulationTrace(dict):
    """Trace metadata plus lazily-created time-series buffers.

    Keeping buffer creation here avoids a hundred-line list declaration in the
    simulation setup while retaining the existing dict/NPZ interface.
    """

    def __missing__(self, key):
        value = []
        self[key] = value
        return value


def record_simulation_step(
    trace,
    *,
    step,
    track,
    policies,
    mpcs,
    states,
    actions,
    car_s,
    car_e,
    ego_idx,
    track_half_width,
    offroad_recovery_threshold,
    control_update,
    offroad_recovery_active,
    offroad_recovery_heading_error,
    low_speed_boundary_recovery_active,
    deadlock_reset_event,
    deadlock_reset_blocked,
    deadlock_reset_pre_lateral_error,
    deadlock_reset_pre_speed,
    ref_component_snapshot,
):
    """Record one pre-integration simulation snapshot and return its reference point."""
    ego_policy = policies[ego_idx]
    ego_mpc = mpcs[ego_idx]
    ego_state = states[ego_idx]
    ref_traj = ego_policy.last_ref_traj
    ref_fx, ref_fy = np.nan, np.nan
    ref_horizon_x = np.full(ego_mpc.horizon + 1, np.nan)
    ref_horizon_y = np.full(ego_mpc.horizon + 1, np.nan)
    raceline_horizon_x = np.full(ego_mpc.horizon + 1, np.nan)
    raceline_horizon_y = np.full(ego_mpc.horizon + 1, np.nan)

    # MPC horizons can differ in mixed-policy races.  Pad every diagnostic to
    # the largest horizon so NPZ arrays remain numeric rather than ragged
    # object arrays.
    max_horizon = max(mpc.horizon for mpc in mpcs)

    def padded_plan(source, rows, columns):
        result = np.full((rows, columns), np.nan, dtype=float)
        if source is None:
            return result
        values = np.asarray(source, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        row_count = min(rows, values.shape[0])
        column_count = min(columns, values.shape[1])
        result[:row_count, :column_count] = values[:row_count, :column_count]
        return result
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
    trace["car_mpc_track_slack_loss"].append(
        [mpc.last_track_slack_loss for mpc in mpcs]
    )
    trace["car_mpc_collision_slack_loss"].append(
        [mpc.last_collision_slack_loss for mpc in mpcs]
    )
    trace["car_mpc_predicted_state_plan"].append([
        padded_plan(mpc.last_state_plan, mpc.n_states, max_horizon + 1)
        for mpc in mpcs
    ])
    trace["car_mpc_predicted_control_plan"].append([
        padded_plan(mpc.last_control_plan, mpc.n_controls, max_horizon)
        for mpc in mpcs
    ])
    trace["car_mpc_reference_plan"].append([
        padded_plan(
            getattr(policy, "last_ref_traj", None),
            8,
            max_horizon + 1,
        )
        for policy in policies
    ])
    trace["car_mpc_track_slack_plan"].append([
        padded_plan(mpc.last_track_slack_plan, 1, max_horizon)[0]
        for mpc in mpcs
    ])
    trace["car_mpc_collision_slack_plan"].append([
        padded_plan(
            mpc.last_collision_slack_plan,
            mpc.max_constrained_opponents,
            max_horizon,
        )
        for mpc in mpcs
    ])
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
    trace["track_boundary_violation"].append(
        abs(car_e[ego_idx]) > track_half_width
    )
    trace["offroad"].append(abs(car_e[ego_idx]) > offroad_recovery_threshold)
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
    trace["car_parameter_ibr_score"].append(
        [getattr(policy, "last_parameter_ibr_score", np.nan) for policy in policies]
    )
    trace["car_parameter_ibr_predicted_progress"].append([
        getattr(policy, "last_parameter_ibr_diagnostics", {}).get(
            "predicted_progress", np.nan
        )
        for policy in policies
    ])
    for field in (
        "progress_reward",
        "lateral_cost",
        "control_rate_cost",
        "track_slack_cost",
        "collision_slack_cost",
    ):
        trace[f"car_parameter_ibr_{field}"].append([
            getattr(policy, "last_parameter_ibr_diagnostics", {}).get(
                field, np.nan
            )
            for policy in policies
        ])
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
    trace["car_explicit_rule_mode"].append(
        [getattr(policy, "last_explicit_rule_mode", "") for policy in policies]
    )
    trace["car_explicit_rule_consistent"].append(
        [getattr(policy, "last_explicit_rule_consistent", True) for policy in policies]
    )
    trace["car_track_boundary_violation"].append(
        [abs(error) > track_half_width for error in car_e]
    )
    # Backward-compatible ``car_offroad`` means recovery-triggering offroad,
    # not merely crossing the nominal track boundary.
    trace["car_offroad"].append(
        [abs(error) > offroad_recovery_threshold for error in car_e]
    )
    trace["car_offroad_recovery_active"].append(offroad_recovery_active)
    trace["car_offroad_recovery_heading_error"].append(
        offroad_recovery_heading_error
    )
    trace["car_low_speed_boundary_recovery_active"].append(
        low_speed_boundary_recovery_active
    )
    trace["car_deadlock_reset_event"].append(deadlock_reset_event)
    trace["car_deadlock_reset_blocked"].append(deadlock_reset_blocked)
    trace["car_deadlock_reset_pre_lateral_error"].append(
        deadlock_reset_pre_lateral_error
    )
    trace["car_deadlock_reset_pre_speed"].append(deadlock_reset_pre_speed)
    diagnostic_fields = (
        "max_collision_constraint",
        "min_predicted_longitudinal_distance",
        "min_predicted_lateral_distance",
    )
    for field in diagnostic_fields:
        trace[f"car_ibr_{field}"].append(
            [
                float(getattr(policy, "last_ibr_diagnostics", {}).get(field, np.nan))
                for policy in policies
            ]
        )
    trace["car_ibr_max_collision_slack"].append(
        [
            float(np.max(diagnostics["collision_slack"]))
            if (diagnostics := getattr(policy, "last_ibr_diagnostics", {})).get(
                "collision_slack"
            ) is not None
            and np.size(diagnostics["collision_slack"])
            else np.nan
            for policy in policies
        ]
    )
    trace["car_ibr_max_track_slack"].append(
        [
            float(np.max(diagnostics["track_slack"]))
            if (diagnostics := getattr(policy, "last_ibr_diagnostics", {})).get(
                "track_slack"
            ) is not None
            and np.size(diagnostics["track_slack"])
            else np.nan
            for policy in policies
        ]
    )
    return ref_fx, ref_fy


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
        car_mpc_track_slack_loss=np.asarray(
            trace["car_mpc_track_slack_loss"], dtype=float
        ),
        car_mpc_collision_slack_loss=np.asarray(
            trace["car_mpc_collision_slack_loss"], dtype=float
        ),
        car_mpc_predicted_state_plan=np.asarray(
            trace["car_mpc_predicted_state_plan"], dtype=float
        ),
        car_mpc_predicted_control_plan=np.asarray(
            trace["car_mpc_predicted_control_plan"], dtype=float
        ),
        car_mpc_reference_plan=np.asarray(
            trace["car_mpc_reference_plan"], dtype=float
        ),
        car_mpc_track_slack_plan=np.asarray(
            trace["car_mpc_track_slack_plan"], dtype=float
        ),
        car_mpc_collision_slack_plan=np.asarray(
            trace["car_mpc_collision_slack_plan"], dtype=float
        ),
        car_mpc_max_track_slack=np.asarray(trace["car_mpc_max_track_slack"]),
        car_mpc_max_collision_slack=np.asarray(
            trace["car_mpc_max_collision_slack"]
        ),
        car_mpc_solver_success=np.asarray(trace["car_mpc_solver_success"]),
        car_mpc_solver_status=np.asarray(trace["car_mpc_solver_status"]),
        car_mpc_recovery_active=np.asarray(trace["car_mpc_recovery_active"]),
        car_mpc_recovery_reason=np.asarray(trace["car_mpc_recovery_reason"]),
        car_ibr_max_collision_constraint=np.asarray(
            trace["car_ibr_max_collision_constraint"], dtype=float
        ),
        car_ibr_min_predicted_longitudinal_distance=np.asarray(
            trace["car_ibr_min_predicted_longitudinal_distance"], dtype=float
        ),
        car_ibr_min_predicted_lateral_distance=np.asarray(
            trace["car_ibr_min_predicted_lateral_distance"], dtype=float
        ),
        car_ibr_max_collision_slack=np.asarray(
            trace["car_ibr_max_collision_slack"], dtype=float
        ),
        car_ibr_max_track_slack=np.asarray(
            trace["car_ibr_max_track_slack"], dtype=float
        ),
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
        track_boundary_violation=np.asarray(
            trace["track_boundary_violation"], dtype=bool
        ),
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
        car_alpharacer_weight_paths=np.asarray(
            trace.get("car_alpharacer_weight_paths", []), dtype=str
        ),
        car_alpharacer_weight_parameterizations=np.asarray(
            trace.get("car_alpharacer_weight_parameterizations", []), dtype=str
        ),
        car_ibr_implementations=np.asarray(
            trace.get("car_ibr_implementations", []), dtype=str
        ),
        llm_model_names=np.asarray(trace.get("llm_model_names", []), dtype=str),
        llm_temperatures=np.asarray(trace.get("llm_temperatures", []), dtype=float),
        llm_safeguard_enabled=np.asarray(trace.get("llm_safeguard_enabled", False), dtype=bool),
        car_llm_safeguard_enabled=np.asarray(
            trace.get("car_llm_safeguard_enabled", []), dtype=bool
        ),
        car_llm_prompt_variants=np.asarray(
            trace.get("car_llm_prompt_variants", []), dtype=str
        ),
        car_llm_system_prompt_paths=np.asarray(
            trace.get("car_llm_system_prompt_paths", []), dtype=str
        ),
        car_llm_system_prompts=np.asarray(
            trace.get("car_llm_system_prompts", []), dtype=str
        ),
        llm_parameter_names=np.asarray(
            trace.get("llm_parameter_names", params.ALPHARACER_PARAMETER_NAMES),
            dtype=str,
        ),
        llm_parameter_bounds=np.asarray(
            trace.get("llm_parameter_bounds", params.ALPHARACER_PARAMETER_BOUNDS),
            dtype=float,
        ),
        llm_raw_call_indices=np.asarray(trace.get("llm_raw_call_indices", []), dtype=int),
        llm_raw_prompts=np.asarray(trace.get("llm_raw_prompts", []), dtype=str),
        llm_raw_responses=np.asarray(trace.get("llm_raw_responses", []), dtype=str),
        llm_raw_errors=np.asarray(trace.get("llm_raw_errors", []), dtype=str),
        llm_raw_done_reasons=np.asarray(trace.get("llm_raw_done_reasons", []), dtype=str),
        llm_raw_eval_counts=np.asarray(trace.get("llm_raw_eval_counts", []), dtype=int),
        llm_raw_prompt_eval_counts=np.asarray(
            trace.get("llm_raw_prompt_eval_counts", []), dtype=int
        ),
        llm_raw_latency_seconds=np.asarray(
            trace.get("llm_raw_latency_seconds", []), dtype=float
        ),
        car_policy_decision_counts=np.asarray(
            trace.get("car_policy_decision_counts", []), dtype=int
        ),
        car_policy_params=np.asarray(trace.get("car_policy_params", []), dtype=float),
        car_parameter_ibr_score=np.asarray(
            trace.get("car_parameter_ibr_score", []), dtype=float
        ),
        car_parameter_ibr_predicted_progress=np.asarray(
            trace.get("car_parameter_ibr_predicted_progress", []), dtype=float
        ),
        car_parameter_ibr_progress_reward=np.asarray(
            trace.get("car_parameter_ibr_progress_reward", []), dtype=float
        ),
        car_parameter_ibr_lateral_cost=np.asarray(
            trace.get("car_parameter_ibr_lateral_cost", []), dtype=float
        ),
        car_parameter_ibr_control_rate_cost=np.asarray(
            trace.get("car_parameter_ibr_control_rate_cost", []), dtype=float
        ),
        car_parameter_ibr_track_slack_cost=np.asarray(
            trace.get("car_parameter_ibr_track_slack_cost", []), dtype=float
        ),
        car_parameter_ibr_collision_slack_cost=np.asarray(
            trace.get("car_parameter_ibr_collision_slack_cost", []), dtype=float
        ),
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
        car_explicit_rule_mode=np.asarray(
            trace.get("car_explicit_rule_mode", []), dtype=str
        ),
        car_explicit_rule_consistent=np.asarray(
            trace.get("car_explicit_rule_consistent", []), dtype=bool
        ),
        car_offroad=np.asarray(trace.get("car_offroad", []), dtype=bool),
        car_track_boundary_violation=np.asarray(
            trace.get("car_track_boundary_violation", []), dtype=bool
        ),
        car_offroad_recovery_active=np.asarray(
            trace.get("car_offroad_recovery_active", []), dtype=bool
        ),
        car_offroad_recovery_heading_error=np.asarray(
            trace.get("car_offroad_recovery_heading_error", []), dtype=float
        ),
        car_low_speed_boundary_recovery_active=np.asarray(
            trace.get("car_low_speed_boundary_recovery_active", []), dtype=bool
        ),
        car_deadlock_reset_event=np.asarray(
            trace.get("car_deadlock_reset_event", []), dtype=bool
        ),
        car_deadlock_reset_blocked=np.asarray(
            trace.get("car_deadlock_reset_blocked", []), dtype=bool
        ),
        car_deadlock_reset_pre_lateral_error=np.asarray(
            trace.get("car_deadlock_reset_pre_lateral_error", []), dtype=float
        ),
        car_deadlock_reset_pre_speed=np.asarray(
            trace.get("car_deadlock_reset_pre_speed", []), dtype=float
        ),
        ego_index=np.asarray(trace.get("ego_index", 0)),
        ego_position=np.asarray(trace.get("ego_position", 1)),
        start_ranks=np.asarray(trace.get("start_ranks", [])),
        start_gap=np.asarray(trace.get("start_gap", params.CAR_START_GAP)),
        initial_longitudinal_offsets=np.asarray(
            trace.get("initial_longitudinal_offsets", []), dtype=float
        ),
        initial_lateral_offsets=np.asarray(
            trace.get("initial_lateral_offsets", []), dtype=float
        ),
        initial_speeds=np.asarray(trace.get("initial_speeds", []), dtype=float),
        track_csv=np.asarray(params.TRACK_CSV),
        raceline_csv=np.asarray(trace.get("raceline_csv", params.RACELINE_CSV)),
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
        offroad_recovery_threshold=np.asarray(params.OFFROAD_RECOVERY_THRESHOLD),
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
        low_speed_boundary_recovery_enabled=np.asarray(
            params.LOW_SPEED_BOUNDARY_RECOVERY_ENABLED, dtype=bool
        ),
        low_speed_boundary_trigger_fraction=np.asarray(
            params.LOW_SPEED_BOUNDARY_TRIGGER_FRACTION
        ),
        low_speed_boundary_exit_fraction=np.asarray(
            params.LOW_SPEED_BOUNDARY_EXIT_FRACTION
        ),
        low_speed_boundary_trigger_speed=np.asarray(
            params.LOW_SPEED_BOUNDARY_TRIGGER_SPEED
        ),
        low_speed_boundary_exit_speed=np.asarray(
            params.LOW_SPEED_BOUNDARY_EXIT_SPEED
        ),
        low_speed_boundary_trigger_seconds=np.asarray(
            params.LOW_SPEED_BOUNDARY_TRIGGER_SECONDS
        ),
        deadlock_reset_enabled=np.asarray(
            params.DEADLOCK_RESET_ENABLED, dtype=bool
        ),
        deadlock_reset_trigger_speed=np.asarray(
            params.DEADLOCK_RESET_TRIGGER_SPEED
        ),
        deadlock_reset_trigger_seconds=np.asarray(
            params.DEADLOCK_RESET_TRIGGER_SECONDS
        ),
        deadlock_reset_min_progress=np.asarray(
            params.DEADLOCK_RESET_MIN_PROGRESS
        ),
        deadlock_reset_restart_speed=np.asarray(
            params.DEADLOCK_RESET_RESTART_SPEED
        ),
        deadlock_reset_min_sim_time=np.asarray(
            params.DEADLOCK_RESET_MIN_SIM_TIME
        ),
        deadlock_reset_cooldown_seconds=np.asarray(
            params.DEADLOCK_RESET_COOLDOWN_SECONDS
        ),
        deadlock_reset_collision_margin=np.asarray(
            params.DEADLOCK_RESET_COLLISION_MARGIN
        ),
        velocity_track_weight=np.asarray(
            trace.get("velocity_track_weight", params.VELOCITY_TRACK_WEIGHT)
        ),
        heading_track_weight=np.asarray(
            trace.get("heading_track_weight", params.HEADING_TRACK_WEIGHT)
        ),
        track_slack_weight=np.asarray(
            trace.get("track_slack_weight", params.MPC_TRACK_SLACK_WEIGHT)
        ),
        track_length=np.asarray(track.total_length),
        vehicle_model=np.asarray(trace["vehicle_model"]),
        simulation_dt=np.asarray(params.SIM_DT),
        control_dt=np.asarray(params.CONTROL_DT),
        mpc_dt=np.asarray(params.MPC_DT),
        mpc_horizon=np.asarray(
            trace.get("mpc_horizon", params.MPC_HORIZON)
        ),
        mpc_formulation=np.asarray(
            trace.get("mpc_formulation", params.MPC_FORMULATION)
        ),
        mpc_state_coordinates=np.asarray(
            trace.get("mpc_state_coordinates", "unknown")
        ),
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
        ibr_horizon=np.asarray(params.IBR_HORIZON),
        ibr_iterations=np.asarray(params.IBR_ITERATIONS),
        ibr_implementation=np.asarray(
            trace.get("ibr_implementation", params.IBR_IMPLEMENTATION)
        ),
        ibr_progress_weight=np.asarray(params.IBR_PROGRESS_WEIGHT),
        ibr_lateral_weight=np.asarray(params.IBR_LATERAL_WEIGHT),
        ibr_lateral_exponent=np.asarray(params.IBR_LATERAL_EXPONENT),
        ibr_heading_weight=np.asarray(params.IBR_HEADING_WEIGHT),
        ibr_allow_track_slack=np.asarray(params.IBR_ALLOW_TRACK_SLACK),
        ibr_allow_collision_slack=np.asarray(params.IBR_ALLOW_COLLISION_SLACK),
        ibr_speed_weight=np.asarray(params.IBR_SPEED_WEIGHT),
        ibr_speed_cap_mode=np.asarray(params.IBR_SPEED_CAP_MODE),
        ibr_raceline_speed_cap_factor=np.asarray(
            params.IBR_RACELINE_SPEED_CAP_FACTOR
        ),
        ibr_control_weight=np.asarray(params.IBR_CONTROL_WEIGHT),
        ibr_control_reference=np.asarray(params.IBR_CONTROL_REFERENCE),
        ibr_control_rate_weight=np.asarray(params.IBR_CONTROL_RATE_WEIGHT),
        ibr_convergence_tolerance=np.asarray(params.IBR_CONVERGENCE_TOLERANCE),
        parameter_ibr_rounds=np.asarray(
            trace.get("parameter_ibr_rounds", params.PARAMETER_IBR_ROUNDS)
        ),
        parameter_ibr_search_points=np.asarray(
            trace.get("parameter_ibr_search_points", params.PARAMETER_IBR_SEARCH_POINTS)
        ),
        parameter_ibr_search_passes=np.asarray(
            trace.get("parameter_ibr_search_passes", params.PARAMETER_IBR_SEARCH_PASSES)
        ),
        parameter_ibr_update_interval_seconds=np.asarray(
            trace.get(
                "parameter_ibr_update_interval_seconds",
                params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS,
            )
        ),
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
