"""Audit whether a saved MPC deadlock depends on the IPOPT initial guess.

This detachable diagnostic reconstructs the current shared Frenet MPC from an
NPZ trace and resolves one stopped state from several control-plan seeds. It
does not modify the runtime controller or its objective.

Default usage (the saved P3 deterministic-rule failure):

    .venv/bin/python evaluation/mpc_deadlock_multistart_audit.py

Useful options:

    --trace PATH       alternate NPZ trace
    --car-index 0      audited car
    --random-seeds 16  number of smooth random control plans
    --output PATH      JSON result path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import make_vehicle_params, params
from mpc_controller import MPCController
from track import Track, load_centerline_csv


DEFAULT_TRACE = REPO_ROOT / (
    "result/frenet_llm_alpha_screen_rule_only/summary_traces/trial_003.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "result/mpc_deadlock_multistart_audit/audit.json"
ALPHA_SUCCESS_TRACE = REPO_ROOT / (
    "result/frenet_mpc_alpha_25s_w0/summary_traces/trial_001.npz"
)


def _scalar(data, name, default):
    return float(np.asarray(data[name]).item()) if name in data.files else default


def _text(data, name, default):
    return str(np.asarray(data[name]).item()) if name in data.files else default


def _select_deadlock_frame(data, car_index):
    time = np.asarray(data["time"], dtype=float)
    speed = np.asarray(data["car_vx"][:, car_index], dtype=float)
    progress = np.asarray(data["car_px"][:, car_index], dtype=float)
    lateral = np.asarray(data["car_py"][:, car_index], dtype=float)
    updates = np.flatnonzero(np.asarray(data["control_update"], dtype=bool))
    candidates = updates[
        (speed[updates] < 0.03)
        & (np.abs(lateral[updates]) > 0.42)
        & (progress[updates] > 60.0)
        & (progress[updates] < 70.0)
    ]
    if not len(candidates):
        raise RuntimeError("No stopped hairpin control-update frame found")
    return int(candidates[0])


def _opponents_from_trace(data, frame, car_index):
    opponents = []
    car_count = data["car_px"].shape[1]
    for other in range(car_count):
        if other == car_index:
            continue
        opponents.append(
            {
                "px": float(data["car_px"][frame, other]),
                "py": float(data["car_py"][frame, other]),
                "vx": float(data["car_vx"][frame, other]),
            }
        )
    return opponents


def _controller_from_trace(data, track):
    horizon = (
        int(np.asarray(data["mpc_horizon"]).item())
        if "mpc_horizon" in data.files
        else int(data["car_mpc_predicted_control_plan"].shape[-1])
    )
    vehicle_model = _text(data, "vehicle_model", "kinematic")
    return MPCController(
        dt=_scalar(data, "mpc_dt", params.MPC_DT),
        horizon=horizon,
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_delta=params.MPC_MAX_STEER_RATE * _scalar(data, "mpc_dt", params.MPC_DT),
        db_params=make_vehicle_params(vehicle_model),
        track_width=2.0 * _scalar(data, "track_half_width", params.TRACK_HALF_WIDTH),
        track_constraint_half_width=_scalar(
            data, "mpc_constraint_half_width", params.MPC_CONSTRAINT_HALF_WIDTH
        ),
        max_state_speed=params.MAX_STATE_SPEED,
        velocity_track_weight=_scalar(
            data, "velocity_track_weight", params.VELOCITY_TRACK_WEIGHT
        ),
        heading_track_weight=_scalar(
            data, "heading_track_weight", params.HEADING_TRACK_WEIGHT
        ),
        track_length=track.total_length,
        min_longitudinal_separation=params.MPC_MIN_LONGITUDINAL_SEPARATION,
        min_lateral_separation=params.MPC_MIN_LATERAL_SEPARATION,
        max_constrained_opponents=params.MPC_MAX_CONSTRAINED_OPPONENTS,
        collision_superellipse_exponent=params.MPC_COLLISION_SUPERELLIPSE_EXPONENT,
        enforce_track_boundary=bool(
            np.asarray(data["mpc_enforce_track_boundary"]).item()
        ),
        enforce_collision_avoidance=bool(
            np.asarray(data["mpc_enforce_collision_avoidance"]).item()
        ),
        track_slack_weight=_scalar(
            data, "mpc_track_slack_weight", params.MPC_TRACK_SLACK_WEIGHT
        ),
        collision_slack_weight=_scalar(
            data, "mpc_collision_slack_weight", params.MPC_COLLISION_SLACK_WEIGHT
        ),
        enforce_friction_speed_limit=bool(
            np.asarray(data["mpc_enforce_friction_speed_limit"]).item()
        ),
        friction_coefficient=_scalar(
            data, "mpc_friction_coefficient", params.MPC_FRICTION_COEFFICIENT
        ),
        track=track,
    )


def _rollout(controller, x0_mpc, controls):
    states = np.zeros((controller.n_states, controller.horizon + 1))
    states[:, 0] = x0_mpc
    for k in range(controller.horizon):
        states[:, k + 1] = controller._vehicle_dynamics_np(
            states[:, k], controls[:, k]
        )
    return states


def _bounds(controller):
    lbx = np.full(controller.n_opt, -np.inf)
    ubx = np.full(controller.n_opt, np.inf)
    for k in range(1, controller.horizon + 1):
        lbx[k * controller.n_states + 3] = 0.0
        ubx[k * controller.n_states + 3] = controller.max_speed
    for k in range(controller.horizon):
        start = controller.n_x + 2 * k
        lbx[start : start + 2] = -1.0
        ubx[start : start + 2] = 1.0
    lbx[controller.n_x + controller.n_u :] = 0.0
    lbg = np.zeros(controller.n_total)
    ubg = np.zeros(controller.n_total)
    lbg[controller.n_eq :] = -np.inf
    return lbx, ubx, lbg, ubg


def _solve_seed(
    controller, name, controls, x0_mpc, ref, p_val, opponents, current_s,
):
    controls = np.clip(np.asarray(controls, dtype=float), -1.0, 1.0)
    states = _rollout(controller, x0_mpc, controls)
    track_slack, collision_slack = controller._initial_slack_guess(
        states, ref, opponents=opponents, current_s=current_s
    )
    initial = np.concatenate(
        [
            states.T.reshape(-1),
            controls.T.reshape(-1),
            track_slack,
            collision_slack.T.reshape(-1),
        ]
    )
    lbx, ubx, lbg, ubg = _bounds(controller)
    try:
        solution = controller.solver(
            x0=initial, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val
        )
        stats = controller.solver.stats()
        optimum = np.asarray(solution["x"]).reshape(-1)
        state_plan = optimum[: controller.n_x].reshape(
            controller.horizon + 1, controller.n_states
        ).T
        control_plan = optimum[
            controller.n_x : controller.n_x + controller.n_u
        ].reshape(controller.horizon, controller.n_controls).T
        slack_start = controller.n_x + controller.n_u
        solved_track_slack = optimum[
            slack_start : slack_start + controller.n_track_slack
        ]
        solved_collision_slack = optimum[
            slack_start + controller.n_track_slack :
        ].reshape(controller.horizon, controller.max_constrained_opponents).T
        parts = [float(value) for value in controller.loss_fn(optimum, p_val)]
        return {
            "seed": name,
            "success": bool(stats.get("success", False)),
            "status": str(stats.get("return_status", "unknown")),
            "iterations": int(stats.get("iter_count", -1)),
            "objective": parts[0],
            "track_cost": parts[1],
            "action_cost": parts[2],
            "velocity_cost": parts[3],
            "heading_cost": parts[4],
            "violation_cost": parts[5],
            "track_slack_cost": parts[6],
            "collision_slack_cost": parts[7],
            "first_control": control_plan[:, 0].tolist(),
            "terminal_state": state_plan[:, -1].tolist(),
            "predicted_progress": float(state_plan[0, -1] - state_plan[0, 0]),
            "max_track_slack": float(np.max(solved_track_slack)),
            "max_collision_slack": float(np.max(solved_collision_slack)),
        }
    except Exception as exc:
        return {"seed": name, "success": False, "status": str(exc)}


def _alpha_success_seed(horizon, current_s):
    if not ALPHA_SUCCESS_TRACE.exists():
        return None
    with np.load(ALPHA_SUCCESS_TRACE, allow_pickle=True) as alpha:
        updates = np.flatnonzero(alpha["control_update"])
        start = updates[np.argmin(np.abs(alpha["car_px"][updates, 0] - current_s))]
        selected = updates[updates >= start][:horizon]
        if not len(selected):
            return None
        controls = np.asarray(
            [alpha["car_throttle"][selected, 0], alpha["car_steer"][selected, 0]],
            dtype=float,
        )
        if controls.shape[1] < horizon:
            controls = np.column_stack(
                [controls, np.repeat(controls[:, -1:], horizon - controls.shape[1], axis=1)]
            )
        return controls


def _make_seeds(data, frame, car_index, controller, random_count):
    horizon = controller.horizon
    updates = np.flatnonzero(data["control_update"])
    update_position = int(np.searchsorted(updates, frame))
    previous_frame = int(updates[max(update_position - 1, 0)])
    previous_plan = np.asarray(
        data["car_mpc_predicted_control_plan"][previous_frame, car_index], dtype=float
    )
    current_plan = np.asarray(
        data["car_mpc_predicted_control_plan"][frame, car_index], dtype=float
    )
    u_prev = np.asarray(
        [
            data["car_throttle"][previous_frame, car_index],
            data["car_steer"][previous_frame, car_index],
        ],
        dtype=float,
    )
    shifted = np.column_stack([previous_plan[:, 1:], previous_plan[:, -1]])
    seeds = {
        "runtime_shifted_previous_plan": shifted,
        "recorded_current_solution": current_plan,
        "repeat_previous_control": np.repeat(u_prev[:, None], horizon, axis=1),
    }
    inward_steer = -np.sign(float(data["car_py"][frame, car_index]))
    for throttle in (0.35, 0.60, 1.00):
        seed = np.empty((2, horizon))
        seed[0] = throttle
        seed[1] = inward_steer
        seeds[f"inward_steer_throttle_{throttle:.2f}"] = seed
    brake_then_go = np.empty((2, horizon))
    brake_then_go[0] = 1.0
    brake_then_go[0, : min(2, horizon)] = -1.0
    brake_then_go[1] = inward_steer
    seeds["brake_two_steps_then_go"] = brake_then_go
    alpha_seed = _alpha_success_seed(horizon, float(data["car_px"][frame, car_index]))
    if alpha_seed is not None:
        seeds["matched_successful_alpha_controls"] = alpha_seed

    rng = np.random.default_rng(20260825)
    max_delta = controller.max_steer_delta
    for idx in range(random_count):
        controls = np.zeros((2, horizon))
        controls[0] = rng.uniform(-1.0, 1.0, horizon)
        steer = float(u_prev[1])
        for k in range(horizon):
            steer = np.clip(steer + rng.uniform(-max_delta, max_delta), -1.0, 1.0)
            controls[1, k] = steer
        seeds[f"random_smooth_{idx:02d}"] = controls
    return seeds, u_prev


def run_audit(trace_path, car_index, random_count, output_path):
    with np.load(trace_path, allow_pickle=True) as data:
        frame = _select_deadlock_frame(data, car_index)
        track_scale = _scalar(data, "track_scale", 1.0)
        track = Track(load_centerline_csv(_text(data, "track_csv", params.TRACK_CSV), scale=track_scale))
        controller = _controller_from_trace(data, track)
        current_s = float(data["car_px"][frame, car_index])
        x0 = np.asarray(
            [
                data["car_x"][frame, car_index],
                data["car_y"][frame, car_index],
                data["car_psi"][frame, car_index],
                data["car_vx"][frame, car_index],
            ],
            dtype=float,
        )
        x0_mpc = controller._make_x0_mpc(x0, current_s)
        ref = controller._localize_reference(
            np.asarray(data["car_mpc_reference_plan"][frame, car_index], dtype=float),
            current_s,
        )
        opponents = _opponents_from_trace(data, frame, car_index)
        seeds, u_prev = _make_seeds(data, frame, car_index, controller, random_count)
        q_weight = float(data["car_policy_params"][frame, car_index, 0])
        p_val = controller._make_p_val(
            x0_mpc, ref, u_prev, q_weight,
            opponents=opponents, current_s=current_s,
        )
        results = [
            _solve_seed(
                controller, name, controls, x0_mpc, ref, p_val,
                opponents, current_s,
            )
            for name, controls in seeds.items()
        ]
        successful = [item for item in results if item.get("success")]
        successful.sort(key=lambda item: item["objective"])
        report = {
            "trace": str(Path(trace_path).resolve()),
            "frame": frame,
            "time": float(data["time"][frame]),
            "car_index": car_index,
            "horizon": controller.horizon,
            "measured_frenet_state": x0_mpc.tolist(),
            "previous_control": u_prev.tolist(),
            "q_weight": q_weight,
            "track_slack_weight": controller.track_slack_weight,
            "results_by_objective": successful,
            "failed_results": [item for item in results if not item.get("success")],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Audit state: t={report['time']:.2f}s, "
        f"z={np.round(report['measured_frenet_state'], 4).tolist()}"
    )
    print(
        "seed                              objective   progress  "
        "first throttle/steer   max track/collision slack"
    )
    for item in successful:
        print(
            f"{item['seed']:<33} {item['objective']:>10.4f}  "
            f"{item['predicted_progress']:>8.4f}  "
            f"({item['first_control'][0]:>7.3f},{item['first_control'][1]:>7.3f})  "
            f"({item['max_track_slack']:.5f},{item['max_collision_slack']:.5f})"
        )
    if report["failed_results"]:
        print(f"Failed starts: {len(report['failed_results'])}")
    try:
        display_path = output_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = output_path
    print(f"Saved {display_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--car-index", type=int, default=0)
    parser.add_argument("--random-seeds", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(
        args.trace.expanduser().resolve(),
        args.car_index,
        args.random_seeds,
        args.output.expanduser().resolve(),
    )
