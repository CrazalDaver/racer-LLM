import numpy as np

from config import db_params, params
from mpc_controller import MPCController
from plotting import plot_simulation_results
from policies import AlphaRacerPolicy, CenterLinePolicy
from state import CarState
from track import Track, load_centerline_csv, load_raceline_csv
from vehicle_model import DynamicBicycleModel


def create_policy(policy_type, track, mpc, raceline=None):
    if policy_type == "centerline":
        return CenterLinePolicy(track, mpc, step_distance=params.DT * params.MAX_RACELINE_SPEED)
    if policy_type == "alpharacer":
        return AlphaRacerPolicy(
            track,
            mpc,
            step_distance=params.DT * params.MAX_RACELINE_SPEED,
            params=np.array([1.0, 1.0, 0.8, 0.1, 0.5]),  # q, zeta, s1, s2, s3
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            search_back_steps=params.RACELINE_SEARCH_BACK_STEPS,
            search_forward_steps=params.RACELINE_SEARCH_FORWARD_STEPS,
        )
    raise ValueError("Unknown policy type")


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
        track_csv=np.asarray(params.TRACK_CSV),
        raceline_csv=np.asarray(params.RACELINE_CSV),
        track_width=np.asarray(params.TRACK_WIDTH),
        track_length=np.asarray(track.total_length),
    )
    print(f"Saved simulation trace to {params.TRACE_OUTPUT}")


def run_simulation(policy_type="centerline", save_trace=True, show_plot=True):
    waypoints = load_centerline_csv(params.TRACK_CSV)
    raceline = load_raceline_csv(params.RACELINE_CSV)
    track = Track(waypoints)

    mpc = MPCController(
        dt=params.MPC_DT,
        horizon=params.MPC_HORIZON,
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_rate=params.MAX_STEER_RATE,
        db_params=db_params,
        track_width=params.TRACK_WIDTH,
        max_state_speed=params.MAX_STATE_SPEED,
    )
    policy = create_policy(policy_type, track, mpc, raceline=raceline)
    db_model = DynamicBicycleModel(dt=params.DT, db_params=db_params)

    x0, y0, psi0 = track.interpolate(0.0)
    state = CarState(x=x0, y=y0, vx=0.0, vy=0.0, psi=psi0, omega=0.0)

    total_steps = int(params.TOTAL_TIME / params.DT)
    track_half_width = params.TRACK_WIDTH / 2.0
    progress_s = 0.0

    traj_x, traj_y, velocities, lateral_errors = [], [], [], []
    longitudinal_positions = []
    mpc_losses, mpc_track_losses, mpc_action_losses = [], [], []
    frenet_ref_lines = []
    trace = {
        "time": [],
        "x": [],
        "y": [],
        "psi": [],
        "vx": [],
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
    }

    print(f"Simulation with {policy_type} policy started...")
    for step in range(total_steps):
        s, e_before = track.project(
            state.x,
            state.y,
            s_hint=progress_s,
            search_radius=params.PROJECTION_SEARCH_RADIUS,
            psi_hint=state.psi,
            max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
        )
        progress_s = s
        action = policy.get_action(state, s)
        ref_traj = policy.last_ref_traj
        ref_fx, ref_fy = np.nan, np.nan
        ref_horizon_x = np.full(mpc.horizon + 1, np.nan)
        ref_horizon_y = np.full(mpc.horizon + 1, np.nan)
        raceline_horizon_x = np.full(mpc.horizon + 1, np.nan)
        raceline_horizon_y = np.full(mpc.horizon + 1, np.nan)
        if ref_traj is not None:
            if policy.last_ref_world_horizon is not None:
                ref_horizon_x = policy.last_ref_world_horizon[0].copy()
                ref_horizon_y = policy.last_ref_world_horizon[1].copy()
                ref_fx, ref_fy = ref_horizon_x[0], ref_horizon_y[0]
            else:
                ref_fx, ref_fy = track.frenet_to_world(ref_traj[0, 0], ref_traj[1, 0])
                for k in range(mpc.horizon + 1):
                    ref_horizon_x[k], ref_horizon_y[k] = track.frenet_to_world(ref_traj[0, k], ref_traj[1, k])

            trace["ref_px"].append(ref_traj[0, 0])
            trace["ref_py"].append(ref_traj[1, 0])
            trace["ref_speed"].append(ref_traj[2, 0])
            if policy.last_raceline_world_horizon is not None:
                raceline_horizon_x = policy.last_raceline_world_horizon[0].copy()
                raceline_horizon_y = policy.last_raceline_world_horizon[1].copy()
        else:
            trace["ref_px"].append(np.nan)
            trace["ref_py"].append(np.nan)
            trace["ref_speed"].append(np.nan)

        trace["time"].append(step * params.DT)
        trace["x"].append(state.x)
        trace["y"].append(state.y)
        trace["psi"].append(state.psi)
        trace["vx"].append(state.vx)
        trace["px"].append(s)
        trace["py"].append(e_before)
        trace["ref_x"].append(ref_fx)
        trace["ref_y"].append(ref_fy)
        trace["ref_raceline_s"].append(getattr(policy, "last_raceline_s", np.nan))
        trace["raceline_horizon_x"].append(raceline_horizon_x)
        trace["raceline_horizon_y"].append(raceline_horizon_y)
        trace["ref_horizon_x"].append(ref_horizon_x)
        trace["ref_horizon_y"].append(ref_horizon_y)
        trace["offroad"].append(abs(e_before) > track_half_width)
        trace["throttle"].append(action.throttle)
        trace["steer"].append(action.steer)
        trace["mpc_loss"].append(mpc.last_loss)
        trace["mpc_track_loss"].append(mpc.last_track_loss)
        trace["mpc_action_loss"].append(mpc.last_action_loss)

        if (step + 1) % 100 == 0 and policy.last_ref_traj is not None:
            frenet_ref_lines.append((state.x, state.y, ref_fx, ref_fy))

        mpc_losses.append(mpc.last_loss)
        mpc_track_losses.append(mpc.last_track_loss)
        mpc_action_losses.append(mpc.last_action_loss)
        next_state_arr = db_model.step(state.to_array(), action.throttle, action.steer)
        state = CarState.from_array(next_state_arr)

        # compute lateral error
        s, e = track.project(
            state.x,
            state.y,
            s_hint=progress_s,
            search_radius=params.PROJECTION_SEARCH_RADIUS,
            psi_hint=state.psi,
            max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
        )
        progress_s = s
        lateral_errors.append(e)
        longitudinal_positions.append(progress_s)

        # out-of-track penalty
        if abs(e) > track_half_width:
            overshoot = abs(e) - track_half_width
            factor = np.exp(-params.OFFROAD_STRENGTH * overshoot)
            state.vx *= factor
            state.vy *= factor

        if step % 2 == 0:
            traj_x.append(state.x)
            traj_y.append(state.y)
            velocities.append(state.vx)

        if (step + 1) % 200 == 0:
            print(
                f"Step {step+1}/{total_steps}, t={(step+1)*params.DT:.1f}s, "
                f"vx={state.vx:.2f} m/s, e={e:.3f}, "
                f"throttle={action.throttle:.2f}, steer={action.steer:.2f}"
            )

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
            longitudinal_positions,
            frenet_ref_lines,
            params,
        )


if __name__ == "__main__":
    # Options: "centerline" or "alpharacer"
    run_simulation(policy_type="alpharacer")
