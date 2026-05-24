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
            params=np.array([1.0, 0.0, 0.8, 0.1, 0.5]),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
        )
    raise ValueError("Unknown policy type")


def run_simulation(policy_type="centerline"):
    waypoints = load_centerline_csv(params.TRACK_CSV)
    raceline = load_raceline_csv(params.RACELINE_CSV)
    track = Track(waypoints)

    mpc = MPCController(
        dt=params.DT,
        horizon=params.MPC_HORIZON,
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_rate=params.MAX_STEER_RATE,
        db_params=db_params,
        track_width=params.TRACK_WIDTH,
        max_state_vel_rate=params.MAX_STATE_VEL_RATE,
        max_state_speed=params.MAX_STATE_SPEED,
        max_lateral_speed=params.MAX_LATERAL_SPEED,
        max_yaw_rate=params.MAX_YAW_RATE,
    )
    policy = create_policy(policy_type, track, mpc, raceline=raceline)
    db_model = DynamicBicycleModel(dt=params.DT, db_params=db_params)

    x0, y0, psi0 = track.interpolate(0.0)
    state = CarState(x=x0, y=y0, vx=0.0, vy=0.0, psi=psi0, omega=0.0)

    total_steps = int(params.TOTAL_TIME / params.DT)
    track_half_width = params.TRACK_WIDTH / 2.0
    progress_s = 0.0

    traj_x, traj_y, velocities, lateral_errors = [], [], [], []

    print(f"Simulation with {policy_type} policy started...")
    for step in range(total_steps):
        s, _ = track.project(
            state.x,
            state.y,
            s_hint=progress_s,
            search_radius=params.PROJECTION_SEARCH_RADIUS,
        )
        progress_s = s
        action = policy.get_action(state, s)
        next_state_arr = db_model.step(state.to_array(), action.throttle, action.steer)
        state = CarState.from_array(next_state_arr)

        s, e = track.project(
            state.x,
            state.y,
            s_hint=progress_s,
            search_radius=params.PROJECTION_SEARCH_RADIUS,
        )
        progress_s = s
        lateral_errors.append(e)
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

    plot_simulation_results(waypoints, raceline, traj_x, traj_y, velocities, lateral_errors, params)


if __name__ == "__main__":
    # Options: "centerline" or "alpharacer"
    run_simulation(policy_type="alpharacer")
