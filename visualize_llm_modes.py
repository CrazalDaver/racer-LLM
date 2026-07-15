"""Compare the five discrete LLM modes in one fixed short driving scenario.

The LLM in :mod:`policies` selects one of five parameter presets.  This script
does not call an LLM; it freezes each preset in turn so that mode is the only
experimental variable.  Ego starts before a curved section and approaches a
stationary opponent on the centerline.

Example:
    python visualize_llm_modes.py
    python visualize_llm_modes.py --track modena --duration 4 --save modes.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import make_vehicle_params, params
from mpc_controller import MPCController
from policies import StaticAlphaRacerPolicy
from simulation import progress_delta, project_vehicle_state
from state import CarState
from track import Track, configure_track, load_centerline_csv, load_raceline_csv
from vehicle_model import KinematicBicycleModel


# Keep this table next to the experiment so the plotted comparison is explicit.
MODE_PRESETS = {
    "stable": np.array([1.3, 1.00, 0.05, 1.00, 0.0]),
    "push": np.array([1.8, 1.25, 0.05, 1.00, 0.0]),
    "overtake": np.array([1.0, 1.35, 1.75, 0.10, 0.0]),
    "block": np.array([1.6, 0.90, 0.05, 0.45, 2.0]),
    "recover": np.array([3.0, 0.70, 0.00, 1.50, 0.0]),
}

MODE_COLORS = {
    "stable": "tab:blue",
    "push": "tab:red",
    "overtake": "tab:orange",
    "block": "tab:purple",
    "recover": "tab:green",
}


def _curvature(track, s, ds=0.25):
    """Signed centerline curvature estimated from heading change."""
    psi0 = track.interpolate(s - ds)[2]
    psi1 = track.interpolate(s + ds)[2]
    return np.arctan2(np.sin(psi1 - psi0), np.cos(psi1 - psi0)) / (2.0 * ds)


def choose_curve_s(track):
    """Choose a pronounced curve, away from the closed-path seam."""
    samples = np.linspace(0.08 * track.total_length, 0.92 * track.total_length, 600)
    curvatures = np.array([abs(_curvature(track, s)) for s in samples])
    return float(samples[np.argmax(curvatures)])


def make_mpc(track, vehicle_params, horizon):
    return MPCController(
        dt=params.MPC_DT,
        horizon=horizon,
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_rate=params.MAX_STEER_RATE,
        db_params=vehicle_params,
        track_width=params.TRACK_WIDTH,
        track_constraint_half_width=params.MPC_CONSTRAINT_HALF_WIDTH,
        max_state_speed=params.MAX_STATE_SPEED,
        velocity_track_weight=params.VELOCITY_TRACK_WEIGHT,
        heading_track_weight=params.HEADING_TRACK_WEIGHT,
        track_length=track.total_length,
        min_longitudinal_separation=params.MPC_MIN_LONGITUDINAL_SEPARATION,
        min_lateral_separation=params.MPC_MIN_LATERAL_SEPARATION,
        max_constrained_opponents=1,
        collision_superellipse_exponent=params.MPC_COLLISION_SUPERELLIPSE_EXPONENT,
    )


def simulate_mode(track, raceline, mode, ego_s0, opponent_s, duration, v0, horizon):
    """Run one deterministic rollout with a fixed, stationary opponent."""
    vehicle_params = make_vehicle_params("kinematic")
    mpc = make_mpc(track, vehicle_params, horizon)
    policy = StaticAlphaRacerPolicy(
        track,
        mpc,
        car_idx=0,
        policy_name=f"fixed_{mode}",
        params=MODE_PRESETS[mode].copy(),
        raceline=raceline,
        max_speed=params.MAX_RACELINE_SPEED,
        ref_lateral_clip=params.REF_LATERAL_CLIP,
    )
    policy.current_mode = mode
    policy.log_fn = lambda _message: None
    model = KinematicBicycleModel(params.DT, vehicle_params, params.MAX_STATE_SPEED)

    x0, y0 = track.frenet_to_world(ego_s0, 0.0)
    psi0 = track.interpolate(ego_s0)[2]
    state = CarState(x=x0, y=y0, psi=psi0, vx=v0)
    opponent_x, opponent_y = track.frenet_to_world(opponent_s, 0.0)
    opponent_psi = track.interpolate(opponent_s)[2]

    history = {key: [] for key in ("time", "x", "y", "s", "e", "vx")}
    progress_s = ego_s0
    for step in range(int(round(duration / params.DT)) + 1):
        progress_s, lateral_e = project_vehicle_state(
            track,
            state,
            progress_s,
            params.PROJECTION_SEARCH_RADIUS,
            params.PROJECTION_MAX_HEADING_ERROR,
        )
        history["time"].append(step * params.DT)
        history["x"].append(state.x)
        history["y"].append(state.y)
        history["s"].append(progress_s)
        history["e"].append(lateral_e)
        history["vx"].append(state.vx)
        if step == int(round(duration / params.DT)):
            break

        opponent = [{
            "px": opponent_s,
            "py": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "omega": 0.0,
            "psi": opponent_psi,
            "x": opponent_x,
            "y": opponent_y,
        }]
        action = policy.get_action(
            state, progress_s, opponents=opponent, sim_time=step * params.DT
        )
        state = CarState.from_array(model.step(state.to_array(), action.throttle, action.steer))

    return {key: np.asarray(values) for key, values in history.items()}


def _sample_track_band(track, start_s, end_s, half_width, count=500):
    ss = np.linspace(start_s, end_s, count)
    center = np.array([track.frenet_to_world(s, 0.0) for s in ss])
    left = np.array([track.frenet_to_world(s, half_width) for s in ss])
    right = np.array([track.frenet_to_world(s, -half_width) for s in ss])
    return ss, center, left, right


def plot_results(track, results, ego_s0, opponent_s, duration, save_path=None, show=True):
    furthest_s = max(float(run["s"].max()) for run in results.values())
    view_start = ego_s0 - 0.5
    view_end = max(opponent_s + 1.5, furthest_s + 0.5)
    _, center, left, right = _sample_track_band(
        track, view_start, view_end, params.TRACK_HALF_WIDTH
    )
    opponent_xy = track.frenet_to_world(opponent_s, 0.0)
    ego_xy = track.frenet_to_world(ego_s0, 0.0)

    fig, (ax_map, ax_e, ax_gap) = plt.subplots(1, 3, figsize=(16, 5.2))
    ax_map.fill(
        np.r_[left[:, 0], right[::-1, 0]],
        np.r_[left[:, 1], right[::-1, 1]],
        color="0.92",
        label="track",
    )
    ax_map.plot(left[:, 0], left[:, 1], "k-", lw=1)
    ax_map.plot(right[:, 0], right[:, 1], "k-", lw=1)
    ax_map.plot(center[:, 0], center[:, 1], "k--", lw=0.8, alpha=0.5)
    for mode, run in results.items():
        label = f"{mode}  {MODE_PRESETS[mode].tolist()}"
        ax_map.plot(run["x"], run["y"], color=MODE_COLORS[mode], lw=2.2, label=label)
        ax_e.plot(run["time"], run["e"], color=MODE_COLORS[mode], lw=2, label=mode)
        gaps = np.array([progress_delta(opponent_s, s, track.total_length) for s in run["s"]])
        ax_gap.plot(run["time"], gaps, color=MODE_COLORS[mode], lw=2, label=mode)

    ax_map.scatter(*ego_xy, marker="o", s=55, color="black", zorder=5, label="ego start")
    ax_map.scatter(*opponent_xy, marker="X", s=110, color="black", zorder=6, label="stationary opponent")
    ax_map.set_title("World trajectories (curve crop)")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(fontsize=7, loc="best")

    ax_e.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax_e.axhline(params.TRACK_HALF_WIDTH, color="k", lw=0.8)
    ax_e.axhline(-params.TRACK_HALF_WIDTH, color="k", lw=0.8)
    ax_e.set_title("Lateral response")
    ax_e.set_xlabel("time [s]")
    ax_e.set_ylabel("Frenet lateral e [m]")
    ax_e.grid(alpha=0.25)
    ax_e.legend(fontsize=8)

    ax_gap.axhline(0.0, color="k", ls="--", lw=0.8)
    ax_gap.set_title("Distance to stationary opponent")
    ax_gap.set_xlabel("time [s]")
    ax_gap.set_ylabel("opponent s - ego s [m]")
    ax_gap.grid(alpha=0.25)
    ax_gap.legend(fontsize=8)
    fig.suptitle(f"Fixed LLM mode comparison ({duration:g} s; only mode preset changes)")
    fig.tight_layout()

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="berlin", choices=("berlin", "modena", "rounded_rectangle"))
    parser.add_argument("--duration", type=float, default=3.0, help="rollout duration in seconds")
    parser.add_argument("--gap", type=float, default=2.0, help="initial distance to stationary opponent [m]")
    parser.add_argument("--v0", type=float, default=2.0, help="ego initial speed [m/s]")
    parser.add_argument("--horizon", type=int, default=params.MPC_HORIZON)
    parser.add_argument("--curve-s", type=float, default=None, help="opponent progress; default finds a sharp curve")
    parser.add_argument("--save", default="llm_modes_comparison.png")
    parser.add_argument("--no-show", action="store_true", help="save without opening a plot window")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0 or args.gap <= 0 or args.horizon <= 0:
        raise ValueError("duration, gap, and horizon must be positive")
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    raceline = load_raceline_csv(params.RACELINE_CSV, scale=params.TRACK_SCALE)
    opponent_s = choose_curve_s(track) if args.curve_s is None else args.curve_s
    ego_s0 = opponent_s - args.gap
    print(
        f"Scenario: track={args.track}, ego_s={ego_s0:.3f}, opponent_s={opponent_s:.3f}, "
        f"gap={args.gap:.2f} m, opponent_speed=0, curve={_curvature(track, opponent_s):.3f} 1/m"
    )
    results = {}
    for mode in MODE_PRESETS:
        print(f"Running {mode:8s}: q,zeta,s1,s2,s3={MODE_PRESETS[mode].tolist()}")
        results[mode] = simulate_mode(
            track, raceline, mode, ego_s0, opponent_s, args.duration, args.v0, args.horizon
        )
    plot_results(
        track, results, ego_s0, opponent_s, args.duration,
        save_path=args.save, show=not args.no_show,
    )


if __name__ == "__main__":
    main()
