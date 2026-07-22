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
import base64
import html
import json
from pathlib import Path

import matplotlib

# The script primarily generates files and may run without a GUI session.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize

from config import make_vehicle_params, params
from mpc_controller import MPCController
from simulation import create_policy, create_vehicle_model, progress_delta, project_vehicle_state
from state import CarState
from track import Track, configure_track, load_centerline_csv, load_raceline_csv


# Previous hand-tuned values (kept as comments for comparison):
# stable   = [1.3, 1.00, 0.05, 1.00, 0.0]
# push     = [1.8, 1.25, 0.05, 1.00, 0.0]
# overtake = [1.0, 1.35, 1.75, 0.10, 0.0]
# block    = [1.6, 0.90, 0.05, 0.45, 2.0]
# recover  = [3.0, 0.70, 0.00, 1.50, 0.0]
# Active values come from config so this visualization and the executable LLM
# policy cannot silently diverge.
MODE_PRESETS = params.LLM_MODE_PRESETS

OPPONENT_PARAMS = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

MODE_COLORS = {
    "stable": "tab:blue",
    "push": "tab:red",
    "overtake": "tab:orange",
    "block": "tab:purple",
    "recover": "tab:green",
}

# Official alpha-RACER sampling bounds mapped to this project's
# [q, zeta, s1, s2, s3] representation:
# q=2*lookahead_factor and zeta=speed_factor**2.
PARAMETER_SWEEP_RANGES = params.LLM_PARAMETER_BOUNDS
PARAMETER_NAMES = tuple(PARAMETER_SWEEP_RANGES)
PARAMETER_BASELINE = np.array([
    np.mean(PARAMETER_SWEEP_RANGES[name]) for name in PARAMETER_NAMES
])


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


def centerline_pose_at_track_s(track, target_s):
    x, y, psi = track.interpolate(target_s)
    return float(x), float(y), float(psi), float(target_s), 0.0


def make_mpc(track, vehicle_params, horizon):
    return MPCController(
        dt=params.MPC_DT,
        horizon=horizon,
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_delta=params.MPC_MAX_STEER_DELTA,
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


def simulate_mode(
    track, raceline, mode, ego_start, opponent_start,
    duration, v0, opponent_v0, stationary_opponent, horizon, ego_params=None,
):
    """Run one rollout against either a moving or stationary opponent."""
    vehicle_params = make_vehicle_params("kinematic")
    ego_mpc = make_mpc(track, vehicle_params, horizon)
    opponent_mpc = make_mpc(track, vehicle_params, horizon)
    # Construct the policy through the same factory as simulation.run_simulation().
    policy = create_policy(
        "static_alpharacer",
        track,
        ego_mpc,
        raceline=raceline,
        car_idx=0,
        static_policy_params=(
            MODE_PRESETS[mode] if ego_params is None else np.asarray(ego_params, dtype=float)
        ),
    )
    policy.current_mode = mode
    policy.log_fn = lambda _message: None
    opponent_policy = create_policy(
        "static_alpharacer",
        track,
        opponent_mpc,
        raceline=raceline,
        car_idx=1,
        static_policy_params=OPPONENT_PARAMS,
    )
    opponent_policy.current_mode = "raceline_baseline"
    opponent_policy.log_fn = lambda _message: None
    # Construct the vehicle through the same factory as run_simulation().
    ego_model = create_vehicle_model(
        "kinematic", params.DT, vehicle_params, params.MAX_STATE_SPEED
    )
    opponent_model = create_vehicle_model(
        "kinematic", params.DT, vehicle_params, params.MAX_STATE_SPEED
    )

    x0, y0, psi0, ego_s0, ego_e0 = ego_start
    state = CarState(x=x0, y=y0, psi=psi0, vx=v0)
    opponent_x, opponent_y, opponent_psi, opponent_s0, opponent_e0 = opponent_start
    opponent_state = CarState(
        x=opponent_x,
        y=opponent_y,
        psi=opponent_psi,
        vx=opponent_v0,
    )

    history = {key: [] for key in (
        "time", "x", "y", "psi", "s", "e", "vx", "throttle", "steer",
        "mpc_success", "mpc_status", "mpc_recovery", "ref_horizon_x", "ref_horizon_y",
        "opponent_x", "opponent_y", "opponent_psi", "opponent_s", "opponent_e", "opponent_vx",
    )}
    progress_s = ego_s0
    opponent_progress_s = opponent_s0
    for step in range(int(round(duration / params.DT)) + 1):
        progress_s, lateral_e = project_vehicle_state(
            track,
            state,
            progress_s,
            params.PROJECTION_SEARCH_RADIUS,
            params.PROJECTION_MAX_HEADING_ERROR,
        )
        opponent_progress_s, opponent_e = project_vehicle_state(
            track,
            opponent_state,
            opponent_progress_s,
            params.PROJECTION_SEARCH_RADIUS,
            params.PROJECTION_MAX_HEADING_ERROR,
        )
        opponent_for_ego = [{
            "car_idx": 1,
            "px": opponent_progress_s,
            "py": opponent_e,
            "vx": max(opponent_state.vx, 0.0),
            "vy": opponent_state.vy,
            "omega": opponent_state.omega,
            "psi": opponent_state.psi,
            "state": opponent_state,
        }]
        ego_for_opponent = [{
            "car_idx": 0,
            "px": progress_s,
            "py": lateral_e,
            "vx": max(state.vx, 0.0),
            "vy": state.vy,
            "omega": state.omega,
            "psi": state.psi,
            "state": state,
        }]
        action = policy.get_action(
            state, progress_s, opponents=opponent_for_ego, sim_time=step * params.DT
        )
        ref_world = policy.last_ref_world_horizon
        if ref_world is None:
            ref_x = np.full(horizon + 1, np.nan)
            ref_y = np.full(horizon + 1, np.nan)
        else:
            ref_x = np.asarray(ref_world[0], dtype=float).copy()
            ref_y = np.asarray(ref_world[1], dtype=float).copy()

        history["time"].append(step * params.DT)
        history["x"].append(state.x)
        history["y"].append(state.y)
        history["psi"].append(state.psi)
        history["s"].append(progress_s)
        history["e"].append(lateral_e)
        history["vx"].append(state.vx)
        history["throttle"].append(action.throttle)
        history["steer"].append(action.steer)
        history["mpc_success"].append(ego_mpc.last_solver_success)
        history["mpc_status"].append(ego_mpc.last_solver_status)
        history["mpc_recovery"].append(ego_mpc.last_recovery_active)
        history["ref_horizon_x"].append(ref_x)
        history["ref_horizon_y"].append(ref_y)
        history["opponent_x"].append(opponent_state.x)
        history["opponent_y"].append(opponent_state.y)
        history["opponent_psi"].append(opponent_state.psi)
        history["opponent_s"].append(opponent_progress_s)
        history["opponent_e"].append(opponent_e)
        history["opponent_vx"].append(opponent_state.vx)
        if step == int(round(duration / params.DT)):
            break

        state = CarState.from_array(
            ego_model.step(state.to_array(), action.throttle, action.steer)
        )
        if not stationary_opponent:
            opponent_action = opponent_policy.get_action(
                opponent_state,
                opponent_progress_s,
                opponents=ego_for_opponent,
                sim_time=step * params.DT,
            )
            opponent_state = CarState.from_array(
                opponent_model.step(
                    opponent_state.to_array(),
                    opponent_action.throttle,
                    opponent_action.steer,
                )
            )

    return {key: np.asarray(values) for key, values in history.items()}


def _sample_track_band(track, start_s, end_s, half_width, count=500):
    ss = np.linspace(start_s, end_s, count)
    center = np.array([track.frenet_to_world(s, 0.0) for s in ss])
    left = np.array([track.frenet_to_world(s, half_width) for s in ss])
    right = np.array([track.frenet_to_world(s, -half_width) for s in ss])
    return ss, center, left, right


def plot_results(track, raceline, results, ego_start, opponent_start, duration, save_path=None, show=True):
    ego_x0, ego_y0, _, ego_s0, ego_e0 = ego_start
    opponent_x0, opponent_y0, _, opponent_s, _ = opponent_start
    furthest_s = max(
        max(float(run["s"].max()), float(run["opponent_s"].max()))
        for run in results.values()
    )
    view_start = ego_s0 - 0.5
    view_end = max(opponent_s + 1.5, furthest_s + 0.5)
    _, center, left, right = _sample_track_band(
        track, view_start, view_end, params.TRACK_HALF_WIDTH
    )
    opponent_xy = (opponent_x0, opponent_y0)
    ego_xy = (ego_x0, ego_y0)

    fig, (ax_map, ax_e, ax_gap) = plt.subplots(1, 3, figsize=(16, 5.2))
    ax_map.fill(
        np.r_[left[:, 0], right[::-1, 0]],
        np.r_[left[:, 1], right[::-1, 1]],
        color="0.92",
        label="track",
    )
    ax_map.plot(left[:, 0], left[:, 1], "k-", lw=1)
    ax_map.plot(right[:, 0], right[:, 1], "k-", lw=1)
    ax_map.plot(center[:, 0], center[:, 1], color="0.65", ls=":", lw=0.9, label="centerline")
    closed_raceline_x = np.r_[raceline.x, raceline.x[0]]
    closed_raceline_y = np.r_[raceline.y, raceline.y[0]]
    ax_map.plot(
        closed_raceline_x,
        closed_raceline_y,
        color="0.4",
        ls="--",
        lw=1.4,
        label="raceline",
        zorder=2,
    )
    for mode, run in results.items():
        label = f"{mode}  {MODE_PRESETS[mode].tolist()}"
        ax_map.plot(run["x"], run["y"], color=MODE_COLORS[mode], lw=2.2, label=label)
        ax_map.plot(
            run["opponent_x"],
            run["opponent_y"],
            color=MODE_COLORS[mode],
            ls="--",
            lw=1.2,
            alpha=0.65,
            label="opponent trajectories" if mode == next(iter(results)) else "_nolegend_",
        )
        ax_e.plot(run["time"], run["e"], color=MODE_COLORS[mode], lw=2, label=mode)
        gaps = np.array([
            progress_delta(opponent_s, ego_s, track.total_length)
            for opponent_s, ego_s in zip(run["opponent_s"], run["s"])
        ])
        ax_gap.plot(run["time"], gaps, color=MODE_COLORS[mode], lw=2, label=mode)

    ax_map.scatter(*ego_xy, marker="o", s=55, color="black", zorder=5, label="ego start")
    ax_map.scatter(*opponent_xy, marker="X", s=110, color="black", zorder=6, label="opponent start")
    ax_map.set_title("World trajectories (curve crop)")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal", adjustable="box")
    map_x = np.r_[left[:, 0], right[:, 0]]
    map_y = np.r_[left[:, 1], right[:, 1]]
    margin = 0.15
    ax_map.set_xlim(map_x.min() - margin, map_x.max() + margin)
    ax_map.set_ylim(map_y.min() - margin, map_y.max() + margin)
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
    ax_gap.set_title("Distance to opponent")
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


def _car_corners(x, y, psi):
    half_length = float(params.VEHICLE_HALF_LENGTH)
    half_width = float(params.VEHICLE_HALF_WIDTH)
    local = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width],
    ])
    rotation = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)],
    ])
    return local @ rotation.T + np.array([x, y])


def save_mode_animation(track, raceline, mode, run, output_path, stride=3, interval=50, dpi=90):
    """Save one mode rollout GIF with the instantaneous MPC reference horizon."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    view_start = min(float(run["s"].min()), float(run["opponent_s"].min())) - 0.7
    view_end = max(float(run["s"].max()), float(run["opponent_s"].max())) + 0.7
    _, center, left, right = _sample_track_band(
        track, view_start, view_end, params.TRACK_HALF_WIDTH, count=600
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.fill(
        np.r_[left[:, 0], right[::-1, 0]],
        np.r_[left[:, 1], right[::-1, 1]],
        color="0.93",
        zorder=0,
    )
    ax.plot(left[:, 0], left[:, 1], "k-", lw=1)
    ax.plot(right[:, 0], right[:, 1], "k-", lw=1)
    ax.plot(center[:, 0], center[:, 1], color="0.65", ls=":", lw=1, label="centerline")
    ax.plot(raceline.x, raceline.y, color="0.45", ls="--", lw=1.2, label="raceline")
    ego_path, = ax.plot([], [], color=MODE_COLORS[mode], lw=1.8, label="ego path")
    opponent_path, = ax.plot([], [], color="#0B3C8C", lw=1.4, alpha=0.8, label="opponent path")
    ref_line, = ax.plot([], [], color="magenta", marker=".", ms=4, lw=1.8, label="MPC ref horizon")
    ego_car = Polygon(_car_corners(run["x"][0], run["y"][0], run["psi"][0]),
                      closed=True, color=MODE_COLORS[mode], zorder=6, label="ego")
    opponent_car = Polygon(
        _car_corners(run["opponent_x"][0], run["opponent_y"][0], run["opponent_psi"][0]),
        closed=True, color="#0B3C8C", zorder=5, label="opponent",
    )
    ax.add_patch(ego_car)
    ax.add_patch(opponent_car)
    status_text = ax.text(
        0.98, 0.98, "", transform=ax.transAxes, va="top", ha="right",
        fontsize=9, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    map_x = np.r_[left[:, 0], right[:, 0]]
    map_y = np.r_[left[:, 1], right[:, 1]]
    ax.set_xlim(map_x.min() - 0.2, map_x.max() + 0.2)
    ax.set_ylim(map_y.min() - 0.2, map_y.max() + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{mode}: params={MODE_PRESETS[mode].tolist()}")
    ax.legend(loc="lower right", fontsize=8)
    frames = np.arange(0, len(run["time"]), max(1, int(stride)))
    if frames[-1] != len(run["time"]) - 1:
        frames = np.r_[frames, len(run["time"]) - 1]

    def update(frame):
        ego_path.set_data(run["x"][: frame + 1], run["y"][: frame + 1])
        opponent_path.set_data(run["opponent_x"][: frame + 1], run["opponent_y"][: frame + 1])
        ref_line.set_data(run["ref_horizon_x"][frame], run["ref_horizon_y"][frame])
        ego_car.set_xy(_car_corners(run["x"][frame], run["y"][frame], run["psi"][frame]))
        opponent_car.set_xy(_car_corners(
            run["opponent_x"][frame], run["opponent_y"][frame], run["opponent_psi"][frame]
        ))
        gap = progress_delta(run["opponent_s"][frame], run["s"][frame], track.total_length)
        status_text.set_text(
            f"t={run['time'][frame]:.2f} s  gap={gap:+.2f} m\n"
            f"ego v={run['vx'][frame]:.2f} m/s  e={run['e'][frame]:+.3f} m\n"
            f"u=({run['throttle'][frame]:+.2f}, {run['steer'][frame]:+.2f})  "
            f"MPC={run['mpc_status'][frame]}"
        )
        return ego_path, opponent_path, ref_line, ego_car, opponent_car, status_text

    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=max(1, round(1000 / interval))), dpi=dpi)
    plt.close(fig)
    print(f"Saved {mode} GIF to {output_path}")


def _mode_metrics(track, run):
    gaps = np.array([
        progress_delta(opponent_s, ego_s, track.total_length)
        for opponent_s, ego_s in zip(run["opponent_s"], run["s"])
    ])
    return {
        "final_progress": float(run["s"][-1] - run["s"][0]),
        "final_gap": float(gaps[-1]),
        "minimum_abs_gap": float(np.min(np.abs(gaps))),
        "maximum_abs_lateral_error": float(np.max(np.abs(run["e"]))),
        "mpc_failure_rate": float(np.mean(~run["mpc_success"].astype(bool))),
        "recovery_rate": float(np.mean(run["mpc_recovery"].astype(bool))),
    }


def save_mode_report(track, raceline, results, summary_png, output_dir, report_path, animation_stride=3):
    """Save per-mode NPZ/GIF files, a JSON index, and a self-contained HTML report."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "parameter_order": list(PARAMETER_NAMES),
        "parameter_bounds": {name: list(bounds) for name, bounds in PARAMETER_SWEEP_RANGES.items()},
        "modes": {},
    }
    for mode, run in results.items():
        trace_path = output_dir / f"{mode}.npz"
        gif_path = output_dir / f"{mode}.gif"
        np.savez(
            trace_path,
            **run,
            mode=np.asarray(mode),
            params=np.asarray(MODE_PRESETS[mode]),
            track_half_width=np.asarray(params.TRACK_HALF_WIDTH),
            mpc_constraint_half_width=np.asarray(params.MPC_CONSTRAINT_HALF_WIDTH),
            ref_lateral_clip=np.asarray(params.REF_LATERAL_CLIP),
            vehicle_half_length=np.asarray(params.VEHICLE_HALF_LENGTH),
            vehicle_half_width=np.asarray(params.VEHICLE_HALF_WIDTH),
        )
        save_mode_animation(
            track, raceline, mode, run, gif_path, stride=animation_stride
        )
        metrics = _mode_metrics(track, run)
        summary["modes"][mode] = {
            "params": MODE_PRESETS[mode].tolist(),
            "metrics": metrics,
            "trace": str(trace_path),
            "gif": str(gif_path),
        }
        rows.append((mode, gif_path, metrics))

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_uri = "data:image/png;base64," + base64.b64encode(Path(summary_png).read_bytes()).decode("ascii")
    cards = []
    for mode, gif_path, metrics in rows:
        gif_uri = "data:image/gif;base64," + base64.b64encode(gif_path.read_bytes()).decode("ascii")
        metric_rows = "".join(
            f"<tr><th>{html.escape(name.replace('_', ' '))}</th><td>{value:.4f}</td></tr>"
            for name, value in metrics.items()
        )
        cards.append(f"""
        <section><h2>{html.escape(mode)} — {html.escape(str(MODE_PRESETS[mode].tolist()))}</h2>
        <div class="grid"><img src="{gif_uri}" alt="{html.escape(mode)} animation">
        <table>{metric_rows}</table></div></section>""")
    document = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>LLM mode preset comparison</title><style>
    body{{font-family:Arial,sans-serif;background:#f3f5f7;color:#202124;margin:0;padding:24px}}
    main{{max-width:1500px;margin:auto}} section{{background:white;margin:20px 0;padding:18px;border-radius:10px}}
    img{{width:100%;border:1px solid #ddd;border-radius:6px}} .grid{{display:grid;grid-template-columns:3fr 1fr;gap:18px;align-items:start}}
    table{{width:100%;border-collapse:collapse}} th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}
    code{{background:#eee;padding:2px 5px}} @media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
    </style></head><body><main><h1>LLM mode preset comparison</h1>
    <p>Intermediate traces: <code>{html.escape(str(output_dir))}</code>; summary JSON: <code>{html.escape(str(json_path))}</code></p>
    <section><h2>Combined static summary</h2><img src="{summary_uri}" alt="Combined mode summary"></section>
    {''.join(cards)}</main></body></html>"""
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(document, encoding="utf-8")
    print(f"Saved mode report to {report_path}")
    return report_path


def plot_parameter_sweeps(
    track,
    raceline,
    ego_start,
    opponent_start,
    duration,
    v0,
    opponent_v0,
    stationary_opponent,
    horizon,
    points=5,
    save_path="llm_parameter_sweeps.png",
    show=True,
):
    """Plot five one-at-a-time parameter sweeps using continuous colorbars."""
    if points < 2:
        raise ValueError("parameter sweep requires at least two points")

    ego_s0 = ego_start[3]
    opponent_s0 = opponent_start[3]
    sweep_results = {}
    furthest_s = opponent_s0
    for parameter_idx, name in enumerate(PARAMETER_NAMES):
        low, high = PARAMETER_SWEEP_RANGES[name]
        values = np.linspace(low, high, points)
        runs = []
        for value in values:
            trial_params = PARAMETER_BASELINE.copy()
            trial_params[parameter_idx] = value
            print(
                f"Sweep {name:5s}={value:.4f}: "
                f"q,zeta,s1,s2,s3={trial_params.tolist()}"
            )
            run = simulate_mode(
                track,
                raceline,
                f"sweep_{name}",
                ego_start,
                opponent_start,
                duration,
                v0,
                opponent_v0,
                stationary_opponent,
                horizon,
                ego_params=trial_params,
            )
            runs.append(run)
            furthest_s = max(
                furthest_s,
                float(run["s"].max()),
                float(run["opponent_s"].max()),
            )
        sweep_results[name] = (values, runs)

    view_start = ego_s0 - 0.5
    view_end = max(opponent_s0 + 1.5, furthest_s + 0.5)
    _, center, left, right = _sample_track_band(
        track, view_start, view_end, params.TRACK_HALF_WIDTH
    )
    closed_raceline_x = np.r_[raceline.x, raceline.x[0]]
    closed_raceline_y = np.r_[raceline.y, raceline.y[0]]
    map_x = np.r_[left[:, 0], right[:, 0]]
    map_y = np.r_[left[:, 1], right[:, 1]]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()
    cmap = plt.get_cmap("plasma")
    for ax, name in zip(axes, PARAMETER_NAMES):
        values, runs = sweep_results[name]
        norm = Normalize(vmin=float(values[0]), vmax=float(values[-1]))
        ax.fill(
            np.r_[left[:, 0], right[::-1, 0]],
            np.r_[left[:, 1], right[::-1, 1]],
            color="0.94",
        )
        ax.plot(left[:, 0], left[:, 1], "k-", lw=0.8)
        ax.plot(right[:, 0], right[:, 1], "k-", lw=0.8)
        ax.plot(center[:, 0], center[:, 1], color="0.72", ls=":", lw=0.8)
        ax.plot(
            closed_raceline_x,
            closed_raceline_y,
            color="0.45",
            ls="--",
            lw=1.1,
        )
        for value, run in zip(values, runs):
            color = cmap(norm(float(value)))
            ax.plot(run["x"], run["y"], color=color, lw=2.0)
        # The first opponent rollout is sufficient when stationary; when moving,
        # this gray line provides context without mixing it into the parameter colors.
        ax.plot(
            runs[0]["opponent_x"],
            runs[0]["opponent_y"],
            color="#0B3C8C",
            ls="--",
            lw=1.6,
        )
        ax.scatter(ego_start[0], ego_start[1], color="black", s=25, zorder=5)
        ax.scatter(
            opponent_start[0], opponent_start[1],
            color="black", marker="X", s=55, zorder=5,
        )
        ax.set_title(f"Effect of {name}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(map_x.min() - 0.15, map_x.max() + 0.15)
        ax.set_ylim(map_y.min() - 0.15, map_y.max() + 0.15)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label=name,
            fraction=0.046,
            pad=0.04,
        )

    axes[-1].axis("off")
    axes[-1].text(
        0.03,
        0.95,
        "One-at-a-time sweep\n"
        f"baseline [q, zeta, s1, s2, s3]\n{PARAMETER_BASELINE.tolist()}\n\n"
        "solid colored: ego\n"
        "dark-blue dashed: opponent\n"
        "gray dashed track curve: raceline",
        va="top",
        fontsize=11,
    )
    fig.suptitle("alpha-RACER policy-parameter trajectory sensitivity")
    fig.tight_layout()
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved parameter sweep figure to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="berlin", choices=("berlin", "modena", "rounded_rectangle"))
    parser.add_argument("--duration", type=float, default=3.0, help="rollout duration in seconds")
    parser.add_argument("--gap", type=float, default=2.0, help="initial distance to opponent [m]")
    parser.add_argument("--v0", type=float, default=2.0, help="ego initial speed [m/s]")
    parser.add_argument(
        "--opponent-v0", type=float, default=1.0, help="opponent initial speed [m/s]"
    )
    parser.add_argument(
        "--stationary-opponent",
        action="store_true",
        help="freeze opponent at its raceline start instead of running its policy",
    )
    parser.add_argument("--horizon", type=int, default=params.MPC_HORIZON)
    parser.add_argument("--curve-s", type=float, default=None, help="opponent progress; default finds a sharp curve")
    parser.add_argument("--save", default="llm_modes_comparison.png")
    parser.add_argument("--report", default="llm_modes_report.html")
    parser.add_argument("--output-dir", default="llm_mode_report_assets")
    parser.add_argument("--animation-stride", type=int, default=3)
    parser.add_argument(
        "--no-report", action="store_true",
        help="skip per-mode NPZ/GIF files and the self-contained HTML report",
    )
    parser.add_argument(
        "--parameter-sweep",
        action="store_true",
        help="plot one-at-a-time q/zeta/s1/s2/s3 trajectory sweeps instead of modes",
    )
    parser.add_argument("--sweep-points", type=int, default=5)
    parser.add_argument("--sweep-save", default="llm_parameter_sweeps.png")
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
    # A common centerline start isolates interaction-mode effects. Starting on
    # different sides of a clipped raceline can make the cars already separated
    # laterally and therefore fails to exercise the overtake preset.
    ego_start = centerline_pose_at_track_s(track, ego_s0)
    opponent_start = centerline_pose_at_track_s(track, opponent_s)
    print(
        f"Scenario: track={args.track}, ego_s={ego_start[3]:.3f}, "
        f"opponent_s={opponent_start[3]:.3f}, ego_e={ego_start[4]:+.3f}, "
        f"opponent_e={opponent_start[4]:+.3f}, both_on_centerline=True, gap={args.gap:.2f} m, "
        f"opponent_motion={'stationary' if args.stationary_opponent else 'moving'}, "
        f"opponent_v0={0.0 if args.stationary_opponent else args.opponent_v0:.2f}, "
        f"opponent_params={OPPONENT_PARAMS.tolist()}, "
        f"curve={_curvature(track, opponent_s):.3f} 1/m"
    )
    if args.parameter_sweep:
        plot_parameter_sweeps(
            track,
            raceline,
            ego_start,
            opponent_start,
            args.duration,
            args.v0,
            0.0 if args.stationary_opponent else args.opponent_v0,
            args.stationary_opponent,
            args.horizon,
            points=args.sweep_points,
            save_path=args.sweep_save,
            show=not args.no_show,
        )
        return

    results = {}
    for mode in MODE_PRESETS:
        print(f"Running {mode:8s}: q,zeta,s1,s2,s3={MODE_PRESETS[mode].tolist()}")
        results[mode] = simulate_mode(
            track, raceline, mode, ego_start, opponent_start,
            args.duration,
            args.v0,
            0.0 if args.stationary_opponent else args.opponent_v0,
            args.stationary_opponent,
            args.horizon,
        )
    plot_results(
        track, raceline, results, ego_start, opponent_start, args.duration,
        save_path=args.save, show=not args.no_show,
    )
    if not args.no_report:
        save_mode_report(
            track,
            raceline,
            results,
            summary_png=args.save,
            output_dir=args.output_dir,
            report_path=args.report,
            animation_stride=args.animation_stride,
        )


if __name__ == "__main__":
    main()
