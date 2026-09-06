#!/usr/bin/env python3
"""TEMPORARY alpha-potential overtake/block probe on a very long straight.

The two opponent plants are scripted constant-velocity bodies.  They ignore
policy controls and remain on their initial lateral lines, while the alpha ego
uses the ordinary local kinematic plant, potential network, reference builder,
and MPC.  The long closed rectangle is only a Track-compatible approximation
to an infinite straight; no tested car reaches a corner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import params
import simulation


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


class ConstantVelocityPlant:
    """Straight-line opponent plant with an externally fixed speed."""

    def __init__(self, base_model, speed):
        self.dt = base_model.dt
        self.p = base_model.p
        self.max_speed = getattr(base_model, "max_speed", None)
        self.speed = float(speed)

    def step(self, state, _throttle, _steer):
        x, y, psi, _vx, _vy, _omega = np.asarray(state, dtype=float)
        return np.asarray(
            [
                x + self.speed * np.cos(psi) * self.dt,
                y + self.speed * np.sin(psi) * self.dt,
                psi,
                self.speed,
                0.0,
                0.0,
            ]
        )


def write_long_straight(output_dir: Path):
    """Write a dense 600 m straight embedded in a distant closed rectangle."""
    bottom_x = np.arange(-300.0, 300.0 + 2.0, 2.0)
    right_y = np.arange(2.0, 120.0 + 2.0, 2.0)
    top_x = np.arange(298.0, -300.0 - 2.0, -2.0)
    left_y = np.arange(118.0, 0.0, -2.0)
    xy = np.vstack(
        [
            np.column_stack([bottom_x, np.zeros_like(bottom_x)]),
            np.column_stack([np.full_like(right_y, 300.0), right_y]),
            np.column_stack([top_x, np.full_like(top_x, 120.0)]),
            np.column_stack([np.full_like(left_y, -300.0), left_y]),
        ]
    )
    centerline = output_dir / "long_straight_centerline.csv"
    raceline = output_dir / "long_straight_raceline.csv"
    np.savetxt(centerline, xy, delimiter=",", header="x_m,y_m", comments="# ")
    np.savetxt(
        raceline,
        # A 4 m/s ego raceline leaves room for a 5 m/s scripted rear vehicle
        # to remain a persistent blocking threat throughout the probe.
        np.column_stack([xy, np.full(len(xy), 4.0)]),
        delimiter=",",
        header="x_m,y_m,vx_mps",
        comments="# ",
    )
    return centerline, raceline


def make_local_animation(trace, output, behavior, controller, stride, interval=80, dpi=90):
    """Animate only the local interaction window instead of the 600 m track."""
    time = np.asarray(trace["time"])
    car_x = np.asarray(trace["car_x"])
    car_y = np.asarray(trace["car_y"])
    frames = np.arange(0, len(time), max(1, int(stride)))
    colors = ("tab:blue", "tab:orange")
    labels = (f"ego ({controller})", "constant-speed target")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhspan(-0.44, 0.44, color="0.90", zorder=0)
    ax.axhline(-0.44, color="0.35", linewidth=1.0)
    ax.axhline(0.44, color="0.35", linewidth=1.0)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    boxes = []
    trails = []
    for idx in range(2):
        box = Rectangle((0, 0), 0.36, 0.24, facecolor=colors[idx], edgecolor="black", label=labels[idx])
        ax.add_patch(box)
        boxes.append(box)
        trail, = ax.plot([], [], color=colors[idx], alpha=0.65, linewidth=1.2)
        trails.append(trail)
    info = ax.text(0.015, 0.96, "", transform=ax.transAxes, va="top")
    ax.set_ylim(-0.72, 0.72)
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral e (m)")
    ax.set_title(f"{controller.title()} {behavior} probe — local view")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.18)

    def update(frame_number):
        i = int(frames[frame_number])
        ego_x = float(car_x[i, 0])
        ax.set_xlim(ego_x - 6.0, ego_x + 8.0)
        for idx in range(2):
            boxes[idx].set_xy((car_x[i, idx] - 0.18, car_y[i, idx] - 0.12))
            begin = max(0, i - 150)
            trails[idx].set_data(car_x[begin : i + 1, idx], car_y[begin : i + 1, idx])
        gap = car_x[i, 0] - car_x[i, 1]
        info.set_text(f"t={time[i]:.2f} s\nego lead={gap:+.2f} m")
        return (*boxes, *trails, info)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    animation.save(str(output), writer=PillowWriter(fps=max(1, round(1000 / interval))), dpi=dpi)
    plt.close(fig)


def run_case(args, behavior, controller, centerline, raceline, fixed_params=None):
    start_s = 100.0
    if behavior == "overtake":
        desired_s = np.asarray([start_s, start_s + 5.0, start_s - 30.0])
        lateral = np.asarray([-0.12, 0.08, 0.0])
        speeds = np.asarray([3.4, 2.2, 3.0])
        target_idx = 1
    else:
        desired_s = np.asarray([start_s, start_s - 3.5, start_s + 30.0])
        lateral = np.asarray([-0.12, 0.18, 0.0])
        speeds = np.asarray([3.2, 5.0, 3.0])
        target_idx = 1

    params.TRACK_CSV = str(centerline)
    params.RACELINE_CSV = str(raceline)
    params.TRACK_SCALE = 1.0
    params.TRACK_HALF_WIDTH = 0.44
    params.TRACK_WIDTH = 0.88
    params.MPC_CONSTRAINT_HALF_WIDTH = 0.44
    params.REF_LATERAL_CLIP = 0.44 - params.REF_BOUNDARY_MARGIN
    params.DEADLOCK_RESET_ENABLED = False
    params.LOW_SPEED_BOUNDARY_RECOVERY_ENABLED = False

    stem = f"{controller}_{behavior}"
    trace_path = args.output_dir / f"{stem}.npz"
    old_trace_output = params.TRACE_OUTPUT
    old_factory = simulation.create_vehicle_model
    created = 0

    def create_vehicle_model(*factory_args, **factory_kwargs):
        nonlocal created
        base = old_factory(*factory_args, **factory_kwargs)
        car_idx = created
        created += 1
        return base if car_idx == 0 else ConstantVelocityPlant(base, speeds[car_idx])

    simulation.create_vehicle_model = create_vehicle_model
    params.TRACE_OUTPUT = str(trace_path)
    try:
        ranks = np.arange(3, dtype=float)
        offsets = desired_s + ranks * 6.0
        trace = simulation.run_simulation(
            track_name=None,
            raceline_csv=str(raceline),
            save_trace=True,
            show_plot=False,
            num_cars=3,
            policy_names=[
                "alpharacer" if controller == "potential" else "static_alpharacer",
                "centerline",
                "centerline",
            ],
            static_policy_params=np.asarray(
                params.LLM_MODE_PRESETS[behavior] if fixed_params is None else fixed_params,
                dtype=float,
            ),
            alpharacer_weight_path=args.weight,
            ego_position=1,
            start_gap=6.0,
            initial_longitudinal_offsets=offsets,
            initial_lateral_offsets=lateral,
            initial_speeds=speeds,
            total_time=args.total_time,
            horizon=args.mpc_horizon,
            velocity_track_weight=args.velocity_track_weight,
            track_slack_weight=args.track_slack_weight,
            quiet=True,
        )
    finally:
        simulation.create_vehicle_model = old_factory
        params.TRACE_OUTPUT = old_trace_output

    if getattr(args, "make_gif", True):
        make_local_animation(
            trace,
            args.output_dir / f"{stem}.gif",
            behavior,
            controller,
            args.gif_stride,
        )
    return trace, target_idx, speeds


def summarize(behavior, controller, trace, target_idx, scripted_speeds):
    time = np.asarray(trace["time"])
    s = np.asarray(trace["car_px"])
    e = np.asarray(trace["car_py"])
    theta = np.asarray(trace["car_policy_params"])[:, 0]
    gap = s[:, 0] - s[:, target_idx]
    crossing_indices = np.flatnonzero(np.signbit(gap[1:]) != np.signbit(gap[:-1])) + 1
    ref_ot = np.asarray(trace["car_ref_p_ot_front"])[:, 0]
    ref_bl = np.asarray(trace["car_ref_p_bl"])[:, 0]
    collision = np.asarray(trace["collision"], dtype=bool)
    return {
        "controller": controller,
        "behavior": behavior,
        "scripted_opponent_speed_mps": float(scripted_speeds[target_idx]),
        "initial_lead_m": float(gap[0]),
        "final_lead_m": float(gap[-1]),
        "relative_gain_m": float(gap[-1] - gap[0]),
        "crossing_times_seconds": time[crossing_indices].tolist(),
        "minimum_abs_gap_m": float(np.min(np.abs(gap))),
        "collision": bool(np.any(collision)),
        "collision_time_seconds": time[np.flatnonzero(collision)[0]].item() if np.any(collision) else None,
        "offroad": bool(np.any(np.asarray(trace["car_offroad"])[:, 0])),
        "boundary_fraction": float(np.mean(np.asarray(trace["car_track_boundary_violation"])[:, 0])),
        "progress_m": float(s[-1, 0] - s[0, 0]),
        "final_speed_mps": float(np.asarray(trace["car_vx"])[-1, 0]),
        "max_abs_lateral_m": float(np.max(np.abs(e[:, 0]))),
        "max_abs_overtake_reference_m": float(np.nanmax(np.abs(ref_ot))),
        "max_abs_block_reference_m": float(np.nanmax(np.abs(ref_bl))),
        "initial_params": theta[0].tolist(),
        "final_params": theta[-1].tolist(),
        "mean_params": np.mean(theta, axis=0).tolist(),
    }


def plot_results(output_dir, traces):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, behavior in enumerate(("overtake", "block")):
        for controller in ("potential", "preset"):
            trace = traces[(controller, behavior)]
            time = np.asarray(trace["time"])
            s = np.asarray(trace["car_px"])
            e = np.asarray(trace["car_py"])
            axes[0, column].plot(time, s[:, 0] - s[:, 1], label=controller)
            axes[1, column].plot(time, e[:, 0], label=f"ego: {controller}")
        axes[0, column].axhline(0.0, color="0.4", linewidth=0.8)
        axes[0, column].set_title(behavior)
        axes[0, column].set_ylabel("Lead (m)")
        axes[1, column].set_ylabel("Lateral e (m)")
        axes[1, column].set_xlabel("Time (s)")
        for axis in axes[:, column]:
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
    fig.savefig(output_dir / "interaction_time_series.svg", bbox_inches="tight")
    plt.close(fig)

    bounds = np.asarray(params.ALPHARACER_PARAMETER_BOUNDS)
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for (controller, behavior), trace in traces.items():
        time = np.asarray(trace["time"])
        theta = np.asarray(trace["car_policy_params"])[:, 0]
        for dim, axis in enumerate(axes):
            axis.plot(time, theta[:, dim], label=f"{controller}: {behavior}", linewidth=1.6)
    for dim, axis in enumerate(axes):
        axis.set_ylabel(PARAMETER_NAMES[dim], rotation=0, labelpad=18)
        axis.set_ylim(bounds[dim])
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(output_dir / "parameter_time_series.svg", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary straight-line alpha overtake/block probe.")
    parser.add_argument("--weight", type=Path, default=Path("train/weights/alpha_racer/potential_local.pth"))
    parser.add_argument("--total-time", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("result/tmp_05_alpha_straight_interaction_probe"))
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--gif-stride", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centerline, raceline = write_long_straight(args.output_dir)
    traces = {}
    rows = []
    for controller in ("potential", "preset"):
        for behavior in ("overtake", "block"):
            print(f"Running {controller} {behavior} straight-line probe", flush=True)
            trace, target_idx, speeds = run_case(args, behavior, controller, centerline, raceline)
            traces[(controller, behavior)] = trace
            rows.append(summarize(behavior, controller, trace, target_idx, speeds))
    plot_results(args.output_dir, traces)
    payload = {
        "scope": "Long-straight probe with scripted constant-velocity opponents; no recovery/reset.",
        "total_time_seconds": args.total_time,
        "weight": str(args.weight),
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved probe to {args.output_dir}")


if __name__ == "__main__":
    main()
