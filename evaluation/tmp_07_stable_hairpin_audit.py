#!/usr/bin/env python3
"""TEMPORARY Berlin-hairpin audit for fixed stable candidates.

The ego starts 10 m before the sharp turn around centerline s=64 m.  There are
no opponents and both recovery mechanisms are disabled.  The audit separates
the effect of q from zeta and compares them with the current push preset.

Usage:
  .venv/bin/python evaluation/tmp_07_stable_hairpin_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import params
from simulation import run_simulation
from track import Track, configure_track, load_centerline_csv


START_S = 54.0
ENTRY_S = 61.0
APEX_S = 64.0
EXIT_S = 70.0


def candidates():
    stable = np.asarray(params.LLM_MODE_PRESETS["stable"], dtype=float)
    push = np.asarray(params.LLM_MODE_PRESETS["push"], dtype=float)
    return {
        "current_stable": stable,
        "q2_only": np.asarray([2.0, stable[1], stable[2], stable[3], stable[4]]),
        "zeta09_only": np.asarray([stable[0], 0.9, stable[2], stable[3], stable[4]]),
        "q2_zeta09": np.asarray([2.0, 0.9, stable[2], stable[3], stable[4]]),
        "q2_zeta085": np.asarray([2.0, 0.85, stable[2], stable[3], stable[4]]),
        "push_control": push,
    }


def first_crossing(time, s, threshold):
    indices = np.flatnonzero(s >= threshold)
    if not len(indices):
        return None, None
    i = int(indices[0])
    return float(time[i]), i


def summarize(name, theta, initial_speed, trace):
    time = np.asarray(trace["time"], dtype=float)
    s = np.asarray(trace["car_px"], dtype=float)[:, 0]
    e = np.asarray(trace["car_py"], dtype=float)[:, 0]
    speed = np.asarray(trace["car_vx"], dtype=float)[:, 0]
    entry_time, entry_i = first_crossing(time, s, ENTRY_S)
    apex_time, apex_i = first_crossing(time, s, APEX_S)
    exit_time, exit_i = first_crossing(time, s, EXIT_S)
    segment = (s >= ENTRY_S) & (s <= EXIT_S)
    return {
        "candidate": name,
        "initial_speed_mps": initial_speed,
        "params": theta.tolist(),
        "progress_m": float(s[-1] - s[0]),
        "entry_time_s": entry_time,
        "apex_time_s": apex_time,
        "exit_time_s": exit_time,
        "corner_time_s": None if entry_time is None or exit_time is None else exit_time - entry_time,
        "entry_speed_mps": None if entry_i is None else float(speed[entry_i]),
        "apex_speed_mps": None if apex_i is None else float(speed[apex_i]),
        "exit_speed_mps": None if exit_i is None else float(speed[exit_i]),
        "minimum_corner_speed_mps": float(np.min(speed[segment])) if np.any(segment) else None,
        "max_abs_corner_lateral_m": float(np.max(np.abs(e[segment]))) if np.any(segment) else None,
        "offroad": bool(np.any(np.asarray(trace["car_offroad"])[:, 0])),
        "boundary_fraction": float(np.mean(np.asarray(trace["car_track_boundary_violation"])[:, 0])),
        "solver_failure_fraction": float(np.mean(~np.asarray(trace["car_mpc_solver_success"])[:, 0])),
        "stalled_fraction": float(np.mean(speed < 0.05)),
    }


def plot_hairpin(output, track, traces, initial_speed):
    sample_s = np.linspace(50.0, 74.0, 500)
    center = np.asarray([track.interpolate(float(s))[:2] for s in sample_s])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.plot(center[:, 0], center[:, 1], "k--", linewidth=1.2, label="centerline")
    for name, trace in traces.items():
        s = np.asarray(trace["car_px"])[:, 0]
        mask = (s >= 50.0) & (s <= 74.0)
        ax.plot(np.asarray(trace["car_x"])[mask, 0], np.asarray(trace["car_y"])[mask, 0], label=name)
    for label, s0 in (("entry", ENTRY_S), ("apex", APEX_S), ("exit", EXIT_S)):
        x, y, _ = track.interpolate(s0)
        ax.scatter([x], [y], s=35, color="black", zorder=5)
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Berlin hairpin trajectories, initial speed={initial_speed:.1f} m/s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary stable-preset audit at Berlin s≈64 m.")
    parser.add_argument("--initial-speeds", type=float, nargs="+", default=[3.5, 4.5])
    parser.add_argument("--total-time", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=Path("result/tmp_07_stable_hairpin_audit"))
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_track(params, "berlin")
    params.DEADLOCK_RESET_ENABLED = False
    params.LOW_SPEED_BOUNDARY_RECOVERY_ENABLED = False
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    sample_s, curvature = track.curvature_profile(sample_step=0.1, heading_window=0.5)
    local = (sample_s >= 50.0) & (sample_s <= 75.0)
    peak_i = np.flatnonzero(local)[int(np.argmax(np.abs(curvature[local])))]
    rows = []
    for initial_speed in args.initial_speeds:
        traces = {}
        for name, theta in candidates().items():
            print(f"v0={initial_speed:.1f}: {name} {theta.tolist()}", flush=True)
            trace = run_simulation(
                track_name="berlin",
                policy_names=["static_alpharacer"],
                num_cars=1,
                static_policy_params=theta,
                initial_longitudinal_offsets=[START_S],
                initial_lateral_offsets=[0.0],
                initial_speeds=[initial_speed],
                total_time=args.total_time,
                horizon=args.mpc_horizon,
                velocity_track_weight=args.velocity_track_weight,
                track_slack_weight=args.track_slack_weight,
                save_trace=False,
                show_plot=False,
                quiet=True,
            )
            traces[name] = trace
            rows.append(summarize(name, theta, initial_speed, trace))
        plot_hairpin(args.output_dir / f"hairpin_v{initial_speed:.1f}.svg", track, traces, initial_speed)

    payload = {
        "purpose": "Check whether fixed stable candidates produce safer/smoother hairpin behavior than push.",
        "track": "berlin",
        "start_s_m": START_S,
        "entry_apex_exit_s_m": [ENTRY_S, APEX_S, EXIT_S],
        "measured_peak_curvature": {
            "s_m": float(sample_s[peak_i]),
            "curvature_1pm": float(curvature[peak_i]),
        },
        "recovery_enabled": False,
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
