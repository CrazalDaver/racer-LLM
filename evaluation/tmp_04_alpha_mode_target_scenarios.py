#!/usr/bin/env python3
"""TEMPORARY four-scenario behavioral target range for local alpha-potential.

The scenarios are labelled by the behavior they are intended to expose, not by
an assumed hidden alpha-RACER mode.  Each run saves a full trace, animation,
outcome metrics, and the ego alpha policy's five-parameter trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animate_trace import make_animation
from config import params
from evaluation.mode_preset_audit import Scenario, _representative_locations, validate_scenario
from simulation import run_simulation
from track import Track, configure_track, load_centerline_csv


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


def build_target_scenarios(track: Track) -> list[tuple[str, Scenario]]:
    straight_s, curve_s = _representative_locations(track)
    return [
        (
            "stable",
            Scenario(
                "stable_curve",
                "Ego approaches the sharpest turn; two support cars remain far behind.",
                (curve_s, curve_s - 15.0, curve_s - 25.0),
                (0.0, 0.0, 0.0),
                (2.2, 2.0, 2.0),
                None,
            ),
        ),
        (
            "push",
            Scenario(
                "push_clear_straight",
                "Ego has a clear straight; two support cars remain far behind.",
                (straight_s, straight_s - 15.0, straight_s - 25.0),
                (0.0, 0.0, 0.0),
                (2.0, 2.0, 2.0),
                None,
            ),
        ),
        (
            "overtake",
            Scenario(
                "overtake_slower_front",
                "A slower front car is 3 m ahead with a small lateral offset.",
                (straight_s, straight_s + 3.0, straight_s - 15.0),
                (-0.12, 0.05, 0.0),
                (2.8, 1.6, 2.0),
                1,
            ),
        ),
        (
            "block",
            Scenario(
                "block_faster_rear",
                "A faster rear car is 4 m behind on the opposite lateral side.",
                (straight_s, straight_s - 4.0, straight_s + 15.0),
                (-0.12, 0.18, 0.0),
                (2.4, 3.4, 2.4),
                1,
            ),
        ),
    ]


def summarize(trace, intended_behavior: str, scene: Scenario) -> dict:
    time = np.asarray(trace["time"])
    s = np.asarray(trace["car_px"])
    e = np.asarray(trace["car_py"])
    vx = np.asarray(trace["car_vx"])
    theta = np.asarray(trace["car_policy_params"])[:, 0, :]
    target = scene.target_car
    gap = None if target is None else s[:, 0] - s[:, target]
    initial_lead = None if gap is None else float(gap[0])
    final_lead = None if gap is None else float(gap[-1])
    passed = None
    if gap is not None:
        passed = bool(np.any(np.signbit(gap[1:]) != np.signbit(gap[:-1])))

    ref_ot = np.asarray(trace.get("car_ref_p_ot_front", []), dtype=float)
    ref_bl = np.asarray(trace.get("car_ref_p_bl", []), dtype=float)
    return {
        "intended_behavior": intended_behavior,
        "scenario": scene.name,
        "description": scene.description,
        "progress_m": float(s[-1, 0] - s[0, 0]),
        "final_speed_mps": float(vx[-1, 0]),
        "max_abs_lateral_m": float(np.max(np.abs(e[:, 0]))),
        "collision": bool(np.any(trace["collision"])),
        "offroad": bool(np.any(np.asarray(trace["car_offroad"])[:, 0])),
        "boundary_fraction": float(np.mean(np.asarray(trace["car_track_boundary_violation"])[:, 0])),
        "solver_failure_fraction": float(np.mean(~np.asarray(trace["car_mpc_solver_success"])[:, 0])),
        "initial_target_lead_m": initial_lead,
        "final_target_lead_m": final_lead,
        "relative_gain_m": None if gap is None else float(gap[-1] - gap[0]),
        "target_crossing_occurred": passed,
        "minimum_abs_target_gap_m": None if gap is None else float(np.min(np.abs(gap))),
        "max_abs_overtake_reference_m": float(np.nanmax(np.abs(ref_ot[:, 0]))) if ref_ot.size else 0.0,
        "max_abs_block_reference_m": float(np.nanmax(np.abs(ref_bl[:, 0]))) if ref_bl.size else 0.0,
        "initial_params": theta[0].tolist(),
        "final_params": theta[-1].tolist(),
        "mean_params": np.mean(theta, axis=0).tolist(),
        "duration_seconds": float(time[-1] - time[0]),
    }


def opponent_parameters(intended_behavior: str) -> np.ndarray:
    """Keep the target car's speed role persistent during the short audit."""
    stable = np.asarray(params.LLM_MODE_PRESETS["stable"], dtype=float)
    per_car = np.vstack([stable, stable, stable])
    if intended_behavior == "overtake":
        per_car[1, 1] = params.ALPHARACER_PARAMETER_BOUNDS[1, 0]
    elif intended_behavior == "block":
        per_car[1, 1] = params.ALPHARACER_PARAMETER_BOUNDS[1, 1]
    return per_car


def plot_parameters(output: Path, traces: list[tuple[str, dict]]):
    bounds = np.asarray(params.ALPHARACER_PARAMETER_BOUNDS, dtype=float)
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for behavior, trace in traces:
        time = np.asarray(trace["time"])
        theta = np.asarray(trace["car_policy_params"])[:, 0, :]
        for dim, axis in enumerate(axes):
            axis.plot(time, theta[:, dim], linewidth=1.5, label=behavior)
    for dim, axis in enumerate(axes):
        axis.set_ylabel(PARAMETER_NAMES[dim], rotation=0, labelpad=18)
        axis.set_ylim(bounds[dim])
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.38), frameon=False)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_interactions(output: Path, traces: list[tuple[str, dict]], scenes: dict[str, Scenario]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (behavior, trace) in zip(axes.ravel(), traces):
        time = np.asarray(trace["time"])
        s = np.asarray(trace["car_px"])
        e = np.asarray(trace["car_py"])
        scene = scenes[behavior]
        if scene.target_car is None:
            axis.plot(time, np.asarray(trace["car_vx"])[:, 0], label="ego speed")
            axis.set_ylabel("Speed (m/s)")
        else:
            target = scene.target_car
            axis.plot(time, s[:, 0] - s[:, target], label="ego longitudinal lead")
            axis.plot(time, e[:, 0] - e[:, target], label="ego lateral offset")
            axis.axhline(0.0, color="0.4", linewidth=0.8)
            axis.set_ylabel("Relative position (m)")
        axis.set_title(behavior)
        axis.set_xlabel("Simulation time (s)")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary local alpha behavioral target-range audit.")
    parser.add_argument("--track", default="berlin")
    parser.add_argument("--weight", type=Path, default=Path("train/weights/alpha_racer/potential_local.pth"))
    parser.add_argument("--total-time", type=float, default=6.0)
    parser.add_argument("--output-dir", type=Path, default=Path("result/tmp_04_alpha_mode_target_scenarios"))
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--gif-stride", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    cases = build_target_scenarios(track)
    for _, scene in cases:
        validate_scenario(track, scene)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params.DEADLOCK_RESET_ENABLED = False
    params.LOW_SPEED_BOUNDARY_RECOVERY_ENABLED = False
    rows = []
    traces = []
    old_output = params.TRACE_OUTPUT
    try:
        for index, (behavior, scene) in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {behavior}: {scene.name}", flush=True)
            trace_path = args.output_dir / f"{behavior}.npz"
            gif_path = args.output_dir / f"{behavior}.gif"
            params.TRACE_OUTPUT = str(trace_path)
            ranks = np.arange(3, dtype=float)
            offsets = np.asarray(scene.s) + ranks * 6.0
            trace = run_simulation(
                track_name=args.track,
                save_trace=True,
                show_plot=False,
                num_cars=3,
                policy_names=["alpharacer", "static", "static"],
                static_policy_params_per_car=opponent_parameters(behavior),
                alpharacer_weight_path=args.weight,
                ego_position=1,
                start_gap=6.0,
                initial_longitudinal_offsets=offsets,
                initial_lateral_offsets=np.asarray(scene.e),
                initial_speeds=np.asarray(scene.speed),
                total_time=args.total_time,
                horizon=args.mpc_horizon,
                velocity_track_weight=args.velocity_track_weight,
                track_slack_weight=args.track_slack_weight,
                quiet=True,
            )
            rows.append(summarize(trace, behavior, scene))
            traces.append((behavior, trace))
            make_animation(trace_path, stride=args.gif_stride, interval=80, save_path=gif_path, dpi=65)
    finally:
        params.TRACE_OUTPUT = old_output

    payload = {
        "scope": "Four behavior-labelled target scenarios; labels are desired effects, not alpha ground truth.",
        "track": args.track,
        "weight": str(args.weight),
        "total_time": args.total_time,
        "scenarios": {behavior: asdict(scene) for behavior, scene in cases},
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    plot_parameters(args.output_dir / "parameter_time_series.svg", traces)
    plot_interactions(args.output_dir / "behavior_time_series.svg", traces, dict(cases))
    print(f"Saved target-range audit to {args.output_dir}")


if __name__ == "__main__":
    main()
