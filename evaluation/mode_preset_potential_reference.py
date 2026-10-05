#!/usr/bin/env python3
"""Run the locally trained alpha-potential on controlled preset-audit scenes.

This standalone diagnostic does not change any policy, simulator, preset, or
training code.  The potential network expects a three-car 39-D input, so each
one/two-car controlled scene receives fixed distant support cars rather than
the runtime's coincident zero-state padding.  Only the ego executes parameters
selected by the potential; both other cars use the fixed stable preset.
"""

from __future__ import annotations

import argparse
import csv
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

from config import params
from evaluation.mode_preset_audit import (
    MODES,
    Scenario,
    build_scenarios,
    validate_scenario,
)
from simulation import run_simulation
from simulation_core.config import SimulationConfig
from track import Track, configure_track, load_centerline_csv


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


def add_support_cars(scene: Scenario, gap: float) -> Scenario:
    """Return the same diagnostic relation embedded in a three-car scene."""
    s = list(scene.s)
    e = list(scene.e)
    speed = list(scene.speed)
    if len(s) == 1:
        s.extend((s[0] - gap, s[0] - 2.0 * gap))
        e.extend((0.0, 0.0))
        speed.extend((2.0, 2.0))
    elif len(s) == 2:
        # Put the support car on the opposite longitudinal side from the
        # target, leaving the intended ego/target relation unchanged.
        target_is_ahead = s[1] > s[0]
        s.append(s[0] - gap if target_is_ahead else s[0] + gap)
        e.append(0.0)
        speed.append(2.0)
    else:
        raise ValueError(f"{scene.name}: expected a one- or two-car source scene")
    return Scenario(
        name=scene.name,
        description=scene.description,
        s=tuple(s),
        e=tuple(e),
        speed=tuple(speed),
        target_car=scene.target_car,
    )


def normalized_preset_distances(theta: np.ndarray) -> dict[str, float]:
    bounds = np.asarray(params.ALPHARACER_PARAMETER_BOUNDS, dtype=float)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    return {
        mode: float(np.linalg.norm((theta - preset) / span))
        for mode, preset in params.LLM_MODE_PRESETS.items()
    }


def _summarize(trace, scene: Scenario, weight_path: Path):
    s = np.asarray(trace["car_px"], dtype=float)
    e = np.asarray(trace["car_py"], dtype=float)
    vx = np.asarray(trace["car_vx"], dtype=float)
    theta_series = np.asarray(trace["car_policy_params"], dtype=float)[:, 0, :]
    collision = np.asarray(trace["collision"], dtype=bool)
    offroad = np.asarray(trace["car_offroad"], dtype=bool)[:, 0]
    boundary = np.asarray(trace["car_track_boundary_violation"], dtype=bool)[:, 0]
    solver_success = np.asarray(trace["car_mpc_solver_success"], dtype=bool)[:, 0]
    max_track_slack = np.asarray(trace["car_mpc_max_track_slack"], dtype=float)[:, 0]

    progress = s[-1] - s[0]
    target_gain = None
    if scene.target_car is not None:
        target_gain = float(progress[0] - progress[scene.target_car])

    finite_rows = np.all(np.isfinite(theta_series), axis=1)
    theta_series = theta_series[finite_rows]
    if not len(theta_series):
        raise RuntimeError(f"{scene.name}: potential policy produced no finite parameters")
    first_theta = theta_series[0]
    final_theta = theta_series[-1]
    mean_theta = np.mean(theta_series, axis=0)
    min_theta = np.min(theta_series, axis=0)
    max_theta = np.max(theta_series, axis=0)
    distances = normalized_preset_distances(final_theta)

    row = {
        "scenario": scene.name,
        "scenario_description": scene.description,
        "weight_path": str(weight_path),
        "first_params": first_theta.tolist(),
        "final_params": final_theta.tolist(),
        "mean_params": mean_theta.tolist(),
        "min_params": min_theta.tolist(),
        "max_params": max_theta.tolist(),
        "nearest_preset": min(distances, key=distances.get),
        "nearest_preset_normalized_distance": min(distances.values()),
        "preset_distances": distances,
        "ego_progress_m": float(progress[0]),
        "target_relative_gain_m": target_gain,
        "final_speed_mps": float(vx[-1, 0]),
        "max_abs_lateral_m": float(np.max(np.abs(e[:, 0]))),
        "boundary_fraction": float(np.mean(boundary)),
        "physical_collision": bool(np.any(collision)),
        "offroad": bool(np.any(offroad)),
        "solver_failure_fraction": float(np.mean(~solver_success)),
        "max_track_slack_m": float(np.nanmax(max_track_slack)),
    }
    row["unsafe"] = bool(row["physical_collision"] or row["offroad"])
    return row, theta_series


def run_reference(args):
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    scenes = [add_support_cars(scene, args.support_gap) for scene in build_scenarios(track)]
    if args.scenario:
        wanted = set(args.scenario)
        scenes = [scene for scene in scenes if scene.name in wanted]
        missing = wanted - {scene.name for scene in scenes}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")
    for scene in scenes:
        validate_scenario(track, scene)

    params.POTENTIAL_FUNCTION_OPT_STYLE = args.optimization
    params.POTENTIAL_FUNCTION_SEARCH_ITERS = args.search_iterations
    params.ALPHARACER_WEIGHT_SEARCH_ITERS = args.search_iterations
    params.POTENTIAL_FUNCTION_GRADIENT_RATE = args.gradient_rate

    stable = np.asarray(params.LLM_MODE_PRESETS["stable"], dtype=float)
    rows = []
    parameter_traces = []
    time = None
    for index, scene in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] potential reference: {scene.name}")
        ranks = np.arange(3, dtype=float)
        initial_offsets = np.asarray(scene.s, dtype=float) + ranks * args.start_gap
        trace = run_simulation(SimulationConfig(
            track_name=args.track,
            save_trace=False,
            show_plot=False,
            policies=["alpharacer", "static_alpharacer", "static_alpharacer"],
            static_params_by_car=np.vstack([stable, stable, stable]),
            alpha_weight_path=args.weight,
            ego_position=1,
            start_gap=args.start_gap,
            start_s_offsets=initial_offsets,
            start_e_offsets=np.asarray(scene.e, dtype=float),
            start_speeds=np.asarray(scene.speed, dtype=float),
            duration=args.rollout_seconds,
            mpc_horizon=args.mpc_horizon,
            mpc_velocity_weight=args.velocity_track_weight,
            mpc_track_slack=args.track_slack_weight,
            quiet=True,
        ))
        row, theta_series = _summarize(trace, scene, args.weight)
        rows.append(row)
        parameter_traces.append(theta_series)
        if time is None:
            time = np.asarray(trace["time"], dtype=float)[: len(theta_series)]
    return scenes, rows, time, np.asarray(parameter_traces, dtype=float)


def _write_csv(path: Path, rows: list[dict]):
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"first_params", "final_params", "mean_params", "min_params", "max_params", "preset_distances"}
        }
        for field in ("first", "final", "mean", "min", "max"):
            for name, value in zip(PARAMETER_NAMES, row[f"{field}_params"]):
                flat[f"{field}_{name}"] = value
        for mode, value in row["preset_distances"].items():
            flat[f"distance_to_{mode}"] = value
        flat_rows.append(flat)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _plot(path: Path, scenes: list[Scenario], rows: list[dict]):
    bounds = np.asarray(params.ALPHARACER_PARAMETER_BOUNDS, dtype=float)
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    final = np.asarray([row["final_params"] for row in rows], dtype=float)
    normalized = (final - bounds[:, 0]) / span
    distances = np.asarray(
        [[row["preset_distances"][mode] for mode in MODES] for row in rows],
        dtype=float,
    )
    labels = [scene.name.replace("_", "\n") for scene in scenes]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    image = axes[0].imshow(normalized, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(5), PARAMETER_NAMES)
    axes[0].set_yticks(np.arange(len(scenes)), labels, fontsize=8)
    axes[0].set_title("Final potential-selected parameters\n(normalized within allowed bounds)")
    for i in range(len(scenes)):
        for j in range(5):
            axes[0].text(j, i, f"{final[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if normalized[i, j] < 0.35 or normalized[i, j] > 0.75 else "black")
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="normalized value")

    distance_image = axes[1].imshow(distances, aspect="auto", cmap="magma_r")
    axes[1].set_xticks(np.arange(len(MODES)), MODES)
    axes[1].set_yticks(np.arange(len(scenes)), labels, fontsize=8)
    axes[1].set_title("Normalized distance to hand-written presets")
    for i in range(len(scenes)):
        for j in range(len(MODES)):
            axes[1].text(j, i, f"{distances[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(distance_image, ax=axes[1], fraction=0.046, label="Euclidean distance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(args, scenes, rows, time, parameter_traces):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": (
            "Reference only: local alpha-potential parameter choices are not "
            "treated as ground-truth or globally optimal presets."
        ),
        "track": args.track,
        "weight_path": str(args.weight),
        "rollout_seconds": args.rollout_seconds,
        "support_gap_m": args.support_gap,
        "mpc_horizon": args.mpc_horizon,
        "velocity_track_weight": args.velocity_track_weight,
        "track_slack_weight": args.track_slack_weight,
        "optimization": args.optimization,
        "search_iterations": args.search_iterations,
        "gradient_rate": args.gradient_rate,
        "parameter_names": PARAMETER_NAMES,
        "parameter_bounds": np.asarray(params.ALPHARACER_PARAMETER_BOUNDS).tolist(),
        "mode_presets": {
            mode: np.asarray(params.LLM_MODE_PRESETS[mode], dtype=float).tolist()
            for mode in MODES
        },
        "scenarios": [asdict(scene) for scene in scenes],
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "potential_reference.csv", rows)
    np.savez_compressed(
        args.output_dir / "parameter_trajectories.npz",
        time=time,
        scenario_names=np.asarray([scene.name for scene in scenes]),
        parameter_names=np.asarray(PARAMETER_NAMES),
        ego_parameter_trajectories=parameter_traces,
    )
    _plot(args.output_dir / "potential_reference.png", scenes, rows)
    print(f"Saved {len(rows)} potential-reference rollouts to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Observe local alpha-potential choices in controlled three-car scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs
  summary.json                Settings, scenes, outcomes, and parameter summaries.
  potential_reference.csv     One flattened result row per scene.
  parameter_trajectories.npz  Full ego five-parameter time series.
  potential_reference.png     Final parameters and distances to fixed presets.

Example
  .venv/bin/python evaluation/mode_preset_potential_reference.py \\
    --track berlin \\
    --weight policy/alpharacer/train/weights/potential_local.pth \\
    --output-dir result/mode_preset_potential_reference_berlin
""",
    )
    parser.add_argument("--track", choices=("berlin", "modena", "rounded_rectangle"), default="berlin",
                        help="Registered track (default: berlin).")
    parser.add_argument("--weight", type=Path,
                        default=Path("policy/alpharacer/train/weights/potential_local.pth"),
                        help="Local metadata-bearing potential checkpoint.")
    parser.add_argument("--rollout-seconds", type=float, default=3.0,
                        help="Duration of each scene (default: 3 s).")
    parser.add_argument("--scenario", action="append", default=None,
                        help="Run only this scenario; repeat to select several.")
    parser.add_argument("--support-gap", type=float, default=6.0,
                        help="Distance from ego to each fixed support car (default: 6 m).")
    parser.add_argument("--start-gap", type=float, default=6.0,
                        help="Bookkeeping gap used to encode absolute Frenet starts.")
    parser.add_argument("--mpc-horizon", type=int, default=10,
                        help="MPC horizon in 0.1 s steps (default: 10).")
    parser.add_argument("--velocity-track-weight", type=float, default=10.0,
                        help="Shared MPC speed-tracking weight (default: 10).")
    parser.add_argument("--track-slack-weight", type=float, default=30000.0,
                        help="Shared MPC track-slack weight (default: 30000).")
    parser.add_argument("--optimization", choices=("grad", "linear"), default="grad",
                        help="Existing alpha-potential online optimizer (default: grad).")
    parser.add_argument("--search-iterations", type=int, default=6,
                        help="Existing potential optimization iterations per control update.")
    parser.add_argument("--gradient-rate", type=float, default=1.0e-4,
                        help="Step size in normalized parameter coordinates.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("result/mode_preset_potential_reference_berlin"),
                        help="Output directory for JSON, CSV, NPZ, and PNG files.")
    args = parser.parse_args()
    if not args.weight.is_file():
        parser.error(f"weight file does not exist: {args.weight}")
    positive = (args.rollout_seconds, args.support_gap, args.start_gap, args.mpc_horizon,
                args.search_iterations, args.gradient_rate)
    if any(float(value) <= 0.0 for value in positive):
        parser.error("durations, gaps, horizon, iterations, and gradient rate must be positive")
    return args


def main():
    args = parse_args()
    scenes, rows, time, parameter_traces = run_reference(args)
    write_outputs(args, scenes, rows, time, parameter_traces)


if __name__ == "__main__":
    main()
