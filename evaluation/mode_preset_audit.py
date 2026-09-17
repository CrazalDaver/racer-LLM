#!/usr/bin/env python3
"""Audit whether the four hand-written LLM mode presets have distinct effects.

This is deliberately an external evaluation harness.  It does not modify the
online LLM policies or search for new parameters.  For each controlled traffic
scenario it branches from the same initial state, holds one ego preset for a
short rollout, and keeps the optional opponent on the stable preset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collision import oriented_rectangle_overlap_cost
from config import params
from simulation import run_simulation
from simulation_core.config import SimulationConfig
from state import CarState
from track import Track, configure_track, load_centerline_csv


MODES = ("stable", "push", "overtake", "block")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    s: tuple[float, ...]
    e: tuple[float, ...]
    speed: tuple[float, ...]
    target_car: int | None


def _representative_locations(track: Track) -> tuple[float, float]:
    """Choose a quiet straight and the approach to the sharpest turn."""
    sample_s, curvature = track.curvature_profile(sample_step=0.1, heading_window=0.5)
    sample_s = sample_s[:-1]
    curvature = curvature[:-1]
    sharp_s = float(sample_s[int(np.argmax(np.abs(curvature)))])
    curve_approach_s = (sharp_s - 2.0) % track.total_length

    distance_from_sharp = np.abs(
        (sample_s - sharp_s + 0.5 * track.total_length) % track.total_length
        - 0.5 * track.total_length
    )
    candidates = np.where(distance_from_sharp > 10.0)[0]
    straight_idx = candidates[int(np.argmin(np.abs(curvature[candidates])))]
    return float(sample_s[straight_idx]), curve_approach_s


def build_scenarios(track: Track) -> list[Scenario]:
    """Return six controlled one- or two-car diagnostic scenarios."""
    straight_s, curve_s = _representative_locations(track)

    return [
        Scenario(
            "clear_straight",
            "Ego alone on a clear straight.",
            (straight_s,),
            (0.0,),
            (2.2,),
            None,
        ),
        Scenario(
            "clear_curve_approach",
            "Ego alone two metres before the track's sharpest turn.",
            (curve_s,),
            (0.0,),
            (2.2,),
            None,
        ),
        Scenario(
            "closing_on_front",
            "A slower front car is 2 m ahead; closing speed is +0.8 m/s.",
            (straight_s, straight_s + 2.0),
            (0.0, 0.0),
            (2.5, 1.7),
            1,
        ),
        Scenario(
            "front_pulling_away",
            "A faster front car is 2 m ahead; closing speed is -0.8 m/s.",
            (straight_s, straight_s + 2.0),
            (0.0, 0.0),
            (1.7, 2.5),
            1,
        ),
        Scenario(
            "rear_vehicle_closing",
            "A rear car offset to the other side is 2 m behind and closing at +0.8 m/s.",
            (straight_s, straight_s - 2.0),
            (-0.12, 0.12),
            (2.0, 2.8),
            1,
        ),
        Scenario(
            "front_side_vehicle",
            "A front-side car is 1.2 m ahead with a safe lateral separation.",
            (straight_s, straight_s + 1.2),
            (-0.12, 0.12),
            (2.2, 2.0),
            1,
        ),
    ]


def validate_scenario(track: Track, scenario: Scenario):
    """Reject an invalid initial scene before comparing its four branches."""
    car_count = len(scenario.s)
    if not (len(scenario.e) == len(scenario.speed) == car_count):
        raise ValueError(f"{scenario.name}: inconsistent state-vector lengths")
    if scenario.target_car is not None and not 0 <= scenario.target_car < car_count:
        raise ValueError(f"{scenario.name}: invalid target car index")
    if any(abs(value) >= params.TRACK_HALF_WIDTH for value in scenario.e):
        raise ValueError(f"{scenario.name}: a car starts outside the nominal boundary")
    if any(value < 0.0 or value > params.MAX_STATE_SPEED for value in scenario.speed):
        raise ValueError(f"{scenario.name}: an initial speed is outside simulator limits")

    states = []
    for s, e, speed in zip(scenario.s, scenario.e, scenario.speed):
        x, y = track.frenet_to_world(s, e)
        _, _, psi = track.interpolate(s)
        states.append(CarState(x=x, y=y, vx=speed, vy=0.0, psi=psi, omega=0.0))
    for first in range(car_count):
        for second in range(first + 1, car_count):
            overlap = oriented_rectangle_overlap_cost(
                states[first],
                states[second],
                params.VEHICLE_HALF_LENGTH,
                params.VEHICLE_HALF_WIDTH,
            )
            if overlap > 0.0:
                raise ValueError(
                    f"{scenario.name}: cars {first} and {second} overlap initially"
                )


def _as_array(trace, key, dtype=float):
    return np.asarray(trace.get(key, []), dtype=dtype)


def summarize_rollout(trace, scenario: Scenario, mode: str) -> dict:
    s = _as_array(trace, "car_px")
    e = _as_array(trace, "car_py")
    speed = _as_array(trace, "car_vx")
    collision = _as_array(trace, "collision", bool)
    offroad = _as_array(trace, "car_offroad", bool)
    boundary = _as_array(trace, "car_track_boundary_violation", bool)
    solver_success = _as_array(trace, "car_mpc_solver_success", bool)
    max_track_slack = _as_array(trace, "car_mpc_max_track_slack")

    progress = s[-1] - s[0]
    target_gain = None
    if scenario.target_car is not None:
        target_gain = float(progress[0] - progress[scenario.target_car])

    ego_offroad = bool(np.any(offroad[:, 0])) if offroad.size else False
    ego_collision = bool(np.any(collision)) if collision.size else False
    ego_solver_failure_fraction = (
        float(np.mean(~solver_success[:, 0])) if solver_success.size else None
    )
    unsafe = ego_collision or ego_offroad

    return {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "mode": mode,
        "params": [float(value) for value in params.LLM_MODE_PRESETS[mode]],
        "ego_progress_m": float(progress[0]),
        "target_relative_gain_m": target_gain,
        "final_speed_mps": float(speed[-1, 0]),
        "max_abs_lateral_m": float(np.max(np.abs(e[:, 0]))),
        "boundary_fraction": float(np.mean(boundary[:, 0])),
        "physical_collision": ego_collision,
        "offroad": ego_offroad,
        "unsafe": unsafe,
        "solver_failure_fraction": ego_solver_failure_fraction,
        "max_track_slack_m": (
            float(np.nanmax(max_track_slack[:, 0])) if max_track_slack.size else None
        ),
    }


def run_audit(args) -> tuple[list[Scenario], list[dict]]:
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    scenarios = build_scenarios(track)
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")
    for scenario in scenarios:
        validate_scenario(track, scenario)

    # This audit must expose failures rather than allowing a supervisor to
    # repair them after the selected preset has acted.

    stable = np.asarray(params.LLM_MODE_PRESETS["stable"], dtype=float)
    rows = []
    for scenario in scenarios:
        car_count = len(scenario.s)
        # run_simulation assigns starting ranks [0, 1, ...] for ego_position=1.
        # Offsets below cancel the ordinary rank gaps and recover the absolute
        # Frenet positions specified by the scenario.
        ranks = np.arange(car_count, dtype=float)
        initial_offsets = np.asarray(scenario.s) + ranks * args.start_gap
        for mode in MODES:
            print(f"[{len(rows) + 1}/{len(scenarios) * len(MODES)}] {scenario.name}: {mode}")
            per_car_params = np.vstack(
                [np.asarray(params.LLM_MODE_PRESETS[mode], dtype=float)]
                + [stable] * (car_count - 1)
            )
            trace = run_simulation(SimulationConfig(
                track_name=args.track,
                raceline=args.raceline_csv,
                save_trace=False,
                show_plot=False,
                policies=["static_alpharacer"] * car_count,
                static_params_by_car=per_car_params,
                ego_position=1,
                start_gap=args.start_gap,
                start_s_offsets=initial_offsets,
                start_e_offsets=np.asarray(scenario.e, dtype=float),
                start_speeds=np.asarray(scenario.speed, dtype=float),
                duration=args.rollout_seconds,
                mpc_horizon=args.mpc_horizon,
                mpc_velocity_weight=args.velocity_track_weight,
                mpc_track_slack=args.track_slack_weight,
                quiet=True,
            ))
            rows.append(summarize_rollout(trace, scenario, mode))
    return scenarios, rows


def _write_csv(path: Path, rows: list[dict]):
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, scenarios: list[Scenario], rows: list[dict]):
    progress = np.full((len(scenarios), len(MODES)), np.nan)
    relative = np.full_like(progress, np.nan)
    lateral = np.full_like(progress, np.nan)
    unsafe = np.zeros_like(progress)
    for row in rows:
        i = next(idx for idx, item in enumerate(scenarios) if item.name == row["scenario"])
        j = MODES.index(row["mode"])
        progress[i, j] = row["ego_progress_m"]
        if row["target_relative_gain_m"] is not None:
            relative[i, j] = row["target_relative_gain_m"]
        lateral[i, j] = row["max_abs_lateral_m"]
        unsafe[i, j] = float(row["unsafe"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)
    labels = [scenario.name.replace("_", "\n") for scenario in scenarios]
    x = np.arange(len(scenarios))
    width = 0.19
    for j, mode in enumerate(MODES):
        axes[0].bar(x + (j - 1.5) * width, progress[:, j], width, label=mode)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylabel("Ego progress over rollout (m)")
    axes[0].set_title("Absolute progress effect")
    axes[0].legend(ncols=2, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    image = axes[1].imshow(relative, aspect="auto", cmap="RdYlGn")
    axes[1].set_xticks(np.arange(len(MODES)), MODES)
    axes[1].set_yticks(np.arange(len(scenarios)), labels, fontsize=8)
    axes[1].set_title("Gain relative to relevant opponent (m)")
    for i in range(len(scenarios)):
        for j in range(len(MODES)):
            text = "--" if np.isnan(relative[i, j]) else f"{relative[i, j]:.2f}"
            if unsafe[i, j]:
                text += " !"
            axes[1].text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="Ego gain minus target gain")
    lateral_image = axes[2].imshow(lateral, aspect="auto", cmap="Blues")
    axes[2].set_xticks(np.arange(len(MODES)), MODES)
    axes[2].set_yticks(np.arange(len(scenarios)), labels, fontsize=8)
    axes[2].set_title("Maximum absolute lateral position (m)")
    for i in range(len(scenarios)):
        for j in range(len(MODES)):
            text = f"{lateral[i, j]:.2f}"
            if unsafe[i, j]:
                text += " !"
            axes[2].text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(lateral_image, ax=axes[2], fraction=0.046, label="max |e|")
    fig.suptitle("Mode preset audit (! marks collision or off-road)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(args, scenarios: list[Scenario], rows: list[dict]):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": (
            "Diagnostic audit of the current fixed presets; this does not "
            "estimate globally optimal five-parameter values."
        ),
        "track": args.track,
        "raceline_csv": (
            str(args.raceline_csv)
            if args.raceline_csv is not None
            else params.RACELINE_CSV
        ),
        "rollout_seconds": args.rollout_seconds,
        "mpc_horizon": args.mpc_horizon,
        "velocity_track_weight": args.velocity_track_weight,
        "track_slack_weight": args.track_slack_weight,
        "mode_presets": {
            mode: [float(value) for value in params.LLM_MODE_PRESETS[mode]]
            for mode in MODES
        },
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "rollouts": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "rollouts.csv", rows)
    _plot(args.output_dir / "preset_audit.png", scenarios, rows)

    unsafe_count = sum(bool(row["unsafe"]) for row in rows)
    print(f"Saved {len(rows)} rollouts to {args.output_dir}")
    print(f"Unsafe branches: {unsafe_count}/{len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a detachable audit of the four current mode presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs
  summary.json      Full settings, scenarios, presets, and rollout metrics.
  rollouts.csv      One row per scenario/preset branch.
  preset_audit.png  Absolute and opponent-relative progress comparison.

Example
  .venv/bin/python evaluation/mode_preset_audit.py \\
    --track berlin --rollout-seconds 3 --output-dir result/mode_preset_audit
""",
    )
    parser.add_argument(
        "--track",
        choices=["berlin", "modena", "rounded_rectangle"],
        default="berlin",
        help="Registered track used to construct and simulate the scenarios.",
    )
    parser.add_argument(
        "--rollout-seconds", type=float, default=3.0,
        help="Duration of each fixed-mode branch (default: 3 s).",
    )
    parser.add_argument(
        "--raceline-csv", type=Path, default=None,
        help=(
            "Override the registered raceline for this audit without changing "
            "track.py; the selected path is recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--scenario", action="append", default=None,
        help="Run only this scenario; repeat the option to select several.",
    )
    parser.add_argument(
        "--mpc-horizon", type=int, default=10,
        help="MPC horizon in 0.1 s steps (default: 10).",
    )
    parser.add_argument(
        "--velocity-track-weight", type=float, default=10.0,
        help="Shared MPC reference-speed tracking weight (default: 10).",
    )
    parser.add_argument(
        "--track-slack-weight", type=float, default=30000.0,
        help="Shared MPC quadratic track-slack weight (default: 30000).",
    )
    parser.add_argument(
        "--start-gap", type=float, default=6.0,
        help="Bookkeeping gap used when converting absolute scenario s positions.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("result/mode_preset_audit"),
        help="Directory for JSON, CSV, and PNG results.",
    )
    args = parser.parse_args()
    if args.rollout_seconds <= 0 or args.mpc_horizon <= 0 or args.start_gap <= 0:
        parser.error("rollout-seconds, mpc-horizon, and start-gap must be positive")
    return args


def main():
    args = parse_args()
    scenarios, rows = run_audit(args)
    write_outputs(args, scenarios, rows)


if __name__ == "__main__":
    main()
