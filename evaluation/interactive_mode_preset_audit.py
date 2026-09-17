#!/usr/bin/env python3
"""Closed-loop semantic audit of the fixed block preset.

Unlike the older constant-velocity probe, the rear challenger here is a normal
local vehicle controlled by the same constrained MPC as the ego.  Its high-level
intent is held at the fixed ``overtake`` preset, but its reference and control
are recomputed from the current ego state every control cycle.  This isolates
whether the ego's block component delays a responsive overtaking attempt.

This is an evaluation harness only.  It does not tune presets or modify online
policies.

Example
-------
    .venv/bin/python evaluation/interactive_mode_preset_audit.py \
        --output-dir result/interactive_mode_preset_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import params
from simulation import run_simulation
from simulation_core.config import SimulationConfig


@dataclass(frozen=True)
class RearChallenge:
    name: str
    gap_m: float
    ego_speed_mps: float
    challenger_speed_mps: float


CHALLENGES = (
    RearChallenge("near_moderate", 1.75, 3.2, 3.8),
    RearChallenge("nominal", 2.50, 3.2, 4.0),
    RearChallenge("far_aggressive", 3.00, 3.2, 4.2),
)
MIRRORS = (-1, 1)
EGO_VARIANTS = ("stable", "push", "no_block_s3_zero", "block")


def _write_long_straight(output_dir: Path) -> tuple[Path, Path]:
    """Embed a 600 m test straight in a distant closed rectangle."""
    bottom_x = np.arange(-300.0, 302.0, 2.0)
    right_y = np.arange(2.0, 122.0, 2.0)
    top_x = np.arange(298.0, -302.0, -2.0)
    left_y = np.arange(118.0, 0.0, -2.0)
    xy = np.vstack((
        np.column_stack((bottom_x, np.zeros_like(bottom_x))),
        np.column_stack((np.full_like(right_y, 300.0), right_y)),
        np.column_stack((top_x, np.full_like(top_x, 120.0))),
        np.column_stack((np.full_like(left_y, -300.0), left_y)),
    ))
    centerline = output_dir / "controlled_straight_centerline.csv"
    raceline = output_dir / "controlled_straight_raceline.csv"
    np.savetxt(centerline, xy, delimiter=",", header="x_m,y_m", comments="# ")
    np.savetxt(
        raceline,
        np.column_stack((xy, np.full(len(xy), 4.0))),
        delimiter=",",
        header="x_m,y_m,vx_mps",
        comments="# ",
    )
    return centerline, raceline


def _variant_params(name: str) -> np.ndarray:
    if name == "no_block_s3_zero":
        theta = np.asarray(params.LLM_MODE_PRESETS["block"], dtype=float).copy()
        theta[4] = 0.0
        return theta
    return np.asarray(params.LLM_MODE_PRESETS[name], dtype=float).copy()


def _first_sustained_time(
    time: np.ndarray,
    condition: np.ndarray,
    duration_seconds: float,
) -> Optional[float]:
    if condition.size == 0:
        return None
    dt = float(np.median(np.diff(time))) if time.size > 1 else params.SIM_DT
    required = max(1, int(math.ceil(duration_seconds / max(dt, 1.0e-9))))
    if condition.size < required:
        return None
    counts = np.convolve(condition.astype(int), np.ones(required, dtype=int), mode="valid")
    found = np.flatnonzero(counts == required)
    return None if not found.size else float(time[int(found[0])])


def _summarize(
    trace: dict,
    challenge: RearChallenge,
    mirror: int,
    variant: str,
    theta: np.ndarray,
    pass_margin_m: float,
    pass_hold_seconds: float,
) -> dict:
    time = np.asarray(trace["time"], dtype=float)
    progress = np.asarray(trace["car_px"], dtype=float)
    lateral = np.asarray(trace["car_py"], dtype=float)
    speed = np.asarray(trace["car_vx"], dtype=float)
    challenger_lead = progress[:, 1] - progress[:, 0]
    pass_time = _first_sustained_time(
        time,
        challenger_lead >= pass_margin_m,
        pass_hold_seconds,
    )
    collision = bool(np.any(np.asarray(trace["collision"], dtype=bool)))
    offroad_by_car = np.any(np.asarray(trace["car_offroad"], dtype=bool), axis=0)
    solver_success = np.asarray(trace["car_mpc_solver_success"], dtype=bool)
    block_reference = np.asarray(trace["car_ref_p_bl"], dtype=float)[:, 0]
    overtake_reference = np.asarray(trace["car_ref_p_ot_front"], dtype=float)[:, 1]
    return {
        "challenge": challenge.name,
        "mirror": "ego_left" if mirror < 0 else "ego_right",
        "ego_variant": variant,
        "ego_params": theta.tolist(),
        "challenger_mode": "overtake",
        "initial_gap_m": challenge.gap_m,
        "initial_ego_speed_mps": challenge.ego_speed_mps,
        "initial_challenger_speed_mps": challenge.challenger_speed_mps,
        "initial_challenger_lead_m": float(challenger_lead[0]),
        "final_challenger_lead_m": float(challenger_lead[-1]),
        "challenger_relative_gain_m": float(challenger_lead[-1] - challenger_lead[0]),
        "ego_progress_m": float(progress[-1, 0] - progress[0, 0]),
        "challenger_progress_m": float(progress[-1, 1] - progress[0, 1]),
        "challenger_passed": pass_time is not None,
        "sustained_pass_time_s": pass_time,
        "minimum_abs_longitudinal_gap_m": float(np.min(np.abs(challenger_lead))),
        "final_ego_speed_mps": float(speed[-1, 0]),
        "final_challenger_speed_mps": float(speed[-1, 1]),
        "max_abs_ego_lateral_m": float(np.max(np.abs(lateral[:, 0]))),
        "max_abs_challenger_lateral_m": float(np.max(np.abs(lateral[:, 1]))),
        "max_abs_ego_block_reference_m": float(np.nanmax(np.abs(block_reference))),
        "max_abs_challenger_overtake_reference_m": float(np.nanmax(np.abs(overtake_reference))),
        "collision": collision,
        "ego_offroad": bool(offroad_by_car[0]),
        "challenger_offroad": bool(offroad_by_car[1]),
        "unsafe": bool(collision or np.any(offroad_by_car)),
        "ego_solver_failure_fraction": float(np.mean(~solver_success[:, 0])),
        "challenger_solver_failure_fraction": float(np.mean(~solver_success[:, 1])),
    }


def _run_case(
    args,
    centerline: Path,
    raceline: Path,
    challenge: RearChallenge,
    mirror: int,
    variant: str,
) -> tuple[dict, dict]:
    start_s = 100.0
    desired_s = np.asarray([start_s, start_s - challenge.gap_m], dtype=float)
    initial_offsets = desired_s + np.arange(2, dtype=float) * args.start_gap
    lateral_magnitude = args.initial_lateral_separation / 2.0
    initial_lateral = np.asarray(
        [mirror * lateral_magnitude, -mirror * lateral_magnitude], dtype=float
    )
    initial_speeds = np.asarray(
        [challenge.ego_speed_mps, challenge.challenger_speed_mps], dtype=float
    )
    ego_theta = _variant_params(variant)
    challenger_theta = np.asarray(params.LLM_MODE_PRESETS["overtake"], dtype=float)

    params.TRACK_CSV = str(centerline)
    params.RACELINE_CSV = str(raceline)
    params.TRACK_SCALE = 1.0
    params.TRACK_HALF_WIDTH = args.track_half_width
    params.TRACK_WIDTH = 2.0 * args.track_half_width
    params.MPC_CONSTRAINT_HALF_WIDTH = args.track_half_width
    params.REF_LATERAL_CLIP = args.track_half_width - params.REF_BOUNDARY_MARGIN

    stem = f"{challenge.name}_{'left' if mirror < 0 else 'right'}_{variant}"
    trace_path = args.output_dir / "traces" / f"{stem}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = run_simulation(SimulationConfig(
            raceline=str(raceline),
            save_trace=True,
            trace_output=trace_path,
            show_plot=False,
            policies=["static_alpharacer", "static_alpharacer"],
            static_params_by_car=np.vstack((ego_theta, challenger_theta)),
            ego_position=1,
            start_gap=args.start_gap,
            start_s_offsets=initial_offsets,
            start_e_offsets=initial_lateral,
            start_speeds=initial_speeds,
            duration=args.total_time,
            mpc_horizon=args.mpc_horizon,
            mpc_velocity_weight=args.velocity_track_weight,
            mpc_track_slack=args.track_slack_weight,
            quiet=True,
        ))
    row = _summarize(
        trace,
        challenge,
        mirror,
        variant,
        ego_theta,
        args.pass_margin,
        args.pass_hold_seconds,
    )
    row["trace"] = str(trace_path)
    return trace, row


def _block_effects(rows: list[dict], total_time: float) -> list[dict]:
    effects = []
    groups = {(row["challenge"], row["mirror"]) for row in rows}
    for challenge, mirror in sorted(groups):
        group = {
            row["ego_variant"]: row
            for row in rows
            if row["challenge"] == challenge and row["mirror"] == mirror
        }
        block = group["block"]
        control = group["no_block_s3_zero"]
        block_time = total_time if block["sustained_pass_time_s"] is None else block["sustained_pass_time_s"]
        control_time = total_time if control["sustained_pass_time_s"] is None else control["sustained_pass_time_s"]
        effects.append({
            "challenge": challenge,
            "mirror": mirror,
            "block_safe": not block["unsafe"],
            "control_safe": not control["unsafe"],
            "block_prevented_pass": bool(
                control["challenger_passed"] and not block["challenger_passed"]
            ),
            "censored_pass_delay_s": float(block_time - control_time),
            "challenger_relative_gain_reduction_m": float(
                control["challenger_relative_gain_m"]
                - block["challenger_relative_gain_m"]
            ),
            "ego_progress_cost_m": float(
                control["ego_progress_m"] - block["ego_progress_m"]
            ),
        })
    return effects


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict]):
    labels = [f"{c.name}\n{side}" for c in CHALLENGES for side in ("ego_left", "ego_right")]
    relative = np.full((len(labels), len(EGO_VARIANTS)), np.nan)
    pass_time = np.full_like(relative, np.nan)
    unsafe = np.zeros_like(relative, dtype=bool)
    for row in rows:
        label = f"{row['challenge']}\n{row['mirror']}"
        i = labels.index(label)
        j = EGO_VARIANTS.index(row["ego_variant"])
        relative[i, j] = row["challenger_relative_gain_m"]
        pass_time[i, j] = (
            row["sustained_pass_time_s"]
            if row["sustained_pass_time_s"] is not None
            else np.nan
        )
        unsafe[i, j] = row["unsafe"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), constrained_layout=True)
    im = axes[0].imshow(relative, aspect="auto", cmap="RdYlGn_r")
    axes[0].set_title("Challenger relative gain (lower is better defense)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="metres")
    axes[1].set_title("Sustained pass time (blank = no completed pass)")
    for axis in axes:
        axis.set_xticks(np.arange(len(EGO_VARIANTS)), EGO_VARIANTS, rotation=25, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(EGO_VARIANTS)):
            suffix = " !" if unsafe[i, j] else ""
            axes[0].text(j, i, f"{relative[i, j]:.2f}{suffix}", ha="center", va="center", fontsize=8)
            text = "--" if np.isnan(pass_time[i, j]) else f"{pass_time[i, j]:.1f}"
            axes[1].text(j, i, f"{text}{suffix}", ha="center", va="center", fontsize=8)
    axes[1].imshow(np.isnan(pass_time), aspect="auto", cmap="Greys", vmin=0, vmax=2)
    fig.suptitle("Interactive block audit; ! marks collision or off-road")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit fixed ego presets against a responsive fixed-overtake challenger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs
  summary.json             Settings, rollout metrics, and block-vs-s3=0 effects.
  rollouts.csv             One row per challenge/mirror/ego-preset branch.
  block_effects.csv        Paired block minus no-block comparisons.
  interactive_audit.png    Relative gain and pass-time visualization.
  traces/*.npz             Full trace for every branch.

Important arguments
  --total-time SECONDS           Duration of every branch (default: 6).
  --pass-margin METERS           Lead required for a completed pass (default: vehicle length).
  --pass-hold-seconds SECONDS    Required sustained lead (default: 0.5).
  --velocity-track-weight W      Shared low-level MPC speed weight (default: 10).
  --track-slack-weight W         Shared track-slack weight (default: 30000).
""",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("result/interactive_mode_preset_audit"))
    parser.add_argument("--total-time", type=float, default=6.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--start-gap", type=float, default=6.0)
    parser.add_argument("--track-half-width", type=float, default=0.44)
    parser.add_argument("--initial-lateral-separation", type=float, default=0.28)
    parser.add_argument("--pass-margin", type=float, default=2.0 * params.VEHICLE_HALF_LENGTH)
    parser.add_argument("--pass-hold-seconds", type=float, default=0.5)
    args = parser.parse_args()
    positive = (
        args.total_time,
        args.mpc_horizon,
        args.start_gap,
        args.track_half_width,
        args.initial_lateral_separation,
        args.pass_margin,
        args.pass_hold_seconds,
    )
    if any(value <= 0 for value in positive):
        parser.error("time, horizon, geometry, gap, and pass thresholds must be positive")
    if args.velocity_track_weight < 0 or args.track_slack_weight < 0:
        parser.error("MPC weights must be non-negative")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centerline, raceline = _write_long_straight(args.output_dir)
    rows = []
    for challenge in CHALLENGES:
        for mirror in MIRRORS:
            for variant in EGO_VARIANTS:
                print(
                    f"[{len(rows) + 1}/{len(CHALLENGES) * len(MIRRORS) * len(EGO_VARIANTS)}] "
                    f"{challenge.name}, mirror={mirror:+d}, ego={variant}",
                    flush=True,
                )
                _, row = _run_case(args, centerline, raceline, challenge, mirror, variant)
                rows.append(row)

    effects = _block_effects(rows, args.total_time)
    payload = {
        "scope": (
            "Controlled closed-loop block audit. Both cars use the shared MPC; "
            "the challenger holds the overtake preset while responding to ego state."
        ),
        "limitations": (
            "This tests block semantics against one fixed overtaking intent; it is "
            "not a globally optimal-mode label or a natural race-frequency estimate."
        ),
        "total_time_seconds": args.total_time,
        "mpc_horizon": args.mpc_horizon,
        "velocity_track_weight": args.velocity_track_weight,
        "track_slack_weight": args.track_slack_weight,
        "pass_margin_m": args.pass_margin,
        "pass_hold_seconds": args.pass_hold_seconds,
        "challenger_params": np.asarray(params.LLM_MODE_PRESETS["overtake"], dtype=float).tolist(),
        "challenges": [asdict(challenge) for challenge in CHALLENGES],
        "rollouts": rows,
        "block_vs_s3_zero": effects,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "rollouts.csv", rows)
    _write_csv(args.output_dir / "block_effects.csv", effects)
    _plot(args.output_dir / "interactive_audit.png", rows)
    safe_effects = sum(effect["block_safe"] for effect in effects)
    prevented = sum(effect["block_prevented_pass"] for effect in effects)
    print(f"Saved {len(rows)} rollouts to {args.output_dir}")
    print(f"Block safe in {safe_effects}/{len(effects)} paired scenes; prevented pass in {prevented}/{len(effects)}")


if __name__ == "__main__":
    main()
