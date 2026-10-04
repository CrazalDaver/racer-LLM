#!/usr/bin/env python3
"""Scan all four fixed modes from matched controlled initial conditions.

The script measures whether the current mode library produces meaningfully
different closed-loop outcomes.  It is intentionally independent of the LLM:
the ego holds one mode for the complete rollout, while every opponent holds a
configurable fixed high-level mode and continues to react through its MPC.

This is a mode-separation audit, not yet an estimate of switching value.  Its
output identifies effect-bearing scenarios that are suitable for a later
fixed-mode versus switching-policy comparison.
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


def load_scenario_bank(args) -> tuple[Track, list[Scenario]]:
    """Load the selected track, build its controlled scenes, and validate them."""
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    scenarios = build_scenarios(track)

    if args.scenario:
        requested = set(args.scenario)
        scenarios = [scene for scene in scenarios if scene.name in requested]
        missing = requested - {scene.name for scene in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")

    for scene in scenarios:
        validate_scenario(track, scene)
    return track, scenarios


def absolute_start_offsets(scene: Scenario, bookkeeping_gap: float) -> np.ndarray:
    """Convert desired absolute Frenet positions to run_simulation offsets."""
    ranks = np.arange(len(scene.s), dtype=float)
    return np.asarray(scene.s, dtype=float) + ranks * bookkeeping_gap


def fixed_mode_parameters(
    scene: Scenario,
    ego_mode: str,
    opponent_mode: str,
) -> np.ndarray:
    """Return one preset row per car: ego first, then fixed-mode opponents."""
    ego = np.asarray(params.LLM_MODE_PRESETS[ego_mode], dtype=float)
    opponent = np.asarray(params.LLM_MODE_PRESETS[opponent_mode], dtype=float)
    return np.vstack([ego] + [opponent] * (len(scene.s) - 1))


def branch_trace_path(
    output_dir: Path,
    scene: Scenario,
    mode: str,
    horizon_seconds: float,
) -> Path:
    """Create a stable, human-readable filename for an optional branch trace."""
    horizon_label = f"{horizon_seconds:g}".replace(".", "p")
    return output_dir / "traces" / f"{scene.name}_{mode}_{horizon_label}s.npz"


def build_branch_config(
    args,
    scene: Scenario,
    ego_mode: str,
    horizon_seconds: float,
) -> SimulationConfig:
    """Build the complete simulator configuration for one matched branch."""
    trace_path = branch_trace_path(
        args.output_dir, scene, ego_mode, horizon_seconds
    )
    if args.save_traces:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    return SimulationConfig(
        track_name=args.track,
        raceline=(str(args.raceline_csv) if args.raceline_csv else None),
        policies=["static_alpharacer"] * len(scene.s),
        static_params_by_car=fixed_mode_parameters(
            scene, ego_mode, args.opponent_mode
        ),
        ego_position=1,
        start_gap=args.start_gap,
        start_s_offsets=absolute_start_offsets(scene, args.start_gap),
        start_e_offsets=np.asarray(scene.e, dtype=float),
        start_speeds=np.asarray(scene.speed, dtype=float),
        duration=horizon_seconds,
        random_seed=args.random_seed,
        mpc_horizon=args.mpc_horizon,
        mpc_velocity_weight=args.velocity_track_weight,
        mpc_track_slack=args.track_slack_weight,
        save_trace=args.save_traces,
        trace_output=trace_path if args.save_traces else None,
        show_plot=False,
        quiet=True,
    )


def _trace_array(trace: dict, key: str, dtype=float) -> np.ndarray:
    """Read one required trace field as a NumPy array with a useful error."""
    if key not in trace:
        raise KeyError(f"Simulation trace is missing required field {key!r}")
    return np.asarray(trace[key], dtype=dtype)


def summarize_branch(
    trace: dict,
    scene: Scenario,
    mode: str,
    opponent_mode: str,
    horizon_seconds: float,
    trace_path: Path | None,
) -> dict:
    """Reduce a full trajectory to safety, progress, and interaction metrics."""
    progress_s = _trace_array(trace, "car_px")
    lateral_e = _trace_array(trace, "car_py")
    speed = _trace_array(trace, "car_vx")
    collision = _trace_array(trace, "collision", bool)
    offroad = _trace_array(trace, "car_offroad", bool)
    boundary = _trace_array(trace, "car_track_boundary_violation", bool)
    solver_success = _trace_array(trace, "car_mpc_solver_success", bool)
    max_track_slack = _trace_array(trace, "car_mpc_max_track_slack")

    gains = progress_s[-1] - progress_s[0]
    ego_progress = float(gains[0])
    relative_gain = None
    if scene.target_car is not None:
        relative_gain = float(gains[0] - gains[scene.target_car])

    physical_collision = bool(np.any(collision))
    ego_offroad = bool(np.any(offroad[:, 0]))
    solver_failure_fraction = float(np.mean(~solver_success[:, 0]))
    physically_safe = not physical_collision and not ego_offroad
    eligible = physically_safe and solver_failure_fraction == 0.0

    # With traffic, relative progress is the primary comparison.  In a
    # single-car scene, absolute progress is the only meaningful progress score.
    primary_metric = relative_gain if relative_gain is not None else ego_progress
    return {
        "scenario": scene.name,
        "scenario_description": scene.description,
        "horizon_seconds": float(horizon_seconds),
        "mode": mode,
        "mode_params": [
            float(value) for value in params.LLM_MODE_PRESETS[mode]
        ],
        "opponent_mode": None if len(scene.s) == 1 else opponent_mode,
        "primary_metric_name": (
            "target_relative_gain_m"
            if relative_gain is not None
            else "ego_progress_m"
        ),
        "primary_metric_m": float(primary_metric),
        "ego_progress_m": ego_progress,
        "target_relative_gain_m": relative_gain,
        "final_speed_mps": float(speed[-1, 0]),
        "max_abs_lateral_m": float(np.max(np.abs(lateral_e[:, 0]))),
        "boundary_fraction": float(np.mean(boundary[:, 0])),
        "physical_collision": physical_collision,
        "ego_offroad": ego_offroad,
        "physically_safe": physically_safe,
        "solver_failure_fraction": solver_failure_fraction,
        "eligible_for_ranking": eligible,
        "max_track_slack_m": float(np.nanmax(max_track_slack[:, 0])),
        "trace_path": str(trace_path) if trace_path is not None else None,
    }


def run_mode_branch(
    args,
    scene: Scenario,
    mode: str,
    horizon_seconds: float,
) -> dict:
    """Run and summarize one ego-mode branch without changing online policies."""
    config = build_branch_config(args, scene, mode, horizon_seconds)
    trace = run_simulation(config)
    return summarize_branch(
        trace,
        scene,
        mode,
        args.opponent_mode,
        horizon_seconds,
        Path(config.trace_output) if config.trace_output else None,
    )


def run_scan(args, scenarios: list[Scenario]) -> list[dict]:
    """Execute every scenario x horizon x mode branch in a fixed order."""
    total = len(scenarios) * len(args.horizon) * len(MODES)
    rows = []
    for scene in scenarios:
        for horizon_seconds in args.horizon:
            for mode in MODES:
                print(
                    f"[{len(rows) + 1}/{total}] {scene.name}: "
                    f"{mode}, {horizon_seconds:g} s"
                )
                rows.append(
                    run_mode_branch(args, scene, mode, horizon_seconds)
                )
    return rows


def compare_modes(rows: list[dict]) -> list[dict]:
    """Rank safe branches and quantify mode separation per scene and horizon."""
    groups = sorted(
        {(row["scenario"], row["horizon_seconds"]) for row in rows}
    )
    comparisons = []
    for scenario, horizon_seconds in groups:
        group = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["horizon_seconds"] == horizon_seconds
        ]
        eligible = [row for row in group if row["eligible_for_ranking"]]
        ranked = sorted(
            eligible, key=lambda row: row["primary_metric_m"], reverse=True
        )
        best = ranked[0] if ranked else None
        worst = ranked[-1] if ranked else None
        best_modes = (
            [
                row["mode"]
                for row in ranked
                if best["primary_metric_m"] - row["primary_metric_m"] <= 1.0e-9
            ]
            if best is not None
            else []
        )
        spread = (
            float(best["primary_metric_m"] - worst["primary_metric_m"])
            if best is not None and worst is not None
            else None
        )
        comparisons.append({
            "scenario": scenario,
            "horizon_seconds": float(horizon_seconds),
            "primary_metric_name": group[0]["primary_metric_name"],
            "best_eligible_mode": best["mode"] if best else None,
            "best_eligible_modes_within_tolerance": best_modes,
            "best_eligible_metric_m": (
                float(best["primary_metric_m"]) if best else None
            ),
            "worst_eligible_mode": worst["mode"] if worst else None,
            "worst_eligible_metric_m": (
                float(worst["primary_metric_m"]) if worst else None
            ),
            "eligible_mode_count": len(eligible),
            "unsafe_mode_count": sum(not row["physically_safe"] for row in group),
            "physical_safety_separates_modes": 0 < sum(
                not row["physically_safe"] for row in group
            ) < len(group),
            "solver_failure_mode_count": sum(
                row["solver_failure_fraction"] > 0.0 for row in group
            ),
            "mode_spread_m": spread,
            "ranking_best_to_worst": [row["mode"] for row in ranked],
        })
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of flat dictionaries, encoding nested values as JSON."""
    encoded = []
    for row in rows:
        encoded.append({
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(encoded[0]))
        writer.writeheader()
        writer.writerows(encoded)


def plot_scan(path: Path, rows: list[dict], comparisons: list[dict]) -> None:
    """Plot primary metric by mode and the resulting safe-mode spread."""
    groups = [
        (row["scenario"], row["horizon_seconds"])
        for row in comparisons
    ]
    values = np.full((len(groups), len(MODES)), np.nan)
    eligible = np.zeros_like(values, dtype=bool)
    for row in rows:
        i = groups.index((row["scenario"], row["horizon_seconds"]))
        j = MODES.index(row["mode"])
        values[i, j] = row["primary_metric_m"]
        eligible[i, j] = row["eligible_for_ranking"]

    # Color encodes within-row loss relative to the best eligible mode.  The
    # annotations retain the raw primary metric.  This makes 1 s and 3 s rows,
    # and absolute versus relative progress rows, visually comparable.
    relative_to_best = np.full_like(values, np.nan)
    for i in range(len(groups)):
        if np.any(eligible[i]):
            best = float(np.max(values[i, eligible[i]]))
            relative_to_best[i] = values[i] - best

    labels = [f"{scene}\n{horizon:g} s" for scene, horizon in groups]
    spreads = np.asarray([
        np.nan if row["mode_spread_m"] is None else row["mode_spread_m"]
        for row in comparisons
    ])
    fig_height = max(5.0, 0.55 * len(groups) + 1.8)
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, fig_height),
        gridspec_kw={"width_ratios": (3.2, 1.2)}, constrained_layout=True,
    )

    finite = relative_to_best[np.isfinite(relative_to_best)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 0.05)
    image = axes[0].imshow(
        relative_to_best, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=0.0
    )
    axes[0].set_xticks(np.arange(len(MODES)), MODES)
    axes[0].set_yticks(np.arange(len(groups)), labels, fontsize=8)
    axes[0].set_title("Primary progress metric by fixed ego mode")
    for i in range(len(groups)):
        for j in range(len(MODES)):
            suffix = "" if eligible[i, j] else " !"
            axes[0].text(
                j, i, f"{values[i, j]:+.2f}{suffix}",
                ha="center", va="center", fontsize=8,
            )
    fig.colorbar(
        image,
        ax=axes[0],
        fraction=0.04,
        label="primary metric minus best eligible mode (m)",
    )

    y = np.arange(len(groups))
    axes[1].barh(y, spreads, color="#4c78a8")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("best - worst eligible mode (m)")
    axes[1].set_title("Mode separation")
    axes[1].grid(axis="x", alpha=0.25)
    for i, value in enumerate(spreads):
        if np.isfinite(value):
            axes[1].text(value, i, f" {value:.2f}", va="center", fontsize=8)

    fig.suptitle("Four-mode library scan (! = unsafe or solver failure)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(
    args,
    scenarios: list[Scenario],
    rows: list[dict],
    comparisons: list[dict],
) -> None:
    """Write machine-readable results and one compact visual summary."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": (
            "Fixed-mode separation audit only; this does not estimate the "
            "value of an online switching policy."
        ),
        "track": args.track,
        "track_csv": params.TRACK_CSV,
        "raceline_csv": (
            str(args.raceline_csv)
            if args.raceline_csv is not None
            else params.RACELINE_CSV
        ),
        "horizons_seconds": [float(value) for value in args.horizon],
        "opponent_model": (
            f"fixed high-level {args.opponent_mode!r} preset with responsive "
            "closed-loop MPC"
        ),
        "opponent_mode": args.opponent_mode,
        "random_seed": args.random_seed,
        "mpc_horizon": args.mpc_horizon,
        "velocity_track_weight": args.velocity_track_weight,
        "track_slack_weight": args.track_slack_weight,
        "ranking_rule": (
            "Rank collision-free, ego-offroad-free, solver-failure-free branches "
            "by target-relative progress when an opponent is present, otherwise "
            "by ego progress."
        ),
        "mode_presets": {
            mode: [float(value) for value in params.LLM_MODE_PRESETS[mode]]
            for mode in MODES
        },
        "scenarios": [asdict(scene) for scene in scenarios],
        "comparisons": comparisons,
        "rollouts": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "rollouts.csv", rows)
    write_csv(args.output_dir / "mode_comparisons.csv", comparisons)
    plot_scan(args.output_dir / "mode_library_scan.png", rows, comparisons)

    effect_bearing = sum(
        row["physical_safety_separates_modes"]
        or (
            row["mode_spread_m"] is not None
            and row["mode_spread_m"] >= args.informative_spread
        )
        for row in comparisons
    )
    unsafe = sum(not row["physically_safe"] for row in rows)
    print(f"Saved {len(rows)} branches to {args.output_dir}")
    print(
        "Effect-bearing groups (physical safety differs or safe-mode spread "
        f">= {args.informative_spread:g} m): {effect_bearing}/{len(comparisons)}"
    )
    print(f"Physically unsafe branches: {unsafe}/{len(rows)}")


def parse_args():
    """Define and validate the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Scan the four fixed mode presets on matched controlled scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Outputs
  summary.json             Settings, scenes, comparisons, and all rollouts.
  rollouts.csv             One row per scene/horizon/mode branch.
  mode_comparisons.csv     Best mode, ranking, and spread per scene/horizon.
  mode_library_scan.png    Heatmap of outcomes plus mode separation.
  traces/*.npz             Optional full trajectories when --save-traces is set.

Example
  .venv/bin/python evaluation/mode_library_scan.py \\
    --track berlin --horizon 1 --horizon 3 \\
    --output-dir result/mode_library_scan
""",
    )
    parser.add_argument(
        "--track",
        choices=("berlin", "modena", "rounded_rectangle"),
        default="berlin",
        help="Registered track used to construct the controlled scenes.",
    )
    parser.add_argument(
        "--horizon", action="append", type=float, default=None,
        help="Fixed-mode rollout duration in seconds; repeatable (default: 1 and 3).",
    )
    parser.add_argument(
        "--scenario", action="append", default=None,
        help="Run only this named scenario; repeat to select several.",
    )
    parser.add_argument(
        "--opponent-mode", choices=MODES, default="stable",
        help="Fixed high-level mode used by all opponents (default: stable).",
    )
    parser.add_argument(
        "--raceline-csv", type=Path, default=None,
        help="Optional raceline override; the registered raceline is the default.",
    )
    parser.add_argument(
        "--mpc-horizon", type=int, default=params.MPC_HORIZON,
        help=f"MPC horizon in control steps (default: {params.MPC_HORIZON}).",
    )
    parser.add_argument(
        "--velocity-track-weight", type=float,
        default=params.VELOCITY_TRACK_WEIGHT,
        help=(
            "Shared MPC reference-speed tracking weight "
            f"(default: {params.VELOCITY_TRACK_WEIGHT:g})."
        ),
    )
    parser.add_argument(
        "--track-slack-weight", type=float,
        default=params.MPC_TRACK_SLACK_WEIGHT,
        help=(
            "Shared quadratic track-slack weight "
            f"(default: {params.MPC_TRACK_SLACK_WEIGHT:g})."
        ),
    )
    parser.add_argument(
        "--start-gap", type=float, default=6.0,
        help="Bookkeeping gap used to express absolute Frenet starts (default: 6).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0,
        help="Simulator seed shared by every matched branch (default: 0).",
    )
    parser.add_argument(
        "--informative-spread", type=float, default=0.05,
        help="Spread threshold used only in the printed summary (default: 0.05 m).",
    )
    parser.add_argument(
        "--save-traces", action="store_true",
        help="Save the complete NPZ trajectory for every branch.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("result/mode_library_scan"),
        help="Directory for JSON, CSV, plot, and optional traces.",
    )
    args = parser.parse_args()
    args.horizon = args.horizon or [1.0, 3.0]
    if any(value <= 0.0 for value in args.horizon):
        parser.error("Every --horizon value must be positive")
    if len(set(args.horizon)) != len(args.horizon):
        parser.error("Duplicate --horizon values are not allowed")
    if args.mpc_horizon <= 0 or args.start_gap <= 0.0:
        parser.error("--mpc-horizon and --start-gap must be positive")
    if args.velocity_track_weight < 0.0 or args.track_slack_weight < 0.0:
        parser.error("MPC weights must be non-negative")
    if args.informative_spread < 0.0:
        parser.error("--informative-spread must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    _, scenarios = load_scenario_bank(args)
    rows = run_scan(args, scenarios)
    comparisons = compare_modes(rows)
    write_outputs(args, scenarios, rows, comparisons)


if __name__ == "__main__":
    main()
