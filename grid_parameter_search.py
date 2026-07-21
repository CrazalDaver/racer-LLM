import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from config import params
from plotting import compute_lap_stats
from simulation import run_simulation


def _parse_float_list(values):
    result = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                result.append(float(part))
    return result


def _parse_int_list(values):
    return [int(value) for value in _parse_float_list(values)]


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(~np.isfinite(values)):
        return np.nan
    return float(np.nanmean(values))


def _safe_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.min(values))


def _finite_or_inf(value):
    return float(value) if np.isfinite(value) else float("inf")


def evaluate_trace(trace_path):
    data = np.load(trace_path, allow_pickle=True)
    time = np.asarray(data["time"], dtype=float)
    track_length = float(np.asarray(data["track_length"]).item())
    px = np.asarray(data["px"], dtype=float)
    py = np.asarray(data["py"], dtype=float)
    ref_py = np.asarray(data["ref_py"], dtype=float)
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    ref_x = np.asarray(data["ref_x"], dtype=float)
    ref_y = np.asarray(data["ref_y"], dtype=float)

    lap_stats = compute_lap_stats(time, px, track_length)
    lap_times = lap_stats["lap_times"]
    completed_laps = len(lap_times)

    finite_ref = np.isfinite(ref_x) & np.isfinite(ref_y)
    world_pos_sq = (x[finite_ref] - ref_x[finite_ref]) ** 2 + (y[finite_ref] - ref_y[finite_ref]) ** 2
    lateral_abs = np.abs(py - ref_py)

    total_progress = completed_laps * track_length + lap_stats.get("current_progress", 0.0)
    elapsed = float(time[-1] - time[0]) if len(time) else np.nan
    estimated_lap_time = np.nan
    if total_progress > 1e-9 and np.isfinite(elapsed):
        estimated_lap_time = elapsed * track_length / total_progress

    return {
        "completed_laps": completed_laps,
        "first_lap_time": float(lap_times[0]) if lap_times else np.nan,
        "best_lap_time": _safe_min(lap_times),
        "estimated_lap_time": estimated_lap_time,
        "current_lap": lap_stats["current_lap"],
        "current_lap_elapsed": lap_stats["current_elapsed"],
        "total_progress_m": float(total_progress),
        "mean_world_pos_loss": _safe_mean(world_pos_sq),
        "mean_world_pos_error": float(np.sqrt(_safe_mean(world_pos_sq))) if np.isfinite(_safe_mean(world_pos_sq)) else np.nan,
        "mean_abs_lateral_ref_error": _safe_mean(lateral_abs),
        "mean_mpc_track_loss": _safe_mean(data["mpc_track_loss"]) if "mpc_track_loss" in data.files else np.nan,
        "mean_mpc_loss": _safe_mean(data["mpc_loss"]) if "mpc_loss" in data.files else np.nan,
        "offroad_rate": _safe_mean(data["offroad"].astype(float)) if "offroad" in data.files else np.nan,
        "steer_saturation_rate": _safe_mean(np.abs(data["steer"]) > 0.95) if "steer" in data.files else np.nan,
        "mean_vx": _safe_mean(data["vx"]) if "vx" in data.files else np.nan,
    }


def _restore_params(snapshot):
    for name, value in snapshot.items():
        setattr(params, name, value)


def _params_snapshot():
    return {
        "VELOCITY_TRACK_WEIGHT": params.VELOCITY_TRACK_WEIGHT,
        "MPC_HORIZON": params.MPC_HORIZON,
        "STATIC_POLICY_PARAMS": params.STATIC_POLICY_PARAMS.copy(),
        "TOTAL_TIME": params.TOTAL_TIME,
        "TRACE_OUTPUT": params.TRACE_OUTPUT,
        "NUM_CARS": params.NUM_CARS,
    }


def run_grid_search(args):
    velocity_weights = _parse_float_list(args.velocity_weights)
    q_values = _parse_float_list(args.q_values)
    horizons = _parse_int_list(args.horizons)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "grid_search_summary.csv"
    json_path = output_dir / "grid_search_summary.json"

    base_snapshot = _params_snapshot()
    rows = []
    combinations = list(itertools.product(velocity_weights, q_values, horizons))

    try:
        for run_idx, (velocity_weight, q_value, horizon) in enumerate(combinations, start=1):
            run_name = f"vw{velocity_weight:g}_q{q_value:g}_h{horizon}"
            trace_path = output_dir / f"{run_idx:03d}_{run_name}.npz"

            params.VELOCITY_TRACK_WEIGHT = float(velocity_weight)
            params.MPC_HORIZON = int(horizon)
            params.TOTAL_TIME = float(args.total_time)
            params.TRACE_OUTPUT = str(trace_path)
            params.NUM_CARS = int(args.num_cars)
            alpha_params = params.STATIC_POLICY_PARAMS.copy()
            alpha_params[0] = float(q_value)
            params.STATIC_POLICY_PARAMS = alpha_params

            print(
                f"[{run_idx}/{len(combinations)}] "
                f"VELOCITY_TRACK_WEIGHT={velocity_weight:g}, q={q_value:g}, horizon={horizon}"
            )
            run_simulation(
                policy_type=args.policy_type,
                save_trace=True,
                show_plot=False,
                num_cars=args.num_cars,
                policy_names=args.policy_names,
                vehicle_model_type=args.vehicle_model,
                alpharacer_params=alpha_params,
                v0=args.v0,
            )

            metrics = evaluate_trace(trace_path)
            row = {
                "run": run_idx,
                "trace_path": str(trace_path),
                "velocity_track_weight": float(velocity_weight),
                "q": float(q_value),
                "horizon": int(horizon),
                **metrics,
            }
            rows.append(row)
            print(
                "  result: "
                f"best_lap={row['best_lap_time']:.3f}, "
                f"est_lap={row['estimated_lap_time']:.3f}, "
                f"pos_loss={row['mean_world_pos_loss']:.4f}, "
                f"offroad={row['offroad_rate']:.3f}"
            )
    finally:
        _restore_params(base_snapshot)

    if rows:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)

    ranked = sorted(
        rows,
        key=lambda row: (
            _finite_or_inf(row["best_lap_time"])
            if np.isfinite(row["best_lap_time"])
            else _finite_or_inf(row["estimated_lap_time"]),
            _finite_or_inf(row["mean_world_pos_loss"]),
        ),
    )
    if ranked:
        best = ranked[0]
        print("\nBest by lap time then positional loss:")
        print(
            f"  q={best['q']}, velocity_weight={best['velocity_track_weight']}, "
            f"horizon={best['horizon']}, best_lap={best['best_lap_time']:.3f}, "
            f"est_lap={best['estimated_lap_time']:.3f}, pos_loss={best['mean_world_pos_loss']:.4f}"
        )
    print(f"\nSaved summary to {summary_path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search AlphaRacer MPC parameters.")
    parser.add_argument("--velocity-weights", nargs="+", default=["0.5", "1.0", "2.0"], help="Velocity loss weights.")
    parser.add_argument("--q-values", nargs="+", default=["1.0", "2.0", "4.0"], help="AlphaRacer q values.")
    parser.add_argument("--horizons", nargs="+", default=["8", "10", "12"], help="MPC horizon values.")
    parser.add_argument("--total-time", type=float, default=25.0, help="Simulation duration in seconds.")
    parser.add_argument("--output-dir", default="result/grid_search", help="Directory for traces and summary files.")
    parser.add_argument("--vehicle-model", choices=["kinematic", "kinetic", "kinetics", "dbm"], default="dbm")
    parser.add_argument("--policy-type", default="alpharacer", help="Policy type used when policy_names is not set.")
    parser.add_argument("--policy-names", nargs="+", default=None, help="Optional per-car policy list.")
    parser.add_argument("--num-cars", type=int, default=1)
    parser.add_argument("--v0", type=float, default=0, help="Initial longitudinal velocity.")
    return parser.parse_args()


def main():
    run_grid_search(parse_args())


if __name__ == "__main__":
    main()
