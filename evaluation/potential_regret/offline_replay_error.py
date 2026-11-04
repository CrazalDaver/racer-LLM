"""Measure one-second offline reconstruction error from standard race traces.

This diagnostic does not optimize an oracle action.  It first answers the
prerequisite question for post-hoc regret: can the recorded LLM branch itself
be reproduced after rebuilding the runtime from trace fields?

The replay uses the recorded vehicle state and high-level parameters at each
LLM decision, plus the previous tick's feasible control and MPC warm start.
All high-level parameters are held fixed over the one-second window, matching
the recorded decision interval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config import params
from simulation import Race
from simulation_core.config import SimulationConfig
from state import CarState


DEFAULT_HORIZON_SECONDS = 1.0
DEFAULT_ACCEPTANCE_METERS = 0.05
DEFAULT_APPROXIMATE_METERS = 0.20


@dataclass(frozen=True)
class WindowError:
    trace: str
    step: int
    time_s: float
    comparator: int
    recorded_relative_gain_m: float
    replay_relative_gain_m: float
    relative_gain_error_m: float
    max_car_progress_error_m: float
    sensitive: bool


def _scalar(trace, key):
    return np.asarray(trace[key]).item()


def _track_name(trace) -> str:
    csv_name = Path(str(_scalar(trace, "track_csv"))).stem.lower()
    aliases = {
        "berlin_2018": "berlin",
        "modena_2019": "modena",
        "rounded_rectangle": "rounded_rectangle",
    }
    try:
        return aliases[csv_name]
    except KeyError as exc:
        raise ValueError(
            f"Cannot map recorded track_csv={csv_name!r} to a registered track"
        ) from exc


def _decision_steps(trace, ego_idx: int) -> np.ndarray:
    counts = np.asarray(trace["car_policy_decision_count"], dtype=int)[:, ego_idx]
    previous = np.r_[0, counts[:-1]]
    return np.flatnonzero(counts > previous)


def _endpoint_progress(trace, step: int) -> np.ndarray:
    progress = np.asarray(trace["car_px"], dtype=float)
    if step < len(progress):
        return progress[step].copy()
    if step == len(progress):
        terminal = np.asarray(trace["terminal_car_progress"], dtype=float)
        if terminal.shape == (progress.shape[1],):
            return terminal.copy()
    raise IndexError(f"No recorded progress state for endpoint step {step}")


def _state_row(trace, step: int) -> list[CarState]:
    names = ("car_x", "car_y", "car_psi", "car_vx", "car_vy", "car_omega")
    values = [np.asarray(trace[name], dtype=float)[step] for name in names]
    return [
        CarState(
            x=values[0][idx],
            y=values[1][idx],
            psi=values[2][idx],
            vx=values[3][idx],
            vy=values[4][idx],
            omega=values[5][idx],
        )
        for idx in range(len(values[0]))
    ]


def _last_feasible_action(trace, before_step: int, car_idx: int) -> np.ndarray:
    if before_step < 0:
        return np.zeros(2, dtype=float)
    success = np.asarray(trace["car_mpc_solver_success"], dtype=bool)
    candidates = np.flatnonzero(success[: before_step + 1, car_idx])
    if not candidates.size:
        return np.zeros(2, dtype=float)
    row = int(candidates[-1])
    return np.array(
        [trace["car_throttle"][row, car_idx], trace["car_steer"][row, car_idx]],
        dtype=float,
    )


def _make_replay_race(trace) -> Race:
    policy_names = np.asarray(trace["policy_names"], dtype=str)
    initial_params = np.asarray(trace["car_policy_params"], dtype=float)[0]
    config = SimulationConfig(
        policies=tuple("static_alpharacer" for _ in policy_names),
        duration=DEFAULT_HORIZON_SECONDS,
        ego_position=int(_scalar(trace, "ego_position")),
        random_seed=int(_scalar(trace, "random_seed")),
        track_name=_track_name(trace),
        track_scale=float(_scalar(trace, "track_scale")),
        raceline=str(_scalar(trace, "raceline_csv")),
        start_gap=float(_scalar(trace, "start_gap")),
        vehicle_model=str(_scalar(trace, "vehicle_model")),
        mpc_formulation=str(_scalar(trace, "mpc_formulation")),
        mpc_horizon=int(_scalar(trace, "mpc_horizon")),
        mpc_min_state_speed=(
            float(_scalar(trace, "mpc_min_state_speed"))
            if "mpc_min_state_speed" in trace.files
            else 0.0
        ),
        mpc_velocity_weight=float(_scalar(trace, "velocity_track_weight")),
        mpc_track_slack=float(_scalar(trace, "mpc_track_slack_weight")),
        mpc_collision_slack=float(_scalar(trace, "mpc_collision_slack_weight")),
        static_params_by_car=initial_params,
        save_trace=False,
        quiet=True,
    )
    return Race(config)


def restore_window_start(
    race: Race,
    trace,
    step: int,
    *,
    joint_params: np.ndarray | None = None,
) -> None:
    states = _state_row(trace, step)
    race.states = states
    race.environment.states = states
    race.environment.progress_s = np.asarray(trace["car_px"], dtype=float)[step].copy()
    race.environment.step_index = None
    race.environment.car_s = None
    race.environment.car_e = None

    theta = (
        np.asarray(trace["car_policy_params"], dtype=float)[step]
        if joint_params is None
        else np.asarray(joint_params, dtype=float)
    )
    previous_step = step - 1
    previous_plans = np.asarray(
        trace["car_mpc_predicted_control_plan"], dtype=float
    )
    previous_states = np.asarray(
        trace["car_mpc_predicted_state_plan"], dtype=float
    )
    for idx, (policy, mpc) in enumerate(zip(race.policies, race.mpcs)):
        policy.params = theta[idx].copy()
        feasible = _last_feasible_action(trace, previous_step, idx)
        policy.last_u = feasible.copy()
        policy.last_feasible_u = feasible.copy()
        if previous_step < 0:
            mpc.last_control_plan = None
            mpc.last_state_plan = None
        else:
            control_plan = previous_plans[previous_step, idx, :, : mpc.horizon]
            state_plan = previous_states[
                previous_step, idx, :, : mpc.horizon + 1
            ]
            mpc.last_control_plan = control_plan.copy()
            mpc.last_state_plan = state_plan.copy()


def rollout_endpoint(
    race: Race,
    trace,
    step: int,
    horizon_steps: int,
    *,
    joint_params: np.ndarray | None = None,
) -> np.ndarray:
    """Restore one recorded decision point and return replayed final progress."""
    restore_window_start(race, trace, step, joint_params=joint_params)
    for offset in range(horizon_steps):
        race.environment.begin_step(int(step + offset))
        observation = race.environment.capture(race.policies)
        actions = tuple(policy.get_action(observation) for policy in race.policies)
        race.environment.step(actions)
    return np.asarray(race.environment.progress_s, dtype=float).copy()


def _is_sensitive_window(trace, start: int, end: int) -> bool:
    collision = np.asarray(trace["collision"], dtype=bool)[start:end]
    recovery = np.asarray(
        trace["car_offroad_recovery_active"], dtype=bool
    )[start:end]
    return bool(np.any(collision) or np.any(recovery))


def replay_trace(path: Path, horizon_seconds: float) -> list[WindowError]:
    with np.load(path, allow_pickle=False) as trace:
        dt = float(_scalar(trace, "simulation_dt")) if "simulation_dt" in trace else params.SIM_DT
        horizon_steps_float = horizon_seconds / dt
        horizon_steps = int(round(horizon_steps_float))
        if not np.isclose(horizon_steps, horizon_steps_float, atol=1e-12, rtol=0.0):
            raise ValueError("Replay horizon must be an integer number of simulation steps")

        ego_idx = int(_scalar(trace, "ego_index"))
        race = _make_replay_race(trace)
        results = []
        for step in _decision_steps(trace, ego_idx):
            end_step = int(step + horizon_steps)
            try:
                recorded_end = _endpoint_progress(trace, end_step)
            except IndexError:
                continue
            start_progress = _endpoint_progress(trace, int(step))
            opponents = [idx for idx in range(len(start_progress)) if idx != ego_idx]
            comparator = max(opponents, key=lambda idx: start_progress[idx])

            replay_end = rollout_endpoint(
                race, trace, int(step), horizon_steps
            )

            recorded_gain = float(
                (recorded_end[ego_idx] - recorded_end[comparator])
                - (start_progress[ego_idx] - start_progress[comparator])
            )
            replay_gain = float(
                (replay_end[ego_idx] - replay_end[comparator])
                - (start_progress[ego_idx] - start_progress[comparator])
            )
            results.append(
                WindowError(
                    trace=str(path),
                    step=int(step),
                    time_s=float(step * dt),
                    comparator=int(comparator),
                    recorded_relative_gain_m=recorded_gain,
                    replay_relative_gain_m=replay_gain,
                    relative_gain_error_m=float(replay_gain - recorded_gain),
                    max_car_progress_error_m=float(
                        np.max(np.abs(replay_end - recorded_end))
                    ),
                    sensitive=_is_sensitive_window(trace, int(step), end_step),
                )
            )
        return results


def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths = []
    for raw in inputs:
        path = Path(raw)
        if path.suffix.lower() == ".npz":
            paths.append(path)
            continue
        if path.name != "summary.json":
            raise ValueError(f"Expected an NPZ trace or summary.json, got {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        trials = summary.get("trials", [])
        for ordinal, trial in enumerate(trials, start=1):
            trial_number = int(trial.get("trial", ordinal))
            candidate = path.parent / "summary_traces" / f"trial_{trial_number:03d}.npz"
            if not candidate.is_file():
                raise FileNotFoundError(f"Trace referenced by summary is missing: {candidate}")
            paths.append(candidate)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trace(s): " + ", ".join(missing))
    return paths


def _summary(rows: list[WindowError], threshold: float, approximate: float) -> dict:
    errors = np.abs([row.relative_gain_error_m for row in rows])
    recorded = np.abs([row.recorded_relative_gain_m for row in rows])
    sensitive = np.asarray([row.sensitive for row in rows], dtype=bool)
    denom = float(np.sum(recorded))
    result = {
        "windows": len(rows),
        "mean_absolute_error_m": float(np.mean(errors)),
        "median_absolute_error_m": float(np.median(errors)),
        "p95_absolute_error_m": float(np.percentile(errors, 95)),
        "max_absolute_error_m": float(np.max(errors)),
        "wape_percent": float(100.0 * np.sum(errors) / denom) if denom else None,
        "within_acceptance_percent": float(100.0 * np.mean(errors <= threshold)),
        "within_approximate_percent": float(100.0 * np.mean(errors <= approximate)),
        "sensitive_windows": int(np.sum(sensitive)),
    }
    for label, mask in (("clean", ~sensitive), ("sensitive", sensitive)):
        selected = errors[mask]
        result[f"{label}_windows"] = int(selected.size)
        result[f"{label}_mean_absolute_error_m"] = (
            float(np.mean(selected)) if selected.size else None
        )
        result[f"{label}_within_acceptance_percent"] = (
            float(100.0 * np.mean(selected <= threshold)) if selected.size else None
        )
    return result


def _print_summary(summary: dict, threshold: float, approximate: float) -> None:
    print("Offline one-second reconstruction error")
    print(f"windows: {summary['windows']}")
    print(
        "absolute relative-gain error: "
        f"mean={summary['mean_absolute_error_m']:.6f} m, "
        f"median={summary['median_absolute_error_m']:.6f} m, "
        f"p95={summary['p95_absolute_error_m']:.6f} m, "
        f"max={summary['max_absolute_error_m']:.6f} m"
    )
    wape = summary["wape_percent"]
    print("relative error (WAPE): " + ("undefined" if wape is None else f"{wape:.3f}%"))
    print(
        f"within {threshold:.3f} m: {summary['within_acceptance_percent']:.1f}%"
    )
    print(
        f"within {approximate:.3f} m: {summary['within_approximate_percent']:.1f}%"
    )
    print(
        f"clean windows: {summary['clean_windows']}, "
        f"mean={summary['clean_mean_absolute_error_m']:.6f} m, "
        f"within {threshold:.3f} m={summary['clean_within_acceptance_percent']:.1f}%"
    )
    if summary["sensitive_windows"]:
        print(
            f"collision/recovery windows: {summary['sensitive_windows']}, "
            f"mean={summary['sensitive_mean_absolute_error_m']:.6f} m, "
            f"within {threshold:.3f} m="
            f"{summary['sensitive_within_acceptance_percent']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure one-second offline replay error from standard traces."
    )
    parser.add_argument("inputs", nargs="+", help="NPZ trace(s) or summary.json")
    parser.add_argument("--horizon-seconds", type=float, default=DEFAULT_HORIZON_SECONDS)
    parser.add_argument("--acceptance-m", type=float, default=DEFAULT_ACCEPTANCE_METERS)
    parser.add_argument("--approximate-m", type=float, default=DEFAULT_APPROXIMATE_METERS)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    rows = [
        row
        for path in paths
        for row in replay_trace(path, horizon_seconds=args.horizon_seconds)
    ]
    if not rows:
        raise SystemExit("No complete LLM decision windows found")
    summary = _summary(rows, args.acceptance_m, args.approximate_m)
    _print_summary(summary, args.acceptance_m, args.approximate_m)
    if args.json_output is not None:
        payload = {
            "metric": "post_hoc_recorded-branch_reconstruction_error",
            "horizon_seconds": args.horizon_seconds,
            "acceptance_m": args.acceptance_m,
            "approximate_m": args.approximate_m,
            "summary": summary,
            "windows": [row.__dict__ for row in rows],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"saved: {args.json_output}")


if __name__ == "__main__":
    main()
