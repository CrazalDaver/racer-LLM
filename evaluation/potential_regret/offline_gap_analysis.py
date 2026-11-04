"""Post-hoc potential-reference gap analysis for standard race traces.

The learned-potential reference is optimized at every recorded LLM decision.
Its parameters are then replayed for one second from the reconstructed runtime.
The reported gap is reference relative-progress gain minus the recorded LLM
relative-progress gain.  It is an exploratory reconstructed reference gap, not
proof of globally optimal control and not an exact online snapshot result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from config import params
from policy.alpharacer.network import ALPHA_PARAMETER_BOUNDS
from evaluation.potential_regret.offline_replay_error import (
    _decision_steps,
    _endpoint_progress,
    _expand_inputs,
    _is_sensitive_window,
    _make_replay_race,
    _scalar,
    restore_window_start,
    rollout_endpoint,
)
from evaluation.potential_regret.optimizer import (
    LearnedPotential,
    PotentialOracleOptimizer,
    observation_cars,
)


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


def _model_label(path: Path, trace) -> str:
    names = np.asarray(trace.get("llm_model_names", []), dtype=str).reshape(-1)
    ego = int(_scalar(trace, "ego_index"))
    if names.size > ego and names[ego]:
        return str(names[ego])
    return path.parent.parent.name


def _relative_gain(start, end, ego, comparator) -> float:
    return float(
        (end[ego] - end[comparator]) - (start[ego] - start[comparator])
    )


def analyze_trace(path, weight_path, horizon_seconds, sobol_points, max_iterations):
    with np.load(path, allow_pickle=False) as trace:
        dt = float(_scalar(trace, "simulation_dt")) if "simulation_dt" in trace else params.SIM_DT
        horizon_steps = int(round(horizon_seconds / dt))
        if not np.isclose(horizon_steps * dt, horizon_seconds, atol=1e-12):
            raise ValueError("Horizon must be an integer number of simulation steps")
        ego = int(_scalar(trace, "ego_index"))
        race = _make_replay_race(trace)
        potential = LearnedPotential(
            race.track, dt, int(_scalar(trace, "mpc_horizon")), weight_path
        )
        optimizer = PotentialOracleOptimizer(
            potential,
            sobol_points=sobol_points,
            max_iterations=max_iterations,
            seed=0,
        )
        rows = []
        for step_value in _decision_steps(trace, ego):
            step = int(step_value)
            end_step = step + horizon_steps
            try:
                recorded_end = _endpoint_progress(trace, end_step)
            except IndexError:
                continue
            start = _endpoint_progress(trace, step)
            opponents = [idx for idx in range(len(start)) if idx != ego]
            comparator = max(opponents, key=lambda idx: start[idx])
            joint = np.asarray(trace["car_policy_params"], dtype=float)[step].copy()
            llm_theta = joint[ego].copy()

            restore_window_start(race, trace, step, joint_params=joint)
            race.environment.begin_step(step)
            observation = race.environment.capture(race.policies)
            optimization = optimizer.maximize(
                observation_cars(observation), joint, ego, llm_theta
            )
            reference_theta = np.asarray(optimization.theta, dtype=float)
            reference_joint = joint.copy()
            reference_joint[ego] = reference_theta
            reference_end = rollout_endpoint(
                race,
                trace,
                step,
                horizon_steps,
                joint_params=reference_joint,
            )

            llm_gain = _relative_gain(start, recorded_end, ego, comparator)
            reference_gain = _relative_gain(start, reference_end, ego, comparator)
            rows.append(
                {
                    "trace": str(path),
                    "model": _model_label(path, trace),
                    "ego_position": int(_scalar(trace, "ego_position")),
                    "step": step,
                    "time_s": float(step * dt),
                    "comparison_opponent_index": int(comparator),
                    "llm_theta": llm_theta.tolist(),
                    "reference_theta": reference_theta.tolist(),
                    "predicted_potential_gap": float(optimization.predicted_difference),
                    "llm_relative_progress_gain_m": llm_gain,
                    "reference_relative_progress_gain_m": reference_gain,
                    "rollout_gap_m": float(reference_gain - llm_gain),
                    "sensitive": _is_sensitive_window(trace, step, end_step),
                    "optimizer_successful": bool(optimization.successful),
                }
            )
        return rows, optimizer.settings_dict()


def _variation(theta: np.ndarray) -> dict:
    bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
    spans = bounds[:, 1] - bounds[:, 0]
    normalized_iqr = (
        np.percentile(theta, 75, axis=0) - np.percentile(theta, 25, axis=0)
    ) / spans
    normalized_std = np.std(theta, axis=0) / spans
    return {
        "unique_rounded_3dp": int(len(np.unique(np.round(theta, 3), axis=0))),
        "mean_normalized_iqr": float(np.mean(normalized_iqr)),
        "mean_normalized_std": float(np.mean(normalized_std)),
        "normalized_iqr_by_parameter": dict(zip(PARAMETER_NAMES, normalized_iqr.tolist())),
        "normalized_std_by_parameter": dict(zip(PARAMETER_NAMES, normalized_std.tolist())),
    }


def summarize(rows):
    llm = np.asarray([row["llm_theta"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_theta"] for row in rows], dtype=float)
    gaps = np.asarray([row["rollout_gap_m"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_potential_gap"] for row in rows], dtype=float)
    sensitive = np.asarray([row["sensitive"] for row in rows], dtype=bool)
    bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
    spans = bounds[:, 1] - bounds[:, 0]
    distances = np.mean(np.abs(reference - llm) / spans, axis=1)
    correlation = spearmanr(predicted, gaps).statistic if len(rows) > 1 else np.nan
    result = {
        "decisions": len(rows),
        "llm_action_variation": _variation(llm),
        "reference_action_variation": _variation(reference),
        "mean_normalized_action_distance": float(np.mean(distances)),
        "mean_rollout_gap_m": float(np.mean(gaps)),
        "median_rollout_gap_m": float(np.median(gaps)),
        "p90_rollout_gap_m": float(np.percentile(gaps, 90)),
        "positive_gap_percent": float(100.0 * np.mean(gaps > 1e-9)),
        "gap_over_5cm_percent": float(100.0 * np.mean(gaps > 0.05)),
        "negative_gap_percent": float(100.0 * np.mean(gaps < -1e-9)),
        "mean_predicted_potential_gap": float(np.mean(predicted)),
        "predicted_vs_rollout_gap_spearman": (
            None if not np.isfinite(correlation) else float(correlation)
        ),
        "sensitive_decisions": int(np.sum(sensitive)),
    }
    for label, mask in (("clean", ~sensitive), ("sensitive", sensitive)):
        selected = gaps[mask]
        result[f"{label}_decisions"] = int(selected.size)
        result[f"{label}_mean_rollout_gap_m"] = (
            float(np.mean(selected)) if selected.size else None
        )
        result[f"{label}_positive_gap_percent"] = (
            float(100.0 * np.mean(selected > 1e-9)) if selected.size else None
        )
    return result


def _print_one(label, summary):
    lv = summary["llm_action_variation"]
    rv = summary["reference_action_variation"]
    print(f"\n[{label}] decisions={summary['decisions']}")
    print(
        "action variation (mean normalized IQR): "
        f"LLM={lv['mean_normalized_iqr']:.4f}, "
        f"reference={rv['mean_normalized_iqr']:.4f}"
    )
    print(
        "unique theta @3dp: "
        f"LLM={lv['unique_rounded_3dp']}, "
        f"reference={rv['unique_rounded_3dp']}"
    )
    print(
        f"mean normalized LLM-reference distance: "
        f"{summary['mean_normalized_action_distance']:.4f}"
    )
    print(
        "reconstructed 1s gap: "
        f"mean={summary['mean_rollout_gap_m']:.4f} m, "
        f"median={summary['median_rollout_gap_m']:.4f} m, "
        f"p90={summary['p90_rollout_gap_m']:.4f} m"
    )
    print(
        f"positive gap={summary['positive_gap_percent']:.1f}%, "
        f">5 cm={summary['gap_over_5cm_percent']:.1f}%, "
        f"negative gap={summary['negative_gap_percent']:.1f}%"
    )
    corr = summary["predicted_vs_rollout_gap_spearman"]
    print(
        "potential-gap vs rollout-gap Spearman: "
        + ("undefined" if corr is None else f"{corr:.3f}")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="NPZ traces or summary.json files")
    parser.add_argument(
        "--weight-path",
        type=Path,
        required=True,
        help=(
            "Explicit learned-potential checkpoint. Standard traces do not record "
            "a regret-reference checkpoint, so silently using the current default "
            "would make results incomparable with earlier evaluations."
        ),
    )
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--sobol-points", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    all_rows = []
    settings = None
    for path in _expand_inputs(args.inputs):
        print(f"analyzing: {path}", flush=True)
        rows, settings = analyze_trace(
            path,
            args.weight_path,
            args.horizon_seconds,
            args.sobol_points,
            args.max_iterations,
        )
        all_rows.extend(rows)
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row["model"], []).append(row)
    summaries = {label: summarize(rows) for label, rows in grouped.items()}
    for label, summary in summaries.items():
        _print_one(label, summary)

    if args.json_output:
        payload = {
            "metric": "exploratory_post_hoc_reconstructed_potential_reference_gap",
            "claim_limit": (
                "Reference maximizes the selected learned potential, not true rollout "
                "return; reconstructed branches are not exact online snapshots."
            ),
            "weight_path": str(args.weight_path),
            "optimizer": settings,
            "summaries": summaries,
            "decisions": all_rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nsaved: {args.json_output}")


if __name__ == "__main__":
    main()
