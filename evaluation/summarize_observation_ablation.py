#!/usr/bin/env python3
"""Summarize closed-loop LLM observation-ablation traces."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _scalar(data, key, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def _safe_mean(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _discover(paths):
    found = set()
    for path in paths:
        if path.is_file() and path.suffix == ".npz":
            found.add(path.resolve())
        elif path.is_dir():
            found.update(item.resolve() for item in path.rglob("*.npz"))
    return sorted(found)


def _mode_metrics(data, ego_idx, sim_dt):
    if "car_policy_mode" not in data.files:
        return {}, 0, None
    modes = np.asarray(data["car_policy_mode"], dtype=str)[:, ego_idx]
    modes = modes[modes != ""]
    if modes.size == 0:
        return {}, 0, None
    counts = Counter(modes.tolist())
    fractions = {mode: count / modes.size for mode, count in sorted(counts.items())}
    switches = int(np.sum(modes[1:] != modes[:-1]))
    change_indices = np.r_[0, np.flatnonzero(modes[1:] != modes[:-1]) + 1, modes.size]
    mean_duration = float(np.mean(np.diff(change_indices)) * sim_dt)
    return fractions, switches, mean_duration


def _llm_metrics(data, ego_idx):
    calls = np.asarray(data.get("llm_raw_call_indices", np.empty((0, 0))), dtype=int)
    if calls.ndim != 2 or ego_idx >= calls.shape[0]:
        return {"llm_calls": 0, "llm_errors": 0, "mean_latency_s": None,
                "prompt_tokens": 0, "output_tokens": 0}
    valid = calls[ego_idx] >= 0
    errors = np.asarray(data.get("llm_raw_errors", np.empty_like(calls, dtype=str)), dtype=str)
    latency = np.asarray(data.get("llm_raw_latency_seconds", np.full(calls.shape, np.nan)), dtype=float)
    prompt_tokens = np.asarray(data.get("llm_raw_prompt_eval_counts", np.full(calls.shape, -1)), dtype=int)
    output_tokens = np.asarray(data.get("llm_raw_eval_counts", np.full(calls.shape, -1)), dtype=int)
    latency_values = latency[ego_idx][valid & np.isfinite(latency[ego_idx])]
    return {
        "llm_calls": int(np.sum(valid)),
        "llm_errors": int(np.sum(errors[ego_idx][valid] != "")),
        "mean_latency_s": float(np.mean(latency_values)) if latency_values.size else None,
        "prompt_tokens": int(np.sum(prompt_tokens[ego_idx][valid & (prompt_tokens[ego_idx] >= 0)])),
        "output_tokens": int(np.sum(output_tokens[ego_idx][valid & (output_tokens[ego_idx] >= 0)])),
    }


def summarize_trace(path):
    with np.load(path, allow_pickle=False) as data:
        progress = np.asarray(data["car_px"], dtype=float)
        ego_idx = int(_scalar(data, "ego_index", 0))
        ego_position = int(_scalar(data, "ego_position", 1))
        sim_dt = float(_scalar(data, "simulation_dt", 0.1))
        opponents = [idx for idx in range(progress.shape[1]) if idx != ego_idx]
        ego_progress = progress[:, ego_idx]
        opponent_best = np.max(progress[:, opponents], axis=1)
        lead = ego_progress - opponent_best
        final_order = np.argsort(-progress[-1])
        ego_rank = int(np.where(final_order == ego_idx)[0][0]) + 1
        ranks = 1 + np.sum(progress > ego_progress[:, None], axis=1)

        policies = np.asarray(data.get("policy_names", []), dtype=str)
        variants = np.asarray(
            data.get("car_llm_observation_variants", np.full(len(policies), "unknown")),
            dtype=str,
        )
        observation_variant = (
            str(variants[ego_idx]) if ego_idx < len(variants) else "unknown"
        )
        mode_fractions, mode_switches, mean_mode_duration = _mode_metrics(
            data, ego_idx, sim_dt
        )

        collision = np.asarray(data.get("collision", []), dtype=bool)
        boundary = np.asarray(data.get("car_track_boundary_violation", []), dtype=bool)
        recovery = np.asarray(data.get("car_offroad_recovery_active", []), dtype=bool)
        solver_success = np.asarray(data.get("car_mpc_solver_success", []), dtype=bool)
        row = {
            "case": path.parent.parent.name,
            "trace": str(path),
            "observation_variant": observation_variant,
            "ego_policy": str(policies[ego_idx]) if ego_idx < len(policies) else "unknown",
            "opponent_policies": "+".join(str(policies[idx]) for idx in opponents),
            "ego_position": ego_position,
            "ego_won": ego_rank == 1,
            "ego_final_rank": ego_rank,
            "ego_progress_gain_m": float(ego_progress[-1] - ego_progress[0]),
            "final_lead_m": float(lead[-1]),
            "lead_time_fraction": float(np.mean(lead > 0.0)),
            "mean_ego_rank": float(np.mean(ranks)),
            "rank_change_count": int(np.sum(ranks[1:] != ranks[:-1])),
            "collision_seconds": float(np.sum(collision) * sim_dt) if collision.size else 0.0,
            "boundary_fraction": float(np.mean(boundary[:, ego_idx])) if boundary.size else 0.0,
            "recovery_fraction": float(np.mean(recovery[:, ego_idx])) if recovery.size else 0.0,
            "solver_failure_fraction": float(np.mean(~solver_success[:, ego_idx])) if solver_success.size else 0.0,
            "mode_switches": mode_switches,
            "mean_mode_duration_s": mean_mode_duration,
            "mode_fractions": mode_fractions,
        }
        row.update(_llm_metrics(data, ego_idx))
        return row


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["observation_variant"], row["ego_policy"], row["opponent_policies"])
        groups[key].append(row)
    output = []
    for (variant, ego_policy, opponents), items in sorted(groups.items()):
        output.append({
            "observation_variant": variant,
            "ego_policy": ego_policy,
            "opponent_policies": opponents,
            "race_count": len(items),
            "win_count": int(sum(item["ego_won"] for item in items)),
            "win_rate": float(np.mean([item["ego_won"] for item in items])),
            "mean_final_lead_m": _safe_mean(item["final_lead_m"] for item in items),
            "mean_progress_gain_m": _safe_mean(item["ego_progress_gain_m"] for item in items),
            "mean_lead_time_fraction": _safe_mean(item["lead_time_fraction"] for item in items),
            "mean_ego_rank": _safe_mean(item["mean_ego_rank"] for item in items),
            "collision_seconds": float(sum(item["collision_seconds"] for item in items)),
            "mean_boundary_fraction": _safe_mean(item["boundary_fraction"] for item in items),
            "mean_solver_failure_fraction": _safe_mean(item["solver_failure_fraction"] for item in items),
            "total_mode_switches": int(sum(item["mode_switches"] for item in items)),
            "mean_llm_latency_s": _safe_mean(item["mean_latency_s"] for item in items),
            "llm_error_count": int(sum(item["llm_errors"] for item in items)),
            "prompt_tokens": int(sum(item["prompt_tokens"] for item in items)),
            "output_tokens": int(sum(item["output_tokens"] for item in items)),
        })
    return output


def _write_csv(path, rows):
    flattened = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("mode_fractions"), dict):
            modes = item.pop("mode_fractions")
            for mode in ("stable", "push", "overtake", "block", "custom"):
                item[f"mode_fraction_{mode}"] = modes.get(mode, 0.0)
        flattened.append(item)
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    traces = _discover(args.inputs)
    if not traces:
        parser.error("No NPZ traces found under the supplied inputs")

    rows = [summarize_trace(path) for path in traces]
    aggregate_rows = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"races": rows, "aggregate": aggregate_rows}, indent=2),
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "races.csv", rows)
    _write_csv(args.output_dir / "aggregate.csv", aggregate_rows)
    print(f"Summarized {len(rows)} traces in {args.output_dir}")


if __name__ == "__main__":
    main()
