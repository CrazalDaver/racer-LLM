#!/usr/bin/env python3
"""Run the explicit llm-param-minimal-v1 experiment configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import params
from main import (
    _trace_llm_summary,
    _trace_safety_summary,
    params_snapshot,
    run_position_sweep,
)


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
EXPECTED_NEUTRAL = np.array([1.0 / 0.75, 0.90, 0.20, 0.30, 0.00])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_config(config: dict) -> None:
    if config["protocol_version"] != "K-Constrained-v1":
        raise ValueError("This runner only accepts K-Constrained-v1")
    if config["interface_schema"] != "direct_params_modular_v1":
        raise ValueError("Unexpected direct-parameter interface schema")
    neutral = np.asarray(config["neutral_parameters"], dtype=float)
    fallback = np.asarray(
        config["llm"]["initial_and_fallback_parameters"], dtype=float
    )
    opponents = np.asarray(config["opponents"]["parameters"], dtype=float)
    if not (
        np.array_equal(neutral, EXPECTED_NEUTRAL)
        and np.array_equal(fallback, EXPECTED_NEUTRAL)
        and np.array_equal(opponents, EXPECTED_NEUTRAL)
    ):
        raise ValueError("Neutral, fallback, and opponent parameters must match")
    modules = config["llm"]["input_modules"]
    if modules != {
        "base": True,
        "raceline_speed": True,
        "current_parameters": False,
        "decision_feedback": False,
    }:
        raise ValueError("llm-param-minimal-v1 input modules changed")
    if config["llm"]["output_fields"] != list(PARAMETER_NAMES):
        raise ValueError("Direct-parameter output fields changed")
    if config["llm"]["output_mode"] != "params":
        raise ValueError("The minimal arm must use direct parameter output")
    if config["llm"]["temperature"] != 0.0:
        raise ValueError("The minimal arm requires temperature 0")
    if config["llm"]["update_interval_s"] != 1.0:
        raise ValueError("The frozen update interval is 1.0 simulated seconds")
    runtime = config["frozen_runtime"]
    expected_runtime = {
        "vehicle_model": "kinematic",
        "mpc_formulation": "constrained_local",
        "mpc_horizon_steps": 10,
        "velocity_tracking_weight": 10.0,
        "track_slack_weight": 1000.0,
        "collision_slack_weight": 10000.0,
    }
    if runtime != expected_runtime:
        raise ValueError("Frozen runtime settings changed")
    if config["opponents"]["policies"] != [
        "static_alpharacer",
        "static_alpharacer",
    ]:
        raise ValueError("Opponent roster changed")
    push = np.asarray(config["fixed_push_parameters"], dtype=float)
    if not np.array_equal(push, np.asarray(params.LLM_MODE_PRESETS["push"])):
        raise ValueError("Recorded fixed-push vector differs from the frozen preset")


def _model_metadata(config: dict) -> dict:
    payload = json.dumps({"name": config["llm"]["model"]}).encode("utf-8")
    request = urllib.request.Request(
        f"{config['llm']['base_url'].rstrip('/')}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _arm_overrides(config: dict, arm_id: str, track: str, duration: float) -> dict:
    arm = config["arms"][arm_id]
    runtime = config["frozen_runtime"]
    scenario = config["scenario"]
    neutral = list(config["neutral_parameters"])
    values = {
        "track_name": track,
        "duration": float(duration),
        "random_seed": int(scenario["seed"]),
        "start_gap": float(scenario["start_gap_m"]),
        "start_speeds": list(scenario["initial_speeds_mps"]),
        "vehicle_model": runtime["vehicle_model"],
        "mpc_formulation": runtime["mpc_formulation"],
        "mpc_horizon": int(runtime["mpc_horizon_steps"]),
        "mpc_velocity_weight": float(runtime["velocity_tracking_weight"]),
        "mpc_track_slack": float(runtime["track_slack_weight"]),
        "mpc_collision_slack": float(runtime["collision_slack_weight"]),
        "static_params": neutral,
        "show_plot": False,
    }
    if arm["ego_policy"] == "llm_alpharacer":
        llm = config["llm"]
        values.update({
            "llm_params": neutral,
            "llm_output_mode": "params",
            "llm_model": llm["model"],
            "llm_temperature": float(llm["temperature"]),
            "llm_update_interval": float(llm["update_interval_s"]),
            "llm_observation_variant": "baseline",
            "llm_include_raceline_speed": True,
            "llm_include_current_parameters": False,
            "llm_include_decision_feedback": False,
        })
    return values


def _parameter_statistics(values: np.ndarray) -> dict:
    bounds = np.asarray(params.ALPHARACER_PARAMETER_BOUNDS, dtype=float)
    result = {}
    for idx, name in enumerate(PARAMETER_NAMES):
        column = values[:, idx]
        deltas = np.abs(np.diff(column))
        tolerance = 0.01 * (bounds[idx, 1] - bounds[idx, 0])
        result[name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "near_lower_bound_fraction": float(
                np.mean(column <= bounds[idx, 0] + tolerance)
            ),
            "near_upper_bound_fraction": float(
                np.mean(column >= bounds[idx, 1] - tolerance)
            ),
            "mean_absolute_change": float(np.mean(deltas)) if deltas.size else 0.0,
            "max_absolute_change": float(np.max(deltas)) if deltas.size else 0.0,
            "total_variation": float(np.sum(deltas)),
        }
    return result


def _analyze_trace(trace_path: Path, arm_id: str) -> dict:
    with np.load(trace_path, allow_pickle=True) as data:
        progress = np.asarray(data["car_px"], dtype=float)
        final = progress[-1]
        control_steps = np.flatnonzero(np.asarray(data["control_update"], dtype=bool))
        all_params = np.asarray(data["car_policy_params"], dtype=float)
        if arm_id == "llm-param-minimal-v1":
            calls = np.asarray(data["llm_raw_call_indices"], dtype=int)[0]
            calls = calls[calls >= 1] - 1
            calls = calls[calls < control_steps.size]
            sample_steps = control_steps[calls]
        else:
            sample_steps = control_steps
        parameter_values = all_params[sample_steps, 0]
        solver_success = np.asarray(data["car_mpc_solver_success"], dtype=bool)
        return {
            "trace": str(trace_path.relative_to(REPOSITORY_ROOT)),
            "ego_position": int(np.asarray(data["ego_position"])),
            "initial_progress_m": float(progress[0, 0]),
            "final_progress_m": float(final[0]),
            "progress_gain_m": float(final[0] - progress[0, 0]),
            "finish_position": int(1 + np.sum(final > final[0])),
            "llm_error_count": int(
                np.sum(np.asarray(data["llm_raw_errors"], dtype=str)[0] != "")
            ),
            "mpc_solver_failure_count_by_car": np.sum(
                ~solver_success, axis=0
            ).astype(int).tolist(),
            "safety": _trace_safety_summary(data),
            "decision_timing_by_car": _trace_llm_summary(data),
            "parameter_decision_count": int(parameter_values.shape[0]),
            "parameter_statistics": _parameter_statistics(parameter_values),
        }


def _resolved_record(
    config: dict,
    config_path: Path,
    stage_name: str,
    stage: dict,
    arm_id: str,
    track: str,
    overrides: dict,
    model_metadata: dict,
) -> dict:
    prompt_path = REPOSITORY_ROOT / "policy/llm/prompts/params_system.txt"
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "arm_id": arm_id,
        "result_class": stage["result_class"],
        "protocol_version": config["protocol_version"],
        "interface_schema": config["interface_schema"],
        "requested_config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "requested_config_sha256": _sha256(config_path),
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
        "stage": stage_name,
        "track": track,
        "ego_positions": stage["ego_positions"],
        "policy_roster": [
            config["arms"][arm_id]["ego_policy"],
            *config["opponents"]["policies"],
        ],
        "runtime_overrides": overrides,
        "input_modules": config["llm"]["input_modules"],
        "memory": config["llm"]["memory"],
        "update_interval_s": config["llm"]["update_interval_s"],
        "repository_defaults": params_snapshot(),
        "system_prompt_path": str(prompt_path.relative_to(REPOSITORY_ROOT)),
        "system_prompt_sha256": _sha256(prompt_path),
        "system_prompt": prompt_path.read_text(encoding="utf-8").strip(),
        "output_schema": {
            "type": "object",
            "required": list(PARAMETER_NAMES),
            "properties": {
                name: {
                    "type": "number",
                    "minimum": float(params.LLM_PARAMETER_BOUNDS[name][0]),
                    "maximum": float(params.LLM_PARAMETER_BOUNDS[name][1]),
                }
                for name in PARAMETER_NAMES
            },
            "additionalProperties": False,
        },
        "parser_clipping_retry_timeout_and_fallback": (
            "Ollama JSON schema; parsed JSON object; each value clipped to frozen "
            "bounds; no retry; 20 s HTTP timeout; retain last valid parameters, "
            "initialized to the neutral vector"
        ),
        "model_metadata": model_metadata if arm_id == "llm-param-minimal-v1" else None,
    }


def _stage_summary(records: list[dict]) -> dict:
    cells = {}
    wins = {}
    for record in records:
        arm = record["arm_id"]
        wins.setdefault(arm, [0, 0])
        for cell in record["cells"]:
            key = (record["track"], cell["ego_position"])
            cells.setdefault(key, {})[arm] = cell["progress_gain_m"]
            wins[arm][0] += int(cell["finish_position"] == 1)
            wins[arm][1] += 1
    paired = []
    for (track, position), values in sorted(cells.items()):
        row = {"track": track, "ego_position": position, "progress_gain_m": values}
        if "llm-param-minimal-v1" in values:
            llm = values["llm-param-minimal-v1"]
            row["llm_minus_static_neutral_m"] = (
                llm - values["static-neutral-v1"]
                if "static-neutral-v1" in values else None
            )
            row["llm_minus_fixed_push_m"] = (
                llm - values["fixed-push-v1"]
                if "fixed-push-v1" in values else None
            )
        paired.append(row)
    return {
        "runs": records,
        "paired_cells": paired,
        "ego_win_rate_by_arm": {
            arm: won / total if total else None
            for arm, (won, total) in wins.items()
        },
    }


def run(config_path: Path, stage_name: str) -> None:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    validate_config(config)
    stage = config["stages"][stage_name]
    git_dirty = bool(_git("status", "--porcelain", "--untracked-files=all"))
    if stage["result_class"] == "core_scan" and git_dirty:
        raise RuntimeError("Stage 2 core scan requires a clean Git worktree")

    stage_dir = EXPERIMENT_DIR / "results" / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    model_metadata = _model_metadata(config)
    _write_json(stage_dir / "model_metadata.json", model_metadata)
    _write_json(stage_dir / "requested_experiment.json", config)

    records = []
    for track in stage["tracks"]:
        for arm_id in stage["arms"]:
            run_dir = stage_dir / track / arm_id
            run_dir.mkdir(parents=True, exist_ok=True)
            overrides = _arm_overrides(
                config, arm_id, track, float(stage["duration_s"])
            )
            roster = [
                config["arms"][arm_id]["ego_policy"],
                *config["opponents"]["policies"],
            ]
            requested = {
                "experiment_id": config["experiment_id"],
                "stage": stage_name,
                "arm_id": arm_id,
                "track": track,
                "ego_positions": stage["ego_positions"],
                "policy_roster": roster,
            }
            _write_json(run_dir / "requested_config.json", requested)
            _write_json(
                run_dir / "resolved_config.json",
                _resolved_record(
                    config,
                    config_path,
                    stage_name,
                    stage,
                    arm_id,
                    track,
                    overrides,
                    model_metadata,
                ),
            )
            run_position_sweep(
                policy_names=roster,
                trials_per_position=int(config["scenario"]["repeat_count"]),
                ego_positions=stage["ego_positions"],
                output_path=run_dir / "summary.json",
                keep_npz=True,
                show_progress=True,
                suppress_simulation_output=True,
                **overrides,
            )
            cells = [
                _analyze_trace(path, arm_id)
                for path in sorted((run_dir / "summary_traces").glob("*.npz"))
            ]
            analysis = {"arm_id": arm_id, "track": track, "cells": cells}
            _write_json(run_dir / "analysis.json", analysis)
            records.append(analysis)

    _write_json(stage_dir / "stage_summary.json", _stage_summary(records))
    print(f"Saved {stage_name} results to {stage_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "stage1", "stage2"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config, arguments.stage)
