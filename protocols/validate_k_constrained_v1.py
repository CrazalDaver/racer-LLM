#!/usr/bin/env python3
"""Validate the frozen K-Constrained-v1 manifest against repository defaults."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DynamicParams, params  # noqa: E402
from track import TRACKS  # noqa: E402


MANIFEST_PATH = Path(__file__).with_name("K-Constrained-v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(actual, expected) -> bool:
    if isinstance(expected, float):
        return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
    return actual == expected


def validate() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    plant = manifest["plant"]
    mpc = manifest["mpc"]
    recovery = manifest["recovery"]
    scenario = manifest["scenario_generation"]
    llm = manifest["llm"]
    vehicle = DynamicParams("kinematic")
    errors: list[str] = []

    checks = {
        "params.PROTOCOL_VERSION": (params.PROTOCOL_VERSION, manifest["protocol_version"]),
        "params.VEHICLE_MODEL": (params.VEHICLE_MODEL, "kinematic"),
        "params.MPC_FORMULATION": (params.MPC_FORMULATION, mpc["formulation"]),
        "params.SIM_DT": (params.SIM_DT, mpc["timestep_s"]),
        "params.CONTROL_DT": (params.CONTROL_DT, mpc["timestep_s"]),
        "params.MPC_DT": (params.MPC_DT, mpc["timestep_s"]),
        "params.MPC_HORIZON": (params.MPC_HORIZON, mpc["horizon_steps"]),
        "params.VELOCITY_TRACK_WEIGHT": (params.VELOCITY_TRACK_WEIGHT, mpc["velocity_tracking_weight"]),
        "params.HEADING_TRACK_WEIGHT": (params.HEADING_TRACK_WEIGHT, mpc["heading_tracking_weight"]),
        "params.MPC_TRACK_SLACK_WEIGHT": (params.MPC_TRACK_SLACK_WEIGHT, mpc["track_slack_weight"]),
        "params.MPC_COLLISION_SLACK_WEIGHT": (params.MPC_COLLISION_SLACK_WEIGHT, mpc["collision_slack_weight"]),
        "params.MPC_MIN_LONGITUDINAL_SEPARATION": (params.MPC_MIN_LONGITUDINAL_SEPARATION, mpc["minimum_longitudinal_separation_m"]),
        "params.MPC_MIN_LATERAL_SEPARATION": (params.MPC_MIN_LATERAL_SEPARATION, mpc["minimum_lateral_separation_m"]),
        "params.MPC_MAX_CONSTRAINED_OPPONENTS": (params.MPC_MAX_CONSTRAINED_OPPONENTS, mpc["maximum_constrained_opponents"]),
        "params.MPC_COLLISION_SUPERELLIPSE_EXPONENT": (params.MPC_COLLISION_SUPERELLIPSE_EXPONENT, 4),
        "params.MPC_ENFORCE_TRACK_BOUNDARY": (params.MPC_ENFORCE_TRACK_BOUNDARY, True),
        "params.MPC_ENFORCE_COLLISION_AVOIDANCE": (params.MPC_ENFORCE_COLLISION_AVOIDANCE, True),
        "params.MPC_ENFORCE_FRICTION_SPEED_LIMIT": (params.MPC_ENFORCE_FRICTION_SPEED_LIMIT, True),
        "params.MPC_FRICTION_COEFFICIENT": (params.MPC_FRICTION_COEFFICIENT, mpc["friction_coefficient"]),
        "params.MAX_STATE_SPEED": (params.MAX_STATE_SPEED, mpc["max_speed_mps"]),
        "params.MAX_VEL_RATE": (params.MAX_VEL_RATE, mpc["max_speed_rate_mps2"]),
        "params.MPC_MAX_STEER_RATE": (params.MPC_MAX_STEER_RATE, mpc["max_steering_command_rate_per_s"]),
        "params.OFFROAD_RECOVERY_THRESHOLD": (params.OFFROAD_RECOVERY_THRESHOLD, recovery["trigger_abs_lateral_error_m"]),
        "params.OFFROAD_STRENGTH": (params.OFFROAD_STRENGTH, recovery["speed_decay_strength"]),
        "params.OFFROAD_HEADING_REALIGN_STRENGTH": (params.OFFROAD_HEADING_REALIGN_STRENGTH, recovery["heading_realign_strength"]),
        "params.OFFROAD_INWARD_STEER": (params.OFFROAD_INWARD_STEER, recovery["inward_steering_command"]),
        "params.OFFROAD_LARGE_HEADING_ERROR": (params.OFFROAD_LARGE_HEADING_ERROR, recovery["large_heading_error_rad"]),
        "params.OFFROAD_THROTTLE_BOOST": (params.OFFROAD_THROTTLE_BOOST, recovery["throttle_boost"]),
        "params.OFFROAD_LOW_SPEED_THRESHOLD": (params.OFFROAD_LOW_SPEED_THRESHOLD, recovery["low_speed_threshold_mps"]),
        "params.COLLISION_FRONT_DECAY": (params.COLLISION_FRONT_DECAY, recovery["collision_front_decay"]),
        "params.COLLISION_REAR_DECAY": (params.COLLISION_REAR_DECAY, recovery["collision_rear_decay"]),
        "params.LLM_UPDATE_INTERVAL_SECONDS": (params.LLM_UPDATE_INTERVAL_SECONDS, llm["update_interval_s"]),
        "params.CAR_START_GAP": (params.CAR_START_GAP, scenario["start_gap_m"]),
        "params.TOTAL_TIME": (params.TOTAL_TIME, scenario["duration_s"]),
        "vehicle.LF": (vehicle.LF, plant["lf_m"]),
        "vehicle.LR": (vehicle.LR, plant["lr_m"]),
        "vehicle.Sa": (vehicle.Sa, plant["steering_gain_rad"]),
        "vehicle.Sb": (vehicle.Sb, plant["steering_bias_rad"]),
        "vehicle.Ta": (vehicle.Ta, plant["acceleration_gain_mps2"]),
        "vehicle.fr": (vehicle.fr, plant["rolling_deceleration_mps2"]),
    }
    for label, (actual, expected) in checks.items():
        if not _same(actual, expected):
            errors.append(f"{label}: expected {expected!r}, found {actual!r}")

    expected_names = tuple(manifest["parameters"]["order"])
    if tuple(params.ALPHARACER_PARAMETER_NAMES) != expected_names:
        errors.append("five-parameter order differs from the frozen manifest")
    expected_bounds = np.asarray(
        [manifest["parameters"]["bounds"][name] for name in expected_names],
        dtype=float,
    )
    if not np.allclose(params.ALPHARACER_PARAMETER_BOUNDS, expected_bounds, rtol=0.0, atol=1e-12):
        errors.append("five-parameter bounds differ from the frozen manifest")
    if not np.array_equal(np.diag(params.Q), np.asarray(mpc["position_weight_diagonal"])):
        errors.append("MPC Q differs from the frozen manifest")
    if not np.array_equal(np.diag(params.R), np.asarray(mpc["control_rate_weight_diagonal"])):
        errors.append("MPC R differs from the frozen manifest")

    for name, expected in manifest["tracks"].items():
        spec = TRACKS.get(name)
        if spec is None:
            errors.append(f"registered track {name!r} is missing")
            continue
        for field in ("centerline", "raceline", "scale"):
            if not _same(getattr(spec, field), expected[field]):
                errors.append(f"TRACKS[{name!r}].{field} differs from manifest")
        if not _same(spec.half_width, expected["half_width_m"]):
            errors.append(f"TRACKS[{name!r}].half_width differs from manifest")
        for field in ("centerline", "raceline"):
            path = ROOT / expected[field]
            if not path.is_file():
                errors.append(f"frozen file is missing: {expected[field]}")
            elif _sha256(path) != expected[f"{field}_sha256"]:
                errors.append(f"frozen file content changed: {expected[field]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = validate()
    if errors:
        print("K-Constrained-v1 validation FAILED:")
        for error in errors:
            print(f"- {error}")
        print("A frozen field changed. Restore it or create a new protocol version.")
        return 1
    print("K-Constrained-v1 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
