"""Small local checks that exercise simulation interfaces without an LLM server."""

import argparse
import difflib
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter

import numpy as np

from config import params
from policy.llm import LLMAlphaRacerPolicy
from simulation import Race
from simulation_core import SimulationConfig
from state import CarState


def _run_case(name, config):
    started = perf_counter()
    race = Race(config)
    trace = race.run()
    expected_steps = round(float(config.duration) / params.SIM_DT)
    assert len(trace["time"]) == expected_steps
    assert len(trace["car_x"]) == expected_steps
    assert len(trace["car_x"][0]) == len(config.policies)
    assert trace is race.recorder.trace
    assert all(
        np.isfinite(state.to_array()).all()
        for state in race.environment.states
    )
    return trace, perf_counter() - started


def run_interface_checks():
    cases = (
        (
            "single_centerline",
            SimulationConfig(
                policies=("centerline",), duration=0.2, track_name="berlin"
            ),
        ),
        (
            "three_static_alpharacer",
            SimulationConfig(
                policies=("static_alpharacer",) * 3,
                duration=0.1,
                track_name="berlin",
            ),
        ),
        (
            "mixed_ibr_centerline",
            SimulationConfig(
                policies=("ibr", "centerline"),
                duration=0.1,
                track_name="rounded_rectangle",
            ),
        ),
        (
            "upstream_simple_mpc",
            SimulationConfig(
                policies=("centerline",),
                duration=0.1,
                track_name="berlin",
                mpc_formulation="upstream_simple",
            ),
        ),
    )
    results = []
    for name, config in cases:
        config = SimulationConfig(
            **{
                **config.__dict__,
                "save_trace": False,
                "show_plot": False,
                "quiet": True,
            }
        )
        trace, elapsed = _run_case(name, config)
        results.append({
            "name": name,
            "steps": len(trace["time"]),
            "cars": len(config.policies),
            "elapsed_seconds": elapsed,
        })
        print(f"PASS {name} ({elapsed:.2f}s)", flush=True)
    return results


def run_observation_checks():
    """Check that rounded magnitudes cannot contradict categorical labels."""
    cases = (
        (-0.11, "opening"),
        (-0.10, "constant"),
        (-0.01, "constant"),
        (0.00, "constant"),
        (0.01, "constant"),
        (0.10, "constant"),
        (0.11, "closing"),
    )
    results = []
    for closing_speed, expected_trend in cases:
        encoded = LLMAlphaRacerPolicy._gap_motion(closing_speed)
        assert encoded["trend"] == expected_trend
        assert encoded["speed_meters_per_second"] == round(abs(closing_speed), 2)
        results.append({
            "closing_speed_mps": closing_speed,
            "trend": encoded["trend"],
        })

    observer = LLMAlphaRacerPolicy.__new__(LLMAlphaRacerPolicy)
    observer.track = SimpleNamespace(total_length=100.0)
    boundary_case = observer._opponent_feature(
        CarState(vx=2.0),
        current_s=0.0,
        current_py=0.0,
        lateral_limit=1.0,
        opponent={"car_idx": 1, "px": 1.0, "py": 0.0, "vx": 1.9},
    )
    assert boundary_case["gap_motion"]["trend"] == "constant"
    assert boundary_case["ttc_seconds"] is None

    near_center_left = LLMAlphaRacerPolicy._side_and_magnitude(
        0.04, "percent_of_half_width", digits=1
    )
    near_center_right = LLMAlphaRacerPolicy._side_and_magnitude(
        -0.04, "percent_of_half_width", digits=1
    )
    visible_left = LLMAlphaRacerPolicy._side_and_magnitude(
        0.06, "percent_of_half_width", digits=1
    )
    assert near_center_left == {"side": "center", "percent_of_half_width": 0.0}
    assert near_center_right == {"side": "center", "percent_of_half_width": 0.0}
    assert visible_left == {"side": "left", "percent_of_half_width": 0.1}

    near_straight = LLMAlphaRacerPolicy._turn_and_magnitude(
        -0.04, "degrees", digits=1
    )
    visible_right = LLMAlphaRacerPolicy._turn_and_magnitude(
        -0.06, "degrees", digits=1
    )
    tiny_curvature = LLMAlphaRacerPolicy._turn_and_magnitude(
        -0.0004, "magnitude_per_meter", digits=3
    )
    assert near_straight == {"direction": "straight", "degrees": 0.0}
    assert visible_right == {"direction": "right", "degrees": 0.1}
    assert tiny_curvature == {
        "direction": "straight",
        "magnitude_per_meter": 0.0,
    }
    print("PASS LLM observation numerical encoding", flush=True)
    return results


def run_observation_snapshot_check():
    """Compare one deterministic three-car LLM scene with its reviewed JSON."""
    race, actual = _capture_llm_scene(
        "baseline",
        car_s=(10.0, 12.0, 8.0),
        car_e=(0.0, 0.1, -0.1),
        car_speed=(2.0, 1.85, 2.25),
    )

    snapshot_path = Path(__file__).with_name("snapshots") / "llm_observation_v1_1.json"
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if actual != expected:
        expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
        actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
        difference = "\n".join(difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile=str(snapshot_path),
            tofile="generated observation",
            lineterm="",
        ))
        raise AssertionError(f"LLM observation snapshot changed:\n{difference}")

    print("PASS LLM observation v1.1 snapshot", flush=True)
    return {
        "snapshot": str(snapshot_path),
        "ego_progress_percent": actual["ego"]["lap_progress_percent"],
        "front_vehicle_id": actual["traffic"]["front"]["id"],
        "rear_vehicle_id": actual["traffic"]["rear"]["id"],
        "road_preview_samples": len(actual["road_preview"]["samples_near_to_far"]),
    }


def _capture_llm_scene(observation_variant, car_s, car_e, car_speed):
    """Build one deterministic prompt scene without contacting the LLM server."""
    start_gap = 0.8
    start_offsets = tuple(
        float(s) + rank * start_gap for rank, s in enumerate(car_s)
    )
    race = Race(SimulationConfig(
        policies=("llm_alpharacer", "static_alpharacer", "static_alpharacer"),
        duration=0.1,
        track_name="berlin",
        ego_position=1,
        start_gap=start_gap,
        start_s_offsets=start_offsets,
        start_e_offsets=car_e,
        start_speeds=car_speed,
        llm_observation_variant=observation_variant,
        save_trace=False,
        show_plot=False,
        quiet=True,
    ))
    race.environment.begin_step(0)
    observation = race.environment.capture(race.policies)
    ego = observation.car(race.ego_idx)
    scene = race.policies[race.ego_idx]._scene_features(
        ego.state,
        ego.s,
        opponents=observation.mpc_opponents(race.ego_idx),
    )
    return race, scene


def run_observation_variant_checks():
    """Verify that each ablation changes only its declared observation field."""
    _, baseline = _capture_llm_scene(
        "baseline", (10.0, 12.0, 14.0), (0.0, 0.1, -0.1), (2.0, 1.85, 1.5)
    )
    all_race, all_opponents = _capture_llm_scene(
        "all_opponents", (10.0, 12.0, 14.0), (0.0, 0.1, -0.1), (2.0, 1.85, 1.5)
    )
    assert "additional_opponents_near_to_far" not in baseline["traffic"]
    assert all_opponents["traffic"]["front"]["id"] == 1
    assert all_opponents["traffic"]["rear"] is None
    assert [
        item["id"]
        for item in all_opponents["traffic"]["additional_opponents_near_to_far"]
    ] == [2]
    all_without_extra = json.loads(json.dumps(all_opponents))
    all_without_extra["traffic"].pop("additional_opponents_near_to_far")
    assert all_without_extra == baseline

    baseline_standard_race, baseline_standard = _capture_llm_scene(
        "baseline", (10.0, 12.0, 8.0), (0.0, 0.1, -0.1), (2.0, 1.85, 2.25)
    )
    no_lap_race, no_lap = _capture_llm_scene(
        "no_lap_progress", (10.0, 12.0, 8.0), (0.0, 0.1, -0.1), (2.0, 1.85, 2.25)
    )
    assert "lap_progress_percent" not in no_lap["ego"]
    no_lap_with_baseline_value = json.loads(json.dumps(no_lap))
    no_lap_with_baseline_value["ego"]["lap_progress_percent"] = (
        baseline_standard["ego"]["lap_progress_percent"]
    )
    assert no_lap_with_baseline_value == baseline_standard
    assert "ego.lap_progress_percent" not in no_lap_race.policies[0].system_prompt

    speed_race, raceline_speed = _capture_llm_scene(
        "raceline_speed", (10.0, 12.0, 8.0), (0.0, 0.1, -0.1), (2.0, 1.85, 2.25)
    )
    speed_samples = raceline_speed["road_preview"]["samples_near_to_far"]
    assert all(
        sample["raceline_target_speed_meters_per_second"] is not None
        for sample in speed_samples
    )
    speed_without_targets = json.loads(json.dumps(raceline_speed))
    for sample in speed_without_targets["road_preview"]["samples_near_to_far"]:
        sample.pop("raceline_target_speed_meters_per_second")
    assert speed_without_targets == baseline_standard
    assert "raceline_target_speed_meters_per_second" in speed_race.policies[0].system_prompt

    feedback_race, first_feedback_scene = _capture_llm_scene(
        "decision_feedback", (10.0, 12.0, 8.0), (0.0, 0.1, -0.1), (2.0, 1.85, 2.25)
    )
    feedback_policy = feedback_race.policies[0]
    feedback_policy._llm_call_count = 1
    # Rebuild once with the correct first decision-call index, then mark that
    # mode as the action executed over the following interval.
    first_observation = feedback_race.environment.capture(feedback_race.policies)
    first_ego = first_observation.car(0)
    first_feedback_scene = feedback_policy._scene_features(
        first_ego.state, first_ego.s, first_observation.mpc_opponents(0)
    )
    assert first_feedback_scene["memory"]["previous_decision_feedback"] is None
    feedback_without_extra = json.loads(json.dumps(first_feedback_scene))
    feedback_without_extra["memory"].pop("previous_decision_feedback")
    assert feedback_without_extra == baseline_standard
    feedback_policy._commit_decision_feedback_state()

    next_s, next_e, next_speed = 11.0, 0.05, 2.2
    next_x, next_y = feedback_race.track.frenet_to_world(next_s, next_e)
    _, _, next_psi = feedback_race.track.interpolate(next_s)
    feedback_policy._llm_call_count = 11
    next_scene = feedback_policy._scene_features(
        CarState(x=next_x, y=next_y, vx=next_speed, psi=next_psi),
        next_s,
        opponents=[
            {"car_idx": 1, "px": 13.2, "py": 0.1, "vx": 1.85},
            {"car_idx": 2, "px": 9.5, "py": -0.1, "vx": 2.25},
        ],
    )
    feedback = next_scene["memory"]["previous_decision_feedback"]
    assert feedback["executed_mode"] == "stable"
    assert feedback["elapsed_seconds"] == 1.0
    assert feedback["ego_progress_change_meters"] == 1.0
    assert feedback["ego_lateral_change_meters_left_positive"] == 0.05
    assert feedback["ego_speed_change_meters_per_second"] == 0.2

    assert all_race.policies[0].observation_variant == "all_opponents"
    assert baseline_standard_race.policies[0].observation_variant == "baseline"
    print("PASS independent LLM observation variants", flush=True)
    return [
        "baseline",
        "all_opponents",
        "no_lap_progress",
        "raceline_speed",
        "decision_feedback",
    ]


def run_track_checks(seconds):
    results = []
    for track_name in ("berlin", "modena", "rounded_rectangle"):
        config = SimulationConfig(
            policies=("static_alpharacer",),
            duration=seconds,
            track_name=track_name,
            initial_speed=2.0,
            save_trace=False,
            show_plot=False,
            quiet=True,
        )
        trace, elapsed = _run_case(f"track_{track_name}", config)
        progress = np.asarray(trace["car_px"], dtype=float)[:, 0]
        result = {
            "track": track_name,
            "duration_seconds": seconds,
            "progress_gain_m": float(progress[-1] - progress[0]),
            "collision_steps": int(np.sum(trace["collision"])),
            "boundary_violation_steps": int(
                np.sum(np.asarray(trace["car_track_boundary_violation"])[:, 0])
            ),
            "recovery_steps": int(
                np.sum(np.asarray(trace["car_offroad_recovery_active"])[:, 0])
            ),
            "solver_failure_steps": int(
                np.sum(~np.asarray(trace["car_mpc_solver_success"], dtype=bool)[:, 0])
            ),
            "elapsed_seconds": elapsed,
        }
        results.append(result)
        print(
            f"PASS track {track_name}: progress={result['progress_gain_m']:.2f}m, "
            f"boundary={result['boundary_violation_steps']}, "
            f"solver_failures={result['solver_failure_steps']} ({elapsed:.2f}s)",
            flush=True,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if args.track_seconds <= 0:
        parser.error("--track-seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulation_dt": params.SIM_DT,
        "observation_checks": run_observation_checks(),
        "observation_snapshot": run_observation_snapshot_check(),
        "observation_variants": run_observation_variant_checks(),
        "interface_checks": run_interface_checks(),
        "track_checks": run_track_checks(args.track_seconds),
    }
    output = args.output_dir / "preflight_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved preflight summary to {output}")


if __name__ == "__main__":
    main()
