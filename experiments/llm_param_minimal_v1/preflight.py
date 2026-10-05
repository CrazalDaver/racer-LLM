#!/usr/bin/env python3
"""Offline acceptance checks for llm-param-minimal-v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulation import Race
from simulation_core import SimulationConfig
from run import EXPECTED_NEUTRAL, _read_json, validate_config


def main() -> None:
    config = _read_json(EXPERIMENT_DIR / "experiment.json")
    validate_config(config)
    race = Race(SimulationConfig(
        policies=("llm_alpharacer", "static_alpharacer", "static_alpharacer"),
        duration=0.1,
        track_name="rounded_rectangle",
        ego_position=2,
        random_seed=0,
        start_gap=6.0,
        start_speeds=(0.0, 0.0, 0.0),
        static_params=EXPECTED_NEUTRAL,
        llm_params=EXPECTED_NEUTRAL,
        llm_output_mode="params",
        llm_model=config["llm"]["model"],
        llm_temperature=0.0,
        llm_update_interval=1.0,
        llm_observation_variant="baseline",
        llm_include_raceline_speed=True,
        llm_include_current_parameters=False,
        llm_include_decision_feedback=False,
        save_trace=False,
        show_plot=False,
        quiet=True,
    ))
    race.environment.begin_step(0)
    observation = race.environment.capture(race.policies)
    ego = observation.car(0)
    opponents = observation.mpc_opponents(0)
    policy = race.policies[0]
    scene = policy._scene_features(ego.state, ego.s, opponents=opponents)

    assert set(scene) == {
        "ego_state", "traffic_state", "road_preview", "raceline_speed"
    }
    assert len(scene["traffic_state"]["all_opponents"]) == 2
    assert "current_parameters" not in scene
    assert "decision_feedback" not in scene
    prompt = json.loads(policy._prompt(ego.state, ego.s, opponents=opponents))
    assert prompt == scene
    assert set(policy._output_schema()["properties"]) == {
        "q", "zeta", "s1", "s2", "s3"
    }
    assert policy._output_schema()["additionalProperties"] is False
    assert np.array_equal(policy.params, EXPECTED_NEUTRAL)
    assert np.array_equal(race.policies[1].params, EXPECTED_NEUTRAL)
    assert np.array_equal(race.policies[2].params, EXPECTED_NEUTRAL)

    raw_invalid = '{"q":"invalid","zeta":0.9,"s1":0.2,"s2":0.3,"s3":0}'

    def invalid_response(_prompt):
        policy._last_raw_llm_response = raw_invalid
        return json.loads(raw_invalid)

    policy._query_llm = invalid_response
    policy.update_params_from_llm(ego.state, ego.s, opponents=opponents)
    assert np.array_equal(policy.params, EXPECTED_NEUTRAL)
    assert policy.llm_raw_io_history[-1]["response"] == raw_invalid
    assert "could not convert string to float" in policy.llm_raw_io_history[-1]["error"]

    trace = race.recorder.trace
    assert trace["car_llm_input_schema_versions"][0] == "direct_params_modular_v1"
    assert trace["car_llm_include_raceline_speed"][0] is True
    assert trace["car_llm_include_current_parameters"][0] is False
    assert trace["car_llm_include_decision_feedback"][0] is False
    print("PASS llm-param-minimal-v1 offline acceptance checks")


if __name__ == "__main__":
    main()
