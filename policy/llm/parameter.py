"""Direct-parameter LLM policy for q, zeta, s1, s2 and s3."""

import json
from pathlib import Path

import numpy as np

from config import params as sim_params
from ..mpc import MPCPolicy
from ..utils import progress_delta

from .base import LLMPolicyBase


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
# System-prompt layers appended after params_system.txt (task, parameter
# effects, input description). Each later module adds one more level of detail.
PROMPT_MODULE_FILES = {
    "base": (),
    "reference": ("params_reference_module.txt",),
    "reference_mpc": ("params_reference_module.txt", "params_mpc_module.txt"),
}


class LLMParameterPolicy(LLMPolicyBase, MPCPolicy):
    """Ask an LLM for the five parameters, then let MPC execute them.

    Input is one immutable ``RaceObservation``. ``observe`` converts its ego,
    opponent and road fields to the JSON object documented by the parameter
    prompt. ``apply_response`` converts the model object to a bounded vector in
    the fixed paper order ``(q, zeta, s1, s2, s3)``.
    """

    response_format = "params"

    def __init__(
        self,
        *args,
        prompt_module="base",
        prompt_variant="raw",
        observation_variant="baseline",
        include_raceline_speed=None,
        include_current_parameters=False,
        include_decision_feedback=False,
        memory_length=None,
        race_duration_seconds=None,
        history_length=None,
        param_bounds=None,
        system_prompt_path=None,
        **kwargs,
    ):
        """Configure optional input modules and initialize the shared LLM flow."""
        if observation_variant != "baseline":
            raise ValueError(
                "Direct-parameter policies use input-module booleans; "
                "observation variants are reserved for mode policies"
            )
        if prompt_module not in PROMPT_MODULE_FILES:
            raise ValueError(f"Unknown parameter prompt module: {prompt_module}")

        self.prompt_module = str(prompt_module)
        self.prompt_module_version = (
            "none" if prompt_module == "base" else f"{prompt_module}_v1"
        )
        self.prompt_variant = (
            "direct_params_v3" if prompt_variant == "raw" else str(prompt_variant)
        )
        self.include_raceline_speed = bool(
            sim_params.LLM_INPUT_INCLUDE_RACELINE_SPEED
            if include_raceline_speed is None
            else include_raceline_speed
        )
        self.include_current_parameters = bool(include_current_parameters)
        self.include_decision_feedback = bool(include_decision_feedback)
        self.memory_length = max(
            1,
            int(
                sim_params.LLM_MEMORY_LENGTH
                if memory_length is None
                else memory_length
            ),
        )
        self.race_duration_seconds = float(
            sim_params.TOTAL_TIME
            if race_duration_seconds is None
            else race_duration_seconds
        )
        self.observation_variant = "baseline"
        self.input_schema_version = (
            "direct_params_v3_recent_decisions_v1"
            if self.include_decision_feedback
            else "direct_params_v3"
        )
        self.input_modules = {
            "base": True,
            "raceline_speed": self.include_raceline_speed,
            "current_parameters": self.include_current_parameters,
            "decision_feedback": self.include_decision_feedback,
        }
        # Recent-decision memory: completed (parameters, outcome) entries and
        # the open entry for the interval that started at the last decision.
        self._recent_decisions = []
        self._open_decision = None
        self._decision_observation = None
        self._interval_collision = False
        self._interval_off_track = False
        self.history_length = max(
            0,
            int(
                sim_params.LLM_RECENT_MODES_MAX_LENGTH
                if history_length is None
                else history_length
            ),
        )

        prompt_dir = Path(__file__).with_name("prompts")
        if system_prompt_path is None:
            system_prompt_path = prompt_dir / "params_system.txt"
        extra_prompts = tuple(
            prompt_dir / name for name in PROMPT_MODULE_FILES[prompt_module]
        )
        super().__init__(
            *args,
            system_prompt_path=system_prompt_path,
            extra_system_prompt_paths=extra_prompts,
            **kwargs,
        )
        self.current_mode = "custom"
        self.last_llm_requested_mode = "custom"
        self.param_bounds = (
            sim_params.LLM_PARAMETER_BOUNDS.copy()
            if param_bounds is None
            else dict(param_bounds)
        )

    def format_observation(self, measured, observation):
        """Render shared scene facts with direct-parameter field names."""
        ego = measured["ego"]
        # Speeds along/across the track are Frenet rates (s_dot, e_dot) for
        # every car; ego body speed is the MPC's own scalar speed state.
        traffic = [
            {
                "id": opponent["id"],
                "relative_longitudinal_position_m": self._rounded(
                    opponent["delta_s"], 2
                ),
                "relative_lateral_position_m": self._rounded(
                    opponent["delta_e"], 3
                ),
                "progress_rate_mps": self._rounded(opponent["s_dot"], 2),
                "lateral_rate_mps": self._rounded(opponent["e_dot"], 2),
            }
            for opponent in measured["opponents"]
        ]
        road_samples = []
        for sample in measured["road"]["samples"]:
            road_sample = {
                "distance_ahead_m": self._rounded(sample["distance"], 1),
                "centerline_curvature_per_m": self._rounded(
                    sample["curvature"], 3
                ),
                "raceline_offset_m": self._rounded(
                    sample["raceline_offset"], 3
                ),
            }
            if self.include_raceline_speed:
                road_sample["raceline_speed_mps"] = self._rounded(
                    sample["raceline_speed"], 2
                )
            road_samples.append(road_sample)
        scene = {
            "race_time": {
                "elapsed_s": self._rounded(observation.sim_time, 1),
                "total_s": self._rounded(self.race_duration_seconds, 1),
            },
            "ego_state": {
                "speed_mps": self._rounded(ego["body_speed"], 2),
                "progress_rate_mps": self._rounded(ego["s_dot"], 2),
                "lateral_rate_mps": self._rounded(ego["e_dot"], 2),
                "lateral_position_m": self._rounded(ego["e"], 3),
                "heading_error_rad": self._rounded(ego["heading_error"], 3),
            },
            "traffic_state": {"all_opponents": traffic},
            "road_preview": {"samples_near_to_far": road_samples},
        }
        if self.include_current_parameters:
            scene["current_parameters"] = {
                name: self._rounded(value)
                for name, value in zip(PARAMETER_NAMES, self.params)
            }
        return scene

    def observe(self, observation):
        """Close the open memory entry with this snapshot, then render it."""
        self._decision_observation = observation
        if self.include_decision_feedback:
            self._close_open_decision(observation)
        return super().observe(observation)

    def _close_open_decision(self, observation):
        """Add the outcome of the interval since the previous decision."""
        entry = self._open_decision
        if entry is None:
            return
        length = self.track.total_length
        travelled = {}
        for car in observation.cars:
            key = "ego" if car.car_idx == self.car_idx else f"car_{car.car_idx}"
            travelled[key] = self._rounded(
                progress_delta(car.s, entry["start_s"][car.car_idx], length), 2
            )
        self._recent_decisions.append(
            {
                "parameters": entry["parameters"],
                "interval_seconds": self._rounded(
                    observation.sim_time - entry["start_time"], 2
                ),
                "distance_travelled_m": travelled,
                "ego_touched_another_car": bool(self._interval_collision),
                "ego_center_outside_track_edges": bool(self._interval_off_track),
            }
        )
        self._recent_decisions = self._recent_decisions[-self.memory_length:]
        self._open_decision = None

    def build_prompt_payload(self, observed):
        """Optionally prepend the recent (parameters, outcome) entries."""
        if not self.include_decision_feedback:
            return observed
        return {
            "recent_decisions": json.loads(json.dumps(self._recent_decisions)),
            "current_observation": observed,
        }

    def response_schema(self):
        """Require exactly five bounded numerical fields from the model."""
        properties = {}
        for name in PARAMETER_NAMES:
            lower, upper = self.param_bounds[name]
            # Four significant digits, as written in the prompt (3.333, 0.7225);
            # apply_response still clips to the exact bounds.
            properties[name] = {
                "type": "number",
                "minimum": float(f"{lower:.4g}"),
                "maximum": float(f"{upper:.4g}"),
            }
        return {
            "type": "object",
            "properties": properties,
            "required": list(PARAMETER_NAMES),
            "additionalProperties": False,
        }

    def apply_response(self, response, observation, observed):
        """Read five response fields in paper order and clip them to bounds."""
        values = []
        for name, fallback in zip(PARAMETER_NAMES, self.params):
            lower, upper = self.param_bounds[name]
            values.append(
                np.clip(float(response.get(name, fallback)), lower, upper)
            )
        self.params = np.asarray(values, dtype=float)
        self.current_mode = "custom"
        self.last_llm_requested_mode = "custom"

    def record_decision(self, observation, observed):
        """Store the selected vector for inspection without changing control."""
        if self.history_length <= 0:
            return
        self.llm_decision_history.append(
            {
                "call": self._llm_call_count,
                "mode": "custom",
                "requested_mode": "custom",
                "basis": self.last_llm_basis,
                "reason": self.last_llm_reason,
                "params": self.params.copy(),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def finish_decision_attempt(self, observed):
        """Open a memory entry for the parameters now in effect."""
        if not self.include_decision_feedback:
            return
        observation = self._decision_observation
        self._open_decision = {
            "parameters": {
                name: self._rounded(value)
                for name, value in zip(PARAMETER_NAMES, self.params)
            },
            "start_time": float(observation.sim_time),
            "start_s": {car.car_idx: float(car.s) for car in observation.cars},
        }
        self._interval_collision = False
        self._interval_off_track = False

    def record_interval_events(self, *, collision, off_track):
        """Accumulate physical events until the next parameter decision."""
        if not self.include_decision_feedback:
            return
        self._interval_collision |= bool(collision)
        self._interval_off_track |= bool(off_track)
