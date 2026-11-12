"""Direct-parameter LLM policy for q, zeta, s1, s2 and s3."""

import json
from pathlib import Path

import numpy as np

from config import params as sim_params
from ..mpc import MPCPolicy

from .base import LLMPolicyBase


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


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
        include_raceline_speed=False,
        include_current_parameters=False,
        include_decision_feedback=False,
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
        if prompt_module not in {"base", "base_plus_mechanics"}:
            raise ValueError(f"Unknown parameter prompt module: {prompt_module}")

        self.prompt_module = str(prompt_module)
        self.prompt_module_version = (
            "mechanics_v1" if prompt_module == "base_plus_mechanics" else "none"
        )
        self.prompt_variant = (
            "direct_params_modular_v1" if prompt_variant == "raw" else str(prompt_variant)
        )
        self.include_raceline_speed = bool(include_raceline_speed)
        self.include_current_parameters = bool(include_current_parameters)
        self.include_decision_feedback = bool(include_decision_feedback)
        if (
            self.include_decision_feedback
            and self.prompt_variant == "direct_params_modular_v1"
        ):
            self.prompt_variant = "direct_params_memory_feedback_v1"
        self.observation_variant = "baseline"
        self.input_schema_version = (
            "direct_params_memory_feedback_v1"
            if self.include_decision_feedback
            else "direct_params_modular_v1"
        )
        self.input_modules = {
            "base": True,
            "raceline_speed": self.include_raceline_speed,
            "current_parameters": self.include_current_parameters,
            "decision_feedback": self.include_decision_feedback,
        }
        self._previous_prompt_observation = None
        self._previous_selected_parameters = None
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
        extra_prompts = (
            (prompt_dir / "params_mechanics_module.txt",)
            if prompt_module == "base_plus_mechanics"
            else ()
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
        traffic = [
            {
                "id": opponent["id"],
                "relative_longitudinal_position_m": self._rounded(
                    opponent["delta_s"], 2
                ),
                "relative_lateral_position_m": self._rounded(
                    opponent["delta_e"], 3
                ),
                "relative_speed_mps": self._rounded(
                    opponent["relative_speed"], 2
                ),
            }
            for opponent in measured["opponents"]
        ]
        road_samples = [
            {
                "distance_ahead_m": self._rounded(sample["distance"], 1),
                "centerline_curvature_per_m": self._rounded(
                    sample["curvature"], 3
                ),
                "raceline_offset_m": self._rounded(
                    sample["raceline_offset"], 3
                ),
            }
            for sample in measured["road"]["samples"]
        ]
        scene = {
            "ego_state": {
                "speed_mps": self._rounded(ego["body_speed"], 2),
                "lateral_position_m": self._rounded(ego["e"], 3),
                "heading_error_rad": self._rounded(ego["heading_error"], 3),
            },
            "traffic_state": {"all_opponents": traffic},
            "road_preview": {"samples_near_to_far": road_samples},
        }
        if self.include_raceline_speed:
            scene["raceline_speed"] = {
                "target_speed_mps_at_preview_points": [
                    self._rounded(sample["raceline_speed"], 2)
                    for sample in measured["road"]["samples"]
                ]
            }
        if self.include_current_parameters:
            scene["current_parameters"] = {
                name: self._rounded(value)
                for name, value in zip(PARAMETER_NAMES, self.params)
            }
        return scene

    def build_prompt_payload(self, observed):
        """Optionally prepend the previous observation, parameters and events."""
        if not self.include_decision_feedback:
            return observed
        previous_step = None
        if self._previous_prompt_observation is not None:
            previous_step = {
                "observation": json.loads(json.dumps(self._previous_prompt_observation)),
                "selected_parameters": dict(self._previous_selected_parameters),
                "events_during_interval": {
                    "collision": bool(self._interval_collision),
                    "off_track": bool(self._interval_off_track),
                },
            }
        return {"previous_step": previous_step, "current_observation": observed}

    def response_schema(self):
        """Require exactly five bounded numerical fields from the model."""
        properties = {}
        for name in PARAMETER_NAMES:
            lower, upper = self.param_bounds[name]
            properties[name] = {
                "type": "number",
                "minimum": float(lower),
                "maximum": float(upper),
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
        """Save this observation and executed parameters for the next prompt."""
        if not self.include_decision_feedback:
            return
        self._previous_prompt_observation = json.loads(json.dumps(observed))
        self._previous_selected_parameters = {
            name: self._rounded(value)
            for name, value in zip(PARAMETER_NAMES, self.params)
        }
        self._interval_collision = False
        self._interval_off_track = False

    def record_interval_events(self, *, collision, off_track):
        """Accumulate physical events until the next parameter decision."""
        if not self.include_decision_feedback:
            return
        self._interval_collision |= bool(collision)
        self._interval_off_track |= bool(off_track)
