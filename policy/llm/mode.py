"""Discrete-mode LLM policy using configured five-parameter presets."""

import hashlib
from pathlib import Path

import numpy as np

from config import params as sim_params
from ..mpc import MPCPolicy
from simulation_core.config import LLM_OBSERVATION_VARIANTS

from .base import LLMPolicyBase


VARIANT_INSTRUCTIONS = {
    "all_opponents": (
        "Observation ablation: traffic.additional_opponents_near_to_far contains "
        "every opponent not already represented by traffic.front or traffic.rear."
    ),
    "no_lap_progress": (
        "Observation ablation: absolute lap progress is intentionally omitted."
    ),
    "raceline_speed": (
        "Observation ablation: road-preview samples include raceline target speed."
    ),
    "decision_feedback": (
        "Observation ablation: memory.previous_decision_feedback reports measured "
        "changes since the previous completed decision interval."
    ),
}


class ModeDecisionMixin:
    """Turn a race observation and LLM response into one parameter preset."""

    response_format = "mode"

    def __init__(
        self,
        *args,
        prompt_module="base",
        prompt_variant="raw",
        observation_variant="baseline",
        history_length=None,
        include_target_memory=True,
        system_prompt_path=None,
        **kwargs,
    ):
        """Configure mode observation variants and initialize the shared flow."""
        if prompt_module not in {"base", "mode_v2"}:
            raise ValueError(f"Unknown mode prompt module: {prompt_module}")
        if observation_variant not in LLM_OBSERVATION_VARIANTS:
            raise ValueError(f"Unknown LLM observation variant: {observation_variant}")
        if history_length is None:
            history_length = sim_params.LLM_RECENT_MODES_MAX_LENGTH

        self.prompt_module = str(prompt_module)
        self.prompt_module_version = "mode_v2" if prompt_module == "mode_v2" else "none"
        self.prompt_variant = str(prompt_variant)
        self.observation_variant = str(observation_variant)
        self.history_length = max(0, int(history_length))
        self.include_target_memory = bool(include_target_memory)
        self.input_schema_version = (
            "mode_all_opponents_v1"
            if observation_variant == "all_opponents"
            else "mode_legacy_v1"
        )
        self.input_modules = {
            "base": True,
            "raceline_speed": observation_variant == "raceline_speed",
            "current_parameters": False,
            "decision_feedback": observation_variant == "decision_feedback",
        }
        self.mode_presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        self._previous_decision_feedback_state = None
        self._pending_decision_feedback_state = None

        prompt_dir = Path(__file__).with_name("prompts")
        if system_prompt_path is None:
            filename = (
                "mode_system_v2.txt"
                if prompt_module == "mode_v2"
                else "mode_system.txt"
            )
            system_prompt_path = prompt_dir / filename
        super().__init__(*args, system_prompt_path=system_prompt_path, **kwargs)
        self.current_mode = "stable"
        self.last_llm_requested_mode = "stable"

        if (
            not self.include_target_memory
            and self.system_prompt_path.name == "mode_system.txt"
        ):
            self.system_prompt = self.system_prompt.split(
                "\n\nPrevious-decision memory:\n", 1
            )[0]
        instruction = VARIANT_INSTRUCTIONS.get(self.observation_variant)
        if instruction:
            self.system_prompt = f"{self.system_prompt}\n\n{instruction}"
        if self.observation_variant == "no_lap_progress":
            self.system_prompt = self.system_prompt.replace(
                "- ego.lap_progress_percent: ego progress around the current lap, from 0 to 100.\n",
                "",
            )
        self.system_prompt_sha256 = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()

    @classmethod
    def _signed_description(cls, value, positive, negative, magnitude_name, digits):
        """Convert a signed scalar to a direction and nonnegative magnitude."""
        if value is None or not np.isfinite(value):
            return None
        value = float(value)
        magnitude = cls._rounded(abs(value), digits)
        direction = "center" if magnitude == 0.0 else (positive if value > 0.0 else negative)
        return {"side": direction, magnitude_name: magnitude}

    def format_observation(self, measured, observation):
        """Render shared scene facts with mode-selection wording and memory."""
        ego = measured["ego"]
        lateral_limit = measured["lateral_limit"]
        opponent_features = [
            self._format_opponent(opponent, lateral_limit)
            for opponent in measured["opponents"]
        ]

        def nearest(position):
            """Return the closest observed car in the requested relative position."""
            candidates = [
                item for item in opponent_features if item["position"] == position
            ]
            if not candidates:
                return None
            result = min(candidates, key=lambda item: item["longitudinal_gap_meters"])
            return {key: value for key, value in result.items() if key != "position"}

        front = nearest("front")
        rear = nearest("rear")
        ego_input = {
            "lap_progress_percent": self._rounded(
                ego["lap_progress_percent"], 1
            ),
            "lateral_position": self._signed_description(
                100.0 * ego["e"] / max(lateral_limit, 1e-6),
                "left",
                "right",
                "percent_of_half_width",
                1,
            ),
            "speed_percent_of_maximum": self._rounded(
                ego["speed_percent_of_maximum"], 1
            ),
        }
        if self.observation_variant == "no_lap_progress":
            ego_input.pop("lap_progress_percent")

        traffic = {"front": front, "rear": rear}
        if self.observation_variant == "all_opponents":
            represented = {
                item["id"] for item in (front, rear) if item is not None
            }
            traffic["additional_opponents_near_to_far"] = sorted(
                [item for item in opponent_features if item["id"] not in represented],
                key=lambda item: item["longitudinal_gap_meters"],
            )

        scene = {
            "ego": ego_input,
            "traffic": traffic,
            "road_preview": self._format_road(measured["road"], lateral_limit),
            "memory": {
                "previous_mode": self.current_mode,
                "recent_modes_oldest_to_newest": (
                    [item["mode"] for item in self.llm_decision_history[-self.history_length:]]
                    if self.history_length > 0
                    else []
                ),
            },
        }
        if self.include_target_memory:
            scene["memory"].update(self._target_memory(measured))
        if self.observation_variant == "decision_feedback":
            scene["memory"]["previous_decision_feedback"] = self._decision_feedback(
                ego, front, rear
            )
        return scene

    def _format_opponent(self, opponent, lateral_limit):
        """Render one shared opponent measurement with mode prompt wording."""
        closing_speed = opponent["closing_speed"]
        trend = (
            "constant"
            if opponent["constant_gap"]
            else ("closing" if closing_speed > 0.0 else "opening")
        )
        return {
            "id": opponent["id"],
            "position": opponent["position"],
            "longitudinal_gap_meters": self._rounded(abs(opponent["delta_s"]), 2),
            "lateral_offset_from_ego": self._signed_description(
                100.0 * opponent["delta_e"] / max(lateral_limit, 1e-6),
                "left",
                "right",
                "percent_of_half_width",
                1,
            ),
            "gap_motion": {
                "trend": trend,
                "speed_meters_per_second": self._rounded(abs(closing_speed), 2),
            },
            "ttc_seconds": self._rounded(opponent["ttc"], 2),
        }

    def _format_road(self, road, lateral_limit):
        """Render shared road samples as directions, magnitudes and percentages."""
        samples = []
        for measured_sample in road["samples"]:
            curvature = measured_sample["curvature"]
            turn = measured_sample["turn_from_current_degrees"]
            rendered = {
                "distance_ahead_meters": self._rounded(
                    measured_sample["distance"], 1
                ),
                "centerline_curvature": {
                    "direction": (
                        "straight"
                        if curvature == 0.0
                        else ("left" if curvature > 0.0 else "right")
                    ),
                    "magnitude_per_meter": self._rounded(abs(curvature), 3),
                },
                "centerline_turn_from_current": {
                    "direction": (
                        "straight"
                        if turn == 0.0
                        else ("left" if turn > 0.0 else "right")
                    ),
                    "degrees": self._rounded(abs(turn), 1),
                },
                "raceline_offset_from_centerline": self._signed_description(
                    (
                        100.0
                        * measured_sample["raceline_offset"]
                        / max(lateral_limit, 1e-6)
                        if measured_sample["raceline_offset"] is not None
                        else None
                    ),
                    "left",
                    "right",
                    "percent_of_half_width",
                    1,
                ),
            }
            if self.observation_variant == "raceline_speed":
                rendered["raceline_target_speed_meters_per_second"] = self._rounded(
                    measured_sample["raceline_speed"], 2
                )
            samples.append(rendered)
        return {
            "preview_length_meters": self._rounded(road["preview_length"], 1),
            "samples_near_to_far": samples,
        }

    def _target_memory(self, measured):
        """Report whether the previous overtake or block target still qualifies."""
        previous = self.llm_decision_history[-1] if self.llm_decision_history else {}
        target_id = previous.get("target_id")
        condition = None
        if target_id is not None and previous.get("mode") in {"overtake", "block"}:
            condition = False
            target = next(
                (
                    opponent
                    for opponent in measured["opponents"]
                    if opponent["id"] == target_id
                ),
                None,
            )
            if target is not None:
                if previous["mode"] == "overtake":
                    condition = (
                        0.0 < target["delta_s"] < sim_params.LLM_CLOSE_DISTANCE
                        and target["relative_speed"] < 0.0
                    )
                else:
                    condition = (
                        target["delta_s"] < 0.0
                        and abs(target["delta_s"])
                        < sim_params.LLM_OVERTAKE_DISTANCE
                        and target["ttc"] is not None
                        and target["ttc"] <= sim_params.LLM_BLOCK_MAX_TTC_SECONDS
                    )
        return {
            "previous_basis": previous.get("basis"),
            "previous_target_id": target_id,
            "previous_condition_still_active": condition,
        }

    def response_schema(self):
        """Require one configured mode plus a short basis and reason."""
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": list(self.mode_presets)},
                "basis": {"type": "string", "maxLength": 40},
                "reason": {"type": "string", "maxLength": 120},
            },
            "required": ["mode", "basis", "reason"],
            "additionalProperties": False,
        }

    def apply_response(self, response, observation, observed):
        """Map the returned mode to its configured five-parameter preset."""
        mode = (
            str(response["mode"])
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if mode not in self.mode_presets:
            raise ValueError(f"Unknown LLM policy mode: {response['mode']}")
        self.current_mode = mode
        self.last_llm_requested_mode = mode
        self.params = self.mode_presets[mode].copy()

    def record_decision(self, observation, observed):
        """Save the executed mode and the selected front or rear target."""
        if self.history_length <= 0:
            return
        target = observed["traffic"].get(
            {"overtake": "front", "block": "rear"}.get(self.current_mode, "")
        )
        target_id = None if target is None or target["id"] < 0 else target["id"]
        self.llm_decision_history.append(
            {
                "call": self._llm_call_count,
                "mode": self.current_mode,
                "requested_mode": self.last_llm_requested_mode,
                "basis": self.last_llm_basis,
                "reason": self.last_llm_reason,
                "params": self.params.copy(),
                **({"target_id": target_id} if self.include_target_memory else {}),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def _decision_feedback(self, ego, front, rear):
        """Compare current progress, lateral state, speed and gaps with the prior decision."""
        gaps = {
            item["id"]: item["longitudinal_gap_meters"]
            for item in (front, rear)
            if item is not None
        }
        current = {
            "control_call": self._llm_call_count,
            "s": ego["s"],
            "e": ego["e"],
            "speed": ego["body_speed"],
            "gaps": gaps,
        }
        self._pending_decision_feedback_state = current
        previous = self._previous_decision_feedback_state
        if previous is None:
            return None
        return {
            "executed_mode": previous["mode"],
            "elapsed_seconds": self._rounded(
                (current["control_call"] - previous["control_call"]) * self.decision_dt,
                2,
            ),
            "ego_progress_change_meters": self._rounded(
                current["s"] - previous["s"], 2
            ),
            "ego_lateral_change_meters_left_positive": self._rounded(
                current["e"] - previous["e"], 3
            ),
            "ego_speed_change_meters_per_second": self._rounded(
                current["speed"] - previous["speed"], 2
            ),
            "opponent_gap_changes": [
                {
                    "id": int(car_idx),
                    "longitudinal_gap_change_meters": self._rounded(
                        gaps[car_idx] - previous["gaps"][car_idx], 2
                    ),
                }
                for car_idx in sorted(gaps.keys() & previous["gaps"].keys())
            ],
        }

    def finish_decision_attempt(self, observed):
        """Associate pending measured feedback with the mode that was executed."""
        if self._pending_decision_feedback_state is None:
            return
        self._previous_decision_feedback_state = {
            **self._pending_decision_feedback_state,
            "mode": self.current_mode,
        }
        self._pending_decision_feedback_state = None


class LLMModePolicy(ModeDecisionMixin, LLMPolicyBase, MPCPolicy):
    """Select a preset with the mode decision layer, then execute it with MPC."""
