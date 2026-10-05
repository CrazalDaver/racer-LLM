import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np

from config import duration_to_steps, params as sim_params
from simulation_core.config import LLM_OBSERVATION_VARIANTS
from state import CarState

from .mpc import MPCPolicy
from .utils import angle_delta, progress_delta, track_curvature_at


EVIDENCE_FIRST_BASIS_LABELS = (
    "no_front_vehicle",
    "closing_on_front_vehicle",
    "front_vehicle_pulling_away",
    "front_gap_approximately_constant",
    "no_rear_vehicle",
    "rear_vehicle_closing",
    "rear_vehicle_falling_behind",
    "rear_gap_approximately_constant",
)

OBSERVATION_VARIANT_INSTRUCTIONS = {
    "all_opponents": (
        "Observation ablation: traffic.additional_opponents_near_to_far contains "
        "every opponent not already represented by traffic.front or traffic.rear. "
        "Each entry uses the same factual fields and also states front or rear."
    ),
    "no_lap_progress": (
        "Observation ablation: absolute lap progress is intentionally omitted; "
        "use only the supplied local road and traffic state."
    ),
    "raceline_speed": (
        "Observation ablation: every road-preview sample additionally contains "
        "raceline_target_speed_meters_per_second at that location."
    ),
    "decision_feedback": (
        "Observation ablation: memory.previous_decision_feedback is null before "
        "the first completed decision interval. Afterwards it reports objective "
        "changes since the previous LLM decision. Lateral change is left-positive; "
        "positive opponent gap change means separation increased."
    ),
}

class LLMAlphaRacerPolicy(MPCPolicy):
    """
    alpha-RACER policy whose five parameters are supplied by a local LLM.

    The LLM only chooses q, zeta, s1, s2, s3. MPC still computes the actual
    throttle/steering command, so bad LLM responses can be clipped or ignored.
    """

    def __init__(
        self,
        *args,
        model_name="qwen2.5:7b",
        base_url="http://localhost:11434",
        timeout=3.0,
        temperature=0.0,
        update_interval_seconds=1.0,
        decision_dt=None,
        history_length=None,
        output_mode="auto", # Public compatibility option selecting the LLM response format: "mode", "params", or "auto".
        param_bounds=None,
        system_prompt_path=None,
        prompt_variant="raw",
        observation_variant="baseline",
        include_raceline_speed=False,
        include_current_parameters=False,
        include_decision_feedback=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = float(np.clip(temperature, 0.0, 2.0))
        self.update_interval_seconds = max(float(update_interval_seconds), 0.0)
        self.decision_dt = sim_params.CONTROL_DT if decision_dt is None else float(decision_dt)
        self._update_interval_calls = duration_to_steps(
            self.update_interval_seconds, self.decision_dt
        )
        if history_length is None:
            history_length = sim_params.LLM_RECENT_MODES_MAX_LENGTH
        self.history_length = max(0, int(history_length))
        self.response_format = self._resolve_response_format(output_mode, model_name)
        self.prompt_variant = str(prompt_variant)
        self.observation_variant = str(observation_variant)
        if self.observation_variant not in LLM_OBSERVATION_VARIANTS:
            raise ValueError(
                f"Unknown LLM observation variant: {self.observation_variant}"
            )
        if self.response_format == "params" and self.prompt_variant == "raw":
            self.prompt_variant = "direct_params_modular_v1"
        self.input_schema_version = (
            "direct_params_modular_v1"
            if self.response_format == "params"
            else "mode_legacy_v1"
        )
        self.include_raceline_speed = bool(include_raceline_speed)
        self.include_current_parameters = bool(include_current_parameters)
        self.include_decision_feedback = bool(include_decision_feedback)
        if self.response_format != "params" and any((
            self.include_raceline_speed,
            self.include_current_parameters,
            self.include_decision_feedback,
        )):
            raise ValueError(
                "Direct-parameter input-module flags require output_mode='params'"
            )
        self.input_modules = {
            "base": True,
            "raceline_speed": self.include_raceline_speed,
            "current_parameters": self.include_current_parameters,
            "decision_feedback": self.include_decision_feedback,
        }
        if self.response_format == "params" and self.observation_variant != "baseline":
            raise ValueError(
                "Direct-parameter policies use independent input-module booleans; "
                "legacy llm_observation_variant values apply only to mode policies"
            )
        if self.response_format not in {"mode", "params"}:
            raise ValueError(f"Unknown LLM response format: {output_mode}")
        self.llm_decision_history = []
        self.llm_raw_io_history = []
        self._last_raw_llm_response = ""
        self._last_llm_done_reason = ""
        self._last_llm_eval_count = -1
        self._last_llm_prompt_eval_count = -1
        self._last_llm_latency_seconds = np.nan
        self._llm_call_count = 0
        self.policy_decision_count = 0
        self.last_llm_reason = "not_called"
        self.last_llm_basis = "not_called"
        self.last_llm_requested_mode = "stable"
        self.param_bounds = (
            param_bounds if param_bounds is not None else sim_params.LLM_PARAMETER_BOUNDS.copy()
        )
        self.last_llm_params = self.params.copy()
        self.last_llm_error = None
        self._warned_llm_error = False
        self.mode_presets = {
            name: values.copy() for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        self.current_mode = "stable"
        self.last_llm_assessment = {}
        self.last_required_assessment = {}
        self.last_assessment_consistent = True
        self.last_explicit_rule_mode = ""
        self.last_explicit_rule_consistent = True
        self._previous_decision_feedback_state = None
        self._pending_decision_feedback_state = None
        self._oracle_rollout_frozen = False
        if system_prompt_path is None:
            filename = "params_system.txt" if self.response_format == "params" else "mode_system.txt"
            system_prompt_path = Path(__file__).with_name("prompts") / filename
        self.system_prompt_path = Path(system_prompt_path)
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        if self.observation_variant == "no_lap_progress":
            self.system_prompt = self.system_prompt.replace(
                "- ego.lap_progress_percent: ego progress around the current lap, from 0 to 100.\n",
                "",
            )
        variant_instruction = (
            OBSERVATION_VARIANT_INSTRUCTIONS.get(self.observation_variant)
            if self.response_format == "mode"
            else None
        )
        if variant_instruction:
            self.system_prompt = f"{self.system_prompt}\n\n{variant_instruction}"

    def _resolve_response_format(self, output_mode, model_name) -> str:
        # Select discrete-mode or direct-parameter responses from the explicit option or model name.
        requested_format = str(output_mode).strip().lower().replace("-", "_")
        if requested_format not in {"", "auto"}:
            return requested_format
        model_key = str(model_name).strip().lower().replace("-", "_")
        if any(token in model_key for token in ("params", "param", "direct")):
            return "params"
        return "mode"

    @staticmethod
    def _rounded(value, digits=3):  # for standarized JSON output and logging
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)

    @classmethod
    def _side_and_magnitude(cls, value, magnitude_name, digits=1):
        """Encode a signed lateral quantity without exposing a sign convention."""
        if value is None or not np.isfinite(value):
            return None
        value = float(value)
        magnitude = cls._rounded(abs(value), digits)
        if magnitude == 0.0:
            side = "center"
        else:
            side = "left" if value > 0.0 else "right"
        return {"side": side, magnitude_name: magnitude}

    @classmethod
    def _turn_and_magnitude(cls, value, magnitude_name, digits=1):
        """Encode signed left-positive road geometry as direction plus magnitude."""
        if value is None or not np.isfinite(value):
            return None
        value = float(value)
        magnitude = cls._rounded(abs(value), digits)
        if magnitude == 0.0:
            direction = "straight"
        else:
            direction = "left" if value > 0.0 else "right"
        return {"direction": direction, magnitude_name: magnitude}

    @staticmethod
    def _gap_speed_is_constant(closing_speed):
        deadband = float(sim_params.LLM_RELATIVE_SPEED_DEADBAND_MPS)
        return abs(float(closing_speed)) <= deadband + 1.0e-12

    @classmethod
    def _gap_motion(cls, closing_speed):
        """Describe whether a longitudinal gap is shrinking or growing."""
        value = float(closing_speed)
        if cls._gap_speed_is_constant(value):
            trend = "constant"
        elif value > 0.0:
            trend = "closing"
        else:
            trend = "opening"
        return {
            "trend": trend,
            "speed_meters_per_second": cls._rounded(abs(value), digits=2),
        }

    def _decision_history_payload(self) -> list:
        """Return only recent executed modes, oldest to newest."""
        return [
            item["mode"]
            for item in self.llm_decision_history[-self.history_length:]
        ]

    def _road_preview(self, current_s: float, lateral_limit: float) -> dict:
        """Describe upcoming road geometry without classifying its difficulty."""
        preview_length = max(float(sim_params.LLM_ROAD_PREVIEW_LENGTH_METERS), 0.0)
        sample_count = max(int(sim_params.LLM_ROAD_PREVIEW_SAMPLE_COUNT), 1)
        distances = np.linspace(0.0, preview_length, sample_count)
        _, _, current_heading = self.track.interpolate(current_s)

        samples = []
        for distance in distances:
            sample_s = current_s + float(distance)
            center_x, center_y, center_heading = self.track.interpolate(sample_s)
            raceline_left_offset_percent = None
            raceline_target_speed = None
            if self.raceline is not None:
                raceline_idx = self.raceline.nearest_index(center_x, center_y)
                raceline_s = float(self.raceline.s[raceline_idx])
                raceline_state = self._raceline_frenet_point(
                    raceline_s,
                    centerline_s_hint=sample_s,
                )
                raceline_left_offset_percent = (
                    100.0 * raceline_state["lateral_e"] / max(lateral_limit, 1e-6)
                )
                raceline_target_speed = raceline_state["speed"]

            sample = {
                "distance_ahead_meters": self._rounded(distance, digits=1),
                "centerline_curvature": self._turn_and_magnitude(
                    track_curvature_at(self.track, sample_s),
                    "magnitude_per_meter",
                    digits=3,
                ),
                "centerline_turn_from_current": self._turn_and_magnitude(
                    math.degrees(angle_delta(center_heading, current_heading)),
                    "degrees",
                    digits=1,
                ),
                "raceline_offset_from_centerline": self._side_and_magnitude(
                    raceline_left_offset_percent,
                    "percent_of_half_width",
                    digits=1,
                ),
            }
            if self.observation_variant == "raceline_speed":
                sample["raceline_target_speed_meters_per_second"] = self._rounded(
                    raceline_target_speed, digits=2
                )
            samples.append(sample)

        return {
            "preview_length_meters": self._rounded(preview_length, digits=1),
            "samples_near_to_far": samples,
        }

    def _opponent_feature(
        self,
        state,
        current_s,
        current_py,
        lateral_limit,
        opponent,
    ) -> dict:
        delta_s = float(progress_delta(opponent["px"], current_s, self.track.total_length))
        position = "front" if delta_s >= 0.0 else "rear"
        ego_speed = max(float(state.vx), 0.0)
        opponent_speed = max(float(opponent.get("vx", 0.0)), 0.0)

        abs_delta_s = abs(delta_s)
        if position == "front":
            closing_speed = ego_speed - opponent_speed
        else:
            closing_speed = opponent_speed - ego_speed
        ttc_seconds = (
            self._rounded(abs_delta_s / closing_speed, digits=2)
            if closing_speed > 0.0
            and not self._gap_speed_is_constant(closing_speed)
            else None
        )

        return {
            "id": int(opponent.get("car_idx", -1)),
            "position": position,
            "longitudinal_gap_meters": self._rounded(abs_delta_s, digits=2),
            "lateral_offset_from_ego": self._side_and_magnitude(
                100.0
                * (float(opponent.get("py", current_py)) - float(current_py))
                / max(lateral_limit, 1e-6),
                "percent_of_half_width",
                digits=1,
            ),
            "gap_motion": self._gap_motion(closing_speed),
            "ttc_seconds": ttc_seconds,  # null if donot approach
        }

    def _direct_parameter_road_modules(self, current_s: float) -> tuple[dict, dict]:
        """Build factual road and optional speed modules on one shared grid."""
        preview_length = max(float(sim_params.LLM_ROAD_PREVIEW_LENGTH_METERS), 0.0)
        sample_count = max(int(sim_params.LLM_ROAD_PREVIEW_SAMPLE_COUNT), 1)
        distances = np.linspace(0.0, preview_length, sample_count)
        road_samples = []
        target_speeds = []
        centerline_s_hint = current_s
        for distance in distances:
            sample_s = current_s + float(distance)
            raceline_offset = None
            raceline_speed = None
            if self.raceline is not None:
                center_x, center_y, _ = self.track.interpolate(sample_s)
                raceline_idx = self.raceline.nearest_index(center_x, center_y)
                raceline_state = self._raceline_frenet_point(
                    float(self.raceline.s[raceline_idx]),
                    centerline_s_hint=centerline_s_hint,
                )
                centerline_s_hint = raceline_state["track_s"]
                raceline_offset = raceline_state["lateral_e"]
                raceline_speed = raceline_state["speed"]
            road_samples.append({
                "distance_ahead_m": self._rounded(distance, digits=1),
                "centerline_curvature_per_m": self._rounded(
                    track_curvature_at(self.track, sample_s), digits=3
                ),
                "raceline_offset_m": self._rounded(raceline_offset, digits=3),
            })
            target_speeds.append(self._rounded(raceline_speed, digits=2))
        return (
            {"samples_near_to_far": road_samples},
            {"target_speed_mps_at_preview_points": target_speeds},
        )

    def _direct_parameter_feedback(
        self, state: CarState, current_s: float, opponents
    ):
        current_gaps = {
            int(opponent.get("car_idx", -1)): abs(
                float(progress_delta(
                    opponent["px"], current_s, self.track.total_length
                ))
            )
            for opponent in opponents
        }
        current = {
            "control_call": self._llm_call_count,
            "s": float(current_s),
            "speed": max(float(state.vx), 0.0),
            "gaps": current_gaps,
        }
        self._pending_decision_feedback_state = current
        previous = self._previous_decision_feedback_state
        if previous is None:
            return None
        gap_changes = [
            {
                "id": car_idx,
                "longitudinal_gap_change_m": self._rounded(
                    current_gaps[car_idx] - previous["gaps"][car_idx], digits=2
                ),
            }
            for car_idx in sorted(current_gaps.keys() & previous["gaps"].keys())
        ]
        return {
            "previous_progress_change_m": self._rounded(
                current["s"] - previous["s"], digits=2
            ),
            "previous_speed_change_mps": self._rounded(
                current["speed"] - previous["speed"], digits=2
            ),
            "previous_gap_changes": gap_changes,
        }

    def _direct_parameter_scene_features(
        self, state: CarState, current_s: float, opponents=None
    ) -> dict:
        """Return the modular, rule-free direct-parameter observation."""
        _, current_e = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        _, _, center_heading = self.track.interpolate(current_s)
        raw_opponents = sorted(
            opponents or [], key=lambda opponent: int(opponent.get("car_idx", -1))
        )
        traffic = []
        for opponent in raw_opponents:
            traffic.append({
                "id": int(opponent.get("car_idx", -1)),
                "relative_longitudinal_position_m": self._rounded(
                    progress_delta(
                        opponent["px"], current_s, self.track.total_length
                    ),
                    digits=2,
                ),
                "relative_lateral_position_m": self._rounded(
                    float(opponent.get("py", current_e)) - float(current_e),
                    digits=3,
                ),
                "relative_speed_mps": self._rounded(
                    max(float(opponent.get("vx", 0.0)), 0.0)
                    - max(float(state.vx), 0.0),
                    digits=2,
                ),
            })
        road_preview, raceline_speed = self._direct_parameter_road_modules(current_s)
        scene = {
            "ego_state": {
                "speed_mps": self._rounded(max(float(state.vx), 0.0), digits=2),
                "lateral_position_m": self._rounded(current_e, digits=3),
                "heading_error_rad": self._rounded(
                    angle_delta(float(state.psi), center_heading), digits=3
                ),
            },
            "traffic_state": {"all_opponents": traffic},
            "road_preview": road_preview,
        }
        if self.include_raceline_speed:
            scene["raceline_speed"] = raceline_speed
        if self.include_current_parameters:
            scene["current_parameters"] = {
                name: self._rounded(value)
                for name, value in zip(("q", "zeta", "s1", "s2", "s3"), self.params)
            }
        if self.include_decision_feedback:
            scene["decision_feedback"] = self._direct_parameter_feedback(
                state, current_s, raw_opponents
            )
        return scene

    def _scene_features(self, state: CarState, current_s: float, opponents=None) -> dict:
        if self.response_format == "params":
            return self._direct_parameter_scene_features(
                state, current_s, opponents=opponents
            )
        _, current_py = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        lateral_limit = float(
            getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        )
        raw_opponents = opponents or []
        opponent_features = [
            self._opponent_feature(
                state,
                current_s,
                current_py,
                lateral_limit,
                opponent,
            )
            for opponent in raw_opponents
        ]

        def nearest(position):
            # Return the nearest opponent feature in the given position ("front" or "rear"),
            candidates = [
                feature
                for feature in opponent_features
                if feature["position"] == position
            ]
            if not candidates:
                return None
            feature = min(candidates, key=lambda item: item["longitudinal_gap_meters"])
            return {
                "id": feature["id"],
                "longitudinal_gap_meters": feature["longitudinal_gap_meters"],
                "lateral_offset_from_ego": feature["lateral_offset_from_ego"],
                "gap_motion": feature["gap_motion"],
                "ttc_seconds": feature["ttc_seconds"],
            }

        front = nearest("front")
        rear = nearest("rear")
        ego_features = {
            "lap_progress_percent": self._rounded(
                100.0 * (float(current_s) % self.track.total_length) / self.track.total_length,
                digits=1,
            ),
            "lateral_position": self._side_and_magnitude(
                100.0 * float(current_py) / max(lateral_limit, 1e-6),
                "percent_of_half_width",
                digits=1,
            ),
            "speed_percent_of_maximum": self._rounded(
                100.0 * max(float(state.vx), 0.0) / max(self.max_speed, 1e-6),
                digits=1,
            ),
        }
        if self.observation_variant == "no_lap_progress":
            ego_features.pop("lap_progress_percent")

        traffic = {"front": front, "rear": rear}
        if self.observation_variant == "all_opponents":
            represented_ids = {
                vehicle["id"] for vehicle in (front, rear) if vehicle is not None
            }
            traffic["additional_opponents_near_to_far"] = sorted(
                [
                    feature
                    for feature in opponent_features
                    if feature["id"] not in represented_ids
                ],
                key=lambda feature: feature["longitudinal_gap_meters"],
            )

        scene = {
            "ego": ego_features,
            "traffic": traffic,
            "road_preview": self._road_preview(current_s, lateral_limit),
        }
        if self.response_format == "mode":
            scene["memory"] = {
                "previous_mode": self.current_mode,
                # Deliberately omit previous_basis in this memory ablation.
                "recent_modes_oldest_to_newest": self._decision_history_payload(),
            }
            if self.observation_variant == "decision_feedback":
                scene["memory"]["previous_decision_feedback"] = (
                    self._decision_feedback(
                        state,
                        current_s,
                        current_py,
                        front,
                        rear,
                    )
                )
        return scene

    def _decision_feedback(self, state, current_s, current_e, front, rear):
        """Return objective changes since the previous LLM decision interval."""
        current_gaps = {
            vehicle["id"]: vehicle["longitudinal_gap_meters"]
            for vehicle in (front, rear)
            if vehicle is not None
        }
        current = {
            "control_call": self._llm_call_count,
            "s": float(current_s),
            "e": float(current_e),
            "speed": max(float(state.vx), 0.0),
            "gaps": current_gaps,
        }
        self._pending_decision_feedback_state = current
        previous = self._previous_decision_feedback_state
        if previous is None:
            return None

        gap_changes = []
        for car_idx in sorted(current_gaps.keys() & previous["gaps"].keys()):
            gap_changes.append({
                "id": int(car_idx),
                "longitudinal_gap_change_meters": self._rounded(
                    current_gaps[car_idx] - previous["gaps"][car_idx], digits=2
                ),
            })
        return {
            "executed_mode": previous["mode"],
            "elapsed_seconds": self._rounded(
                (current["control_call"] - previous["control_call"])
                * self.decision_dt,
                digits=2,
            ),
            "ego_progress_change_meters": self._rounded(
                current["s"] - previous["s"], digits=2
            ),
            "ego_lateral_change_meters_left_positive": self._rounded(
                current["e"] - previous["e"], digits=3
            ),
            "ego_speed_change_meters_per_second": self._rounded(
                current["speed"] - previous["speed"], digits=2
            ),
            "opponent_gap_changes": gap_changes,
        }

    def _commit_decision_feedback_state(self):
        if self._pending_decision_feedback_state is None:
            return
        self._previous_decision_feedback_state = {
            **self._pending_decision_feedback_state,
            "mode": self.current_mode,
        }
        self._pending_decision_feedback_state = None

    def _record_llm_decision(self):
        if self.history_length <= 0:
            return
        self.llm_decision_history.append(
            {
                "call": self._llm_call_count,
                "mode": self.current_mode,
                "requested_mode": self.last_llm_requested_mode,
                "basis": self.last_llm_basis,
                "reason": self.last_llm_reason,
                "params": self.params.copy(),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def _prompt(self, state: CarState, current_s: float, opponents=None) -> str:
        payload = self._scene_features(state, current_s, opponents=opponents)  # real-time scene features
        # Preserve the existing trace field without duplicating a second
        # assessment block inside the model input.
        self.last_required_assessment = json.loads(json.dumps(payload))
        # Whitespace keeps the structured observation readable in raw prompt
        # logs while preserving valid JSON and the exact same data semantics.
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _output_schema(self) -> dict:
        properties = {}
        required = []
        if self.response_format == "mode":
            properties["mode"] = {
                "type": "string",
                "enum": list(self.mode_presets),
            }
            required.append("mode")
        else:
            for name in ("q", "zeta", "s1", "s2", "s3"):
                lo, hi = self.param_bounds[name]
                properties[name] = {"type": "number", "minimum": float(lo), "maximum": float(hi)}
                required.append(name)
        if self.response_format == "mode":
            properties["basis"] = {"type": "string", "maxLength": 40}
            properties["reason"] = {"type": "string", "maxLength": 120}
            required.extend(["basis", "reason"])
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _query_llm(self, prompt: str) -> dict:
        self._last_raw_llm_response = ""
        self._last_llm_done_reason = ""
        self._last_llm_eval_count = -1
        self._last_llm_prompt_eval_count = -1
        self._last_llm_latency_seconds = np.nan
        
        # ollama query
        payload = {
            "model": self.model_name,
            "system": self.system_prompt,  # local file that describes tasks definition and mode explanation
            "prompt": prompt,
            "format": self._output_schema(),
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 192},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        finally:
            self._last_llm_latency_seconds = time.perf_counter() - started
            # print(raw)
        body = json.loads(raw)
        self._last_raw_llm_response = str(body.get("response", ""))
        self._last_llm_done_reason = str(body.get("done_reason", ""))
        self._last_llm_eval_count = int(body.get("eval_count") or -1)
        self._last_llm_prompt_eval_count = int(body.get("prompt_eval_count") or -1)
        return self._extract_json(self._last_raw_llm_response)

    def _record_llm_raw_io(self, prompt: str, error=None):
        # for debugging
        self.llm_raw_io_history.append(
            {
                "call": self._llm_call_count,
                "prompt": prompt,
                "response": self._last_raw_llm_response,
                "error": "" if error is None else str(error),
                "done_reason": self._last_llm_done_reason,
                "eval_count": self._last_llm_eval_count,
                "prompt_eval_count": self._last_llm_prompt_eval_count,
                "latency_seconds": self._last_llm_latency_seconds,
            }
        )

    def _select_explicit_rule_mode(
        self, state: CarState, current_s: float, opponents=None
    ) -> tuple[str, str]:
        """Shared ordered rules for the explicit-prompt and rule-only baselines."""
        context = self._mode_context(state, current_s, opponents)
        lateral_limit = max(float(self.mpc.track_half_width), 1e-6)
        if abs(context["current_py"]) >= 0.9 * lateral_limit:
            return "stable", "boundary_caution"

        tiny_front_gap = (
            context["nearest_front_delta"] is not None
            and context["nearest_front_delta"] < self.mpc.min_longitudinal_separation
            and context["nearest_front_lateral_gap"] is not None
            and context["nearest_front_lateral_gap"] < 0.762 * lateral_limit
        )
        if tiny_front_gap:
            return "stable", "tiny_front_gap"
        if context["close_slow_front"]:
            return "overtake", "slower_close_front"
        if context["faster_rear"]:
            return "block", "faster_closing_rear"

        front_clear = (
            context["nearest_front_delta"] is None
            or context["nearest_front_delta"] > sim_params.LLM_OVERTAKE_DISTANCE
        )
        preview_distances = np.linspace(
            0.0,
            max(float(sim_params.LLM_ROAD_PREVIEW_LENGTH_METERS), 0.0),
            max(int(sim_params.LLM_ROAD_PREVIEW_SAMPLE_COUNT), 1),
        )
        max_preview_curvature = max(
            abs(track_curvature_at(self.track, current_s + float(distance)))
            for distance in preview_distances
        )
        if front_clear and max_preview_curvature < sim_params.LLM_PUSH_MAX_ABS_CURVATURE:
            return "push", "clear_gentle_road"
        return "stable", "default"

    def _sanitize_params(self, raw_params: dict) -> np.ndarray:
        
        if self.response_format == "params":
            self.current_mode = "custom"
            values = []
            for name, fallback in zip(("q", "zeta", "s1", "s2", "s3"), self.params):
                lo, hi = self.param_bounds[name]
                value = float(raw_params.get(name, fallback))
                values.append(np.clip(value, lo, hi))
            return np.asarray(values, dtype=float)

        mode = raw_params.get("mode")
        if mode is not None:
            mode_key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
            if mode_key not in self.mode_presets:
                raise ValueError(f"Unknown LLM policy mode: {mode}")
            self.current_mode = mode_key
            return self.mode_presets[mode_key].copy()

        self.current_mode = "custom"
        values = []
        for name, fallback in zip(("q", "zeta", "s1", "s2", "s3"), self.params):
            lo, hi = self.param_bounds[name]
            value = float(raw_params.get(name, fallback))
            values.append(np.clip(value, lo, hi))
        return np.asarray(values, dtype=float)

    def _sanitize_reason(self, raw_params: dict) -> str:
        """Return one short, single-line explanation without affecting control."""
        reason = " ".join(str(raw_params.get("reason", "not provided")).split())
        if not reason:
            reason = "not provided"
        return reason[:160]

    def _sanitize_basis(self, raw_params: dict) -> str:
        if "basis" in raw_params:
            return str(raw_params["basis"])[:40]
        traffic = str(raw_params.get("traffic_basis", "uncertain"))
        track = str(raw_params.get("track_basis", ""))
        return f"{traffic}|{track}"[:80] if track else traffic[:40]

    def _mode_context(self, state: CarState, current_s: float, opponents=None) -> dict:
        # Compute context information for mode guarding, including nearest opponent distances and relative speeds.
        opponents = opponents or []
        _, current_py = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        close_slow_front = False
        immediate_slow_front = False
        slow_front = False
        faster_rear = False
        nearest_front_delta = None
        nearest_rear_delta = None
        nearest_delta = None
        nearest_lateral_gap = None
        nearest_front_lateral_gap = None
        for opponent in opponents:
            delta_s = progress_delta(opponent["px"], current_s, self.track.total_length)
            if nearest_delta is None or abs(delta_s) < abs(nearest_delta):
                nearest_delta = delta_s
                nearest_lateral_gap = abs(current_py - opponent["py"])
            if delta_s > 0.0:
                if nearest_front_delta is None or delta_s < nearest_front_delta:
                    nearest_front_delta = delta_s
                    nearest_front_lateral_gap = abs(current_py - opponent["py"])
                if opponent["vx"] < state.vx:
                    slow_front = slow_front or delta_s < sim_params.LLM_OVERTAKE_DISTANCE
                    close_slow_front = close_slow_front or delta_s < sim_params.LLM_CLOSE_DISTANCE
                    immediate_slow_front = immediate_slow_front or delta_s < sim_params.LLM_CRITICAL_DISTANCE
            elif delta_s < 0.0 and opponent["vx"] > state.vx:
                rear_closing_speed = float(opponent["vx"]) - float(state.vx)
                rear_ttc = (
                    abs(delta_s) / rear_closing_speed
                    if rear_closing_speed > 0.1
                    else math.inf
                )
                faster_rear = faster_rear or (
                    abs(delta_s) < sim_params.LLM_OVERTAKE_DISTANCE
                    and rear_ttc <= sim_params.LLM_BLOCK_MAX_TTC_SECONDS
                )
                if nearest_rear_delta is None or abs(delta_s) < abs(nearest_rear_delta):
                    nearest_rear_delta = delta_s
        return {
            "close_slow_front": close_slow_front,
            "immediate_slow_front": immediate_slow_front,
            "slow_front": slow_front,
            "faster_rear": faster_rear,
            "nearest_front_delta": nearest_front_delta,
            "nearest_rear_delta": nearest_rear_delta,
            "nearest_delta": nearest_delta,
            "nearest_lateral_gap": nearest_lateral_gap,
            "nearest_front_lateral_gap": nearest_front_lateral_gap,
            "current_py": current_py,
        }

    def update_params_from_llm(self, state: CarState, current_s: float, opponents=None):
        # Call LLM only at the specified update interval; otherwise reuse the last parameters.
        if self._oracle_rollout_frozen:
            self.params = self.last_llm_params.copy()
            return
        self._llm_call_count += 1
        if self._llm_call_count > 1 and (self._llm_call_count - 1) % self._update_interval_calls != 0:
            self.params = self.last_llm_params.copy()
            return

        # user prompt construction
        prompt = self._prompt(state, current_s, opponents=opponents)
        self._last_raw_llm_response = ""

        # LLM query and response handling, with error recovery and logging.
        try:
            self.policy_decision_count += 1
            raw_params = self._query_llm(prompt)
            self.last_llm_reason = self._sanitize_reason(raw_params)
            self.last_llm_basis = self._sanitize_basis(raw_params)
            # Assessment is computed by code and saved directly; asking the
            # model to copy it only lengthens responses and creates a second
            # source of truth.
            self.last_llm_assessment = self.last_required_assessment.copy()
            self.last_assessment_consistent = True
            self.params = self._sanitize_params(raw_params)
            self.last_llm_requested_mode = self.current_mode
            if self.response_format == "mode":
                expected_mode, _ = self._select_explicit_rule_mode(
                    state, current_s, opponents
                )
                self.last_explicit_rule_mode = expected_mode
                self.last_explicit_rule_consistent = self.current_mode == expected_mode
            self.last_llm_params = self.params.copy()
            self.last_llm_error = None
            self._record_llm_decision()
            self._record_llm_raw_io(prompt)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, KeyError) as exc:
            self._record_llm_raw_io(prompt, error=exc)
            self.last_llm_error = str(exc)
            self.last_llm_reason = f"error: {exc}"[:160]
            self.last_llm_basis = "error"
            self.last_assessment_consistent = False
            self.params = self.last_llm_params.copy()
            if not self._warned_llm_error:
                self._log(f"LLM policy failed once, reusing last parameters: {exc}")
                self._warned_llm_error = True
        self._commit_decision_feedback_state()

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        self.update_params_from_llm(state, current_s, opponents=opponents)
        return super().get_reference_trajectory(state, current_s, opponents=opponents)

    def freeze_for_oracle_rollout(self):
        """Freeze external high-level calls inside a hypothetical branch.

        The copied lower-level MPC remains closed-loop and responsive.  This
        prevents candidate order from changing the number or timing of Ollama
        requests; deterministic in-process AlphaRacer/IBR opponents are not
        frozen by the rollout evaluator.
        """
        self._oracle_rollout_frozen = True




class ExplicitRuleLLMPolicy(LLMAlphaRacerPolicy):
    """LLM baseline given the same ordered rules as the rule-only policy."""

    def __init__(self, *args, **kwargs):
        kwargs["output_mode"] = "mode"
        kwargs["prompt_variant"] = "explicit_rules"
        kwargs["system_prompt_path"] = (
            Path(__file__).with_name("prompts") / "mode_explicit_rules_system.txt"
        )
        super().__init__(*args, **kwargs)


class StructuredBasisActionFirstLLMPolicy(LLMAlphaRacerPolicy):
    """Structured-basis control whose mode is generated before its basis."""

    def __init__(self, *args, **kwargs):
        kwargs["output_mode"] = "mode"
        kwargs["prompt_variant"] = "structured_action_first_minimal_v2"
        kwargs["system_prompt_path"] = (
            Path(__file__).with_name("prompts")
            / "mode_structured_action_first_system.txt"
        )
        super().__init__(*args, **kwargs)

    def _output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(self.mode_presets),
                },
                "basis": {
                    "type": "string",
                    "enum": list(EVIDENCE_FIRST_BASIS_LABELS),
                },
                "reason": {"type": "string", "maxLength": 120},
            },
            "required": ["mode", "basis", "reason"],
            "additionalProperties": False,
        }


class EvidenceFirstLLMPolicy(LLMAlphaRacerPolicy):
    """Detachable evidence-first variant of the raw mode policy.

    It receives exactly the same numerical observation and uses the same mode
    presets as :class:`LLMAlphaRacerPolicy`.  Only the response scaffold is
    changed: the model first emits one verifiable scene-relation label, then a
    mode, then a short explanation.  No basis-to-mode rule is supplied.
    """

    def __init__(self, *args, **kwargs):
        kwargs["output_mode"] = "mode"
        kwargs["prompt_variant"] = "evidence_first_minimal_v2"
        kwargs["system_prompt_path"] = (
            Path(__file__).with_name("prompts") / "mode_evidence_first_system.txt"
        )
        super().__init__(*args, **kwargs)

    def _output_schema(self) -> dict:
        # Property order is deliberate for Ollama's constrained generation:
        # the observable basis is generated before the tactical mode.
        return {
            "type": "object",
            "properties": {
                "basis": {
                    "type": "string",
                    "enum": list(EVIDENCE_FIRST_BASIS_LABELS),
                },
                "mode": {
                    "type": "string",
                    "enum": list(self.mode_presets),
                },
                "reason": {"type": "string", "maxLength": 120},
            },
            "required": ["basis", "mode", "reason"],
            "additionalProperties": False,
        }


class RuleBasedPolicy(LLMAlphaRacerPolicy):
    """Deterministic baseline executing the explicit-prompt decision rules.

    It observes the same state and selects from the same four mode presets. No
    language model or network request is used.
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "rule_based"
        kwargs["output_mode"] = "mode"
        kwargs["temperature"] = 0.0
        kwargs["prompt_variant"] = "rule_only"
        kwargs["system_prompt_path"] = (
            Path(__file__).with_name("prompts") / "mode_explicit_rules_system.txt"
        )
        super().__init__(*args, **kwargs)
        self.last_rule_params = self.params.copy()

    def _select_rule_mode(self, state: CarState, current_s: float, opponents=None) -> str:
        mode, basis = self._select_explicit_rule_mode(state, current_s, opponents)
        self.last_llm_basis = basis
        self.last_llm_reason = basis.replace("_", " ")
        self.last_llm_requested_mode = mode
        return mode

    def update_params_from_rules(self, state: CarState, current_s: float, opponents=None):
        if self._oracle_rollout_frozen:
            self.params = self.last_rule_params.copy()
            return
        self._llm_call_count += 1
        if self._llm_call_count > 1 and (self._llm_call_count - 1) % self._update_interval_calls != 0:
            self.params = self.last_rule_params.copy()
            return

        self.current_mode = self._select_rule_mode(state, current_s, opponents=opponents)
        self.policy_decision_count += 1
        self.last_explicit_rule_mode = self.current_mode
        self.last_explicit_rule_consistent = True
        self.params = self.mode_presets[self.current_mode].copy()
        self.last_rule_params = self.params.copy()

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        self.update_params_from_rules(state, current_s, opponents=opponents)
        return MPCPolicy.get_reference_trajectory(
            self, state, current_s, opponents=opponents
        )
