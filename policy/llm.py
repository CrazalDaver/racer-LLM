import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np

from config import duration_to_steps, params as sim_params
from state import CarState

from .mpc import MPCPolicy
from .utils import angle_delta, progress_delta, track_curvature_at

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
        safeguard_enabled=False,  # If True, apply deterministic mode-guarding rules to the LLM output.
        param_bounds=None,
        system_prompt_path=None,
        prompt_variant="raw",
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
        self.safeguard_enabled = bool(safeguard_enabled)
        self.prompt_variant = str(prompt_variant)
        if self.response_format == "params" and self.prompt_variant == "raw":
            self.prompt_variant = "direct_params_effect_aware"
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
        self.last_safeguard_override = ""
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
        if system_prompt_path is None:
            filename = "params_system.txt" if self.response_format == "params" else "mode_system.txt"
            system_prompt_path = Path(__file__).with_name("prompts") / filename
        self.system_prompt_path = Path(system_prompt_path)
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()

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
            if self.raceline is not None:
                raceline_idx = self.raceline.nearest_index(center_x, center_y)
                raceline_s = float(self.raceline.s[raceline_idx])
                raceline_state = self._raceline_frenet_point(
                    raceline_s,
                    centerline_s_hint=sample_s,
                )
                raceline_left_offset_percent = self._rounded(
                    100.0 * raceline_state["lateral_e"] / max(lateral_limit, 1e-6),
                    digits=1,
                )

            samples.append(
                {
                    "distance_ahead_meters": self._rounded(distance, digits=1),
                    "centerline_left_curvature_per_meter": self._rounded(
                        track_curvature_at(self.track, sample_s),
                        digits=3,
                    ),
                    "centerline_left_turn_degrees_from_current": self._rounded(
                        math.degrees(angle_delta(center_heading, current_heading)),
                        digits=1,
                    ),
                    "raceline_left_offset_percent_of_half_width": raceline_left_offset_percent,
                }
            )

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
            if closing_speed > 0.1
            else None
        )

        return {
            "id": int(opponent.get("car_idx", -1)),
            "position": position,
            "longitudinal_gap_meters": self._rounded(abs_delta_s, digits=2),
            "left_offset_from_ego_percent_of_half_width": self._rounded(
                100.0
                * (float(opponent.get("py", current_py)) - float(current_py))
                / max(lateral_limit, 1e-6),
                digits=1,
            ),
            "closing_speed_meters_per_second": self._rounded(closing_speed, digits=2),
            "ttc_seconds": ttc_seconds,  # null if donot approach
        }

    def _scene_features(self, state: CarState, current_s: float, opponents=None) -> dict:
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
                "left_offset_from_ego_percent_of_half_width": feature[
                    "left_offset_from_ego_percent_of_half_width"
                ],
                "closing_speed_meters_per_second": feature[
                    "closing_speed_meters_per_second"
                ],
                "ttc_seconds": feature["ttc_seconds"],
            }

        scene = {
            "ego": {
                "lap_progress_percent": self._rounded(
                    100.0 * (float(current_s) % self.track.total_length) / self.track.total_length,
                    digits=1,
                ),
                "left_position_percent_of_half_width": self._rounded(
                    100.0 * float(current_py) / max(lateral_limit, 1e-6),
                    digits=1,
                ),
                "speed_percent_of_maximum": self._rounded(
                    100.0 * max(float(state.vx), 0.0) / max(self.max_speed, 1e-6),
                    digits=1,
                ),
            },
            "traffic": {
                "front": nearest("front"),
                "rear": nearest("rear"),
            },
            "road_preview": self._road_preview(current_s, lateral_limit),
        }
        if self.response_format == "mode":
            scene["memory"] = {
                "previous_mode": self.current_mode,
                # Deliberately omit previous_basis in this memory ablation.
                "recent_modes_oldest_to_newest": self._decision_history_payload(),
            }
        return scene

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
                "safeguard_override": self.last_safeguard_override,
                "params": self.params.copy(),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def _prompt(self, state: CarState, current_s: float, opponents=None) -> str:
        payload = self._scene_features(state, current_s, opponents=opponents)  # real-time scene features
        # Preserve the existing trace field without duplicating a second
        # assessment block inside the model input.
        self.last_required_assessment = {
            "ego": payload["ego"].copy(),
            "traffic": payload["traffic"].copy(),
            "road_preview": payload["road_preview"].copy(),
        }

        # additional info for param mode, including parameter bounds and current values
        if self.response_format == "params":
            payload["parameter_bounds"] = {
                name: {"min": float(bounds[0]), "max": float(bounds[1])}
                for name, bounds in self.param_bounds.items()
            }
            payload["current_parameters"] = {
                name: self._rounded(value)
                for name, value in zip(("q", "zeta", "s1", "s2", "s3"), self.params)
            }
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
        return str(raw_params.get("basis", "uncertain"))[:40]

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

    def _apply_mode_guard(self, state: CarState, current_s: float, opponents=None):
        self.last_safeguard_override = ""
        if self.response_format != "mode":
            return

        context = self._mode_context(state, current_s, opponents)
        guarded_mode = self.current_mode

        near_contact = context["nearest_delta"] is not None and abs(context["nearest_delta"]) < sim_params.LLM_CRITICAL_DISTANCE
        near_overlap = (
            context["nearest_delta"] is not None
            and abs(context["nearest_delta"]) < sim_params.LLM_CRITICAL_DISTANCE
            and context["nearest_lateral_gap"] is not None
            and context["nearest_lateral_gap"] < self.mpc.min_lateral_separation + sim_params.LLM_SAME_LANE_MARGIN
        )
        front_clear_for_push = context["nearest_front_delta"] is None or context["nearest_front_delta"] > sim_params.LLM_OVERTAKE_DISTANCE
        push_allowed = front_clear_for_push and not near_overlap and abs(context["current_py"]) < 0.45
        tiny_front_gap = (
            context["nearest_front_delta"] is not None
            and context["nearest_front_delta"] < self.mpc.min_longitudinal_separation
            and context["nearest_front_lateral_gap"] is not None
            and context["nearest_front_lateral_gap"] < self.mpc.min_lateral_separation + sim_params.LLM_SAME_LANE_MARGIN
        )

        if state.vx < 0.35 and near_contact:
            guarded_mode = "stable"
        elif tiny_front_gap:
            if guarded_mode in {"push", "overtake", "block"}:
                guarded_mode = "stable"
        elif context["close_slow_front"]:
            if guarded_mode in {"push", "stable", "block"}:
                guarded_mode = "overtake"
        elif guarded_mode == "push" and not push_allowed:
            guarded_mode = "stable"
        elif guarded_mode == "overtake" and not context["slow_front"]:
            guarded_mode = "push" if push_allowed and state.vx > 1.0 else "stable"
        elif guarded_mode == "block" and not context["faster_rear"]:
            guarded_mode = "stable"

        if guarded_mode != self.current_mode:  # Only log when the mode actually changes due to guarding.
            self.last_safeguard_override = f"{self.current_mode}->{guarded_mode}"
            self.current_mode = guarded_mode
            self.params = self.mode_presets[guarded_mode].copy()

    def update_params_from_llm(self, state: CarState, current_s: float, opponents=None):
        # Call LLM only at the specified update interval; otherwise reuse the last parameters.
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
            self.last_safeguard_override = ""
            # Optionally validate/correct the discrete mode against the current race context.
            if self.safeguard_enabled:
                self._apply_mode_guard(state, current_s, opponents=opponents)
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

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        self.update_params_from_llm(state, current_s, opponents=opponents)
        return super().get_reference_trajectory(state, current_s, opponents=opponents)




class ExplicitRuleLLMPolicy(LLMAlphaRacerPolicy):
    """LLM baseline given the same ordered rules as the rule-only policy."""

    def __init__(self, *args, **kwargs):
        kwargs["output_mode"] = "mode"
        kwargs["safeguard_enabled"] = False
        kwargs["prompt_variant"] = "explicit_rules"
        kwargs["system_prompt_path"] = (
            Path(__file__).with_name("prompts") / "mode_explicit_rules_system.txt"
        )
        super().__init__(*args, **kwargs)


class RuleBasedPolicy(LLMAlphaRacerPolicy):
    """Deterministic baseline executing the explicit-prompt decision rules.

    It observes the same state and selects from the same four mode presets. No
    language model, network request, or additional safeguard is used.
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "rule_based"
        kwargs["output_mode"] = "mode"
        kwargs["temperature"] = 0.0
        kwargs["safeguard_enabled"] = False
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


# Backward-compatible name for existing scripts.
RuleBasedSafeguardPolicy = RuleBasedPolicy
