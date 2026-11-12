"""Shared observe -> prompt -> LLM request -> decision control flow."""

import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np

from config import duration_to_steps, params as sim_params
from ..utils import angle_delta, progress_delta, track_curvature_at


class LLMPolicyBase:
    """Add a scheduled LLM decision before another policy computes an action.

    Subclasses turn a complete ``RaceObservation`` into model input and convert
    the parsed response into the five parameters used by the downstream policy.
    The next class in the method-resolution order remains responsible for MPC
    or learned-potential action selection.
    """

    response_format = None

    def __init__(
        self,
        *args,
        model_name="qwen2.5:7b",
        base_url="http://localhost:11434",
        timeout=3.0,
        temperature=0.0,
        thinking=False,
        update_interval_seconds=1.0,
        decision_dt=None,
        system_prompt_path=None,
        extra_system_prompt_paths=(),
        **kwargs,
    ):
        """Initialize the downstream policy, request settings and decision log."""
        super().__init__(*args, **kwargs)
        if self.response_format not in {"mode", "params"}:
            raise ValueError("LLM subclass must declare response_format")

        self.model_name = str(model_name)
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.temperature = float(np.clip(temperature, 0.0, 2.0))
        self.thinking = bool(thinking)
        self.num_predict = 192
        self.retry_count = 0
        self.fallback_behavior = "hold_previous_parameters_on_error"

        self.decision_dt = (
            float(sim_params.CONTROL_DT)
            if decision_dt is None
            else float(decision_dt)
        )
        self.update_interval_seconds = max(float(update_interval_seconds), 0.0)
        self._update_interval_calls = duration_to_steps(
            self.update_interval_seconds, self.decision_dt
        )
        self._llm_call_count = 0
        self.policy_decision_count = 0
        self._oracle_rollout_frozen = False

        self.param_bounds = sim_params.LLM_PARAMETER_BOUNDS.copy()
        self.last_llm_params = self.params.copy()
        self.last_llm_error = None
        self.last_llm_reason = "not_called"
        self.last_llm_basis = "not_called"
        self.last_llm_requested_mode = "custom"
        self.last_llm_assessment = {}
        self.last_required_assessment = {}
        self.last_assessment_consistent = True
        self._warned_llm_error = False

        self.llm_decision_history = []
        self.llm_raw_io_history = []
        self._reset_response_metadata()

        if system_prompt_path is None:
            raise ValueError("LLM subclass must provide a system prompt")
        component_paths = [Path(system_prompt_path)] + [
            Path(path) for path in extra_system_prompt_paths
        ]
        self.system_prompt_path = component_paths[0]
        self.system_prompt_component_paths = [str(path) for path in component_paths]
        self.system_prompt_component_sha256 = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in component_paths
        }
        self.system_prompt = "\n\n".join(
            path.read_text(encoding="utf-8").strip() for path in component_paths
        )
        self.system_prompt_sha256 = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()

    def get_action(self, observation):
        """Update high-level parameters when due, then return the driving action."""
        self.update_decision(observation)
        return super().get_action(observation)

    def update_decision(self, observation):
        """Convert one race snapshot into a model decision and update parameters."""
        if self._oracle_rollout_frozen:
            self.params = self.last_llm_params.copy()
            return

        self._llm_call_count += 1
        due = (
            self._llm_call_count == 1
            or (self._llm_call_count - 1) % self._update_interval_calls == 0
        )
        if not due:
            self.params = self.last_llm_params.copy()
            return

        observed = self.observe(observation)
        prompt_payload = self.build_prompt_payload(observed)
        self.last_required_assessment = json.loads(json.dumps(prompt_payload))
        prompt = json.dumps(prompt_payload, indent=2, ensure_ascii=True)
        self._reset_response_metadata()

        try:
            self.policy_decision_count += 1
            response = self.call_llm(prompt)
            self.apply_response(response, observation, observed)
            self.last_llm_reason = self._short_text(
                response.get("reason", "not provided"), 160
            )
            self.last_llm_basis = self._response_basis(response)
            self.last_llm_assessment = json.loads(json.dumps(prompt_payload))
            self.last_assessment_consistent = True
            self.last_llm_params = self.params.copy()
            self.last_llm_error = None
            self.record_decision(observation, observed)
            self._record_raw_io(prompt)
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            self._record_raw_io(prompt, error=exc)
            self.last_llm_error = str(exc)
            self.last_llm_reason = f"error: {exc}"[:160]
            self.last_llm_basis = "error"
            self.last_assessment_consistent = False
            self.params = self.last_llm_params.copy()
            if not self._warned_llm_error:
                self._log(f"LLM policy failed once, reusing last parameters: {exc}")
                self._warned_llm_error = True
        self.finish_decision_attempt(observed)

    def observe(self, observation):
        """Measure one race snapshot, then render it for the concrete policy."""
        measured = self._measure_scene(observation)
        return self.format_observation(measured, observation)

    def _measure_scene(self, observation):
        """Extract shared ego, opponent and road facts from ``RaceObservation``."""
        if self.car_idx is None:
            raise ValueError(f"{type(self).__name__} requires car_idx")
        ego = observation.car(self.car_idx)
        lateral_limit = float(
            getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        )
        ego_body_speed = max(float(ego.vx_body), 0.0)

        opponents = []
        for opponent in sorted(
            observation.opponents(self.car_idx), key=lambda car: car.car_idx
        ):
            delta_s = float(
                progress_delta(opponent.s, ego.s, self.track.total_length)
            )
            position = "front" if delta_s >= 0.0 else "rear"
            opponent_progress_speed = max(float(opponent.s_dot), 0.0)
            closing_speed = (
                ego_body_speed - opponent_progress_speed
                if position == "front"
                else opponent_progress_speed - ego_body_speed
            )
            deadband = float(sim_params.LLM_RELATIVE_SPEED_DEADBAND_MPS)
            constant_gap = abs(closing_speed) <= deadband + 1e-12
            opponents.append(
                {
                    "id": int(opponent.car_idx),
                    "position": position,
                    "delta_s": delta_s,
                    "delta_e": float(opponent.e - ego.e),
                    "progress_speed": opponent_progress_speed,
                    "relative_speed": opponent_progress_speed - ego_body_speed,
                    "closing_speed": closing_speed,
                    "constant_gap": constant_gap,
                    "ttc": (
                        abs(delta_s) / closing_speed
                        if closing_speed > 0.0 and not constant_gap
                        else None
                    ),
                }
            )

        preview_length = max(float(sim_params.LLM_ROAD_PREVIEW_LENGTH_METERS), 0.0)
        sample_count = max(int(sim_params.LLM_ROAD_PREVIEW_SAMPLE_COUNT), 1)
        _, _, current_heading = self.track.interpolate(ego.s)
        road_samples = []
        for distance in np.linspace(0.0, preview_length, sample_count):
            sample_s = float(ego.s) + float(distance)
            center_x, center_y, center_heading = self.track.interpolate(sample_s)
            raceline_offset = None
            raceline_speed = None
            if self.raceline is not None:
                raceline_idx = self.raceline.nearest_index(center_x, center_y)
                raceline_point = self._raceline_frenet_point(
                    float(self.raceline.s[raceline_idx]), sample_s
                )
                raceline_offset = raceline_point["lateral_e"]
                raceline_speed = raceline_point["speed"]
            road_samples.append(
                {
                    "distance": float(distance),
                    "curvature": float(track_curvature_at(self.track, sample_s)),
                    "turn_from_current_degrees": math.degrees(
                        angle_delta(center_heading, current_heading)
                    ),
                    "raceline_offset": raceline_offset,
                    "raceline_speed": raceline_speed,
                }
            )

        return {
            "ego": {
                "s": float(ego.s),
                "e": float(ego.e),
                "body_speed": ego_body_speed,
                "heading_error": float(ego.heading_error),
                "lap_progress_percent": (
                    100.0
                    * (float(ego.s) % self.track.total_length)
                    / self.track.total_length
                ),
                "speed_percent_of_maximum": (
                    100.0 * ego_body_speed / max(self.max_speed, 1e-6)
                ),
            },
            "opponents": opponents,
            "road": {
                "preview_length": preview_length,
                "samples": road_samples,
            },
            "lateral_limit": lateral_limit,
        }

    def format_observation(self, measured, observation):
        """Render shared measured facts as the concrete policy's JSON object."""
        raise NotImplementedError

    @staticmethod
    def _rounded(value, digits=3):
        """Round one finite value for stable JSON and return null otherwise."""
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)

    def build_prompt_payload(self, observed):
        """Return the final JSON object sent as the user prompt."""
        return observed

    def response_schema(self):
        """Return the JSON schema accepted from the model."""
        raise NotImplementedError

    def apply_response(self, response, observation, observed):
        """Validate one parsed response and update the downstream parameters."""
        raise NotImplementedError

    def record_decision(self, observation, observed):
        """Record subclass-specific decision state after a successful response."""

    def finish_decision_attempt(self, observed):
        """Commit subclass-specific memory after success or handled failure."""

    def call_llm(self, prompt):
        """Send one prompt to Ollama and return the parsed JSON response."""
        request_body = {
            "model": self.model_name,
            "system": self.system_prompt,
            "prompt": prompt,
            "format": self.response_schema(),
            "think": self.thinking,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        finally:
            self._last_llm_latency_seconds = time.perf_counter() - started

        def integer(name):
            """Read one optional Ollama counter using -1 for unavailable data."""
            value = body.get(name)
            return -1 if value is None else int(value)

        self._last_raw_llm_response = str(body.get("response", ""))
        self._last_llm_done_reason = str(body.get("done_reason", ""))
        self._last_llm_eval_count = integer("eval_count")
        self._last_llm_prompt_eval_count = integer("prompt_eval_count")
        self._last_llm_prompt_eval_cached_count = integer(
            "prompt_eval_cached_count"
        )
        self._last_llm_prompt_eval_duration_ns = integer("prompt_eval_duration")
        self._last_llm_load_duration_ns = integer("load_duration")
        self._last_llm_eval_duration_ns = integer("eval_duration")
        return self._extract_json(self._last_raw_llm_response)

    def _extract_json(self, text):
        """Parse a response as JSON, accepting one object embedded in prose."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _response_basis(self, response):
        """Return a short description of the facts used by the model."""
        if "basis" in response:
            return self._short_text(response["basis"], 40)
        traffic = str(response.get("traffic_basis", "uncertain"))
        track = str(response.get("track_basis", ""))
        return self._short_text(f"{traffic}|{track}" if track else traffic, 80)

    @staticmethod
    def _short_text(value, limit):
        """Collapse whitespace and bound one response field for trace storage."""
        text = " ".join(str(value).split()) or "not provided"
        return text[:limit]

    def _reset_response_metadata(self):
        """Clear raw-response fields before recording one request attempt."""
        self._last_raw_llm_response = ""
        self._last_llm_done_reason = ""
        self._last_llm_eval_count = -1
        self._last_llm_prompt_eval_count = -1
        self._last_llm_prompt_eval_cached_count = -1
        self._last_llm_prompt_eval_duration_ns = -1
        self._last_llm_load_duration_ns = -1
        self._last_llm_eval_duration_ns = -1
        self._last_llm_latency_seconds = np.nan

    def _record_raw_io(self, prompt, error=None):
        """Append one prompt, raw response, timing and handled error to history."""
        self.llm_raw_io_history.append(
            {
                "call": self._llm_call_count,
                "prompt": prompt,
                "response": self._last_raw_llm_response,
                "error": "" if error is None else str(error),
                "done_reason": self._last_llm_done_reason,
                "eval_count": self._last_llm_eval_count,
                "prompt_eval_count": self._last_llm_prompt_eval_count,
                "prompt_eval_cached_count": self._last_llm_prompt_eval_cached_count,
                "prompt_eval_duration_ns": self._last_llm_prompt_eval_duration_ns,
                "load_duration_ns": self._last_llm_load_duration_ns,
                "eval_duration_ns": self._last_llm_eval_duration_ns,
                "latency_seconds": self._last_llm_latency_seconds,
            }
        )

    def freeze_for_oracle_rollout(self):
        """Prevent external model requests inside a hypothetical rollout."""
        self._oracle_rollout_frozen = True

    def llm_decision_due(self):
        """Return whether the next control call will request a new decision."""
        return self._llm_call_count % self._update_interval_calls == 0


def resolve_llm_output(output_mode, model_name):
    """Resolve the public auto option to one concrete policy type."""
    requested = str(output_mode).strip().lower().replace("-", "_")
    if requested not in {"", "auto"}:
        if requested not in {"mode", "params"}:
            raise ValueError(f"Unknown LLM output mode: {output_mode!r}")
        return requested
    model_key = str(model_name).strip().lower().replace("-", "_")
    if any(token in model_key for token in ("params", "param", "direct")):
        return "params"
    return "mode"
