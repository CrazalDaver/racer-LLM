import json
import math
from pathlib import Path
import re
import urllib.error
import urllib.request

import numpy as np

from config import params as sim_params
from state import CarState

from .mpc import MPCPolicy
from .utils import progress_delta, track_curvature_at, track_heading_error

class LLMAlphaRacerPolicy(MPCPolicy):
    """
    alpha-RACER policy whose five parameters are supplied by a local LLM.

    The LLM only chooses q, zeta, s1, s2, s3. MPC still computes the actual
    throttle/steering command, so bad LLM responses can be clipped or ignored.
    """

    def __init__(
        self,
        *args,
        model_name="qwen2.5:0.5b",  #"my-racing-ctrl",
        base_url="http://localhost:11434",
        timeout=3.0,
        temperature=0.0,
        update_interval_steps=1,  # Only call the LLM every N steps to reduce latency and instability.
        history_length=5,
        output_mode="auto", # "mode" for discrete mode selection, "params" for direct parameter output, "auto" to infer from model name.
        safeguard_enabled=False,  # If True, apply deterministic mode-guarding rules to the LLM output.
        param_bounds=None,
        system_prompt_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = float(np.clip(temperature, 0.0, 2.0))
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.history_length = max(0, int(history_length))
        self.output_mode = self._resolve_output_mode(output_mode, model_name)
        self.safeguard_enabled = bool(safeguard_enabled)
        if self.output_mode not in {"mode", "params"}:
            raise ValueError(f"Unknown LLM output mode: {output_mode}")
        self.llm_decision_history = []
        self.llm_raw_io_history = []
        self._last_raw_llm_response = ""
        self._last_llm_done_reason = ""
        self._last_llm_eval_count = -1
        self._llm_call_count = 0
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
        self.mode_aliases = {
            "normal": "stable",
            "safe": "stable",
            "aggressive": "push",
            "accelerate": "push",
            "speed_up": "push",
            "pass": "overtake",
            "passing": "overtake",
            "overtaking": "overtake",
            "blocking": "block",
            "defend": "block",
            "defensive": "block",
            "reset": "recover",
            "fallback": "recover",
        }
        self.current_mode = "stable"
        self.last_llm_assessment = {}
        self.last_required_assessment = {}
        self.last_assessment_consistent = True
        if system_prompt_path is None:
            filename = "params_system.txt" if self.output_mode == "params" else "mode_system.txt"
            system_prompt_path = Path(__file__).with_name("prompts") / filename
        self.system_prompt_path = Path(system_prompt_path)
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()

    def _resolve_output_mode(self, output_mode, model_name) -> str:
        mode = str(output_mode).strip().lower().replace("-", "_")
        if mode not in {"", "auto"}:
            return mode
        model_key = str(model_name).strip().lower().replace("-", "_")
        if any(token in model_key for token in ("params", "param", "direct")):
            return "params"
        return "mode"

    @staticmethod
    def _rounded(value, digits=3):
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)

    def _decision_history_payload(self) -> dict:
        modes = []
        for item in self.llm_decision_history[-self.history_length:]:
            entry = {
                "mode": item["mode"],
            }
            if self.output_mode == "params":
                entry["params"] = {
                    name: self._rounded(value)
                    for name, value in zip(("q", "zeta", "s1", "s2", "s3"), item["params"])
                }
            modes.append(entry)
        return {
            "usage": "continuity_only",
            "contains_scene_evidence": False,
            "oldest_to_newest": modes,
        }

    def _opponent_feature(self, state, current_py, current_s, opponent, fallback_id) -> dict:
        delta_s = float(progress_delta(opponent["px"], current_s, self.track.total_length))
        delta_e = float(opponent.get("py", 0.0) - current_py)
        relation = "ahead" if delta_s > 0.05 else "behind" if delta_s < -0.05 else "alongside"
        ego_speed = max(float(state.vx), 0.0)
        opponent_speed = max(float(opponent.get("vx", 0.0)), 0.0)
        if relation == "ahead":
            closing_speed = ego_speed - opponent_speed
        elif relation == "behind":
            closing_speed = opponent_speed - ego_speed
        else:
            closing_speed = abs(ego_speed - opponent_speed)
        ttc = abs(delta_s) / closing_speed if closing_speed > 0.05 else None

        abs_delta_s = abs(delta_s)
        min_longitudinal = float(getattr(self.mpc, "min_longitudinal_separation", 0.36))
        if abs_delta_s < min_longitudinal:
            gap_band = "overlap"
        elif abs_delta_s < 0.8:
            gap_band = "critical"
        elif abs_delta_s < 2.5:
            gap_band = "close"
        elif abs_delta_s < 4.0:
            gap_band = "nearby"
        else:
            gap_band = "far"

        lateral_limit = float(self.ref_lateral_clip)
        min_lateral = float(getattr(self.mpc, "min_lateral_separation", 0.24))
        opponent_py = float(opponent.get("py", 0.0))
        left_pass_possible = lateral_limit - opponent_py >= min_lateral
        right_pass_possible = opponent_py + lateral_limit >= min_lateral
        if left_pass_possible and right_pass_possible:
            pass_space = "both"
        elif left_pass_possible:
            pass_space = "left"
        elif right_pass_possible:
            pass_space = "right"
        else:
            pass_space = "none"

        speed_delta = ego_speed - opponent_speed
        speed_relation = "slower" if speed_delta > 0.1 else "faster" if speed_delta < -0.1 else "similar"
        return {
            "id": int(opponent.get("car_idx", fallback_id)),
            "relation": relation,
            "delta_s": self._rounded(delta_s),
            "delta_e": self._rounded(delta_e),
            "distance_band": gap_band,
            "same_lane": abs(delta_e) < min_lateral,
            "speed": self._rounded(opponent_speed),
            "speed_relation_to_ego": speed_relation,
            "closing_speed": self._rounded(max(closing_speed, 0.0)),
            "ttc": self._rounded(ttc),
            "pass_space": pass_space,
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
        lateral_limit = float(getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width))
        opponents_payload = [
            self._opponent_feature(state, current_py, current_s, opponent, idx)
            for idx, opponent in enumerate(opponents or [])
        ]
        opponents_payload.sort(key=lambda item: abs(item["delta_s"]))
        nearest_front = next((item for item in opponents_payload if item["relation"] == "ahead"), None)
        nearest_rear = next((item for item in opponents_payload if item["relation"] == "behind"), None)

        def interaction_risk(item):
            if item is None:
                return "clear"
            if item["distance_band"] == "overlap" and item["same_lane"]:
                return "overlap"
            if item["ttc"] is not None and item["ttc"] < 1.0 and item["same_lane"]:
                return "critical"
            if item["closing_speed"] > 0.1 and item["distance_band"] != "far" and item["same_lane"]:
                return "closing"
            return "far"

        boundary_ratio = abs(float(current_py)) / max(lateral_limit, 1e-6)
        boundary_risk = "outside" if boundary_ratio > 1.0 else "near" if boundary_ratio >= 0.95 else "low"
        if not self.mpc.last_solver_success or self.mpc.last_recovery_active:
            mpc_health = "failure"
        elif np.isfinite(self.mpc.last_loss):
            mpc_health = "ok"
        else:
            mpc_health = "not_run"

        held_updates = 0
        for item in reversed(self.llm_decision_history):
            if item["mode"] != self.current_mode:
                break
            held_updates += 1

        return {
            "ego": {
                "progress": self._rounded(current_s),
                "speed": self._rounded(max(float(state.vx), 0.0)),
                "lateral_offset": self._rounded(current_py),
                "heading_error": self._rounded(track_heading_error(self.track, state.psi, current_s)),
                "left_boundary_space": self._rounded(lateral_limit - current_py),
                "right_boundary_space": self._rounded(lateral_limit + current_py),
                "boundary_risk": boundary_risk,
                "lookahead_curvature": self._rounded(
                    track_curvature_at(self.track, current_s + max(float(state.vx), 0.0) * self.mpc.dt * self.mpc.horizon)
                ),
                "mpc_health": mpc_health,
                "mpc_status": str(self.mpc.last_solver_status),
            },
            "interaction": {
                "front_risk": interaction_risk(nearest_front),
                "rear_risk": interaction_risk(nearest_rear),
                "nearest_front": nearest_front,
                "nearest_rear": nearest_rear,
                "opponents_by_distance": opponents_payload,
            },
            "limits": {
                "vehicle_center_half_width": self._rounded(lateral_limit),
                "reference_half_width": self._rounded(self.ref_lateral_clip),
                "close_front_distance": 2.5,
                "overtake_distance": 4.0,
                "push_max_abs_curvature": 0.15,
                "critical_ttc": 1.0,
            },
            "decision_state": {
                "previous_mode": self.current_mode,
                "held_updates": held_updates,
                "recent_decisions": self._decision_history_payload(),
            },
            "required_assessment": {
                "front_risk": interaction_risk(nearest_front),
                "rear_risk": interaction_risk(nearest_rear),
                "boundary_risk": boundary_risk,
                "mpc_health": mpc_health,
            },
        }

    def _record_llm_decision(self):
        if self.history_length <= 0:
            return
        self.llm_decision_history.append(
            {
                "call": self._llm_call_count,
                "mode": self.current_mode,
                "requested_mode": self.last_llm_requested_mode,
                "reason": self.last_llm_reason,
                "safeguard_override": self.last_safeguard_override,
                "params": self.params.copy(),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def _prompt(self, state: CarState, current_s: float, opponents=None) -> str:
        # Generate a JSON payload of the current scene and race context
        payload = self._scene_features(state, current_s, opponents=opponents)
        self.last_required_assessment = payload["required_assessment"].copy()
        if self.output_mode == "params":
            payload["parameter_bounds"] = {
                name: {"min": float(bounds[0]), "max": float(bounds[1])}
                for name, bounds in self.param_bounds.items()
            }
            payload["current_parameters"] = {
                name: self._rounded(value)
                for name, value in zip(("q", "zeta", "s1", "s2", "s3"), self.params)
            }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _output_schema(self) -> dict:
        properties = {
            "basis": {
                "type": "string",
                "enum": [
                    "mpc_failure",
                    "outside_boundary",
                    "critical_front",
                    "slower_front_pass",
                    "faster_rear",
                    "clear_track",
                    "uncertain",
                ],
            },
            "reason": {"type": "string", "maxLength": 120},
        }
        required = ["basis", "reason"]
        if self.output_mode == "mode":
            properties["mode"] = {
                "type": "string",
                "enum": list(self.mode_presets),
            }
            required.insert(1, "mode")
        else:
            for name in ("q", "zeta", "s1", "s2", "s3"):
                lo, hi = self.param_bounds[name]
                properties[name] = {"type": "number", "minimum": float(lo), "maximum": float(hi)}
                required.append(name)
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
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
            # print(raw)
        body = json.loads(raw)
        self._last_raw_llm_response = str(body.get("response", ""))
        self._last_llm_done_reason = str(body.get("done_reason", ""))
        self._last_llm_eval_count = int(body.get("eval_count") or -1)
        return self._extract_json(self._last_raw_llm_response)

    def _record_llm_raw_io(self, prompt: str, error=None):
        self.llm_raw_io_history.append(
            {
                "call": self._llm_call_count,
                "prompt": prompt,
                "response": self._last_raw_llm_response,
                "error": "" if error is None else str(error),
                "done_reason": self._last_llm_done_reason,
                "eval_count": self._last_llm_eval_count,
            }
        )
        self.llm_raw_io_history = self.llm_raw_io_history[-5:]

    def _sanitize_params(self, raw_params: dict) -> np.ndarray:
        
        if self.output_mode == "params":
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
            mode_key = self.mode_aliases.get(mode_key, mode_key)
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
                    slow_front = slow_front or delta_s < 3.0
                    close_slow_front = close_slow_front or delta_s < 2.5
                    immediate_slow_front = immediate_slow_front or delta_s < 0.8
            elif delta_s < 0.0 and opponent["vx"] > state.vx:
                faster_rear = faster_rear or abs(delta_s) < 3.0
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
        if self.output_mode != "mode":
            return

        context = self._mode_context(state, current_s, opponents)
        guarded_mode = self.current_mode
        
        near_contact = context["nearest_delta"] is not None and abs(context["nearest_delta"]) < 0.7 # near contact threshold
        near_overlap = (
            context["nearest_delta"] is not None
            and abs(context["nearest_delta"]) < 0.8
            and context["nearest_lateral_gap"] is not None
            and context["nearest_lateral_gap"] < 0.6
        )
        front_clear_for_push = context["nearest_front_delta"] is None or context["nearest_front_delta"] > 4.0
        push_allowed = front_clear_for_push and not near_overlap and abs(context["current_py"]) < 0.45
        tiny_front_gap = (
            context["nearest_front_delta"] is not None
            and context["nearest_front_delta"] < 0.6
            and context["nearest_front_lateral_gap"] is not None
            and context["nearest_front_lateral_gap"] < 0.25
        )

        if state.vx < 0.35 and near_contact:
            guarded_mode = "recover"
        elif tiny_front_gap:
            if guarded_mode in {"push", "overtake", "block"}:
                guarded_mode = "recover" if state.vx < 0.8 else "stable"
        elif context["close_slow_front"]:
            if guarded_mode in {"push", "stable", "block"}:
                guarded_mode = "overtake"
            elif guarded_mode == "recover" and abs(context["current_py"]) < 0.8 and state.vx > 0.5:
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
        # Only call the LLM every update_interval_steps to avoid excessive latency and instability.
        self._llm_call_count += 1
        if self._llm_call_count > 1 and (self._llm_call_count - 1) % self.update_interval_steps != 0:
            self.params = self.last_llm_params.copy()
            return

        prompt = self._prompt(state, current_s, opponents=opponents)
        self._last_raw_llm_response = ""
        try:
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




class RuleBasedSafeguardPolicy(LLMAlphaRacerPolicy):
    """Deterministic ablation of the safeguarded discrete-mode LLM policy.

    It observes the same Frenet interaction state, selects one of the exact
    same five mode presets, and then passes the choice through the same mode
    guard.  No language model or network request is used.
    """

    def __init__(self, *args, **kwargs):
        kwargs["model_name"] = "rule_based_safeguard"
        kwargs["output_mode"] = "mode"
        kwargs["temperature"] = 0.0
        super().__init__(*args, **kwargs)
        self.last_rule_params = self.params.copy()

    def _select_rule_mode(self, state: CarState, current_s: float, opponents=None) -> str:
        context = self._mode_context(state, current_s, opponents)
        nearest_delta = context["nearest_delta"]
        nearest_lateral_gap = context["nearest_lateral_gap"]
        near_contact = nearest_delta is not None and abs(nearest_delta) < 0.7
        near_overlap = (
            nearest_delta is not None
            and abs(nearest_delta) < 0.8
            and nearest_lateral_gap is not None
            and nearest_lateral_gap < 0.6
        )
        tiny_front_gap = (
            context["nearest_front_delta"] is not None
            and context["nearest_front_delta"] < 0.6
            and context["nearest_front_lateral_gap"] is not None
            and context["nearest_front_lateral_gap"] < 0.25
        )

        heading_error = abs(track_heading_error(self.track, state.psi, current_s))
        lookahead_s = current_s + max(float(state.vx), 0.0) * self.mpc.dt * self.mpc.horizon
        lookahead_curvature = abs(track_curvature_at(self.track, lookahead_s))
        near_boundary = abs(context["current_py"]) >= 0.9 * self.mpc.track_half_width

        # Ordered decision tree: recovery and collision avoidance have highest
        # priority, followed by passing, defending, and clear-road pace.
        if near_boundary or heading_error > 0.8 or (state.vx < 0.35 and near_contact):
            return "recover"
        if tiny_front_gap:
            return "recover" if state.vx < 0.8 else "stable"
        if context["close_slow_front"]:
            return "overtake"
        if near_overlap:
            return "overtake" if context["slow_front"] and state.vx > 0.5 else "stable"
        if context["faster_rear"]:
            return "block"
        front_clear = context["nearest_front_delta"] is None or context["nearest_front_delta"] > 4.0
        if front_clear and state.vx > 1.0 and lookahead_curvature < 0.35:
            return "push"
        return "stable"

    def update_params_from_rules(self, state: CarState, current_s: float, opponents=None):
        self._llm_call_count += 1
        if self._llm_call_count > 1 and (self._llm_call_count - 1) % self.update_interval_steps != 0:
            self.params = self.last_rule_params.copy()
            return

        self.current_mode = self._select_rule_mode(state, current_s, opponents=opponents)
        self.params = self.mode_presets[self.current_mode].copy()
        self._apply_mode_guard(state, current_s, opponents=opponents)
        self.last_rule_params = self.params.copy()

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        self.update_params_from_rules(state, current_s, opponents=opponents)
        return MPCPolicy.get_reference_trajectory(
            self, state, current_s, opponents=opponents
        )
