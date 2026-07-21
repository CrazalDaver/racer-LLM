import json
import math
import re
import urllib.error
import urllib.request

import numpy as np

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
        self._llm_call_count = 0
        self.last_llm_reason = "not_called"
        self.last_llm_requested_mode = "stable"
        self.last_safeguard_override = ""
        self.param_bounds = param_bounds if param_bounds is not None else {
            "q": (0.05, 20.0),
            "zeta": (0.1, 3.0),
            "s1": (0.0, self.mpc.track_half_width),
            "s2": (0.001, 5.0),
            "s3": (0.0, 5.0),
        }
        self.last_llm_params = self.params.copy()
        self.last_llm_error = None
        self._warned_llm_error = False
        self.mode_presets = {
            "stable": np.asarray([1.3, 1.0, 0.05, 1.0, 0.0], dtype=float),
            "push": np.asarray([1.8, 1.25, 0.05, 1.0, 0.0], dtype=float),
            "overtake": np.asarray([1.0, 1.35, 1.75, 0.1, 0.0], dtype=float),
            "block": np.asarray([1.6, 0.90, 0.05, 0.45, 2.0], dtype=float),
            "recover": np.asarray([3.0, 0.70, 0.0, 1.5, 0.0], dtype=float),
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

    def _resolve_output_mode(self, output_mode, model_name) -> str:
        mode = str(output_mode).strip().lower().replace("-", "_")
        if mode not in {"", "auto"}:
            return mode
        model_key = str(model_name).strip().lower().replace("-", "_")
        if any(token in model_key for token in ("params", "param", "direct")):
            return "params"
        return "mode"

    def _history_text(self) -> str:
        if not self.llm_decision_history:
            return "none"
        lines = []
        for item in self.llm_decision_history[-self.history_length:]:
            params = item["params"]
            lines.append(
                f"call={item['call']}, requested={item['requested_mode']}, "
                f"executed={item['mode']}, reason={item['reason']}, "
                f"q={params[0]:.3f}, zeta={params[1]:.3f}, "
                f"s1={params[2]:.3f}, s2={params[3]:.3f}, s3={params[4]:.3f}"
            )
        return "\n".join(lines)

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
        opponents = opponents or []
        _, current_py = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        opponent_lines = []
        for idx, opponent in enumerate(opponents):
            delta_s = progress_delta(opponent["px"], current_s, self.track.total_length)
            relation = "ahead" if delta_s > 0.0 else "behind" if delta_s < 0.0 else "alongside"
            speed_relation = "slower" if opponent["vx"] < state.vx else "faster" if opponent["vx"] > state.vx else "same_speed"
            lateral_gap = opponent["py"] - current_py
            opponent_lines.append(
                f"{idx}: delta_px={delta_s:.3f}, relation={relation}, "
                f"speed_relation={speed_relation}, py={opponent['py']:.3f}, "
                f"lateral_gap={lateral_gap:.3f}, vx={opponent['vx']:.3f}"
            )
        opponent_text = "\n".join(opponent_lines) if opponent_lines else "none"
        preset_lines = "\n".join(  # list available modes and their parameters
            f"- {name}: q={values[0]:.2f}, zeta={values[1]:.2f}, "
            f"s1={values[2]:.2f}, s2={values[3]:.2f}, s3={values[4]:.2f}"
            for name, values in self.mode_presets.items()
        )
        if self.output_mode == "params":
            history_text = self._history_text()
            common_context = (
                "The underlying parameter meaning is Q=qI track loss weight; zeta scales raceline speed integration; "
                "s1/s2 tune overtaking lateral offset and longitudinal decay; s3 tunes blocking.\n"
                f"Allowed ranges: q {self.param_bounds['q']}, zeta {self.param_bounds['zeta']}, "
                f"s1 {self.param_bounds['s1']}, s2 {self.param_bounds['s2']}, s3 {self.param_bounds['s3']}.\n"
                f"Current car: px={current_s:.3f}, py={current_py:.3f}, "
                f"x={state.x:.3f}, y={state.y:.3f}, psi={state.psi:.3f}, "
                f"vx={state.vx:.3f}, vy={state.vy:.3f}, omega={state.omega:.3f}.\n"
                f"Opponents in Frenet frame relative to ego:\n{opponent_text}\n"
                f"Previous mode: {self.current_mode}.\n"
                f"Previous parameters: q={self.params[0]:.3f}, zeta={self.params[1]:.3f}, "
                f"s1={self.params[2]:.3f}, s2={self.params[3]:.3f}, s3={self.params[4]:.3f}.\n"
                f"Recent LLM outputs, oldest to newest:\n{history_text}\n"
            )
            return (
                "You choose alpha-RACER MPC policy parameters directly. Return only valid JSON with exactly these keys: "
                "{\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2,"
                "\"reason\":\"clear track, maintain pace\"}. Do not include a mode key. "
                "Keep reason under 12 words and base it only on the supplied state.\n"
                f"{common_context}"
                "Parameter guidance: increase q for tighter tracking/recovery only after off-track, collision, or repeated stall; "
                "do not treat low vx at race start as a recovery condition; reduce zeta when approaching hard turns or unstable; "
                "increase s1 only when lateral interaction is useful, and avoid very large s1 because the usable lane width is narrow; "
                "increase s2 for more local/short-range interaction; increase s3 only when blocking a faster rear car is useful.\n"
                "For overtaking a close slower front car, prefer cautious zeta below 1.0 and enough lateral offset instead of high zeta, "
                "otherwise ego can rear-end the front car before it is laterally separated. "
                "Use push-style parameters only when the nearest front opponent is absent or has delta_px above 4.0 m, and no opponent is near-overlap. "
                "When any opponent has abs(delta_px) below 0.8 m and abs(lateral_gap) below 0.6 m, do not use push-style parameters with high zeta and near-zero lateral interaction. "
                "In that near-overlap case, use overtaking-style parameters if ego is still moving and there is room to pass, or stable/recovery-style parameters if the situation is uncertain. "
                "For blocking, use moderate s3 and avoid forcing large defensive offsets unless a faster rear car is clearly threatening. "
                "If speed is very low while another car is very close, use high q, low zeta, and near-zero interaction to recover.\n"
                "Use recent LLM outputs to avoid oscillation. If the recent output was overtaking and the same slow front car is still close, "
                "keep overtaking-style parameters unless the car is off-track, collided, or already clear of that car.\n"
                "JSON example: {\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2,"
                "\"reason\":\"clear track, maintain pace\"}"
            )
        
        # mode version
        history_text = self._history_text()
        return (
            "You choose one alpha-RACER MPC behavior mode. Return only valid JSON like "
            "{\"mode\":\"stable\",\"reason\":\"close front car, hold position\"}. "
            "Do not invent modes. Keep reason under 12 words and base it only on the supplied state.\n"
            "Available modes:\n"
            f"{preset_lines}\n"
            "Mode meanings: stable follows the raceline with almost no lateral interaction; push tracks harder on clear road; "
            "overtake creates a local lateral offset around a slower front car while avoiding rear-end contact; "
            "block defends against a faster rear car by moving toward the rear car's lane; "
            "recover uses tight tracking and low speed integration to leave contact or off-track states.\n"
            "Hard mode validity rules: never output push unless every push condition below is true. "
            "Push conditions: nearest front opponent is absent or has delta_px above 4.0 m; no opponent is near-overlap; abs(py) is below 0.45. "
            "If any push condition is false, output stable, overtake, block, or recover, but not push. "
            "Choose overtake when an opponent is ahead, close, and slower than ego. Close means positive delta_px below 2.5 m. "
            "Also choose overtake to continue an active pass when an opponent is near-overlap: abs(delta_px) below 0.8 m and abs(lateral_gap) below 0.6 m. "
            "If a car is near-overlap, push is unsafe even when relation is not ahead; choose overtake if ego is still moving and a pass is forming, otherwise choose stable. "
            "If the slower front car is extremely close, delta_px below 0.6 m, and abs(lateral_gap) below 0.25 m, avoid push and avoid a hard high-speed overtake; choose stable or recover to prevent rear-end contact. "
            "Overtake is valid when there is a slower front opponent with positive delta_px below 3.0 m, or when the same opponent is near-overlap during a pass. "
            "If no opponent is ahead/close/slower and no opponent is near-overlap, do not output overtake. "
            "If a slower front opponent has positive delta_px below 2.5 m, overtake has priority over push. "
            "Push is allowed only when the nearest front opponent is absent or has delta_px above 4.0 m, no opponent is near-overlap, and ego is not near the lane boundary. "
            "Push is invalid when any opponent is ahead with positive delta_px below 4.0 m or any opponent is near-overlap. "
            "If recent outputs include overtake and the same opponent is still ahead or near-overlap, keep overtake until ego is clearly past it. "
            "Clearly past means that opponent's delta_px is negative with enough longitudinal separation and the track ahead is clear. "
            "After ego is clearly past the opponent, use push on clear track or stable if uncertain. "
            "Block is valid only when an opponent has relation=behind and speed_relation=faster; use it cautiously, not as the default response to any rear car. "
            "Do not choose recover just because py is offset during an overtake; choose recover when |py| is large, after contact, after off-track behavior, "
            "or when speed is very low and any opponent is still very close. "
            "Use push only on clear track; otherwise use stable.\n"
            "The underlying parameter meaning is Q=qI track loss weight; zeta scales raceline speed integration; "
            "s1/s2 tune overtaking lateral offset and longitudinal decay; s3 tunes blocking.\n"
            f"Allowed ranges: q {self.param_bounds['q']}, zeta {self.param_bounds['zeta']}, "
            f"s1 {self.param_bounds['s1']}, s2 {self.param_bounds['s2']}, s3 {self.param_bounds['s3']}.\n"
            f"Current car: px={current_s:.3f}, py={current_py:.3f}, "
            f"x={state.x:.3f}, y={state.y:.3f}, psi={state.psi:.3f}, "
            f"vx={state.vx:.3f}, vy={state.vy:.3f}, omega={state.omega:.3f}.\n"
            f"Opponents in Frenet frame relative to ego:\n{opponent_text}\n"
            f"Previous mode: {self.current_mode}.\n"
            f"Previous parameters: q={self.params[0]:.3f}, zeta={self.params[1]:.3f}, "
            f"s1={self.params[2]:.3f}, s2={self.params[3]:.3f}, s3={self.params[4]:.3f}.\n"
            f"Recent LLM outputs, oldest to newest:\n{history_text}\n"
            "Use recent LLM outputs to avoid oscillation. "
            "Do not continue overtake from history after the close front opponent is gone.\n"
            "JSON example: {\"mode\":\"stable\",\"reason\":\"clear track, low interaction risk\"}"
        )

    def _extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    def _query_llm(self, prompt: str) -> dict:
        self._last_raw_llm_response = ""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 64},
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
        return self._extract_json(self._last_raw_llm_response)

    def _record_llm_raw_io(self, prompt: str, error=None):
        self.llm_raw_io_history.append(
            {
                "call": self._llm_call_count,
                "prompt": prompt,
                "response": self._last_raw_llm_response,
                "error": "" if error is None else str(error),
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

