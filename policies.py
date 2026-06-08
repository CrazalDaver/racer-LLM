import abc
import json
import math
import re
import urllib.error
import urllib.request

import numpy as np

from state import CarAction, CarState
from track import RaceLine, Track


class BasePolicy(abc.ABC):
    """
    Policy base class that embeds an MPC solver.
    Subclasses must implement get_reference_trajectory.
    """

    def __init__(self, track: Track, mpc, step_distance: float = 0.1):
        self.track = track
        self.mpc = mpc
        self.step_distance = step_distance
        self.last_u = np.array([0.0, self._neutral_steer_command()])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        self.log_fn = print
        self._mpc_failure_active = False
        self._mpc_failure_start_time = None
        self._mpc_failure_last_report_time = None
        self._mpc_failure_count = 0
        self._mpc_failure_report_interval = 1.0
        self._mpc_stall_active = False
        self._mpc_stall_start_time = None
        self._mpc_stall_last_report_time = None
        self._mpc_stall_count = 0
        self._mpc_stall_report_interval = 1.0

    def _neutral_steer_command(self) -> float:
        return np.clip(-self.mpc.p.Sb / self.mpc.p.Sa, -1.0, 1.0)

    @abc.abstractmethod
    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        pass

    def get_q_weight(self) -> float:
        return 1.0

    def _log(self, message: str):
        self.log_fn(message)

    def _format_mpc_time(self, sim_time) -> str:
        if sim_time is None:
            return "t=?"
        return f"t={sim_time:.2f}s"

    def _report_mpc_failure(self, exc: Exception, state: CarState, sim_time=None):
        self._mpc_failure_count += 1
        should_report = False
        if not self._mpc_failure_active:
            self._mpc_failure_active = True
            self._mpc_failure_start_time = sim_time
            self._mpc_failure_last_report_time = sim_time
            should_report = True
        elif sim_time is None:
            should_report = self._mpc_failure_count in {2, 5, 10} or self._mpc_failure_count % 100 == 0
        elif (
            self._mpc_failure_last_report_time is None
            or sim_time - self._mpc_failure_last_report_time >= self._mpc_failure_report_interval
        ):
            self._mpc_failure_last_report_time = sim_time
            should_report = True

        if not should_report:
            return

        if self._mpc_failure_start_time is None or sim_time is None:
            interval = self._format_mpc_time(sim_time)
        else:
            interval = f"t={self._mpc_failure_start_time:.2f}-{sim_time:.2f}s"
        self._log(
            "[MPC warning] solve failed "
            f"({interval}, count={self._mpc_failure_count}, "
            f"x={state.x:.2f}, y={state.y:.2f}, vx={state.vx:.2f}); "
            f"using last control. status={exc}"
        )

    def _report_mpc_recovery(self, sim_time=None):
        if not self._mpc_failure_active:
            return

        if self._mpc_failure_start_time is None or sim_time is None:
            interval = self._format_mpc_time(sim_time)
        else:
            interval = f"t={self._mpc_failure_start_time:.2f}-{sim_time:.2f}s"
        self._log(f"[MPC warning] solve recovered ({interval}, failed_count={self._mpc_failure_count})")
        self._mpc_failure_active = False
        self._mpc_failure_start_time = None
        self._mpc_failure_last_report_time = None
        self._mpc_failure_count = 0

    def _report_mpc_stall(self, state: CarState, ref_speed: float, target_vel: float, target_steer: float, sim_time=None):
        self._mpc_stall_count += 1
        should_report = False
        if not self._mpc_stall_active:
            self._mpc_stall_active = True
            self._mpc_stall_start_time = sim_time
            self._mpc_stall_last_report_time = sim_time
            should_report = True
        elif sim_time is None:
            should_report = self._mpc_stall_count in {2, 5, 10} or self._mpc_stall_count % 100 == 0
        elif (
            self._mpc_stall_last_report_time is None
            or sim_time - self._mpc_stall_last_report_time >= self._mpc_stall_report_interval
        ):
            self._mpc_stall_last_report_time = sim_time
            should_report = True

        if not should_report:
            return

        if self._mpc_stall_start_time is None or sim_time is None:
            interval = self._format_mpc_time(sim_time)
        else:
            interval = f"t={self._mpc_stall_start_time:.2f}-{sim_time:.2f}s"
        self._log(
            "[MPC warning] feasible but stalled "
            f"({interval}, count={self._mpc_stall_count}, "
            f"vx={state.vx:.3f}, ref_v={ref_speed:.3f}, "
            f"cmd={target_vel:.3f}, steer={target_steer:.3f}, "
            f"loss={self.mpc.last_loss:.3g}, status={self.mpc.last_solver_status})"
        )

    def _report_mpc_stall_recovery(self, sim_time=None):
        if not self._mpc_stall_active:
            return

        if self._mpc_stall_start_time is None or sim_time is None:
            interval = self._format_mpc_time(sim_time)
        else:
            interval = f"t={self._mpc_stall_start_time:.2f}-{sim_time:.2f}s"
        self._log(f"[MPC warning] stalled condition recovered ({interval}, count={self._mpc_stall_count})")
        self._mpc_stall_active = False
        self._mpc_stall_start_time = None
        self._mpc_stall_last_report_time = None
        self._mpc_stall_count = 0

    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        ref_traj = self.get_reference_trajectory(state, current_s, opponents=opponents)
        self.last_ref_traj = ref_traj
        state_arr = state.to_array()
        try:
            _, u_first = self.mpc.solve(state_arr, ref_traj, self.last_u, q_weight=self.get_q_weight())
        except Exception as e:
            self._report_mpc_failure(e, state, sim_time=sim_time)
            u_first = self.last_u
            try:
                self.mpc.evaluate_fallback_loss(
                    state_arr,
                    ref_traj,
                    self.last_u,
                    q_weight=self.get_q_weight(),
                )
            except Exception:
                self.mpc.last_loss = np.nan
                self.mpc.last_track_loss = np.nan
                self.mpc.last_action_loss = np.nan
                self.mpc.last_velocity_loss = np.nan
                self.mpc.last_heading_loss = np.nan
        else:
            self._report_mpc_recovery(sim_time=sim_time)

        target_vel = np.clip(u_first[0], -1.0, 1.0)
        target_steer = np.clip(u_first[1], -1.0, 1.0)
        ref_speed = float(ref_traj[2, 0]) if ref_traj is not None else np.nan
        stalled_feasible = (
            self.mpc.last_solver_success
            and np.isfinite(self.mpc.last_loss)
            and state.vx < 0.05
            and ref_speed > 0.5
        )
        if stalled_feasible:
            self._report_mpc_stall(state, ref_speed, target_vel, target_steer, sim_time=sim_time)
        else:
            self._report_mpc_stall_recovery(sim_time=sim_time)

        self.last_u = np.array([target_vel, target_steer])
        return CarAction(throttle=target_vel, steer=target_steer)


class CenterLinePolicy(BasePolicy):
    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        for k in range(horizon + 1):
            s = current_s + k * self.step_distance
            x, y, psi = self.track.interpolate(s)
            ref[0, k] = s % self.track.total_length
            ref[1, k] = 0.0
            ref[2, k] = state.vx
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = -math.sin(psi)
            ref[7, k] = math.cos(psi)
            ref_world[:, k] = [x, y]
        self.last_ref_world_horizon = ref_world
        return ref


class AlphaRacerPolicy(BasePolicy):
    """
    Five-parameter alpha-RACER policy: q, zeta, s1, s2, s3.
    q scales the MPC track loss as Q_eff = q * I when config.Q is identity.
    """

    def __init__(
        self,
        track,
        mpc,
        step_distance=1.2,
        params=None,
        raceline: RaceLine = None,
        max_speed=5.0,
        search_back_steps=20,
        search_forward_steps=200,
    ):
        super().__init__(track, mpc, step_distance)
        self.params = params if params is not None else np.array([1.0, 1.0, 0, 0, 0])
        self.raceline = raceline
        self.max_speed = max_speed
        self.last_raceline_s = None
        self.last_raceline_idx = None
        self.search_back_steps = search_back_steps
        self.search_forward_steps = search_forward_steps

    def get_q_weight(self) -> float:
        return float(self.params[0])

    def _nearest_raceline_s(self, state: CarState) -> float:
        self.last_raceline_idx = self.raceline.nearest_index(
            state.x,
            state.y,
            last_idx=self.last_raceline_idx,
            back_steps=self.search_back_steps,
            forward_steps=self.search_forward_steps,
        )
        return float(self.raceline.s[self.last_raceline_idx])

    def _raceline_frenet_state(self, raceline_s: float, center_hint: float = None) -> dict:
        # Given an s value along the raceline, compute the corresponding Frenet state relative to the track centerline.
        point = self.raceline.interpolate(raceline_s)  # Get the raceline point in world coordinates.
        center_s, lateral_e = self.track.project(
            point["x_m"],
            point["y_m"],
            s_hint=center_hint,
            search_radius=8.0,
            psi_hint=point["psi_rad"],
            max_heading_error=math.pi / 2.0,
        )
        xc, yc, psi_center = self.track.interpolate(center_s)
        return {
            "track_s": center_s,
            "lateral_e": lateral_e,
            "speed": min(point["vx_mps"], self.max_speed),
            "center_x": xc,
            "center_y": yc,
            "psi_center": psi_center,
            "normal_x": -math.sin(psi_center),
            "normal_y": math.cos(psi_center),
        }

    def _raceline_frenet_point(self, raceline_s: float) -> dict:
        point = self.raceline.interpolate(raceline_s)
        p_x = point["s_m"]
        p_y = -point["dev_m"]
        psi = point["psi_rad"]
        normal_x = -math.sin(psi)
        normal_y = math.cos(psi)
        center_x = point["x_m"] - p_y * normal_x
        center_y = point["y_m"] - p_y * normal_y
        return {
            "track_s": p_x,
            "lateral_e": p_y,
            "speed": min(point["vx_mps"], self.max_speed),
            "center_x": center_x,
            "center_y": center_y,
            "psi_center": psi,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "world_x": point["x_m"],
            "world_y": point["y_m"],
        }

    def _progress_delta(self, s_a: float, s_b: float) -> float:
        # Compute the signed shortest distance along the track from s_b to s_a, handling wraparound.
        return math.atan2(
            math.sin((s_a - s_b) * 2.0 * math.pi / self.track.total_length),
            math.cos((s_a - s_b) * 2.0 * math.pi / self.track.total_length),
        ) * self.track.total_length / (2.0 * math.pi)

    def _sign(self, value: float) -> float:
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0

    def _closest_opponents(self, current_s: float, opponents):
        if not opponents:
            return []

        enriched = []
        for opponent in opponents:
            delta = self._progress_delta(opponent["px"], current_s) # Positive delta means ahead
            enriched.append({**opponent, "delta": delta})

        front = [op for op in enriched if op["delta"] > 0.0]
        behind = [op for op in enriched if op["delta"] < 0.0]
        selected = []
        if front:
            selected.append(min(front, key=lambda op: op["delta"]))
        if behind:
            selected.append(max(behind, key=lambda op: op["delta"]))
        return selected

    def _competitive_offsets(
        self,
        p_pert_x: float,
        p_pert_y: float,
        v_pert_x: float,
        ego_py_t: float,
        opponents,
        k: int,
    ) -> tuple:
        _, _, s1, s2, s3 = self.params
        p_ot = 0.0
        p_bl = 0.0

        for opponent in opponents:
            opp_px = opponent["px"] + k * self.mpc.dt * opponent["vx"]
            opp_py = opponent["py"]
            opp_vx = opponent["vx"]
            dx = self._progress_delta(p_pert_x, opp_px)
            dy_current = ego_py_t - opp_py
            ot_mag = max((s1 - abs(dy_current)) * math.exp(-s2 * dx * dx), 0.0)
            p_ot += self._sign(dy_current) * ot_mag

            behind_now = opponent["delta"] < 0.0
            opponent_faster = opp_vx > v_pert_x
            ego_ahead_in_horizon = dx >= 0.0
            if behind_now and opponent_faster and ego_ahead_in_horizon:
                speed_factor = 1.0 - math.exp(-s3 * max(opp_vx - v_pert_x, 0.0))
                proximity = math.exp(-s2 * dx * dx)
                p_bl += (opp_py - p_pert_y) * speed_factor * proximity

        return p_ot, p_bl

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        horizon = self.mpc.horizon
        q, zeta, s1, s2, s3 = self.params
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        raceline_world = np.full((2, horizon + 1), np.nan)
        selected_opponents = self._closest_opponents(current_s, opponents)
        if self.raceline is not None:
            raceline_s = self._nearest_raceline_s(state)
            self.last_raceline_s = raceline_s
            _, ego_py_t = self.track.project(
                state.x,
                state.y,
                s_hint=current_s,
                search_radius=8.0,
                psi_hint=state.psi,
                max_heading_error=math.pi / 2.0,
            )
            source_s = raceline_s
        for k in range(horizon + 1):
            if self.raceline is not None:
                frenet = self._raceline_frenet_point(source_s)
                raceline_speed_k = max(frenet["speed"], 0.0)
                p_pert_x = frenet["track_s"]
                p_pert_y = frenet["lateral_e"]
                p_ot, p_bl = self._competitive_offsets(
                    p_pert_x,
                    p_pert_y,
                    zeta * raceline_speed_k,
                    ego_py_t,
                    selected_opponents,
                    k,
                )
                p_ref_y = float(np.clip(
                    p_pert_y + p_ot + p_bl,
                    -self.mpc.track_half_width,
                    self.mpc.track_half_width,
                ))
                xc, yc = frenet["center_x"], frenet["center_y"]
                psi_center = frenet["psi_center"]
                ref_world[:, k] = [
                    xc + p_ref_y * frenet["normal_x"],
                    yc + p_ref_y * frenet["normal_y"],
                ]
                raceline_world[:, k] = [frenet["world_x"], frenet["world_y"]]
                ref[0, k] = p_pert_x
                ref[1, k] = p_ref_y
                ref[2, k] = raceline_speed_k
                ref[3, k] = psi_center
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = -math.sin(psi_center)
                ref[7, k] = math.cos(psi_center)

                source_s += zeta * raceline_speed_k * self.mpc.dt
            else:
                s = current_s + k * self.step_distance
                xc, yc, psi = self.track.interpolate(s)
                offset = np.clip(
                    s1 * math.sin(s3 * s) * math.exp(-s2 * s),
                    -self.mpc.track_half_width,
                    self.mpc.track_half_width,
                )
                ref[0, k] = s % self.track.total_length
                ref[1, k] = offset
                ref[2, k] = state.vx
                ref[3, k] = psi
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = -math.sin(psi)
                ref[7, k] = math.cos(psi)
                ref_world[:, k] = self.track.frenet_to_world(ref[0, k], ref[1, k])
        self.last_ref_world_horizon = ref_world
        self.last_raceline_world_horizon = raceline_world
        return ref


class LLMAlphaRacerPolicy(AlphaRacerPolicy):
    """
    AlphaRacer policy whose five parameters are supplied by a local LLM.

    The LLM only chooses q, zeta, s1, s2, s3. MPC still computes the actual
    throttle/steering command, so bad LLM responses can be clipped or ignored.
    """

    def __init__(
        self,
        *args,
        model_name="my-racing-ctrl",
        base_url="http://localhost:11434",
        timeout=3.0,
        update_interval_steps=1,  # Only call the LLM every N steps to reduce latency and instability.
        param_bounds=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.update_interval_steps = max(1, int(update_interval_steps))
        self._llm_call_count = 0
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
            "stable": np.asarray([1.2, 1.0, 0.25, 0.8, 0.2], dtype=float),
            "push": np.asarray([1.8, 1.08, 0.20, 0.8, 0.2], dtype=float),
            "overtake": np.asarray([1.4, 1.03, 0.75, 0.18, 0.25], dtype=float),
            "block": np.asarray([1.3, 0.98, 0.55, 0.22, 1.2], dtype=float),
            "recover": np.asarray([2.5, 0.85, 0.05, 1.2, 0.0], dtype=float),
        }
        self.mode_aliases = {
            "normal": "stable",
            "safe": "stable",
            "aggressive": "push",
            "accelerate": "push",
            "speed_up": "push",
            "pass": "overtake",
            "passing": "overtake",
            "blocking": "block",
            "defend": "block",
            "defensive": "block",
            "reset": "recover",
            "fallback": "recover",
        }
        self.current_mode = "stable"

    def _prompt(self, state: CarState, current_s: float, opponents=None) -> str:
        opponents = opponents or []
        opponent_lines = []
        for idx, opponent in enumerate(opponents):
            delta_s = self._progress_delta(opponent["px"], current_s)
            opponent_lines.append(
                f"{idx}: delta_px={delta_s:.3f}, py={opponent['py']:.3f}, vx={opponent['vx']:.3f}"
            )
        opponent_text = "\n".join(opponent_lines) if opponent_lines else "none"

        _, current_py = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        preset_lines = "\n".join(
            f"- {name}: q={values[0]:.2f}, zeta={values[1]:.2f}, "
            f"s1={values[2]:.2f}, s2={values[3]:.2f}, s3={values[4]:.2f}"
            for name, values in self.mode_presets.items()
        )
        return (
            "You choose one alpha-RACER MPC behavior mode. Return only valid JSON like "
            "{\"mode\":\"stable\"}. Do not invent modes.\n"
            "Available modes:\n"
            f"{preset_lines}\n"
            "Mode meanings: stable follows the raceline smoothly; push tracks harder and uses slightly higher speed "
            "integration; overtake creates lateral offset around nearby cars; block defends against a faster rear car; "
            "recover prioritizes returning to the raceline and reducing lateral interaction.\n"
            "Selection rules: use recover when |py| is large, speed is very low after contact, or reference tracking is poor; "
            "use overtake when a slower car is close ahead; use block when a faster car is close behind; "
            "use push only on clear track; otherwise use stable.\n"
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
            "JSON example: {\"mode\":\"stable\"}"
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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
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
        body = json.loads(raw)
        return self._extract_json(body.get("response", ""))

    def _sanitize_params(self, raw_params: dict) -> np.ndarray:
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

    def update_params_from_llm(self, state: CarState, current_s: float, opponents=None):
        # Only call the LLM every update_interval_steps to avoid excessive latency and instability.
        self._llm_call_count += 1
        if self._llm_call_count > 1 and (self._llm_call_count - 1) % self.update_interval_steps != 0:
            self.params = self.last_llm_params.copy()
            return

        prompt = self._prompt(state, current_s, opponents=opponents)
        try:
            raw_params = self._query_llm(prompt)
            self.params = self._sanitize_params(raw_params)
            self.last_llm_params = self.params.copy()
            self.last_llm_error = None
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError, KeyError) as exc:
            self.last_llm_error = str(exc)
            self.params = self.last_llm_params.copy()
            if not self._warned_llm_error:
                self._log(f"LLM policy failed once, reusing last parameters: {exc}")
                self._warned_llm_error = True

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        self.update_params_from_llm(state, current_s, opponents=opponents)
        return super().get_reference_trajectory(state, current_s, opponents=opponents)


class IBRBaselinePolicy(AlphaRacerPolicy):
    """
    Traditional IBR-style opponent baseline.

    It follows the raceline with fixed parameters, adds a lateral shift around
    nearby opponents based on longitudinal distance and speed difference, and
    optionally blocks a faster rear car.
    """

    def __init__(
        self,
        *args,
        q=1.2,
        zeta=1.0,
        sf1=0.1,
        sf2=0.5,
        blocking=0.2,
        **kwargs,
    ):
        super().__init__(*args, params=np.array([q, zeta, sf1, sf2, blocking]), **kwargs)

    def _calc_shift(self, s: float, s_opp: float, v: float, v_opp: float, sf1: float, sf2: float, t: float) -> float:
        if abs(v - v_opp) < 1e-6:
            return 0.0
        eff_s = self._progress_delta(s_opp, s) + (v_opp - v) * t
        return sf1 * math.exp(-sf2 * abs(eff_s) ** 2)

    def _ibr_shift(self, current_s: float, ego_py: float, ego_vx: float, opponents, t: float = 0.5) -> float:
        _, _, sf1, sf2, blocking = self.params
        if not opponents:
            return 0.0

        shifts = []
        enriched = []
        for opponent in opponents:
            delta = self._progress_delta(opponent["px"], current_s)
            opp_py = opponent["py"]
            opp_vx = opponent["vx"]
            shift = self._calc_shift(current_s, opponent["px"], ego_vx, opp_vx, sf1, sf2, t)
            if ego_py > opp_py:
                shift = abs(shift)
            else:
                shift = -abs(shift)
            shifts.append((shift, opponent))
            enriched.append({**opponent, "delta": delta})

        total_shift = sum(shift for shift, _ in shifts)
        dominant_shift, dominant_opponent = max(shifts, key=lambda item: abs(item[0]))
        if abs(dominant_shift) > 0.03:
            candidate = total_shift + dominant_opponent["py"]
            if candidate * total_shift < 0.0:
                total_shift = 0.0
            else:
                total_shift = candidate

        rear_opponents = [op for op in enriched if op["delta"] < 0.0]
        if rear_opponents and sf1 > 1e-6:
            rear = max(rear_opponents, key=lambda op: op["delta"])
            bf = 1.0 - math.exp(-blocking * max(rear["vx"] - ego_vx, 0.0))
            proximity = self._calc_shift(current_s, rear["px"], ego_vx, rear["vx"], sf1, sf2, t) / sf1
            total_shift += (rear["py"] - total_shift) * bf * proximity

        return float(np.clip(total_shift, -self.mpc.track_half_width, self.mpc.track_half_width))

    def _competitive_offsets(
        self,
        p_pert_x: float,
        p_pert_y: float,
        v_pert_x: float,
        ego_py_t: float,
        opponents,
        k: int,
    ) -> tuple:
        horizon_t = k * self.mpc.dt
        shift = self._ibr_shift(p_pert_x, ego_py_t, v_pert_x, opponents, t=horizon_t)
        return shift, 0.0
