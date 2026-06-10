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

    def __init__(self, track: Track, mpc, step_distance: float = 0.1, car_idx=None, policy_name=None):
        self.track = track
        self.mpc = mpc
        self.step_distance = step_distance
        self.car_idx = car_idx
        self.policy_name = policy_name
        car_label = "car?" if car_idx is None else f"car{car_idx}"
        self.warning_label = car_label if policy_name is None else f"{car_label}:{policy_name}"
        self.last_u = np.array([0.0, self._neutral_steer_command()])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        self.log_fn = print
        self._mpc_failure_active = False
        self._mpc_failure_window_start_time = None
        self._mpc_failure_last_report_time = None
        self._mpc_failure_count = 0
        self._mpc_failure_window_count = 0
        self._mpc_failure_report_interval = 3.0
        self._mpc_stall_active = False
        self._mpc_stall_window_start_time = None
        self._mpc_stall_last_report_time = None
        self._mpc_stall_count = 0
        self._mpc_stall_window_count = 0
        self._mpc_stall_report_interval = 3.0

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

    def _format_mpc_interval(self, start_time, end_time) -> str:
        if start_time is None or end_time is None:
            return self._format_mpc_time(end_time)
        return f"t={start_time:.2f}-{end_time:.2f}s"

    def _report_mpc_failure(self, exc: Exception, state: CarState, sim_time=None):
        self._mpc_failure_count += 1
        self._mpc_failure_window_count += 1
        should_report = False
        if not self._mpc_failure_active:
            self._mpc_failure_active = True
            self._mpc_failure_window_start_time = sim_time
            self._mpc_failure_last_report_time = sim_time
        elif sim_time is None:
            should_report = self._mpc_failure_count in {5, 25, 100} or self._mpc_failure_count % 250 == 0
        elif (
            self._mpc_failure_last_report_time is None
            or sim_time - self._mpc_failure_last_report_time >= self._mpc_failure_report_interval
        ):
            self._mpc_failure_last_report_time = sim_time
            should_report = True

        if not should_report:
            return

        interval = self._format_mpc_interval(self._mpc_failure_window_start_time, sim_time)
        self._log(
            f"[MPC warning][{self.warning_label}] solve failed "
            f"({interval}, window_count={self._mpc_failure_window_count}, total_count={self._mpc_failure_count}, "
            f"x={state.x:.2f}, y={state.y:.2f}, vx={state.vx:.2f}); "
            f"using last control. status={exc}"
        )
        self._mpc_failure_window_start_time = sim_time
        self._mpc_failure_window_count = 0

    def _report_mpc_recovery(self, sim_time=None):
        if not self._mpc_failure_active:
            return

        interval = self._format_mpc_interval(self._mpc_failure_window_start_time, sim_time)
        self._log(
            f"[MPC warning][{self.warning_label}] solve recovered "
            f"({interval}, window_count={self._mpc_failure_window_count}, "
            f"total_failed_count={self._mpc_failure_count})"
        )
        self._mpc_failure_active = False
        self._mpc_failure_window_start_time = None
        self._mpc_failure_last_report_time = None
        self._mpc_failure_count = 0
        self._mpc_failure_window_count = 0

    def _report_mpc_stall(self, state: CarState, ref_speed: float, target_vel: float, target_steer: float, sim_time=None):
        self._mpc_stall_count += 1
        self._mpc_stall_window_count += 1
        should_report = False
        if not self._mpc_stall_active:
            self._mpc_stall_active = True
            self._mpc_stall_window_start_time = sim_time
            self._mpc_stall_last_report_time = sim_time
        elif sim_time is None:
            should_report = self._mpc_stall_count in {5, 25, 100} or self._mpc_stall_count % 250 == 0
        elif (
            self._mpc_stall_last_report_time is None
            or sim_time - self._mpc_stall_last_report_time >= self._mpc_stall_report_interval
        ):
            self._mpc_stall_last_report_time = sim_time
            should_report = True

        if not should_report:
            return

        interval = self._format_mpc_interval(self._mpc_stall_window_start_time, sim_time)
        self._log(
            f"[MPC warning][{self.warning_label}] feasible but stalled "
            f"({interval}, window_count={self._mpc_stall_window_count}, total_count={self._mpc_stall_count}, "
            f"vx={state.vx:.3f}, ref_v={ref_speed:.3f}, "
            f"cmd={target_vel:.3f}, steer={target_steer:.3f}, "
            f"loss={self.mpc.last_loss:.3g}, status={self.mpc.last_solver_status})"
        )
        self._mpc_stall_window_start_time = sim_time
        self._mpc_stall_window_count = 0

    def _report_mpc_stall_recovery(self, sim_time=None):
        if not self._mpc_stall_active:
            return

        interval = self._format_mpc_interval(self._mpc_stall_window_start_time, sim_time)
        self._log(
            f"[MPC warning][{self.warning_label}] stalled condition recovered "
            f"({interval}, window_count={self._mpc_stall_window_count}, "
            f"total_count={self._mpc_stall_count})"
        )
        self._mpc_stall_active = False
        self._mpc_stall_window_start_time = None
        self._mpc_stall_last_report_time = None
        self._mpc_stall_count = 0
        self._mpc_stall_window_count = 0

    def flush_mpc_warnings(self, sim_time=None):
        if self._mpc_failure_active and self._mpc_failure_window_count > 0:
            interval = self._format_mpc_interval(self._mpc_failure_window_start_time, sim_time)
            self._log(
                f"[MPC warning][{self.warning_label}] solve failed "
                f"({interval}, window_count={self._mpc_failure_window_count}, "
                f"total_count={self._mpc_failure_count}); still using last control. "
                f"status={self.mpc.last_solver_status}"
            )
            self._mpc_failure_window_start_time = sim_time
            self._mpc_failure_window_count = 0

        if self._mpc_stall_active and self._mpc_stall_window_count > 0:
            interval = self._format_mpc_interval(self._mpc_stall_window_start_time, sim_time)
            self._log(
                f"[MPC warning][{self.warning_label}] feasible but stalled "
                f"({interval}, window_count={self._mpc_stall_window_count}, "
                f"total_count={self._mpc_stall_count}); still stalled. "
                f"loss={self.mpc.last_loss:.3g}, status={self.mpc.last_solver_status}"
            )
            self._mpc_stall_window_start_time = sim_time
            self._mpc_stall_window_count = 0

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
    # Simple baseline policy that tracks the centerline at a constant reference speed.
    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        for k in range(horizon + 1):
            s = current_s + k * self.step_distance
            x, y, psi = self.track.interpolate(s)
            ref[0, k] = s % self.track.total_length
            ref[1, k] = 0.0
            ref[2, k] = state.vx # Track current speed as reference speed to encourage smooth acceleration/deceleration.
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
        car_idx=None,
        policy_name=None,
    ):
        super().__init__(track, mpc, step_distance, car_idx=car_idx, policy_name=policy_name)
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
            plan_idx = None
            if "px_plan" in opponent:
                plan_idx = min(k, len(opponent["px_plan"]) - 1)
            opp_px = (
                opponent["px_plan"][plan_idx]
                if plan_idx is not None
                else opponent["px"] + k * self.mpc.dt * opponent["vx"]
            )
            opp_py = opponent["py_plan"][plan_idx] if plan_idx is not None and "py_plan" in opponent else opponent["py"]
            opp_vx = opponent["vx_plan"][plan_idx] if plan_idx is not None and "vx_plan" in opponent else opponent["vx"]
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
        history_length=5,
        output_mode="auto", # "mode" for discrete mode selection, "params" for direct parameter output, "auto" to infer from model name.
        param_bounds=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.history_length = max(0, int(history_length))
        self.output_mode = self._resolve_output_mode(output_mode, model_name)
        if self.output_mode not in {"mode", "params"}:
            raise ValueError(f"Unknown LLM output mode: {output_mode}")
        self.llm_decision_history = []
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
            "overtake": np.asarray([1.0, 1.0, 1.75, 0.18, 0.25], dtype=float),
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
                f"call={item['call']}, mode={item['mode']}, "
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
                "params": self.params.copy(),
            }
        )
        if len(self.llm_decision_history) > self.history_length:
            self.llm_decision_history = self.llm_decision_history[-self.history_length:]

    def _prompt(self, state: CarState, current_s: float, opponents=None) -> str:
        opponents = opponents or []
        opponent_lines = []
        for idx, opponent in enumerate(opponents):
            delta_s = self._progress_delta(opponent["px"], current_s)
            relation = "ahead" if delta_s > 0.0 else "behind" if delta_s < 0.0 else "alongside"
            speed_relation = "slower" if opponent["vx"] < state.vx else "faster" if opponent["vx"] > state.vx else "same_speed"
            opponent_lines.append(
                f"{idx}: delta_px={delta_s:.3f}, relation={relation}, "
                f"speed_relation={speed_relation}, py={opponent['py']:.3f}, vx={opponent['vx']:.3f}"
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
                "{\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2}. Do not include a mode key.\n"
                f"{common_context}"
                "Parameter guidance: increase q for tighter tracking/recovery only after off-track, collision, or repeated stall; "
                "do not treat low vx at race start as a recovery condition; reduce zeta when approaching hard turns or unstable; "
                "increase s1 for larger lateral overtake/block offset; increase s2 for more local/short-range interaction; "
                "increase s3 only when blocking a faster rear car is useful.\n"
                "Use recent LLM outputs to avoid oscillation. If the recent output was overtaking and the same slow front car is still close, "
                "keep overtaking-style parameters unless the car is off-track, collided, or already clear of that car.\n"
                "JSON example: {\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2}"
            )
        history_text = self._history_text()
        return (
            "You choose one alpha-RACER MPC behavior mode. Return only valid JSON like "
            "{\"mode\":\"stable\"}. Do not invent modes.\n"
            "Critical rule: if any opponent line has relation=ahead, speed_relation=slower, "
            "and positive delta_px below 2.5 m, the only valid output is {\"mode\":\"overtake\"}.\n" # emphasize overtake here
            "Available modes:\n"
            f"{preset_lines}\n"
            "Mode meanings: stable follows the raceline smoothly; push tracks harder and uses slightly higher speed "
            "integration; overtake creates lateral offset around nearby cars; block defends against a faster rear car; "
            "recover prioritizes returning to the raceline and reducing lateral interaction.\n"
            "Selection rules: choose overtake when an opponent is ahead, close, and slower than ego. "
            "Close means positive delta_px below 2.5 m. "
            "Overtake is valid only when there is a slower front opponent with positive delta_px below 3.0 m. "
            "If no opponent line satisfies relation=ahead, speed_relation=slower, and positive delta_px below 3.0 m, do not output overtake. "
            "If a slower front opponent has positive delta_px below 2.5 m, overtake has priority over push. "
            "Push is invalid when any opponent is ahead with positive delta_px below 3.0 m. "
            "If recent outputs include overtake and that slow front opponent is still close, keep overtake until ego is clearly past it. "
            "Clearly past means that opponent's delta_px becomes negative. "
            "After ego is clearly past the slow front opponent, use push on clear track or stable if uncertain. "
            "Block is invalid unless an opponent has relation=behind and speed_relation=faster. "
            "Do not choose recover just because py is offset during an overtake; choose recover only when |py| is large, "
            "after contact, after off-track behavior, or when speed is very low after a crash. "
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
            "Use recent LLM outputs to avoid oscillation while the same close front opponent remains. "
            "Do not continue overtake from history after the close front opponent is gone.\n"
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

    def _mode_context(self, state: CarState, current_s: float, opponents=None) -> dict:
        opponents = opponents or []
        close_slow_front = False
        slow_front = False
        faster_rear = False
        nearest_front_delta = None
        for opponent in opponents:
            delta_s = self._progress_delta(opponent["px"], current_s)
            if delta_s > 0.0:
                if nearest_front_delta is None or delta_s < nearest_front_delta:
                    nearest_front_delta = delta_s
                if opponent["vx"] < state.vx:
                    slow_front = slow_front or delta_s < 3.0
                    close_slow_front = close_slow_front or delta_s < 2.5
            elif delta_s < 0.0 and opponent["vx"] > state.vx:
                faster_rear = faster_rear or abs(delta_s) < 3.0

        _, current_py = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        return {
            "close_slow_front": close_slow_front,
            "slow_front": slow_front,
            "faster_rear": faster_rear,
            "nearest_front_delta": nearest_front_delta,
            "current_py": current_py,
        }

    def _apply_mode_guard(self, state: CarState, current_s: float, opponents=None):
        if self.output_mode != "mode":
            return

        context = self._mode_context(state, current_s, opponents)
        guarded_mode = self.current_mode

        if context["close_slow_front"]:
            if guarded_mode in {"push", "stable", "block"}:
                guarded_mode = "overtake"
            elif guarded_mode == "recover" and abs(context["current_py"]) < 0.8 and state.vx > 0.5:
                guarded_mode = "overtake"
        elif guarded_mode == "overtake" and not context["slow_front"]:
            guarded_mode = "push" if abs(context["current_py"]) < 0.6 and state.vx > 1.0 else "stable"
        elif guarded_mode == "block" and not context["faster_rear"]:
            guarded_mode = "stable"

        if guarded_mode != self.current_mode:
            self.current_mode = guarded_mode
            self.params = self.mode_presets[guarded_mode].copy()

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
            self._apply_mode_guard(state, current_s, opponents=opponents)
            self.last_llm_params = self.params.copy()
            self.last_llm_error = None
            self._record_llm_decision()
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
    Iterated Best Response baseline for the alpha-RACER policy class.
    
    The paper's IBR baseline repeatedly replans each agent against the current
    plans of the other agents. The original implementation is not public, so
    this class implements the published structure: round-robin best responses,
    finite horizon rollouts, and the relative-progress utility.
    """

    def __init__(
        self,
        *args,
        q=1.2,
        zeta=1.0,
        s1=0.2,
        s2=0.2,
        s3=0.2,
        iterations=6,
        gamma=0.99,
        candidate_params=None,
        unsafe_longitudinal=0.45,
        unsafe_lateral=0.35,
        collision_penalty=2.0,
        **kwargs,
    ):
        super().__init__(*args, params=np.array([q, zeta, s1, s2, s3]), **kwargs)
        self.iterations = int(iterations)
        self.gamma = float(gamma)
        self.candidate_params = candidate_params
        self.unsafe_longitudinal = float(unsafe_longitudinal)
        self.unsafe_lateral = float(unsafe_lateral)
        self.collision_penalty = float(collision_penalty)
        self._ibr_cached_action = None
        self._ibr_cached_action_time = None
        self._ibr_action_valid_until = -np.inf
        self._ibr_cached_plan = None
        self._ibr_cached_ref_traj = None
        self._ibr_cached_ref_world_horizon = None
        self._ibr_cached_raceline_world_horizon = None

    def _candidate_param_grid(self):
        candidates = [
            self.params.copy(),
            np.array([1.0, 0.95, 0.0, 0.35, 0.0]),
            np.array([1.2, 1.00, 0.2, 0.25, 0.2]),
            np.array([1.4, 1.05, 0.45, 0.18, 0.35]),
            np.array([1.6, 1.10, 0.65, 0.12, 0.8]),
            np.array([2.0, 0.85, 0.05, 0.70, 0.0]),
        ]
        if self.candidate_params is not None:
            candidates.extend(np.asarray(values, dtype=float) for values in self.candidate_params)

        unique = []
        seen = set()
        for values in candidates:
            clipped = np.asarray(values, dtype=float).copy()
            clipped[0] = np.clip(clipped[0], 0.05, 20.0)
            clipped[1] = np.clip(clipped[1], 0.1, 3.0)
            clipped[2] = np.clip(clipped[2], 0.0, self.mpc.track_half_width)
            clipped[3] = np.clip(clipped[3], 0.001, 5.0)
            clipped[4] = np.clip(clipped[4], 0.0, 5.0)
            key = tuple(np.round(clipped, 4))
            if key not in seen:
                unique.append(clipped)
                seen.add(key)
        return unique

    def _state_plan_to_frenet_plan(self, state_plan, current_s):
        px = np.zeros(state_plan.shape[1])
        py = np.zeros(state_plan.shape[1])
        last_s = current_s
        for k in range(state_plan.shape[1]):
            s, e = self.track.project(
                state_plan[0, k],
                state_plan[1, k],
                s_hint=last_s,
                search_radius=8.0,
                psi_hint=state_plan[2, k],
                max_heading_error=math.pi / 2.0,
            )
            px[k] = s
            py[k] = e
            last_s = s
        return {
            "px": px,
            "py": py,
            "vx": np.maximum(state_plan[3], 0.0),
            "x": state_plan[0].copy(),
            "y": state_plan[1].copy(),
        }

    def _relative_progress_utility(self, own_plan, opponent_plans):
        horizon_lengths = [len(own_plan["px"]) - 1]
        horizon_lengths.extend(len(plan["px"]) - 1 for plan in opponent_plans)
        horizon = min(horizon_lengths)
        value = 0.0
        discount = 1.0
        for k in range(horizon):
            if opponent_plans:
                max_opp_now = max(plan["px"][k] for plan in opponent_plans)
                max_opp_next = max(plan["px"][k + 1] for plan in opponent_plans)
                reward = (own_plan["px"][k + 1] - max_opp_next) - (own_plan["px"][k] - max_opp_now)
            else:
                reward = own_plan["px"][k + 1] - own_plan["px"][k]

            safety_cost = 0.0
            for plan in opponent_plans:
                dx = abs(self._progress_delta(own_plan["px"][k + 1], plan["px"][k + 1]))
                dy = abs(own_plan["py"][k + 1] - plan["py"][k + 1])
                if dx < self.unsafe_longitudinal and dy < self.unsafe_lateral:
                    safety_cost += self.collision_penalty * (
                        (self.unsafe_longitudinal - dx) / self.unsafe_longitudinal
                        + (self.unsafe_lateral - dy) / self.unsafe_lateral
                    )
            offroad = max(abs(own_plan["py"][k + 1]) - self.mpc.track_half_width, 0.0)
            value += discount * (reward - safety_cost - 2.0 * offroad * offroad)
            discount *= self.gamma
        return value

    def _make_opponents_from_plans(self, car_idx, car_s, car_e, states, plans):
        opponents = []
        for other_idx, plan in plans.items():
            if other_idx == car_idx:
                continue
            opponents.append(
                {
                    "car_idx": other_idx,
                    "px": car_s[other_idx],
                    "py": car_e[other_idx],
                    "vx": max(states[other_idx].vx, 0.0),
                    "px_plan": plan["px"],
                    "py_plan": plan["py"],
                    "vx_plan": plan["vx"],
                }
            )
        return opponents

    def _best_response_plan(self, car_idx, state, current_s, car_s, car_e, states, plans):
        opponents = self._make_opponents_from_plans(car_idx, car_s, car_e, states, plans)
        opponent_plans = [plans[j] for j in plans if j != car_idx]
        old_params = self.params.copy()
        best = None

        for params_candidate in self._candidate_param_grid():
            self.params = params_candidate.copy()
            try:
                ref_traj = self.get_reference_trajectory(state, current_s, opponents=opponents)
                u_opt, u_first = self.mpc.solve(
                    state.to_array(),
                    ref_traj,
                    self.last_u,
                    q_weight=self.get_q_weight(),
                )
            except Exception:
                continue

            plan = self._state_plan_to_frenet_plan(self.mpc.last_state_plan, current_s)
            value = self._relative_progress_utility(plan, opponent_plans)
            if best is None or value > best["value"]:
                best = {
                    "value": value,
                    "params": self.params.copy(),
                    "u_opt": u_opt.copy(),
                    "u_first": u_first.copy(),
                    "plan": plan,
                    "ref_traj": ref_traj.copy(),
                    "ref_world": None if self.last_ref_world_horizon is None else self.last_ref_world_horizon.copy(),
                    "raceline_world": (
                        None
                        if self.last_raceline_world_horizon is None
                        else self.last_raceline_world_horizon.copy()
                    ),
                    "loss": self.mpc.last_loss,
                    "track_loss": self.mpc.last_track_loss,
                    "action_loss": self.mpc.last_action_loss,
                    "velocity_loss": self.mpc.last_velocity_loss,
                    "heading_loss": self.mpc.last_heading_loss,
                    "solver_success": self.mpc.last_solver_success,
                    "solver_status": self.mpc.last_solver_status,
                    "state_plan": self.mpc.last_state_plan.copy(),
                    "control_plan": self.mpc.last_control_plan.copy(),
                }

        self.params = old_params
        return best

    def _accept_ibr_plan(self, result, sim_time):
        self.params = result["params"].copy()
        self.last_ref_traj = result["ref_traj"]
        self.last_ref_world_horizon = result["ref_world"]
        self.last_raceline_world_horizon = result["raceline_world"]
        self.mpc.last_loss = result["loss"]
        self.mpc.last_track_loss = result["track_loss"]
        self.mpc.last_action_loss = result["action_loss"]
        self.mpc.last_velocity_loss = result["velocity_loss"]
        self.mpc.last_heading_loss = result["heading_loss"]
        self.mpc.last_solver_success = result["solver_success"]
        self.mpc.last_solver_status = result["solver_status"]
        self.mpc.last_state_plan = result["state_plan"]
        self.mpc.last_control_plan = result["control_plan"]
        self._ibr_cached_plan = result["plan"]
        self._ibr_cached_ref_traj = result["ref_traj"]
        self._ibr_cached_ref_world_horizon = result["ref_world"]
        self._ibr_cached_raceline_world_horizon = result["raceline_world"]
        self._ibr_cached_action = CarAction(
            throttle=float(np.clip(result["u_first"][0], -1.0, 1.0)),
            steer=float(np.clip(result["u_first"][1], -1.0, 1.0)),
        )
        self._ibr_cached_action_time = sim_time
        self._ibr_action_valid_until = (sim_time if sim_time is not None else 0.0) + self.mpc.dt
        self.last_u = np.array([self._ibr_cached_action.throttle, self._ibr_cached_action.steer])

    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        if (
            self._ibr_cached_action is not None
            and sim_time is not None
            and sim_time < self._ibr_action_valid_until - 1e-9
        ):
            self.last_ref_traj = self._ibr_cached_ref_traj
            self.last_ref_world_horizon = self._ibr_cached_ref_world_horizon
            self.last_raceline_world_horizon = self._ibr_cached_raceline_world_horizon
            return self._ibr_cached_action
        return super().get_action(state, current_s, opponents=opponents, sim_time=sim_time)


def _constant_frenet_plan(policy, state, current_s, current_e):
    horizon = policy.mpc.horizon
    px = np.zeros(horizon + 1)
    py = np.full(horizon + 1, current_e)
    vx = np.full(horizon + 1, max(state.vx, 0.0))
    x = np.zeros(horizon + 1)
    y = np.zeros(horizon + 1)
    for k in range(horizon + 1):
        px[k] = current_s + k * policy.mpc.dt * max(state.vx, 0.0)
        x[k], y[k] = policy.track.frenet_to_world(px[k], py[k])
    return {"px": px, "py": py, "vx": vx, "x": x, "y": y}


def _latest_policy_plan(policy, state, current_s, current_e):
    if isinstance(policy, IBRBaselinePolicy) and policy._ibr_cached_plan is not None:
        plan = policy._ibr_cached_plan
        if len(plan["px"]) >= policy.mpc.horizon + 1:
            shifted = {}
            for key, values in plan.items():
                arr = np.asarray(values)
                if len(arr) >= policy.mpc.horizon + 1:
                    shifted[key] = np.r_[arr[1:], arr[-1]]
            if {"px", "py", "vx", "x", "y"}.issubset(shifted):
                shifted["px"][0] = current_s
                shifted["py"][0] = current_e
                shifted["vx"][0] = max(state.vx, 0.0)
                shifted["x"][0] = state.x
                shifted["y"][0] = state.y
                return shifted
    if policy.mpc.last_state_plan is not None:
        try:
            return policy._state_plan_to_frenet_plan(policy.mpc.last_state_plan, current_s)
        except AttributeError:
            pass
    return _constant_frenet_plan(policy, state, current_s, current_e)


def plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=None):
    ibr_indices = [idx for idx, policy in enumerate(policies) if isinstance(policy, IBRBaselinePolicy)]
    if not ibr_indices:
        return

    due_indices = [
        idx for idx in ibr_indices
        if policies[idx]._ibr_cached_action is None
        or sim_time is None
        or sim_time >= policies[idx]._ibr_action_valid_until - 1e-9
    ]
    if not due_indices:
        return

    plans = {
        idx: _latest_policy_plan(policy, states[idx], car_s[idx], car_e[idx])
        for idx, policy in enumerate(policies)
    }
    iterations = max(policies[idx].iterations for idx in due_indices)
    best_results = {}
    for iteration in range(iterations):
        for idx in due_indices:
            policy = policies[idx]
            if iteration >= policy.iterations:
                continue
            result = policy._best_response_plan(idx, states[idx], car_s[idx], car_s, car_e, states, plans)
            if result is None:
                continue
            plans[idx] = result["plan"]
            best_results[idx] = result

    for idx, result in best_results.items():
        policies[idx]._accept_ibr_plan(result, sim_time)
