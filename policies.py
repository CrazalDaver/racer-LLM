import abc
import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

import casadi as cs
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # Torch is only required for the original alpha-RACER weight policy.
    torch = None
    nn = None

from state import CarAction, CarState
from track import RaceLine, Track


class BasePolicy(abc.ABC):
    """Common interface and observable state shared by every driving policy."""

    def __init__(self, track: Track, mpc, step_distance: float = 0.1, car_idx=None, policy_name=None):
        self.track = track
        self.mpc = mpc
        self.step_distance = step_distance
        self.car_idx = car_idx
        self.policy_name = policy_name
        car_label = "car?" if car_idx is None else f"car{car_idx}"
        self.warning_label = car_label if policy_name is None else f"{car_label}:{policy_name}"

        self.last_u = np.zeros(2, dtype=float)
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        self.log_fn = print

    @abc.abstractmethod
    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        """Return the control action selected by this policy."""
        raise NotImplementedError

    def _log(self, message: str):
        self.log_fn(message)

    def flush_solver_warnings(self, sim_time=None):
        """Flush policy-specific solver warnings, if the policy uses a solver."""

    def flush_mpc_warnings(self, sim_time=None):
        """Backward-compatible wrapper for older simulation callers."""
        self.flush_solver_warnings(sim_time=sim_time)


class MPCPolicy(BasePolicy):
    """
    Policy implementation that converts a reference trajectory into an action with MPC.

    Implement main components of an MPC-based policy:
    -   MPC controller: get_action
    -   MPC solver failure recovery: _recovery_control
    -   Status reporting: _report_mpc_failure, _report_mpc_recovery, _report_mpc_stall, _report_mpc_stall_recovery
    -   Interface abstraction: get_q_weight, @abstractmethod get_reference_trajectory

    Subclasses implement get_reference_trajectory to provide a reference trajectory.
    """

    def __init__(self, track: Track, mpc, step_distance: float = 0.1, car_idx=None, policy_name=None):
        super().__init__(track, mpc, step_distance, car_idx=car_idx, policy_name=policy_name)
        self.last_u = np.array([0.0, self._neutral_steer_command()])

        # MPC failure and stall reporting state
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

    # different policy is mainly defined by the reference trajectory it provides to the MPC solver
    @abc.abstractmethod
    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        pass

    # might be a little bit akward way to get q value...
    def get_q_weight(self) -> float:
        return 1.0

    # simple failure recovery
    def _recovery_control(self, state: CarState, current_s: float) -> np.ndarray:
        # 
        _, current_e = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        _, _, track_psi = self.track.interpolate(current_s)
        heading_error = math.atan2(math.sin(state.psi - track_psi), math.cos(state.psi - track_psi))
        half_width = max(getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width), 1e-6)
        steer = np.clip(-1.8 * current_e / half_width - 0.7 * heading_error, -1.0, 1.0)
        throttle = -0.2 if state.vx > 3.0 else 0.45
        return np.array([throttle, steer], dtype=float)

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

    def flush_solver_warnings(self, sim_time=None):
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

    # Main MPC control interface:
    # call get_reference_trajectory to get the reference trajectory, then call mpc.solve to get the control action
    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        ref_traj = self.get_reference_trajectory(state, current_s, opponents=opponents)
        self.last_ref_traj = ref_traj
        state_arr = state.to_array()
        try:
            _, u_first = self.mpc.solve(
                state_arr,
                ref_traj,
                self.last_u,
                q_weight=self.get_q_weight(),
                opponents=opponents,
                current_s=current_s,
            )
        except Exception as e:
            self._report_mpc_failure(e, state, sim_time=sim_time)
            u_first = self._recovery_control(state, current_s)
            self.mpc.last_solver_status = "MPC_recovery"
            try:
                self.mpc.evaluate_fallback_loss(
                    state_arr,
                    ref_traj,
                    u_first,
                    q_weight=self.get_q_weight(),
                    opponents=opponents,
                    current_s=current_s,
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


class CenterLinePolicy(MPCPolicy):
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


class AlphaRacerReferencePolicy(BasePolicy):
    """
    Shared alpha-RACER reference-trajectory logic parameterized by
    theta=(q, zeta, s1, s2, s3).

    This class deliberately does not choose an action. Direct-action policies
    and MPC-backed policies can both reuse its track and interaction helpers.
    """

    def __init__(
        self,
        track,
        mpc,
        step_distance=1.2, # larger step distance to match the original alpha-RACER policy's raceline lookahead
        params=None,
        raceline: RaceLine = None,
        max_speed=5.0,
        ref_lateral_clip=None,
        car_idx=None,
        policy_name=None,
    ):
        super().__init__(track, mpc, step_distance, car_idx=car_idx, policy_name=policy_name)
        self.params = params if params is not None else np.array([1.0, 1.0, 0, 0, 0])
        self.raceline = raceline
        self.max_speed = max_speed
        if ref_lateral_clip is None:
            ref_lateral_clip = getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        self.ref_lateral_clip = float(max(ref_lateral_clip, 0.0))
        self.last_raceline_s = None
        self.last_raceline_idx = None

    def get_q_weight(self) -> float:
        return float(self.params[0])

    def _ref_lateral_limit(self) -> float:
        return self.ref_lateral_clip

    def _raceline_lookahead_distance(self, state: CarState) -> float:
        """Optional forward shift of the raceline reference origin."""
        return 0.0

    def _nearest_raceline_s(self, state: CarState) -> float:
        self.last_raceline_idx = self.raceline.nearest_index(
            state.x,
            state.y,
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

    def _raceline_frenet_point(self, raceline_s: float, center_hint: float = None) -> dict:
        point = self.raceline.interpolate(raceline_s)  # Get the raceline point in world coordinates.
        center_s, lateral_e = self.track.project(
            point["x_m"],
            point["y_m"],
            s_hint=center_hint,
            search_radius=8.0,
            psi_hint=point["psi_rad"],
            max_heading_error=math.pi / 2.0,
        )
        center_x, center_y, psi_center = self.track.interpolate(center_s)
        normal_x = -math.sin(psi_center)
        normal_y = math.cos(psi_center)
        return {
            "track_s": center_s,
            "lateral_e": lateral_e,
            "speed": min(point["vx_mps"], self.max_speed),
            "center_x": center_x,
            "center_y": center_y,
            "psi_center": psi_center,
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

    def _angle_delta(self, angle_a: float, angle_b: float) -> float:
        return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))

    def _track_heading_error(self, psi: float, s: float, reference_minus_heading: bool = True) -> float:
        _, _, ref_psi = self.track.interpolate(s)
        if reference_minus_heading:
            return self._angle_delta(ref_psi, psi)
        return self._angle_delta(psi, ref_psi)

    def _track_curvature_at(self, s: float, ds: float = 0.25) -> float:
        _, _, psi_prev = self.track.interpolate(s - ds)
        _, _, psi_next = self.track.interpolate(s + ds)
        return self._angle_delta(psi_next, psi_prev) / (2.0 * ds)

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
            source_s = raceline_s + self._raceline_lookahead_distance(state)
            center_hint = current_s
        for k in range(horizon + 1):
            if self.raceline is not None:
                frenet = self._raceline_frenet_point(source_s, center_hint=center_hint)
                center_hint = frenet["track_s"]
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
                    -self._ref_lateral_limit(),
                    self._ref_lateral_limit(),
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
                    -self._ref_lateral_limit(),
                    self._ref_lateral_limit(),
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


class ParameterizedAlphaRacerPolicy(AlphaRacerReferencePolicy, MPCPolicy):
    """MPC-backed alpha-RACER policy with parameterized reference generation."""


class StaticAlphaRacerPolicy(ParameterizedAlphaRacerPolicy):
    """
    Fixed-parameter controller from the alpha-RACER policy class.

    The five parameters q, zeta, s1, s2, s3 are held constant during
    simulation, so this is the static-policy baseline.
    """


class SelfPlayRLPolicy(AlphaRacerReferencePolicy):
    """
    Direct-action self-play RL baseline compatible with alpha-RACER's RL branch.

    The upstream comparison loads ``PPO.load("trained_policy9")`` and calls
    ``predict`` on a 15-value observation. The local self_play_rl.py follows
    the same Stable-Baselines3 checkpoint and inference interface.
    """

    OBSERVATION_SIZE = 15

    def __init__(
        self,
        *args,
        model=None,
        model_path=None,
        backend="auto",
        strict_load=False,
        deterministic=True,
        action_order="throttle_steer",
        throttle_bounds=(0.0, 1.0),
        steer_bounds=(-1.0, 1.0),
        obs_mean=None,
        obs_std=None,
        action_scale=None,
        action_bias=None,
        fallback_gain=(0.4, 1.5),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.current_mode = "self_play_rl"
        self.backend = str(backend or "auto").strip().lower()
        self.strict_load = bool(strict_load)
        self.deterministic = bool(deterministic)
        self.action_order = str(action_order or "throttle_steer").strip().lower()
        self.throttle_bounds = tuple(float(v) for v in throttle_bounds)
        self.steer_bounds = tuple(float(v) for v in steer_bounds)
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=float)
        self.obs_std = None if obs_std is None else np.asarray(obs_std, dtype=float)
        self.action_scale = None if action_scale is None else np.asarray(action_scale, dtype=float)
        self.action_bias = None if action_bias is None else np.asarray(action_bias, dtype=float)
        self.fallback_gain = np.asarray(fallback_gain, dtype=float)
        self.model_path = None if model_path in {None, ""} else Path(model_path)
        self.model = model
        self.model_kind = "object" if model is not None else "fallback"
        self.last_observation = np.full(self.OBSERVATION_SIZE, np.nan)
        self.last_raw_action = np.full(2, np.nan)
        self.last_ref_traj = None
        self._warned_model_load = False
        if self.model is None and self.model_path is not None:
            self._load_model_from_path(self.model_path)

    def _warn_load_failure(self, exc):
        if self.strict_load:
            raise exc
        if not self._warned_model_load:
            self._log(
                f"[self-play RL][{self.warning_label}] model unavailable; "
                f"using fallback direct policy. reason={exc}"
            )
            self._warned_model_load = True

    def _load_model_from_path(self, model_path: Path):
        path = Path(model_path)
        if not path.exists():
            self._warn_load_failure(FileNotFoundError(path))
            return

        suffix = path.suffix.lower()
        try:
            if self.backend in {"auto", "npz"} and suffix == ".npz":
                self.model = self._load_npz_policy(path)
                self.model_kind = "npz"
                return
            if self.backend in {"auto", "json"} and suffix == ".json":
                self.model = self._load_json_policy(path)
                self.model_kind = "json"
                return
            if self.backend in {"auto", "npy"} and suffix == ".npy":
                self.model = {"type": "linear", "weights": np.load(path), "bias": np.zeros(2)}
                self.model_kind = "npy"
                return
            if self.backend in {"auto", "torch", "pth", "pt"} and suffix in {".pth", ".pt"}:
                self.model = self._load_torch_policy(path)
                self.model_kind = "torch"
                return
            if self.backend in {"auto", "sb3", "ppo"}:
                self.model = self._load_sb3_policy(path)
                self.model_kind = "sb3"
                return
            raise ValueError(f"Unsupported self-play RL policy file: {path}")
        except Exception as exc:
            self.model = None
            self.model_kind = "fallback"
            self._warn_load_failure(exc)

    def _load_sb3_policy(self, path: Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is only required for SB3/PPO self-play RL models. "
                "Use a .npz/.json exported policy or install stable-baselines3 to load this file."
            ) from exc
        return PPO.load(str(path))

    def _load_torch_policy(self, path: Path):
        try:
            from self_play_rl import load_policy_checkpoint
        except ImportError as exc:
            raise ImportError(
                "PyTorch and self_play_rl.py are required to load native self-play RL checkpoints."
            ) from exc
        model, metadata = load_policy_checkpoint(path, device="cpu")
        if "action_order" in metadata:
            self.action_order = str(metadata["action_order"]).strip().lower()
        if "throttle_bounds" in metadata:
            self.throttle_bounds = tuple(float(v) for v in metadata["throttle_bounds"])
        if "steer_bounds" in metadata:
            self.steer_bounds = tuple(float(v) for v in metadata["steer_bounds"])
        return model

    def _load_npz_policy(self, path: Path):
        data = np.load(path, allow_pickle=False)
        policy = {"type": "linear"}
        if "W" in data:
            policy["weights"] = np.asarray(data["W"], dtype=float)
            policy["bias"] = np.asarray(data["b"], dtype=float) if "b" in data else np.zeros(2)
        elif "weights" in data:
            policy["weights"] = np.asarray(data["weights"], dtype=float)
            policy["bias"] = np.asarray(data["bias"], dtype=float) if "bias" in data else np.zeros(2)
        else:
            layers = []
            idx = 0
            while f"W{idx}" in data:
                bias_key = f"b{idx}"
                layers.append(
                    (
                        np.asarray(data[f"W{idx}"], dtype=float),
                        np.asarray(data[bias_key], dtype=float) if bias_key in data else None,
                    )
                )
                idx += 1
            if not layers:
                raise ValueError(f"{path} must contain W/b, weights/bias, or W0/b0... arrays")
            policy = {"type": "mlp", "layers": layers}

        self._read_optional_npz_arrays(data)
        return policy

    def _read_optional_npz_arrays(self, data):
        if "obs_mean" in data:
            self.obs_mean = np.asarray(data["obs_mean"], dtype=float)
        if "obs_std" in data:
            self.obs_std = np.asarray(data["obs_std"], dtype=float)
        if "action_scale" in data:
            self.action_scale = np.asarray(data["action_scale"], dtype=float)
        if "action_bias" in data:
            self.action_bias = np.asarray(data["action_bias"], dtype=float)

    def _load_json_policy(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            spec = json.load(file)
        if "obs_mean" in spec:
            self.obs_mean = np.asarray(spec["obs_mean"], dtype=float)
        if "obs_std" in spec:
            self.obs_std = np.asarray(spec["obs_std"], dtype=float)
        if "action_scale" in spec:
            self.action_scale = np.asarray(spec["action_scale"], dtype=float)
        if "action_bias" in spec:
            self.action_bias = np.asarray(spec["action_bias"], dtype=float)
        if "action_order" in spec:
            self.action_order = str(spec["action_order"]).strip().lower()

        if "layers" in spec:
            return {
                "type": "mlp",
                "layers": [
                    (
                        np.asarray(layer["weight"], dtype=float),
                        np.asarray(layer.get("bias", 0.0), dtype=float),
                    )
                    for layer in spec["layers"]
                ],
            }
        return {
            "type": "linear",
            "weights": np.asarray(spec["weights"], dtype=float),
            "bias": np.asarray(spec.get("bias", [0.0, 0.0]), dtype=float),
        }

    def _normalize_observation(self, observation):
        obs = np.asarray(observation, dtype=float)
        obs[~np.isfinite(obs)] = 0.0
        if self.obs_mean is not None:
            obs = obs - self.obs_mean
        if self.obs_std is not None:
            obs = obs / np.maximum(self.obs_std, 1e-6)
        return obs

    def _apply_action_transform(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)[:2]
        if self.action_scale is not None:
            action = action * self.action_scale[:2]
        if self.action_bias is not None:
            action = action + self.action_bias[:2]
        if self.action_order in {"steer_throttle", "steering_throttle"}:
            action = np.asarray([action[1], action[0]], dtype=float)
        throttle = float(np.clip(action[0], self.throttle_bounds[0], self.throttle_bounds[1]))
        steer = float(np.clip(action[1], self.steer_bounds[0], self.steer_bounds[1]))
        return CarAction(throttle=throttle, steer=steer)

    def _predict_array_policy(self, policy, observation):
        x = self._normalize_observation(observation)
        if policy["type"] == "linear":
            weights = np.asarray(policy["weights"], dtype=float)
            if weights.shape[0] == x.shape[0] and weights.ndim == 2:
                weights = weights.T
            y = weights @ x + np.asarray(policy["bias"], dtype=float)
            return y

        y = x
        for idx, (weights, bias) in enumerate(policy["layers"]):
            weights = np.asarray(weights, dtype=float)
            if weights.shape[0] == y.shape[0] and weights.ndim == 2:
                weights = weights.T
            y = np.asarray(weights, dtype=float) @ y
            if bias is not None:
                y = y + np.asarray(bias, dtype=float)
            if idx < len(policy["layers"]) - 1:
                y = np.tanh(y)
        return y

    def _predict_model(self, observation):
        if self.model is None:
            return self._fallback_action(observation)
        if isinstance(self.model, dict):
            return self._predict_array_policy(self.model, observation)
        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(np.asarray(observation, dtype=float), deterministic=self.deterministic)
            return action
        if callable(self.model):
            return self.model(np.asarray(observation, dtype=float))
        raise TypeError(f"Unsupported self-play RL model object: {type(self.model)!r}")

    def _fallback_action(self, observation):
        obs = np.asarray(observation, dtype=float)
        lateral_error = obs[1]
        heading_error = obs[3]
        current_speed = max(obs[4], 0.0)
        target_speed = self.max_speed
        if self.raceline is not None and self.last_raceline_s is not None:
            try:
                target_speed = min(self.raceline.interpolate(self.last_raceline_s)["vx_mps"], self.max_speed)
            except Exception:
                target_speed = self.max_speed
        speed_gain = self.fallback_gain[0] if self.fallback_gain.size > 0 else 0.4
        steer_gain = self.fallback_gain[1] if self.fallback_gain.size > 1 else 1.5
        throttle = speed_gain * (target_speed - current_speed)
        steer = -steer_gain * heading_error - 0.8 * lateral_error
        return np.asarray([throttle, steer], dtype=float)

    def _nearest_opponent(self, current_s, opponents):
        if not opponents:
            return None
        return min(
            opponents,
            key=lambda opponent: abs(self._progress_delta(float(opponent["px"]), float(current_s))),
        )

    def _build_original_observation(self, state: CarState, current_s: float, opponents):
        # 15-dim observation: 
        # [delta_s, e, opp_e, own_heading_error, vx, vy, omega, opp_heading_error, opp_vx, opp_vy, opp_omega, kappa_own, kappa_opp, kappa_own_lookahead, kappa_opp_lookahead]
        _, current_e = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        opponent = self._nearest_opponent(current_s, opponents)
        if opponent is None:
            opponent = {
                "px": current_s,
                "py": current_e,
                "vx": state.vx,
                "vy": state.vy,
                "omega": state.omega,
                "psi": state.psi,
            }

        own_heading_error = self._track_heading_error(state.psi, current_s, reference_minus_heading=False)
        opp_heading_error = self._track_heading_error(
            float(opponent.get("psi", state.psi)),
            float(opponent["px"]),
            reference_minus_heading=False,
        )
        lookahead_time = self.mpc.dt * self.mpc.horizon
        own_lookahead_s = current_s + max(float(state.vx), 0.0) * lookahead_time
        opp_lookahead_s = float(opponent["px"]) + max(float(opponent.get("vx", 0.0)), 0.0) * lookahead_time
        observation = np.asarray(
            [
                self._progress_delta(float(opponent["px"]), current_s), # opponent minus ego, as upstream
                current_e,                                              # self lateral error
                float(opponent.get("py", 0.0)),                         # self lateral position
                own_heading_error,                                      # self heading error
                float(state.vx),                                        # self longitudinal velocity
                float(state.vy),                                        # self lateral velocity
                float(state.omega),                                     # self angular velocity
                opp_heading_error,                                      # opponent heading error
                float(opponent.get("vx", 0.0)),                         # opponent longitudinal velocity
                float(opponent.get("vy", 0.0)),                         # opponent lateral velocity
                float(opponent.get("omega", 0.0)),                      # opponent angular velocity
                self._track_curvature_at(current_s),                    # self track curvature
                self._track_curvature_at(float(opponent["px"])),        # opponent track curvature
                self._track_curvature_at(own_lookahead_s),              # self lookahead track curvature
                self._track_curvature_at(opp_lookahead_s),              # opponent lookahead track curvature
            ],
            dtype=float,
        )
        observation[~np.isfinite(observation)] = 0.0
        return observation

    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        self.last_ref_traj = super().get_reference_trajectory(state, current_s, opponents=opponents)
        observation = self._build_original_observation(state, current_s, opponents or [])
        self.last_observation = observation.copy()
        raw_action = self._predict_model(observation)
        self.last_raw_action = np.asarray(raw_action, dtype=float).reshape(-1)[:2].copy()
        action = self._apply_action_transform(raw_action)
        self.last_u = np.array([action.throttle, action.steer])
        self.mpc.last_solver_success = True
        self.mpc.last_solver_status = f"direct_{self.model_kind}"
        self.mpc.last_loss = np.nan
        self.mpc.last_track_loss = np.nan
        self.mpc.last_action_loss = np.nan
        self.mpc.last_velocity_loss = np.nan
        self.mpc.last_heading_loss = np.nan
        self.mpc.last_control_plan = np.repeat(self.last_u.reshape(2, 1), self.mpc.horizon, axis=1)
        self.mpc.last_state_plan = None
        return action


class LLMParameterPolicy(ParameterizedAlphaRacerPolicy):
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
            delta_s = self._progress_delta(opponent["px"], current_s)
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
                "{\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2}. Do not include a mode key.\n"
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
                "JSON example: {\"q\":1.2,\"zeta\":1.0,\"s1\":0.2,\"s2\":0.8,\"s3\":0.2}"
            )
        
        # mode version
        history_text = self._history_text()
        return (
            "You choose one alpha-RACER MPC behavior mode. Return only valid JSON like "
            "{\"mode\":\"stable\"}. Do not invent modes.\n"
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
            "options": {"temperature": self.temperature, "num_predict": 32},
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
            delta_s = self._progress_delta(opponent["px"], current_s)
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
            # Prompt-only experiment: keep the guard implementation available, but do not
            # override the LLM mode here. Re-enable this line if prompt rules are not robust.
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


class LLMAlphaRacerPolicy(LLMParameterPolicy):
    """Backward-compatible name for LLMParameterPolicy."""


class RuleBasedSafeguardPolicy(LLMParameterPolicy):
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

        heading_error = abs(self._track_heading_error(state.psi, current_s))
        lookahead_s = current_s + max(float(state.vx), 0.0) * self.mpc.dt * self.mpc.horizon
        lookahead_curvature = abs(self._track_curvature_at(lookahead_s))
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
        return ParameterizedAlphaRacerPolicy.get_reference_trajectory(
            self, state, current_s, opponents=opponents
        )


class _LearnedAlphaPotentialFunction(nn.Module if nn is not None else object):
    """Neural network approximation of the alpha-potential function."""

    def __init__(self, input_size, hidden_size, output_size):
        if nn is None:
            raise ImportError("PyTorch is required for the learned alpha-potential policy.")
        super().__init__()
        layers = []
        self.batch_norm = nn.BatchNorm1d(input_size, affine=False)
        current = input_size
        for width in hidden_size:
            layers.append(nn.Linear(current, width))
            layers.append(nn.ReLU())
            current = width
        layers.append(nn.Linear(current, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.batch_norm(x)
        elif len(x.shape) == 3:
            flat = self.batch_norm(x.reshape((-1, x.shape[-1])))
            x = flat.reshape(x.shape)
        return self.fc(x)


class AlphaRacerPolicy(ParameterizedAlphaRacerPolicy):
    """
    alpha-RACER online policy that searches for parameters maximizing the
    learned alpha-potential function.

    Implementation of the alpha-potential based policy from the paper:
    https://arxiv.org/pdf/2412.08855v3

    The original model scores 5 policy parameters per car:
    sf1, sf2, lookahead_factor, speed_factor, blocking. This project uses
    q, zeta, s1, s2, s3 in MPC, so the selected parameters are mapped as:
    The upstream implementation stores q/2 as ``lookahead_factor`` and
    sqrt(zeta) as ``speed_factor``.  This project's MPC directly multiplies
    its tracking cost by q_weight, matching the paper's Q=qI definition, so
    the executable map is q=2*lookahead_factor, zeta=speed_factor^2,
    s1=sf1, s2=sf2, s3=blocking.
    """

    ORIGINAL_PARAM_BOUNDS = np.asarray(
        [
            [0.1, 0.5],   # s1
            [0.1, 0.5],   # s2
            [0.15, 0.5],  # lookahead_factor = q/2
            [0.85, 1.1],  # speed_factor = zeta^0.5
            [0.0, 1.0],   # s3
        ],
        dtype=float,
    )

    def __init__(
        self,
        *args,
        weight_path="alpha_racer_weights/p_models_rel/model_multi_myopic_mpc.pth",
        search_points=7,
        search_iters=6,
        opt_style="grad",
        gradient_rate=1e-4,
        opponent_default_original_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if torch is None:
            raise ImportError("PyTorch is required to use AlphaRacerPolicy.")

        self.weight_path = Path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"alpha-RACER weight file not found: {self.weight_path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.potential_function = _LearnedAlphaPotentialFunction(39, [384, 384, 192], 1).to(self.device)
        state_dict = torch.load(str(self.weight_path), map_location=self.device)
        self.potential_function.load_state_dict(state_dict)
        self.potential_function.eval()

        self.search_points = max(2, int(search_points))
        self.search_iters = max(1, int(search_iters))
        self.opt_style = str(opt_style).strip().lower()
        if self.opt_style not in {"grad", "linear"}:
            raise ValueError(f"Unknown AlphaRacer optimization style: {opt_style}")
        self.gradient_rate = float(gradient_rate)
        self.original_params = np.asarray([0.3, 0.2, 0.25, 0.95, 0.0], dtype=float)
        self.opponent_default_original_params = np.asarray(
            [0.3, 0.2, 0.25, 0.95, 0.0]
            if opponent_default_original_params is None
            else opponent_default_original_params,
            dtype=float,
        )
        self.current_mode = "potential"
        self.params = self._map_original_params(self.original_params)
        self.joint_original_params = np.repeat(
            self.opponent_default_original_params[None, :], 3, axis=0
        )
        if self.car_idx is not None and 0 <= self.car_idx < 3:
            self.joint_original_params[self.car_idx] = self.original_params
        self._warned_missing_opponents = False
        self._joint_plan_ready = False
    
    def _map_original_params(self, original_params):
        sf1, sf2, lookahead_factor, speed_factor, blocking = np.asarray(original_params, dtype=float)
        return np.asarray(
            [
                float(np.clip(2.0 * lookahead_factor, 1e-6, None)),
                float(np.clip(speed_factor * speed_factor, 0.1, 3.0)),
                float(np.clip(sf1, 0.0, self.mpc.track_half_width)),
                float(max(sf2, 1e-3)),
                float(np.clip(blocking, 0.0, 5.0)),
            ],
            dtype=float,
        )

    def _competitive_offsets(
        self,
        p_pert_x: float,
        p_pert_y: float,
        v_pert_x: float,
        ego_py_t: float,
        opponents,
        k: int,
    ) -> tuple:
        """Paper alpha-RACER overtaking and blocking shift equations."""
        sf1, sf2, _lookahead_factor, _speed_factor, blocking = self.original_params
        t = k * self.mpc.dt
        overtaking_shift = 0.0
        blocking_shift = 0.0
        for opponent in opponents:
            opponent_vx = float(opponent.get("vx", 0.0))
            opponent_py = float(opponent["py"])
            plan = opponent.get("plan")
            if plan is not None and len(plan):
                opponent_px = float(plan[min(k, len(plan) - 1)][0])
            else:
                opponent_px = float(opponent["px"]) + opponent_vx * t

            delta_x = self._progress_delta(p_pert_x, opponent_px)
            lateral_gap = abs(ego_py_t - opponent_py)
            magnitude = max(
                (sf1 - lateral_gap) * math.exp(-sf2 * delta_x * delta_x),
                0.0,
            )
            if ego_py_t > opponent_py:
                overtaking_shift += magnitude
            elif ego_py_t < opponent_py:
                overtaking_shift -= magnitude

            # The paper activates blocking for a faster trailing opponent.
            # Writing the speed difference as v_opp-v_pert keeps the gain
            # non-negative, as in the official executable implementation.
            is_trailing = float(opponent.get("delta", 0.0)) < 0.0
            speed_gap = opponent_vx - v_pert_x
            ego_still_ahead = delta_x >= 0.0
            if is_trailing and speed_gap > 0.0 and ego_still_ahead:
                block_gain = 1.0 - math.exp(-blocking * speed_gap)
                proximity = math.exp(-sf2 * delta_x * delta_x)
                blocking_shift += (
                    (opponent_py - p_pert_y) * block_gain * proximity
                )

        return overtaking_shift, blocking_shift

    def _wrap_delta(self, s_a, s_b):
        return self._progress_delta(float(s_a), float(s_b))

    def _car_feature_block(self, car):
        s = float(car["px"])
        state = car.get("state")
        psi = float(car.get("psi", getattr(state, "psi", 0.0)))
        vx = float(car.get("vx", getattr(state, "vx", 0.0)))
        vy = float(car.get("vy", getattr(state, "vy", 0.0)))
        omega = float(car.get("omega", getattr(state, "omega", 0.0)))
        return [
            float(car["py"]),
            self._track_heading_error(psi, s),
            vx,
            vy,
            omega,
        ]

    def _ordered_cars(self, state: CarState, current_s: float, opponents):
        cars = [
            {
                "px": current_s,
                "py": self.track.project(
                    state.x,
                    state.y,
                    s_hint=current_s,
                    search_radius=8.0,
                    psi_hint=state.psi,
                    max_heading_error=math.pi / 2.0,
                )[1],
                "vx": state.vx,
                "vy": state.vy,
                "omega": state.omega,
                "psi": state.psi,
                "state": state,
            }
        ]
        for opponent in opponents or []:
            cars.append(dict(opponent))

        if len(cars) < 3:
            if not self._warned_missing_opponents:
                self._log(f"[potential-function policy][{self.warning_label}] expected 3 cars; padding missing opponents.")
                self._warned_missing_opponents = True
            while len(cars) < 3:
                cars.append({"px": current_s, "py": 0.0, "vx": 0.0, "vy": 0.0, "omega": 0.0, "psi": state.psi})
        return cars[:3]

    def _build_joint_input(self, cars, joint_params):
        """Build the canonical 39-D input without reordering vehicle roles."""
        cars = list(cars[:3])
        # Pad missing cars with default values if fewer than 3 are provided.
        while len(cars) < 3:
            cars.append({"px": cars[0]["px"], "py": 0.0, "vx": 0.0, "vy": 0.0, "omega": 0.0, "psi": 0.0})
        s0, s1, s2 = [float(car["px"]) for car in cars]

        # longitudal distance deltas between the three cars, in order: (s2-s1), (s1-s0), (s2-s0)
        # 3 features
        features = [
            self._wrap_delta(s2, s1),
            self._wrap_delta(s1, s0),
            self._wrap_delta(s2, s0),
        ]

        # Frenet states features for each car: 3 * 5 = 15 features
        for car in cars:
            features.extend(self._car_feature_block(car))

        # track curvature features for each car: 3 features
        for car in cars:
            s = float(car["px"])
            features.append(self._track_curvature_at(s))
        
        # lookahead curvature features for each car: 3 features
        for car in cars:
            s = float(car["px"])
            features.append(
                self._track_curvature_at(s + max(float(car.get("vx", 0.0)), 0.0) * self.mpc.dt * self.mpc.horizon)
            )

        # policy parameters for each car: 3 * 5 = 15 features
        param_blocks = np.asarray(joint_params, dtype=float).reshape(3, 5)
        for block in param_blocks:
            features.extend(block.tolist())

        x = np.asarray(features, dtype=np.float32)
        x[~np.isfinite(x)] = 0.0
        if x.shape[0] != 39:
            raise ValueError(f"Potential-function policy input must have 39 values, got {x.shape[0]}")
        return x

    def _build_original_input(self, state: CarState, current_s: float, opponents, original_params):
        """Compatibility path for using one policy outside the joint simulator."""
        cars = self._ordered_cars(state, current_s, opponents)
        opponent_params = [
            np.asarray(op.get("original_params", self.opponent_default_original_params), dtype=float)
            for op in (opponents or [])[:2]
        ]
        while len(opponent_params) < 2:
            opponent_params.append(self.opponent_default_original_params)
        return self._build_joint_input(cars, [original_params, *opponent_params])

    def _score_joint_params(self, cars, joint_params_batch):
        # use the learned alpha-potential function to score a batch of joint parameters, returning a numpy array of scores
        rows = [self._build_joint_input(cars, row) for row in joint_params_batch]
        with torch.no_grad():
            x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=self.device)
            return self.potential_function(x).detach().cpu().numpy().reshape(-1)

    def _maximize_joint_params(self, cars, initial_joint):
        """Warm-started optimizer matching the paper's default online phase."""
        joint = np.asarray(initial_joint, dtype=np.float32).reshape(3, 5).copy()
        bounds = self.ORIGINAL_PARAM_BOUNDS.astype(np.float32)
        if self.opt_style == "linear":
            for _ in range(self.search_iters):
                for flat_idx in range(15):
                    car_idx, dim = divmod(flat_idx, 5)
                    candidates = np.repeat(joint[None, :, :], self.search_points, axis=0)
                    candidates[:, car_idx, dim] = np.linspace(
                        bounds[dim, 0], bounds[dim, 1], self.search_points
                    )
                    scores = self._score_joint_params(cars, candidates)
                    joint = candidates[int(np.argmax(scores))].copy()
            return joint
        # gradient-based optimization, using the learned alpha-potential function as a differentiable surrogate for the true objective
        base = self._build_joint_input(cars, joint)  # 39-D input for the current joint parameters
        x = torch.tensor(base[None, :], dtype=torch.float32, device=self.device)
        lower = torch.tensor(bounds[:, 0], dtype=torch.float32, device=self.device).repeat(3)  # lower bounds for the 15 policy parameters
        upper = torch.tensor(bounds[:, 1], dtype=torch.float32, device=self.device).repeat(3)  # repeat: 3 cars
        for _ in range(self.search_iters):
            x = x.detach().requires_grad_(True)
            score = self.potential_function(x).sum()
            gradient = torch.autograd.grad(score, x)[0]
            with torch.no_grad():
                policy_values = x[0, -15:] + self.gradient_rate * gradient[0, -15:]
                x[0, -15:] = torch.maximum(torch.minimum(policy_values, upper), lower)
        return x.detach().cpu().numpy()[0, -15:].reshape(3, 5)

    def _score_original_params(self, state, current_s, opponents, original_params_batch):
        # alpha-potential function to score given parameters, returning a numpy array of scores
        rows = [self._build_original_input(state, current_s, opponents, row) for row in original_params_batch]
        with torch.no_grad():  # for inference
            x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=self.device)
            scores = self.potential_function(x).detach().cpu().reshape(-1).tolist()
        return np.asarray(scores, dtype=float)

    def _select_original_params(self, state, current_s, opponents):
        # search for the original parameters that maximize the learned alpha-potential function
        best = self.original_params.copy()
        bounds = self.ORIGINAL_PARAM_BOUNDS
        for _ in range(self.search_iters):
            for dim in range(5):
                candidates = np.repeat(best[None, :], self.search_points, axis=0)
                candidates[:, dim] = np.linspace(bounds[dim, 0], bounds[dim, 1], self.search_points)
                scores = self._score_original_params(state, current_s, opponents, candidates)
                best = candidates[int(np.argmax(scores))].copy()
        return best

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        if self._joint_plan_ready:  # cache
            self._joint_plan_ready = False
        else:
            self.original_params = self._select_original_params(state, current_s, opponents or [])
        self.params = self._map_original_params(self.original_params)
        return super().get_reference_trajectory(state, current_s, opponents=opponents)


class PotentialFunctionPolicy(AlphaRacerPolicy):
    """Compatibility alias for the learned-potential AlphaRacer policy."""


class PotentialFunctionMinPolicy(AlphaRacerPolicy):
    """Compatibility alias for users who call this a potential-min policy."""


class WeightedAlphaRacerPolicy(AlphaRacerPolicy):
    """Deprecated compatibility alias for AlphaRacerPolicy."""


def plan_alpharacer_joint_params(policies, states, car_s, car_e):
    """Jointly optimize all three AlphaRacer parameter blocks.

    Vehicle features and parameter blocks remain in global car-index order,
    matching the role ordering used to train the 39-D potential model.
    """
    alpha_indices = [idx for idx, policy in enumerate(policies) if isinstance(policy, AlphaRacerPolicy)]
    if not alpha_indices:
        return

    anchor = policies[alpha_indices[0]]
    cars = [
        {
            "px": float(car_s[idx]),
            "py": float(car_e[idx]),
            "vx": float(states[idx].vx),
            "vy": float(states[idx].vy),
            "omega": float(states[idx].omega),
            "psi": float(states[idx].psi),
            "state": states[idx],
        }
        for idx in range(min(len(states), 3))
    ]
    if not cars:
        return

    joint = anchor.joint_original_params.copy()
    for idx in alpha_indices:
        if idx < 3:
            joint[idx] = policies[idx].original_params
    # The potential is a function of the complete joint policy, even when an
    # opponent executes a non-AlphaRacer baseline.  Those opponent blocks are
    # latent equilibrium estimates; only AlphaRacer-owned blocks are executed.
    joint = anchor._maximize_joint_params(cars, joint)
    anchor.joint_original_params = joint.copy()

    for idx in alpha_indices:
        if idx < 3:
            policies[idx].joint_original_params = joint.copy()
            policies[idx].original_params = joint[idx].copy()
            policies[idx].params = policies[idx]._map_original_params(joint[idx])
            policies[idx]._joint_plan_ready = True


class _StrictIBRSolver:
    """
    Frenet-space IBR MPC matching mpc_ibr function in multi_car_comp_blocking.py.
    For state x = [s, e, psi, v] and control u = [a, delta], the dynamics are:
        s_{k+1} = s_k + (v_k * cos(psi_k)) / (1 - e_k * curvature_k) * dt
        e_{k+1} = e_k + v_k * sin(psi_k) * dt
        psi_{k+1} = psi_k + ((v_k / L) * tan(Sa * delta_k) - v_k * curvature_k) * dt
        v_{k+1} = v_k + (Ta * a_k - fr)
    """

    def __init__(
        self,
        horizon=10,
        dt=0.1,             # dt_ibr
        iterations=3,       # self+opponent optimization iterations
        wheelbase=0.36,     # L
        steer_scale=0.36,   # steering mapping scale Sa
        throttle_gain=20.0 / 4.65,  # Throttle gain Ta
        friction=1.0,       # rolling friction coefficient fr
        mu=1.6,             # friction coefficient mu
        gravity=9.81,       # g
        use_upstream_collision_expression=True,
        lateral_half_width=None,
    ):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.wheelbase = float(wheelbase)
        self.steer_scale = float(steer_scale)
        self.throttle_gain = float(throttle_gain)
        self.friction = float(friction)
        self.mu = float(mu)
        self.gravity = float(gravity)
        self.use_upstream_collision_expression = bool(use_upstream_collision_expression)
        self.lateral_half_width = None if lateral_half_width is None else float(lateral_half_width)
        self._build_solver()    # build CasADi solver for the IBR MPC problem

    def _vehicle_dynamics(self, xk, uk, curvature):
        s = xk[0]
        e = xk[1]
        psi = xk[2]
        v = xk[3]
        a = uk[0]
        delta = uk[1]
        denom = 1.0 - e * curvature     # denominator for curvature correction
        return cs.vertcat(
            s + (v * cs.cos(psi)) / denom * self.dt,
            e + v * cs.sin(psi) * self.dt,
            psi + ((v / self.wheelbase) * cs.tan(self.steer_scale * delta) - v * curvature) * self.dt,
            v + (self.throttle_gain * a - self.friction) * self.dt,
        )

    def _build_solver(self):
        n = self.horizon
        x = cs.MX.sym("x", 4, n + 1)
        u = cs.MX.sym("u", 2, n)
        x0 = cs.MX.sym("x0", 4)
        curvature = cs.MX.sym("curvature", n)
        target_speed = cs.MX.sym("target_speed", 1)
        opp_s = cs.MX.sym("opp_s", n + 1)
        opp_e = cs.MX.sym("opp_e", n + 1)

        cost = 0
        g = [x[:, 0] - x0]  # constraint
        q = np.diag([0.0, 0.0, 0.3, 0.0])
        r = np.diag([1.0, 1.0])
        target_state = np.zeros(4)

        for k in range(n):
            g.append(x[:, k + 1] - self._vehicle_dynamics(x[:, k], u[:, k], curvature[k]))  # kinematic constraints
            state_error = x[:, k] - target_state
            cost += cs.mtimes([state_error.T, q, state_error])
            cost += cs.mtimes([u[:, k].T, r, u[:, k]])
            cost += 10.0 * x[1, k] ** 4

        cost += -10.0 * (x[0, n] - x[0, 0])  # encourage forward progress

        for k in range(n):
            g.append((x[3, k] ** 2) * cs.tan(u[1, k]))
        for k in range(n):
            g.append(x[3, k + 1])
        for k in range(n):
            if self.use_upstream_collision_expression:
                # The upstream source: s - oppo_e
                g.append((x[0, k] - opp_s[k + 1]) ** 2 + (x[0, k] - opp_e[k + 1]) ** 2)
            else:
                # e - oppo_e should be more reasonable
                g.append((x[0, k] - opp_s[k + 1]) ** 2 + (x[1, k] - opp_e[k + 1]) ** 2)
        if self.lateral_half_width is not None:
            for k in range(1, n + 1):
                g.append(x[1, k] - self.lateral_half_width)
                g.append(-x[1, k] - self.lateral_half_width)

        opt_variables = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
        constraints = cs.vertcat(*g)
        params = cs.vertcat(x0, curvature, target_speed, opp_s, opp_e)
        problem = {"f": cost, "x": opt_variables, "g": constraints, "p": params}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 500}
        self.solver = cs.nlpsol("strict_ibr_solver", "ipopt", problem, opts)
        self.n_x = 4 * (n + 1)
        self.n_u = 2 * n
        self.n_constraints = constraints.size1()

    def _solve_best_response(self, x0, opponent_plan, curvature, target_speed):
        n = self.horizon
        x_init = np.tile(np.asarray(x0, dtype=float), (n + 1, 1)).T
        u_init = np.zeros((2, n))
        initial_guess = np.concatenate([x_init.T.reshape(-1), u_init.T.reshape(-1)])
        p_val = np.concatenate(
            [
                np.asarray(x0, dtype=float),
                np.asarray(curvature, dtype=float),
                np.array([float(target_speed)]),
                np.asarray(opponent_plan[0], dtype=float),
                np.asarray(opponent_plan[1], dtype=float),
            ]
        )

        lbg = np.zeros(self.n_constraints)
        ubg = np.zeros(self.n_constraints)
        friction_start = 4 * (n + 1)
        speed_start = friction_start + n
        collision_start = speed_start + n
        lateral_start = collision_start + n
        lbg[friction_start:speed_start] = -self.mu * self.gravity * self.wheelbase
        ubg[friction_start:speed_start] = self.mu * self.gravity * self.wheelbase
        ubg[speed_start:collision_start] = 1.04 * float(target_speed)
        lbg[collision_start:lateral_start] = 0.04
        ubg[collision_start:lateral_start] = 500000.0
        if self.lateral_half_width is not None:
            lbg[lateral_start:] = -np.inf
            ubg[lateral_start:] = 0.0

        solution = self.solver(x0=initial_guess, lbg=lbg, ubg=ubg, p=p_val)
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(stats.get("return_status", "unknown"))
        opt = solution["x"].full().flatten()
        state_plan = opt[: self.n_x].reshape(n + 1, 4).T
        control_plan = opt[self.n_x :].reshape(n, 2).T
        return state_plan, control_plan

    def solve_pair(self, neighbor_x0, own_x0, neighbor_curvature, own_curvature, neighbor_speed, own_speed):
        n = self.horizon
        neighbor_plan = np.repeat(np.asarray(neighbor_x0[:2], dtype=float).reshape(2, 1), n + 1, axis=1)
        own_plan = np.repeat(np.asarray(own_x0[:2], dtype=float).reshape(2, 1), n + 1, axis=1)
        neighbor_u = np.zeros(2)
        own_u = np.zeros(2)
        neighbor_state_plan = None
        own_state_plan = None
        neighbor_control_plan = None
        own_control_plan = None

        for i in range(self.iterations):
            if i % 2 == 0:  # even iterations: solve neighbor
                neighbor_state_plan, neighbor_control_plan = self._solve_best_response(
                    neighbor_x0,
                    own_plan,
                    neighbor_curvature,
                    neighbor_speed,
                )
                neighbor_plan = neighbor_state_plan[:2, :]
                neighbor_u = neighbor_control_plan[:, 0]
            else:  # odd iterations: solve own with opponent known
                own_state_plan, own_control_plan = self._solve_best_response(
                    own_x0,
                    neighbor_plan,
                    own_curvature,
                    own_speed,
                )
                own_plan = own_state_plan[:2, :]
                own_u = own_control_plan[:, 0]

        if own_x0[2] > 0.5:
            own_u = np.array([0.5, -1.0])
        if own_x0[2] < -0.5:
            own_u = np.array([0.5, 1.0])

        return {
            "action": own_u,
            "state_plan": own_state_plan,
            "control_plan": own_control_plan,
            "frenet_plan": own_plan,
            "neighbor_state_plan": neighbor_state_plan,
            "neighbor_control_plan": neighbor_control_plan,
            "neighbor_action": neighbor_u,
        }


class IteratedBestResponsePolicy(AlphaRacerReferencePolicy):
    """
    IBR baseline policy in alpha-RACER paper (multi_car_comp_blocking.py).
    https://arxiv.org/abs/1801.02302 
    The upstream baseline chooses the nearest interaction partner, solves a
    two-car Frenet MPC with 3 alternating best responses, and applies the
    current car's first direct control command.
    """

    def __init__(
        self,
        *args,
        q=1.0,
        zeta=1.0,
        s1=0.0,
        s2=0.0,
        s3=0.0,
        iterations=3,  # N=3*2, total iterations
        gamma=None,
        ibr_horizon=None,
        use_upstream_collision_expression=True,
        lateral_half_width=None,
        **kwargs,
    ):
        super().__init__(*args, params=np.array([q, zeta, s1, s2, s3]), **kwargs)
        solver_horizon = self.mpc.horizon if ibr_horizon is None else int(ibr_horizon)
        if lateral_half_width is None:
            lateral_half_width = getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        self.iterations = int(iterations)
        self.ibr_solver = _StrictIBRSolver(
            horizon=solver_horizon,
            dt=self.mpc.dt,
            iterations=self.iterations,
            wheelbase=self.mpc.p.LF + self.mpc.p.LR,
            steer_scale=self.mpc.p.Sa,
            throttle_gain=self.mpc.p.Ta,
            friction=self.mpc.p.fr,
            use_upstream_collision_expression=use_upstream_collision_expression,
            lateral_half_width=float(lateral_half_width),
        )
        self._ibr_cached_action = None
        self._ibr_cached_action_time = None
        self._ibr_action_valid_until = -np.inf
        self._ibr_cached_ref_traj = None
        self._ibr_cached_ref_world_horizon = None
        self._ibr_cached_raceline_world_horizon = None
        self._ibr_failure_active = False
        self._ibr_failure_count = 0
        self._ibr_failure_start_time = None
        self._ibr_last_failure = None

    def _report_ibr_failure(self, exc, sim_time=None):
        self._ibr_failure_count += 1
        self._ibr_last_failure = str(exc)
        if not self._ibr_failure_active:
            self._ibr_failure_active = True
            self._ibr_failure_start_time = sim_time
            self._log(
                f"[IBR warning][{self.warning_label}] solve failed "
                f"({self._format_ibr_time(sim_time)}); using last control. status={exc}"
            )

    def _report_ibr_recovery(self, sim_time=None):
        if not self._ibr_failure_active:
            return
        self._log(
            f"[IBR warning][{self.warning_label}] solve recovered "
            f"({self._format_ibr_interval(sim_time)}, failed_count={self._ibr_failure_count})"
        )
        self._ibr_failure_active = False
        self._ibr_failure_count = 0
        self._ibr_failure_start_time = None
        self._ibr_last_failure = None

    def _format_ibr_time(self, sim_time):
        return "t=?" if sim_time is None else f"t={sim_time:.2f}s"

    def _format_ibr_interval(self, sim_time):
        if self._ibr_failure_start_time is None or sim_time is None:
            return self._format_ibr_time(sim_time)
        return f"t={self._ibr_failure_start_time:.2f}-{sim_time:.2f}s"

    def flush_solver_warnings(self, sim_time=None):
        if self._ibr_failure_active:
            self._log(
                f"[IBR warning][{self.warning_label}] solve still failing "
                f"({self._format_ibr_interval(sim_time)}, failed_count={self._ibr_failure_count}); "
                f"using last control. status={self._ibr_last_failure}"
            )

    def _curvature_lookahead(self, current_s, state):
        curvature = np.zeros(self.ibr_solver.horizon)
        speed = max(float(state.vx), 0.0)
        for k in range(self.ibr_solver.horizon):
            s = current_s + k * self.mpc.dt * speed
            curvature[k] = self._track_curvature_at(s, ds=0.5)
        return curvature

    def _reference_speed(self, state, current_s):
        if self.raceline is None:
            return max(float(state.vx), 0.5)
        try:
            idx = self.raceline.nearest_index(state.x, state.y)
            self.last_raceline_idx = idx
            self.last_raceline_s = float(self.raceline.s[idx])
            return min(float(self.raceline.speed[idx]), self.max_speed)
        except Exception:
            return max(float(state.vx), 0.5)

    def _frenet_x0(self, state, current_s, current_e):
        return np.array(
            [
                float(current_s),
                float(current_e),
                float(self._track_heading_error(state.psi, current_s, reference_minus_heading=False)),
                max(float(state.vx), 0.0),
            ],
            dtype=float,
        )

    def _make_ref_from_plan(self, plan, speed):
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        plan_len = plan.shape[1]
        for k in range(horizon + 1):
            src = min(k, plan_len - 1)
            s = float(plan[0, src])
            e = float(plan[1, src])
            x, y, psi = self.track.interpolate(s)
            ref[0, k] = s % self.track.total_length
            ref[1, k] = e
            ref[2, k] = speed
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = -math.sin(psi)
            ref[7, k] = math.cos(psi)
            ref_world[:, k] = self.track.frenet_to_world(s, e)
        return ref, ref_world

    def plan_against_neighbor(self, own_state, own_s, own_e, neighbor_state, neighbor_s, neighbor_e, sim_time=None):
        own_x0 = self._frenet_x0(own_state, own_s, own_e)
        neighbor_x0 = self._frenet_x0(neighbor_state, neighbor_s, neighbor_e)
        own_speed = self._reference_speed(own_state, own_s)
        neighbor_speed = self._reference_speed(neighbor_state, neighbor_s)
        result = self.ibr_solver.solve_pair(
            neighbor_x0=neighbor_x0,
            own_x0=own_x0,
            neighbor_curvature=self._curvature_lookahead(neighbor_s, neighbor_state),
            own_curvature=self._curvature_lookahead(own_s, own_state),
            neighbor_speed=neighbor_speed,
            own_speed=own_speed,
        )
        self._accept_ibr_result(result, own_speed, sim_time)
        self._report_ibr_recovery(sim_time=sim_time)

    def _accept_ibr_result(self, result, ref_speed, sim_time):
        action = np.asarray(result["action"], dtype=float)
        action = np.clip(action, -1.0, 1.0)
        self._ibr_cached_action = CarAction(throttle=float(action[0]), steer=float(action[1]))
        self._ibr_cached_action_time = sim_time
        self._ibr_action_valid_until = (sim_time if sim_time is not None else 0.0) + self.mpc.dt
        self.last_u = np.array([self._ibr_cached_action.throttle, self._ibr_cached_action.steer])
        self.mpc.last_solver_success = True
        self.mpc.last_solver_status = "Solve_Succeeded"
        self.mpc.last_control_plan = result["control_plan"]

        if result["state_plan"] is not None:
            plan = result["state_plan"][:2, :]
            ref_traj, ref_world = self._make_ref_from_plan(plan, ref_speed)
            self.last_ref_traj = ref_traj
            self.last_ref_world_horizon = ref_world
            self.last_raceline_world_horizon = np.full_like(ref_world, np.nan)
            self._ibr_cached_ref_traj = ref_traj
            self._ibr_cached_ref_world_horizon = ref_world
            self._ibr_cached_raceline_world_horizon = self.last_raceline_world_horizon
            world_plan = np.zeros((4, plan.shape[1]))
            for k in range(plan.shape[1]):
                x, y = self.track.frenet_to_world(plan[0, k], plan[1, k])
                _, _, psi_center = self.track.interpolate(plan[0, k])
                world_plan[:, k] = [x, y, psi_center + result["state_plan"][2, k], result["state_plan"][3, k]]
            self.mpc.last_state_plan = world_plan

    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        # use cached IBR action if it is still valid for the current simulation time
        if (
            self._ibr_cached_action is not None
            and sim_time is not None
            and sim_time < self._ibr_action_valid_until - 1e-9
        ):
            self.last_ref_traj = self._ibr_cached_ref_traj if self._ibr_cached_ref_traj is not None else self.last_ref_traj
            self.last_ref_world_horizon = (
                self._ibr_cached_ref_world_horizon
                if self._ibr_cached_ref_world_horizon is not None
                else self.last_ref_world_horizon
            )
            self.last_raceline_world_horizon = (
                self._ibr_cached_raceline_world_horizon
                if self._ibr_cached_raceline_world_horizon is not None
                else self.last_raceline_world_horizon
            )
            return self._ibr_cached_action
        return CarAction(throttle=float(self.last_u[0]), steer=float(self.last_u[1]))


class IBRBaselinePolicy(IteratedBestResponsePolicy):
    """Backward-compatible name for IteratedBestResponsePolicy."""


def _nearest_interaction_neighbor(policy, car_idx, car_s, states):
    best_idx = None
    best_abs_delta = np.inf
    for other_idx, _ in enumerate(states):
        if other_idx == car_idx:
            continue
        delta = policy._progress_delta(car_s[other_idx], car_s[car_idx])
        abs_delta = abs(delta)
        if abs_delta < best_abs_delta:
            best_idx = other_idx
            best_abs_delta = abs_delta
    return best_idx


def plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=None):
    # main loop to plan IBR actions for all policies that are IteratedBestResponsePolicy
    for idx, policy in enumerate(policies):
        if not isinstance(policy, IteratedBestResponsePolicy):
            continue
        if (
            policy._ibr_cached_action is not None
            and sim_time is not None
            and sim_time < policy._ibr_action_valid_until - 1e-9
        ):
            continue

        neighbor_idx = _nearest_interaction_neighbor(policy, idx, car_s, states)
        if neighbor_idx is None:
            continue
        try:
            policy.plan_against_neighbor(
                own_state=states[idx],
                own_s=car_s[idx],
                own_e=car_e[idx],
                neighbor_state=states[neighbor_idx],
                neighbor_s=car_s[neighbor_idx],
                neighbor_e=car_e[neighbor_idx],
                sim_time=sim_time,
            )
        except Exception as exc:
            policy._report_ibr_failure(exc, sim_time=sim_time)
