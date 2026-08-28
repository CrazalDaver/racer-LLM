import math

import numpy as np

from state import CarAction, CarState
from track import RaceLine, Track

from .base import BasePolicy
from .utils import progress_delta


class MPCPolicy(BasePolicy):
    """
    Convert a reference trajectory into a per-vehicle MPC action.

    By default the trajectory is generated from the paper's five parameters
    theta=(q, zeta, s1, s2, s3). A subclass may instead override
    get_reference_trajectory() and supply a trajectory directly.
    """

    def __init__(
        self,
        track: Track,
        mpc,
        params=None,
        raceline: RaceLine = None,
        max_speed=5.0,
        ref_lateral_clip=None,
        car_idx=None,
        policy_name=None,
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        self.last_u = np.array([0.0, self._neutral_steer_command()])
        # Keep fallback actions separate from the last control produced by a
        # successful solve, so recovery cannot contaminate the next MPC step.
        self.last_feasible_u = self.last_u.copy()
        self.params = np.asarray(
            [1.0, 1.0, 0.0, 0.0, 0.0] if params is None else params,
            dtype=float,
        )
        self.raceline = raceline
        self.max_speed = float(max_speed)
        if ref_lateral_clip is None:
            ref_lateral_clip = getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        self.ref_lateral_clip = float(max(ref_lateral_clip, 0.0))
        self.last_raceline_s = None
        self.last_raceline_idx = None

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

    @staticmethod
    def _recovery_throttle(vx: float) -> float:
        """Simple, interpretable three-level fallback speed control."""
        vx = max(float(vx), 0.0)
        if vx > 2.0:
            return -0.3  # Brake when failure occurs at racing speed.
        if vx > 0.5:
            return 0.0   # Coast down toward a controllable crawl speed.
        return 0.3       # Keep enough drive to avoid a zero-speed deadlock.

    # Conservative failure recovery: preserve the last feasible direction and
    # brake. Only near a boundary do we slew gently toward the centerline.
    def _recovery_control(self, state: CarState, current_s: float) -> np.ndarray:
        _, current_e = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        half_width = max(getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width), 1e-6)
        steer = float(self.last_u[1])
        if abs(current_e) > 0.95 * half_width:
            _, _, centerline_psi = self.track.interpolate(current_s)
            heading_error = math.atan2(
                math.sin(state.psi - centerline_psi),
                math.cos(state.psi - centerline_psi),
            )
            center_steer = np.clip(
                -0.25 * current_e / half_width - 0.20 * heading_error,
                -0.25,
                0.25,
            )
            steer = np.clip(center_steer, steer - 0.05, steer + 0.05)
        throttle = self._recovery_throttle(state.vx)
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
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        try:
            _, u_first = self.mpc.solve(
                state_arr,
                ref_traj,
                self.last_feasible_u,
                q_weight=self.get_q_weight(),
                opponents=opponents,
                current_s=current_s,
            )
        except Exception as e:
            self._report_mpc_failure(e, state, sim_time=sim_time)
            u_first = self._recovery_control(state, current_s)
            # Keep last_solver_status as the original IPOPT return status.
            # Recovery is recorded separately so diagnostics retain the cause.
            self.mpc.last_recovery_active = True
            self.mpc.last_recovery_reason = "solver_failure"
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
                self.mpc.last_violation_loss = np.nan
                self.mpc.last_track_slack_loss = np.nan
                self.mpc.last_collision_slack_loss = np.nan
        else:
            self._report_mpc_recovery(sim_time=sim_time)
            # Update this state only after a genuinely successful solve.
            self.last_feasible_u = np.clip(np.asarray(u_first, dtype=float), -1.0, 1.0)

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

    def _raceline_frenet_state(self, raceline_s: float, centerline_s_hint: float = None) -> dict:
        # Given an s value along the raceline, compute the corresponding Frenet state relative to the track centerline.
        point = self.raceline.interpolate(raceline_s)  # Get the raceline point in world coordinates.
        center_s, lateral_e = self.track.project(
            point["x_m"],
            point["y_m"],
            s_hint=centerline_s_hint,
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

    def _raceline_frenet_point(self, raceline_s: float, centerline_s_hint: float = None) -> dict:
        point = self.raceline.interpolate(raceline_s)  # Get the raceline point in world coordinates.
        center_s, lateral_e = self.track.project(
            point["x_m"],
            point["y_m"],
            s_hint=centerline_s_hint,
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
            delta = progress_delta(opponent["px"], current_s, self.track.total_length) # Positive delta means ahead
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
        """Return the paper's front/rear overtaking and blocking offsets."""
        _, _, s1, s2, s3 = self.params
        p_ot_front = 0.0
        p_ot_rear = 0.0
        p_bl = 0.0

        for opponent in opponents:
            opp_px, opp_py, opp_vx = self._predict_opponent(opponent, k)
            dx = progress_delta(p_pert_x, opp_px, self.track.total_length)
            dy_current = ego_py_t - opp_py
            ot_mag = max((s1 - abs(dy_current)) * math.exp(-s2 * dx * dx), 0.0)
            p_ot = self._sign(dy_current) * ot_mag
            if opponent["delta"] > 0.0:
                p_ot_front += p_ot
            elif opponent["delta"] < 0.0:
                p_ot_rear += p_ot

            behind_now = opponent["delta"] < 0.0
            opponent_faster = opp_vx > v_pert_x
            ego_ahead_in_horizon = dx >= 0.0
            if behind_now and opponent_faster and ego_ahead_in_horizon:
                speed_factor = 1.0 - math.exp(-s3 * max(opp_vx - v_pert_x, 0.0))
                proximity = math.exp(-s2 * dx * dx)
                p_bl += (opp_py - p_pert_y) * speed_factor * proximity

        return p_ot_front, p_ot_rear, p_bl

    def _predict_opponent(self, opponent, k: int) -> tuple:
        """Predict one opponent at horizon step ``k``.

        The paper assumes constant longitudinal velocity and lateral position.
        Explicit plans, when supplied by another planner, take precedence.
        """
        opponent_vx = float(opponent.get("vx", 0.0))
        plan = opponent.get("plan")
        plan_idx = None
        if plan is not None and len(plan):
            row = plan[min(k, len(plan) - 1)]
            opponent_px = float(row[0])
        elif "px_plan" in opponent and len(opponent["px_plan"]):
            plan_idx = min(k, len(opponent["px_plan"]) - 1)
            opponent_px = float(opponent["px_plan"][plan_idx])
        else:
            opponent_px = float(opponent["px"]) + k * self.mpc.dt * opponent_vx

        opponent_py = float(opponent["py"])
        opponent_vx_at_k = opponent_vx
        if plan is None and plan_idx is not None:
            if "py_plan" in opponent and len(opponent["py_plan"]):
                opponent_py = float(opponent["py_plan"][min(plan_idx, len(opponent["py_plan"]) - 1)])
            if "vx_plan" in opponent and len(opponent["vx_plan"]):
                opponent_vx_at_k = float(opponent["vx_plan"][min(plan_idx, len(opponent["vx_plan"]) - 1)])
        return opponent_px, opponent_py, opponent_vx_at_k

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        """
        Compute the reference trajectory: p^ref = p^* + p^ot + p^bl,
        where p^* is the raceline reference,
        p^ot is the overtaking offset, and p^bl is the blocking offset.
        The reference trajectory is returned in Frenet coordinates relative to the track centerline.
        """
        if self.raceline is None:
            raise ValueError(
                "A raceline is required for five-parameter reference generation; "
                "override get_reference_trajectory() to supply a trajectory directly."
            )
        horizon = self.mpc.horizon
        q, zeta, s1, s2, s3 = self.params
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        raceline_world = np.full((2, horizon + 1), np.nan)
        # TEMPORARY DIAGNOSTIC: preserve the scalar terms before clipping.
        ref_components = {
            name: np.full(horizon + 1, np.nan, dtype=float)
            for name in ("raceline_e", "p_ot_front", "p_ot_rear", "p_bl", "p_ref_e")
        }
        
        # Select the nearest leading and trailing opponents (j^*, j_*)
        selected_opponents = self._closest_opponents(current_s, opponents)
        raceline_s = self._nearest_raceline_s(state)  # project to raceline
        self.last_raceline_s = raceline_s  # cache for visualization
        # project to centerline and get py/e for competitive offsets
        _, ego_py_t = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        source_raceline_s = raceline_s + self._raceline_lookahead_distance(state)
        centerline_s_hint = current_s
        for k in range(horizon + 1):
            frenet = self._raceline_frenet_point(
                source_raceline_s,
                centerline_s_hint=centerline_s_hint,
            )
            centerline_s_hint = frenet["track_s"]
            raceline_speed_k = max(frenet["speed"], 0.0)
            p_pert_x = frenet["track_s"]
            p_pert_y = frenet["lateral_e"]
            p_ot_front, p_ot_rear, p_bl = self._competitive_offsets(
                p_pert_x,
                p_pert_y,
                zeta * raceline_speed_k,
                ego_py_t,
                selected_opponents,
                k,
            )
            p_ref_y = float(np.clip(
                p_pert_y + p_ot_front + p_ot_rear + p_bl,
                -self._ref_lateral_limit(),
                self._ref_lateral_limit(),
            ))
            ref_components["raceline_e"][k] = p_pert_y
            ref_components["p_ot_front"][k] = p_ot_front
            ref_components["p_ot_rear"][k] = p_ot_rear
            ref_components["p_bl"][k] = p_bl
            ref_components["p_ref_e"][k] = p_ref_y
            xc, yc = frenet["center_x"], frenet["center_y"]
            psi_center = frenet["psi_center"]
            ref_world[:, k] = [
                xc + p_ref_y * frenet["normal_x"],
                yc + p_ref_y * frenet["normal_y"],
            ]
            raceline_world[:, k] = [frenet["world_x"], frenet["world_y"]]
            ref[0, k] = p_pert_x
            ref[1, k] = p_ref_y
            # zeta controls how quickly the reference horizon advances.  Once
            # velocity tracking is enabled, the same effective progression
            # speed must be used as the speed target; using the raw raceline
            # speed here would silently remove zeta from this part of the MPC.
            ref[2, k] = float(np.clip(
                zeta * raceline_speed_k,
                0.0,
                self.mpc.max_speed,
            ))
            ref[3, k] = psi_center
            ref[4, k] = xc
            ref[5, k] = yc
            ref[6, k] = -math.sin(psi_center)
            ref[7, k] = math.cos(psi_center)

            source_raceline_s += zeta * raceline_speed_k * self.mpc.dt
        self.last_ref_world_horizon = ref_world
        self.last_raceline_world_horizon = raceline_world
        self.last_ref_components = ref_components
        return ref



class CenterLinePolicy(MPCPolicy):
    # Simple baseline policy that tracks the centerline at a constant reference speed.
    def __init__(self, track: Track, mpc, step_distance: float = 0.1, car_idx=None, policy_name=None):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        self.step_distance = float(step_distance)

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
class StaticAlphaRacerPolicy(MPCPolicy):
    """
    Fixed-parameter controller from the alpha-RACER policy class.

    The five parameters q, zeta, s1, s2, s3 are held constant during
    simulation, so this is the static-policy baseline.
    """
