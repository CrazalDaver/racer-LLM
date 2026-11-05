"""
Base class for all MPC-based driving policies, 
featuring the five-parameter (q, zeta, s1, s2, s3) as higher-level controls.

applies to 
    - original alpha-racer policy
    - LLM-based policy
    - etc.

Derived classes must implement get_reference_trajectory() or get_action()
"""

import math

import numpy as np

from mpc_controller import MPCController
from state import CarAction, CarState
from track import RaceLine, Track

from .base import BasePolicy
from .utils import progress_delta


class MPCPolicy(BasePolicy):
    """
    Using MPC to track a reference trajectory.
    Typically the reference trajectory is a race line 
    with competitive offsets for overtaking and blocking
    that follows the methods of alpha-RACER paper.

    Attributes:
        - track: Track: The track on which this policy is operating.
        - mpc: MPC: The MPC instance used by this policy.

    Methods:
        - get_action(observation) -> CarAction: 
            Return the control action selected by this policy, 
            computed by solving the MPC problem based on the reference trajectory.

        - get_reference_trajectory(state, current_s, opponents=None) -> np.ndarray: 
            Compute the reference trajectory for the given state and opponents.
    """

    def __init__(
        self,
        track: Track,
        mpc: MPCController,  # MPC instance
        params=None,  # (q, zeta, s1, s2, s3)
        raceline: RaceLine = None,
        max_speed=5.0,
        ref_lateral_clip=None,
        car_idx=None,
        policy_name=None,
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        self.last_u = np.array([0.0, self._neutral_steer_command()])
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

        # MPC solver failure diagnostics
        self._warnings = _MPCWarnings(self)

    def _neutral_steer_command(self) -> float:
        return np.clip(-self.mpc.p.Sb / self.mpc.p.Sa, -1.0, 1.0)

    def get_q_weight(self) -> float:
        return float(self.params[0])

    def _sign(self, value: float) -> float:
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0

    def flush_solver_warnings(self, sim_time=None):
        self._warnings.flush(sim_time)

    # Action selection: solve, clip, and record.

    # Main MPC control interface:
    # call get_reference_trajectory to get the reference trajectory,
    # then call mpc.solve to get the control action
    def get_action(self, observation) -> CarAction:
        """Read the scene, build a reference, solve, and record the requested action."""
        # take unified observation and extract ego and opponents
        if self.car_idx is None:
            raise ValueError("MPCPolicy requires car_idx when used with RaceObservation")
        ego = observation.car(self.car_idx)  # Unified full racing states
        opponents = observation.mpc_opponents(self.car_idx)
        sim_time = observation.sim_time

        # Get the reference trajectory for the current state and opponents.
        # Keep this virtual call: subclasses may update their five parameters here.
        ref_traj = self.get_reference_trajectory(
            ego.state, ego.s, opponents=opponents, current_e=ego.e)
        u_first = self._solve_mpc(ego, ref_traj, opponents, sim_time)
        action = self._clip_action(u_first)  # clip to valid range
        self._record_action(ego.state, ref_traj, action, sim_time)
        return action

    def _solve_mpc(self, ego, ref_traj, opponents, sim_time):
        """Return the first MPC command; preserve input caches and failure diagnostics.

        Construct the MPC problem from the current ego state, the reference trajectory, and the opponents. Solve the MPC problem and return the first control action.
        """
        self.last_ref_traj = ref_traj  # Cache for visualization and diagnostics
        state_arr = ego.state.to_array()
        # Solve the MPC problem to get the first control action
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        try:
            _, u_first = self.mpc.solve(
                state_arr,
                ref_traj,
                self.last_u,
                q_weight=self.get_q_weight(),
                opponents=opponents,
                current_s=ego.s,
                frenet_state=ego.kinematic_state,
            )
        except Exception as e:
            self._record_solver_failure(e, ego.state, sim_time)
            raise RuntimeError(f"MPC failed for car {self.car_idx}: {e}") from e

        return u_first

    def _clip_action(self, u_first) -> CarAction:
        """Clip the solver command to a bounded requested action."""
        # Clipping to valid range
        target_vel = np.clip(u_first[0], -1.0 if self.mpc.p.ros_kinematic else 0.0, 1.0)
        target_steer = np.clip(u_first[1], -1.0, 1.0)
        return CarAction(throttle=target_vel, steer=target_steer)

    def _record_action(self, state, ref_traj, action, sim_time):
        """Report stalls and cache this request; the environment later feeds back execution."""
        ref_speed = float(ref_traj[2, 0]) if ref_traj is not None else np.nan  # Reference speed at the first step
        self._warnings.update_stall(state, ref_speed, action.throttle, action.steer, sim_time)
        self.last_requested_action = np.array([action.throttle, action.steer])
        self.last_u = self.last_requested_action.copy()

    def _record_solver_failure(self, exc, state, sim_time):
        """Report the failure, retain its metadata, and discard stale prediction plans."""
        self._warnings.failure(exc, state, sim_time)
        self.mpc.last_solver_success = False
        diagnostics = getattr(self.mpc, "last_diagnostics", {})
        if diagnostics.get("success"):
            diagnostics = dict(status=type(exc).__name__, solve_seconds=0.,
                               residual=None, iterations=0, cost=None)
        self.mpc.last_diagnostics = {**diagnostics, "success": False, "error": str(exc)}
        for name in ("last_state_plan", "last_control_plan",
                     "last_track_slack_plan", "last_collision_slack_plan"):
            setattr(self.mpc, name, None)

    # Reference trajectory: sample, project, offset, and record.

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None, *, current_e=None) -> np.ndarray:
        """
        Compute the reference trajectory: p^ref = p^* + p^ot + p^bl,
        where p^* is the raceline reference,
        p^ot is the overtaking offset, and p^bl is the blocking offset.
        The reference trajectory is returned in Frenet coordinates relative to the track centerline.
        Paper-style x/y components here mean centerline s/e, not world XY.
        p_pert is the point before overtaking/blocking offsets; p_ref is the
        final reference. Their names are retained to keep the distinction clear.
        """
        if self.raceline is None:
            raise ValueError(
                "A raceline is required for five-parameter reference generation; "
                "override get_reference_trajectory() to supply a trajectory directly."
            )
        horizon = self.mpc.horizon
        q, zeta, s1, s2, s3 = self.params
        # Select the nearest leading and trailing opponents (j^*, j_*).
        selected_opponents = self._closest_opponents(current_s, opponents)
        raceline_s, ego_py_t = self._reference_origin(state, current_s, current_e)
        ref, ref_world, raceline_world, ref_components = self._build_reference_horizon(
            state, current_s, raceline_s, ego_py_t, selected_opponents, horizon, zeta)
        self._record_reference(ref_world, raceline_world, ref_components)
        return ref

    def _reference_origin(self, state, current_s, current_e):
        """Return starting raceline s and measured ego e; cache the selected raceline origin.

        Reuse observed e. Only legacy/offline callers without it project the ego.
        _nearest_raceline_s also retains the selected sample index for visualization.
        """
        raceline_s = self._nearest_raceline_s(state)  # project to raceline
        self.last_raceline_s = raceline_s  # cache for visualization
        if current_e is None:
            # Compatibility for direct offline reference calls without observation.
            _, current_e = self.track.project(
                state.x, state.y, s_hint=current_s, search_radius=8.0,
                psi_hint=state.psi, max_heading_error=math.pi/2.)
        ego_py_t = float(current_e)  # Measured ego centerline e (m).
        return raceline_s, ego_py_t

    def _nearest_raceline_s(self, state: CarState) -> float:
        self.last_raceline_idx = self.raceline.nearest_index(
            state.x,
            state.y,
        )
        return float(self.raceline.s[self.last_raceline_idx])

    # Compute the coordinate of the raceline point
    # need to notice that s respect to the raceline and centerline are different
    def _raceline_frenet_point(self, raceline_s: float, projection_s_hint: float = None) -> dict:
        """Sample a raceline point and project it into centerline Frenet s/e.

        projection_s_hint is a nearby centerline progress (m), used to select
        the projection branch and unwrap progress. It is a hint, not the result;
        the existing global fallback can lose its lap-count information.
        """
        raceline_point = self.raceline.interpolate(raceline_s)  # World XY, heading and speed.
        # Project the raceline point to the track centerline 
        # to get its Frenet coordinates (s, e).
        centerline_s, lateral_e = self.track.project(
            raceline_point["x_m"],
            raceline_point["y_m"],
            s_hint=projection_s_hint,
            search_radius=8.0,
            psi_hint=raceline_point["psi_rad"],
            max_heading_error=math.pi / 2.0,
        )
        center_x, center_y, psi_center, normal_x, normal_y = self.track.local_frame(centerline_s)
        return {
            "track_s": centerline_s,  # Centerline progress (m), not raceline arc length.
            "lateral_e": lateral_e,  # Signed centerline offset (m), positive to the left.
            "raceline_speed_mps": min(raceline_point["vx_mps"], self.max_speed),
            "speed": min(raceline_point["vx_mps"], self.max_speed),  # legacy reference API alias
            "center_x": center_x,
            "center_y": center_y,
            "psi_center": psi_center,  # Centerline heading in world coordinates (rad).
            "normal_x": normal_x,
            "normal_y": normal_y,
            "world_x": raceline_point["x_m"],
            "world_y": raceline_point["y_m"],
        }

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

    def _predict_opponent(self, opponent, k: int) -> tuple:
        """Predict one opponent at horizon step ``k``.

        The paper assumes that longitudinal velocity and lateral position remain constant.
        """
        opponent_vx = float(opponent.get("s_dot", opponent.get("vx", 0.0)))  # Frenet s_dot; vx is a legacy key.
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

        opponent_py = float(opponent["py"])  # Centerline e (m), held constant unless a plan is supplied.
        opponent_vx_at_k = opponent_vx
        if plan is None and plan_idx is not None:
            if "py_plan" in opponent and len(opponent["py_plan"]):
                opponent_py = float(opponent["py_plan"][min(plan_idx, len(opponent["py_plan"]) - 1)])
            if "vx_plan" in opponent and len(opponent["vx_plan"]):
                opponent_vx_at_k = float(opponent["vx_plan"][min(plan_idx, len(opponent["vx_plan"]) - 1)])
        return opponent_px, opponent_py, opponent_vx_at_k

    def _competitive_offsets(
        self,
        p_pert_x: float,  # Perturbed point: centerline s (m).
        p_pert_y: float,  # Perturbed point: centerline e (m).
        v_pert_x: float,  # Reference centerline progress rate at step k (m/s).
        ego_py_t: float,  # Measured ego centerline e at the current tick (m).
        opponents,
        k: int,
    ) -> tuple:
        """Compute lateral overtaking and blocking offsets at horizon step k.
        Overtaking:
            p_ot = sign(dy) * max((s1 - abs(dy)) * exp(-s2 * dx**2), 0)

        Blocking:
            p_bl += (opponent_py(k) - p_pert_y)
                    * (1 - exp(-s3 * max(dv, 0))) * exp(-s2 * dx**2)
            dv = measured opponent s_dot - reference s_dot at step k.
            Both selected opponents are eligible once behind the future point.
        """
        _, _, s1, s2, s3 = self.params
        p_ot_front = 0.0
        p_ot_rear = 0.0
        p_bl = 0.0

        for opponent in opponents:
            opp_px, opp_py, opp_vx = self._predict_opponent(opponent, k)  # Predicted s, e, s_dot.
            dx = progress_delta(p_pert_x, opp_px, self.track.total_length)  # Periodic longitudinal gap (m).
            dy_current = ego_py_t - opp_py  # Lateral gap (m); uses measured ego e.
            ot_mag = max((s1 - abs(dy_current)) * math.exp(-s2 * dx * dx), 0.0)
            p_ot = self._sign(dy_current) * ot_mag
            if opponent["delta"] > 0.0:
                p_ot_front += p_ot
            elif opponent["delta"] < 0.0:
                p_ot_rear += p_ot

            # Paper compares the reference rate with the opponent's measured
            # longitudinal rate, held constant over the planning window.
            opponent_vx = float(opponent.get("s_dot", opponent.get("vx", 0.0)))
            opponent_faster = opponent_vx > v_pert_x
            ego_ahead_in_horizon = dx >= 0.0
            if opponent_faster and ego_ahead_in_horizon:
                speed_factor = 1.0 - math.exp(-s3 * max(opponent_vx - v_pert_x, 0.0))
                proximity = math.exp(-s2 * dx * dx)
                p_bl += (opp_py - p_pert_y) * speed_factor * proximity

        return p_ot_front, p_ot_rear, p_bl

    def _build_reference_horizon(
        self, state, current_s, raceline_s, ego_py_t, selected_opponents, horizon, zeta,
    ):
        """Return reference, world paths and component arrays without writing caches.

        For each horizon step: sample/project the perturbed point, compute its
        competitive offsets, assemble the final reference, then advance raceline s.
        Paper symbols and the order of these operations are preserved below.
        """
        ref = np.zeros((8, horizon + 1))  # Rows: s, e, speed, heading, center XY, left normal XY.
        ref_world = np.zeros((2, horizon + 1))  # Final reference world XY (m).
        raceline_world = np.full((2, horizon + 1), np.nan)  # Unshifted raceline world XY (m).
        # TEMPORARY DIAGNOSTIC: preserve the scalar terms before clipping.
        ref_components = {
            name: np.full(horizon + 1, np.nan, dtype=float)
            for name in ("raceline_e", "p_ot_front", "p_ot_rear", "p_bl", "p_ref_e")
        }
        
        raceline_sample_s = raceline_s
        projection_s_hint = current_s  # Start from measured centerline progress.
        frenet = self._raceline_frenet_point(
            raceline_sample_s, projection_s_hint=projection_s_hint)
        for k in range(horizon + 1):
            raceline_speed_k = max(frenet["raceline_speed_mps"], 0.0)
            next_raceline_s = raceline_sample_s + zeta * raceline_speed_k * self.mpc.dt
            next_frenet = self._raceline_frenet_point(
                next_raceline_s, projection_s_hint=frenet["track_s"])
            # The path advances along the raceline, then projects to centerline s.
            # Compare its actual s progression with the opponent's measured s_dot.
            v_pert_x = (next_frenet["track_s"] - frenet["track_s"]) / self.mpc.dt
            p_pert_x = frenet["track_s"]  # Perturbed point: centerline s, before competitive offsets.
            p_pert_y = frenet["lateral_e"]  # Perturbed point: centerline e, before competitive offsets.
            p_ot_front, p_ot_rear, p_bl = self._competitive_offsets(
                p_pert_x,
                p_pert_y,
                v_pert_x,
                ego_py_t,
                selected_opponents,
                k,
            )
            p_ref_y = float(np.clip(  # Final reference e after overtaking/blocking and clipping.
                p_pert_y + p_ot_front + p_ot_rear + p_bl,
                -self.ref_lateral_clip,
                self.ref_lateral_clip,
            ))
            ref_components["raceline_e"][k] = p_pert_y
            ref_components["p_ot_front"][k] = p_ot_front
            ref_components["p_ot_rear"][k] = p_ot_rear
            ref_components["p_bl"][k] = p_bl
            ref_components["p_ref_e"][k] = p_ref_y
            xc, yc = frenet["center_x"], frenet["center_y"]  # Centerline world XY (m).
            psi_center = frenet["psi_center"]  # Centerline heading (rad), not vehicle heading.
            ref_world[:, k] = [
                xc + p_ref_y * frenet["normal_x"],
                yc + p_ref_y * frenet["normal_y"],
            ]
            raceline_world[:, k] = [frenet["world_x"], frenet["world_y"]]
            ref[0, k] = p_pert_x
            ref[1, k] = p_ref_y
            # Retain the speed target along the raceline. Blocking instead uses
            # the centerline progress rate above; velocity tracking defaults to zero.
            ref[2, k] = float(np.clip(
                zeta * raceline_speed_k,
                0.0,
                self.mpc.max_speed,
            ))
            ref[3, k] = psi_center
            ref[4, k] = xc
            ref[5, k] = yc
            ref[6, k] = frenet["normal_x"]
            ref[7, k] = frenet["normal_y"]

            raceline_sample_s = next_raceline_s
            frenet = next_frenet
        return ref, ref_world, raceline_world, ref_components

    def _record_reference(self, ref_world, raceline_world, ref_components):
        """Store completed reference paths and components for diagnostics/visualization."""
        self.last_ref_world_horizon = ref_world
        self.last_raceline_world_horizon = raceline_world
        self.last_ref_components = ref_components



class CenterLinePolicy(MPCPolicy):
    # Simple test policy that tracks the centerline at a constant reference speed.
    def __init__(self, track: Track, mpc, step_distance: float = 0.1, car_idx=None, policy_name=None):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        self.step_distance = float(step_distance)
    # override: here we simply use the centerline as the reference trajectory
    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None, *, current_e=None) -> np.ndarray:
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        for k in range(horizon + 1):
            s = current_s + k * self.step_distance
            x, y, psi, normal_x, normal_y = self.track.local_frame(s)
            ref[0, k] = s % self.track.total_length
            ref[1, k] = 0.0
            ref[2, k] = state.vx # Track current speed as reference speed to encourage smooth acceleration/deceleration.
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = normal_x
            ref[7, k] = normal_y
            ref_world[:, k] = [x, y]
        self.last_ref_world_horizon = ref_world
        return ref

class StaticAlphaRacerPolicy(MPCPolicy):
    """
    Fixed-parameter controller from the alpha-RACER policy class.

    The five parameters q, zeta, s1, s2, s3 are held constant during
    simulation, so this is the static-policy baseline.
    """


class _MPCWarnings:
    """Console reports only; solver diagnostics remain on the MPC instance.

    A failed solve stops the race and is reported immediately. A feasible
    low-speed stall is reported on entry, every three seconds, and on exit.
    """

    def __init__(self, policy):
        self.policy = policy
        self.stall_count = 0
        self.last_report_time = None
        self.pending_stall = False

    def failure(self, exc, state, sim_time):
        time = "t=?" if sim_time is None else f"t={sim_time:.2f}s"
        self.policy._log(
            f"[MPC warning][{self.policy.warning_label}] solve failed "
            f"({time}, x={state.x:.2f}, y={state.y:.2f}, vx={state.vx:.2f}); "
            f"race stopped. status={exc}"
        )

    def update_stall(self, state, ref_speed, throttle, steer, sim_time):
        mpc = self.policy.mpc
        stalled = (mpc.last_solver_success and np.isfinite(mpc.last_loss)
                   and state.vx < 0.05 and ref_speed > 0.5)
        time = "t=?" if sim_time is None else f"t={sim_time:.2f}s"
        if not stalled:
            if self.stall_count:
                self.policy._log(
                    f"[MPC warning][{self.policy.warning_label}] stalled condition recovered "
                    f"({time}, total_count={self.stall_count})"
                )
            self.stall_count = 0
            self.last_report_time = None
            self.pending_stall = False
            return

        self.stall_count += 1
        self.pending_stall = True
        if sim_time is None:
            report_due = self.stall_count % 25 == 0
        else:
            report_due = (self.last_report_time is None
                          or sim_time - self.last_report_time >= 3.0)
        if self.stall_count == 1 or report_due:
            self.policy._log(
                f"[MPC warning][{self.policy.warning_label}] feasible but stalled "
                f"({time}, total_count={self.stall_count}, vx={state.vx:.3f}, "
                f"ref_v={ref_speed:.3f}, cmd={throttle:.3f}, steer={steer:.3f}, "
                f"loss={mpc.last_loss:.3g}, status={mpc.last_solver_status})"
            )
            self.last_report_time = sim_time
            self.pending_stall = False

    def flush(self, sim_time):
        if not self.pending_stall:
            return
        time = "t=?" if sim_time is None else f"t={sim_time:.2f}s"
        self.policy._log(
            f"[MPC warning][{self.policy.warning_label}] feasible but stalled "
            f"({time}, total_count={self.stall_count}); still stalled. "
            f"loss={self.policy.mpc.last_loss:.3g}, "
            f"status={self.policy.mpc.last_solver_status}"
        )
        self.pending_stall = False
        self.last_report_time = sim_time
