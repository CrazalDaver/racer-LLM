import math

import casadi as cs
import numpy as np

from state import CarAction, CarState
from track import RaceLine

from .base import BasePolicy
from .utils import progress_delta, track_curvature_at, track_heading_error

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


class IteratedBestResponsePolicy(BasePolicy):
    """
    IBR baseline policy in alpha-RACER paper (multi_car_comp_blocking.py).
    https://arxiv.org/abs/1801.02302 
    The upstream baseline chooses the nearest interaction partner, solves a
    two-car Frenet MPC with 3 alternating best responses, and applies the
    current car's first direct control command.
    """

    def __init__(
        self,
        track,
        mpc,
        raceline: RaceLine,
        max_speed=5.0,
        car_idx=None,
        policy_name=None,
        iterations=3,  # N=3*2, total iterations
        ibr_horizon=None,
        use_upstream_collision_expression=True,
        lateral_half_width=None,
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        if raceline is None:
            raise ValueError("IteratedBestResponsePolicy requires a raceline")
        self.raceline = raceline
        self.max_speed = float(max_speed)
        self.last_raceline_s = None
        self.last_raceline_idx = None
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
            curvature[k] = track_curvature_at(self.track, s, ds=0.5)
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
                float(track_heading_error(self.track, state.psi, current_s, reference_minus_heading=False)),
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


def _nearest_interaction_neighbor(policy, car_idx, car_s, states):
    best_idx = None
    best_abs_delta = np.inf
    for other_idx, _ in enumerate(states):
        if other_idx == car_idx:
            continue
        delta = progress_delta(car_s[other_idx], car_s[car_idx], policy.track.total_length)
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
