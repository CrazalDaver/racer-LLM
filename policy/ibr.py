import math

import casadi as cs
import numpy as np

from config import GRAVITY, params as sim_params
from state import CarAction, CarState
from track import RaceLine
from vehicle_model import kinematic_frenet_step, minimum_reachable_speed, neutral_throttle

from .alpha_model import ALPHA_PARAMETER_BOUNDS
from .base import BasePolicy
from .mpc import MPCPolicy
from .utils import progress_delta, track_curvature_at, track_heading_error


IBR_IMPLEMENTATIONS = ("local_wang", "historical_strict")


class _IBRPolicyMixin:
    """Shared marker and scheduling helpers for the two IBR policy levels."""

    ibr_level = None

    def _ibr_update_due(self, sim_time, last_update_time, interval):
        if sim_time is None or last_update_time is None:
            return True
        return sim_time >= last_update_time + interval - 1.0e-9


class _StrictIBRSolver:
    """Restored locally adapted IBR used in repository history.

    This matches the `_StrictIBRSolver` present from commit 5ab4f74 through
    the July policy refactors. It retains the upstream objective and optional
    upstream collision typo, but uses local vehicle parameters, enforces local
    lateral track bounds, solves neighbor first, and applies the historical
    heading-error correction to the executed own action.
    """

    def __init__(
        self, *, vehicle_params, horizon=10, dt=0.1, iterations=3,
        mu=1.6, gravity=9.81, use_upstream_collision_expression=True,
        lateral_half_width=None,
    ):
        self.p = vehicle_params
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.mu = float(mu)
        self.gravity = float(gravity)
        self.use_upstream_collision_expression = bool(
            use_upstream_collision_expression
        )
        self.lateral_half_width = (
            None if lateral_half_width is None else float(lateral_half_width)
        )
        self._build_solver()

    def _vehicle_dynamics(self, xk, uk, curvature):
        return kinematic_frenet_step(
            xk,
            uk,
            curvature,
            self.dt,
            self.p,
            curvature_mode="upstream_approx",
            symbolic=True,
        )

    def _build_solver(self):
        n = self.horizon
        x = cs.MX.sym("strict_x", 4, n + 1)
        u = cs.MX.sym("strict_u", 2, n)
        x0 = cs.MX.sym("strict_x0", 4)
        curvature = cs.MX.sym("strict_curvature", n)
        target_speed = cs.MX.sym("strict_target_speed", 1)
        opp_s = cs.MX.sym("strict_opp_s", n + 1)
        opp_e = cs.MX.sym("strict_opp_e", n + 1)

        cost = 0
        constraints = [x[:, 0] - x0]
        q = np.diag([0.0, 0.0, 0.3, 0.0])
        r = np.diag([1.0, 1.0])
        for k in range(n):
            constraints.append(
                x[:, k + 1] - self._vehicle_dynamics(x[:, k], u[:, k], curvature[k])
            )
            cost += cs.mtimes([x[:, k].T, q, x[:, k]])
            cost += cs.mtimes([u[:, k].T, r, u[:, k]])
            cost += 10.0 * x[1, k] ** 4
        cost += -10.0 * (x[0, n] - x[0, 0])

        for k in range(n):
            constraints.append(x[3, k] ** 2 * cs.tan(u[1, k]))
        for k in range(n):
            constraints.append(x[3, k + 1])
        for k in range(n):
            if self.use_upstream_collision_expression:
                constraints.append(
                    (x[0, k] - opp_s[k + 1]) ** 2
                    + (x[0, k] - opp_e[k + 1]) ** 2
                )
            else:
                constraints.append(
                    (x[0, k] - opp_s[k + 1]) ** 2
                    + (x[1, k] - opp_e[k + 1]) ** 2
                )
        if self.lateral_half_width is not None:
            for k in range(1, n + 1):
                constraints.append(x[1, k] - self.lateral_half_width)
                constraints.append(-x[1, k] - self.lateral_half_width)

        opt_variables = cs.vertcat(cs.reshape(x, -1, 1), cs.reshape(u, -1, 1))
        constraint_vector = cs.vertcat(*constraints)
        problem = {
            "f": cost,
            "x": opt_variables,
            "g": constraint_vector,
            "p": cs.vertcat(x0, curvature, target_speed, opp_s, opp_e),
        }
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 500}
        self.solver = cs.nlpsol("historical_strict_ibr_solver", "ipopt", problem, opts)
        self.n_x = 4 * (n + 1)
        self.n_u = 2 * n
        self.n_constraints = int(constraint_vector.size1())

    def _solve_best_response(self, x0, opponent_plan, curvature, target_speed):
        n = self.horizon
        x_init = np.tile(np.asarray(x0, dtype=float), (n + 1, 1)).T
        u_init = np.zeros((2, n))
        initial_guess = np.concatenate([x_init.T.reshape(-1), u_init.T.reshape(-1)])
        parameters = np.concatenate(
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
        wheelbase = self.p.LF + self.p.LR
        lbg[friction_start:speed_start] = -self.mu * self.gravity * wheelbase
        ubg[friction_start:speed_start] = self.mu * self.gravity * wheelbase
        ubg[speed_start:collision_start] = 1.04 * float(target_speed)
        lbg[collision_start:lateral_start] = 0.04
        ubg[collision_start:lateral_start] = 500000.0
        if self.lateral_half_width is not None:
            lbg[lateral_start:] = -np.inf
            ubg[lateral_start:] = 0.0

        solution = self.solver(x0=initial_guess, lbg=lbg, ubg=ubg, p=parameters)
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(stats.get("return_status", "unknown"))
        optimum = solution["x"].full().flatten()
        state_plan = optimum[: self.n_x].reshape(n + 1, 4).T
        control_plan = optimum[self.n_x :].reshape(n, 2).T
        return state_plan, control_plan

    def solve_pair(
        self, neighbor_x0, own_x0, neighbor_curvature, own_curvature,
        neighbor_speed, own_speed, neighbor_previous_control=None,
        own_previous_control=None,
    ):
        del neighbor_previous_control, own_previous_control
        n = self.horizon
        neighbor_plan = np.repeat(
            np.asarray(neighbor_x0[:2]).reshape(2, 1), n + 1, axis=1
        )
        own_plan = np.repeat(np.asarray(own_x0[:2]).reshape(2, 1), n + 1, axis=1)
        neighbor_action = own_action = np.zeros(2)
        neighbor_state_plan = own_state_plan = None
        neighbor_control_plan = own_control_plan = None

        # Historical ordering: neighbor on even iterations, own on odd.
        for iteration in range(self.iterations):
            if iteration % 2 == 0:
                neighbor_state_plan, neighbor_control_plan = self._solve_best_response(
                    neighbor_x0, own_plan, neighbor_curvature, neighbor_speed
                )
                neighbor_plan = neighbor_state_plan[:2]
                neighbor_action = neighbor_control_plan[:, 0]
            else:
                own_state_plan, own_control_plan = self._solve_best_response(
                    own_x0, neighbor_plan, own_curvature, own_speed
                )
                own_plan = own_state_plan[:2]
                own_action = own_control_plan[:, 0]

        # Historical local adaptation: correct the action actually executed.
        if own_x0[2] > 0.5:
            own_action = np.array([0.5, -1.0])
        elif own_x0[2] < -0.5:
            own_action = np.array([0.5, 1.0])

        if own_state_plan is None:
            own_state_plan = np.repeat(np.asarray(own_x0).reshape(4, 1), n + 1, axis=1)
            own_control_plan = np.zeros((2, n))
        return {
            "action": own_action,
            "state_plan": own_state_plan,
            "control_plan": own_control_plan,
            "frenet_plan": own_plan,
            "neighbor_state_plan": neighbor_state_plan,
            "neighbor_control_plan": neighbor_control_plan,
            "neighbor_action": neighbor_action,
            "collision_multipliers": np.zeros(n),
        }


class _WangIBRSolver:
    """Two-player receding-horizon IBR based on Wang et al. (RSS 2019).

    The optional SE term is the paper's first-order approximation

        alpha * mu_j * (d gamma_j / d theta_i) * theta_i,

    where ``mu_j`` is the opponent collision-constraint multiplier from its
    latest best response.  Setting ``sensitivity_weight=0`` gives ordinary
    IBR with exactly the same dynamics, constraints, and iteration order.

    EXPERIMENTAL: the local soft collision constraint makes its multiplier
    depend directly on the slack penalty (lambda ~= 2*rho*slack when active).
    Therefore positive sensitivity weights are not used in the primary paper
    experiments unless that coupling is explicitly resolved and validated.
    """

    def __init__(
        self,
        *,
        horizon,
        dt,
        iterations,
        vehicle_params,
        mu,
        gravity,
        lateral_half_width,
        collision_longitudinal,
        collision_lateral,
        collision_exponent,
        sensitivity_weight,
        track_slack_weight,
        collision_slack_weight,
        allow_track_slack,
        allow_collision_slack,
        progress_weight,
        lateral_weight,
        lateral_exponent,
        heading_weight,
        speed_weight,
        control_weight,
        control_reference,
        control_rate_weight,
        convergence_tolerance,
        max_steer_delta,
        max_speed,
        speed_cap_mode,
        raceline_speed_cap_factor,
        enforce_track_boundary,
        enforce_collision_avoidance,
        enforce_friction_speed_limit,
    ):
        self.p = vehicle_params
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.mu = float(mu)
        self.gravity = float(gravity)
        self.lateral_half_width = float(lateral_half_width)
        self.collision_longitudinal = float(collision_longitudinal)
        self.collision_lateral = float(collision_lateral)
        self.collision_exponent = int(collision_exponent)
        self.sensitivity_weight = max(float(sensitivity_weight), 0.0)
        self.track_slack_weight = float(track_slack_weight)
        self.collision_slack_weight = float(collision_slack_weight)
        self.allow_track_slack = bool(allow_track_slack)
        self.allow_collision_slack = bool(allow_collision_slack)
        self.progress_weight = float(progress_weight)
        self.lateral_weight = float(lateral_weight)
        self.lateral_exponent = int(lateral_exponent)
        if self.lateral_exponent not in {2, 4}:
            raise ValueError("IBR lateral_exponent must be 2 or 4")
        self.heading_weight = float(heading_weight)
        self.speed_weight = float(speed_weight)
        self.control_weight = float(control_weight)
        self.control_reference = str(control_reference)
        if self.control_reference not in {"neutral", "zero"}:
            raise ValueError("IBR control_reference must be 'neutral' or 'zero'")
        self.control_rate_weight = float(control_rate_weight)
        self.convergence_tolerance = max(float(convergence_tolerance), 0.0)
        self.max_steer_delta = max(float(max_steer_delta), 0.0)
        self.max_speed = float(max_speed)
        self.speed_cap_mode = str(speed_cap_mode)
        self.raceline_speed_cap_factor = float(raceline_speed_cap_factor)
        if self.speed_cap_mode not in {"global", "raceline"}:
            raise ValueError("IBR speed_cap_mode must be 'global' or 'raceline'")
        self.enforce_track_boundary = bool(enforce_track_boundary)
        self.enforce_collision_avoidance = bool(enforce_collision_avoidance)
        self.enforce_friction_speed_limit = bool(enforce_friction_speed_limit)
        self._build_solver()    # build CasADi solver for the IBR MPC problem

    def _vehicle_dynamics(self, xk, uk, curvature):
        return kinematic_frenet_step(
            xk,
            uk,
            curvature,
            self.dt,
            self.p,
            curvature_mode="geometric",
            symbolic=True,
        )

    def _build_solver(self):
        n = self.horizon
        x = cs.MX.sym("x", 4, n + 1)
        u = cs.MX.sym("u", 2, n)
        x0 = cs.MX.sym("x0", 4)
        curvature = cs.MX.sym("curvature", n)
        target_speed = cs.MX.sym("target_speed", 1)
        neutral_control = cs.MX.sym("neutral_control", 2)
        opp_s = cs.MX.sym("opp_s", n + 1)
        opp_e = cs.MX.sym("opp_e", n + 1)
        se_gradient = cs.MX.sym("se_gradient", 2, n + 1)
        se_anchor = cs.MX.sym("se_anchor", 2, n + 1)
        u_prev = cs.MX.sym("u_prev", 2)
        track_slack = cs.MX.sym("track_slack", n)
        collision_slack = cs.MX.sym("collision_slack", n)

        cost = 0
        g = [x[:, 0] - x0]  # constraint
        identity = np.eye(2)

        for k in range(n):
            g.append(x[:, k + 1] - self._vehicle_dynamics(x[:, k], u[:, k], curvature[k]))  # kinematic constraints
            cost += self.lateral_weight * x[1, k] ** self.lateral_exponent
            cost += self.heading_weight * x[2, k] ** 2
            cost += self.speed_weight * (x[3, k] - target_speed) ** 2
            # Absolute effort is measured relative to the command that holds
            # the reference speed and drives straight. This avoids treating
            # drag compensation as strategically meaningful acceleration.
            effort = (
                u[:, k] - neutral_control
                if self.control_reference == "neutral"
                else u[:, k]
            )
            cost += self.control_weight * cs.mtimes([effort.T, identity, effort])
            du = u[:, k] - (u_prev if k == 0 else u[:, k - 1])
            cost += self.control_rate_weight * cs.mtimes([du.T, identity, du])

        cost += -self.progress_weight * (x[0, n] - x[0, 0])
        cost += self.track_slack_weight * cs.dot(track_slack, track_slack)
        cost += self.collision_slack_weight * cs.dot(collision_slack, collision_slack)
        # Equation (19), expressed as a minimization and centered at the
        # previous ego trajectory for better numerical conditioning.
        cost -= self.sensitivity_weight * cs.dot(
            cs.reshape(se_gradient, -1, 1),
            cs.reshape(x[:2, :] - se_anchor, -1, 1),
        )

        for k in range(n):
            # Keep the same released-alpha-RACER friction proxy used by the
            # local MPCController: normalized steer appears inside tan().
            g.append((x[3, k] ** 2) * cs.tan(u[1, k]))
        for k in range(n):
            g.append(x[3, k + 1])
        for k in range(1, n + 1):
            g.append(x[1, k] - self.lateral_half_width - track_slack[k - 1])
            g.append(-x[1, k] - self.lateral_half_width - track_slack[k - 1])
        for k in range(n):
            ds = (x[0, k + 1] - opp_s[k + 1]) / self.collision_longitudinal
            de = (x[1, k + 1] - opp_e[k + 1]) / self.collision_lateral
            # gamma <= 0 is the coupled constraint used by the SE derivative.
            g.append(
                1.0 - ds ** self.collision_exponent
                - de ** self.collision_exponent - collision_slack[k]
            )
        for k in range(n):
            previous_steer = u_prev[1] if k == 0 else u[1, k - 1]
            g.append(u[1, k] - previous_steer)

        opt_variables = cs.vertcat(
            cs.reshape(x, -1, 1), cs.reshape(u, -1, 1),
            track_slack, collision_slack,
        )
        constraints = cs.vertcat(*g)
        params = cs.vertcat(
            x0, curvature, target_speed, neutral_control, opp_s, opp_e,
            cs.reshape(se_gradient, -1, 1), cs.reshape(se_anchor, -1, 1),
            u_prev,
        )
        problem = {"f": cost, "x": opt_variables, "g": constraints, "p": params}
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 500}
        self.solver = cs.nlpsol("local_wang_ibr_solver", "ipopt", problem, opts)
        self.n_x = 4 * (n + 1)
        self.n_u = 2 * n
        self.n_track_slack = n
        self.n_collision_slack = n
        self.n_constraints = constraints.size1()

    def _collision_sensitivity(self, own_plan, opponent_plan, opponent_multipliers):
        """Return mu_j * d(gamma_j)/d(theta_i) at the previous plans."""
        gradient = np.zeros((2, self.horizon + 1), dtype=float)
        for k in range(1, self.horizon + 1):
            ds = float(own_plan[0, k] - opponent_plan[0, k])
            de = float(own_plan[1, k] - opponent_plan[1, k])
            mu = max(float(opponent_multipliers[k - 1]), 0.0)
            p = self.collision_exponent
            gradient[0, k] = mu * (
                -p * (ds / self.collision_longitudinal) ** (p - 1)
                / self.collision_longitudinal
            )
            gradient[1, k] = mu * (
                -p * (de / self.collision_lateral) ** (p - 1)
                / self.collision_lateral
            )
        return gradient

    def _neutral_throttle(self, speed):
        """Throttle that approximately maintains speed in MPC dynamics."""
        return neutral_throttle(speed, self.p)

    def _solve_best_response(
        self, x0, opponent_plan, curvature, target_speed,
        initial_plan=None, opponent_multipliers=None, previous_control=None,
    ):
        n = self.horizon
        if initial_plan is None:
            x_init = np.tile(np.asarray(x0, dtype=float), (n + 1, 1)).T
        else:
            x_init = np.asarray(initial_plan, dtype=float).copy()
            x_init[:, 0] = np.asarray(x0, dtype=float)
        u_init = np.zeros((2, n))
        u_init[0, :] = np.clip(self._neutral_throttle(x_init[3, 0]), -1.0, 1.0)
        u_init[1, :] = np.clip(-self.p.Sb / max(self.p.Sa, 1e-6), -1.0, 1.0)
        initial_track_slack = np.maximum(
            np.abs(x_init[1, 1:]) - self.lateral_half_width, 0.0
        ) + 1.0e-4
        ds_init = (x_init[0, 1:] - opponent_plan[0, 1:]) / self.collision_longitudinal
        de_init = (x_init[1, 1:] - opponent_plan[1, 1:]) / self.collision_lateral
        initial_collision_slack = np.maximum(
            1.0 - ds_init ** self.collision_exponent
            - de_init ** self.collision_exponent, 0.0
        ) + 1.0e-4
        initial_guess = np.concatenate([
            x_init.T.reshape(-1), u_init.T.reshape(-1),
            initial_track_slack, initial_collision_slack,
        ])
        opponent_multipliers = (
            np.zeros(n) if opponent_multipliers is None
            else np.asarray(opponent_multipliers, dtype=float)
        )
        if previous_control is None:
            previous_control = np.asarray([
                self._neutral_throttle(x0[3]),
                -self.p.Sb / max(self.p.Sa, 1e-6),
            ])
        previous_control = np.asarray(previous_control, dtype=float)
        neutral_control = np.asarray(
            [self._neutral_throttle(target_speed), -self.p.Sb / max(self.p.Sa, 1e-6)],
            dtype=float,
        )
        se_gradient = self._collision_sensitivity(
            x_init, opponent_plan, opponent_multipliers
        )
        p_val = np.concatenate(
            [
                np.asarray(x0, dtype=float),
                np.asarray(curvature, dtype=float),
                np.array([float(target_speed)]),
                neutral_control,
                np.asarray(opponent_plan[0], dtype=float),
                np.asarray(opponent_plan[1], dtype=float),
                se_gradient.T.reshape(-1),
                x_init[:2, :].T.reshape(-1),
                previous_control,
            ]
        )

        lbg = np.zeros(self.n_constraints)
        ubg = np.zeros(self.n_constraints)
        friction_start = 4 * (n + 1)
        speed_start = friction_start + n
        boundary_start = speed_start + n
        collision_start = boundary_start + 2 * n
        steer_rate_start = collision_start + n
        if self.enforce_friction_speed_limit:
            wheelbase = self.p.LF + self.p.LR
            lbg[friction_start:speed_start] = -self.mu * self.gravity * wheelbase
            ubg[friction_start:speed_start] = self.mu * self.gravity * wheelbase
        else:
            lbg[friction_start:speed_start] = -np.inf
            ubg[friction_start:speed_start] = np.inf
        speed_cap = self.max_speed
        if self.speed_cap_mode == "raceline":
            # A raw v <= factor*v_ref bound can be infeasible when the measured
            # car reaches a slower segment faster than it can brake in one
            # discrete step.  Keep the desired cap, but never demand a next
            # state below what full braking can physically reach.  Recomputing
            # this bound each MPC update makes the cap tighten progressively.
            speed_cap = min(
                self.max_speed,
                max(
                    self.raceline_speed_cap_factor * float(target_speed),
                    minimum_reachable_speed(x0[3], self.dt, self.p),
                ),
            )
        ubg[speed_start:boundary_start] = speed_cap
        lbg[boundary_start:] = -np.inf
        ubg[boundary_start:] = 0.0
        if not self.enforce_track_boundary:
            ubg[boundary_start:collision_start] = np.inf
        if not self.enforce_collision_avoidance:
            ubg[collision_start:steer_rate_start] = np.inf
        lbg[steer_rate_start:] = -self.max_steer_delta
        ubg[steer_rate_start:] = self.max_steer_delta

        lbx = np.full(initial_guess.size, -np.inf)
        ubx = np.full(initial_guess.size, np.inf)
        u_start = self.n_x
        for k in range(n):
            lbx[u_start + 2 * k:u_start + 2 * k + 2] = [-1.0, -1.0]
            ubx[u_start + 2 * k:u_start + 2 * k + 2] = [1.0, 1.0]
        slack_start = self.n_x + self.n_u
        lbx[slack_start:] = 0.0
        track_slack_end = slack_start + self.n_track_slack
        if not self.allow_track_slack:
            ubx[slack_start:track_slack_end] = 0.0
            initial_guess[slack_start:track_slack_end] = 0.0
        if not self.allow_collision_slack:
            ubx[track_slack_end:] = 0.0
            initial_guess[track_slack_end:] = 0.0

        solution = self.solver(
            x0=initial_guess, lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg, p=p_val,
        )
        stats = self.solver.stats()
        if not stats.get("success", False):
            raise RuntimeError(stats.get("return_status", "unknown"))
        opt = solution["x"].full().flatten()
        state_plan = opt[: self.n_x].reshape(n + 1, 4).T
        control_plan = opt[self.n_x : self.n_x + self.n_u].reshape(n, 2).T
        collision_slack_start = self.n_x + self.n_u + self.n_track_slack
        track_slack_plan = opt[
            self.n_x + self.n_u : self.n_x + self.n_u + self.n_track_slack
        ]
        collision_slack_plan = opt[
            collision_slack_start : collision_slack_start + self.n_collision_slack
        ]
        # CasADi uses L = f + lam_g*g for our gamma <= 0 constraint, so the
        # non-negative collision multipliers are directly usable in Eq. (19).
        lam_g = solution["lam_g"].full().flatten()
        collision_multipliers = np.maximum(lam_g[collision_start:collision_start + n], 0.0)
        ds = (state_plan[0, 1:] - opponent_plan[0, 1:]) / self.collision_longitudinal
        de = (state_plan[1, 1:] - opponent_plan[1, 1:]) / self.collision_lateral
        raw_collision_constraint = (
            1.0 - ds ** self.collision_exponent - de ** self.collision_exponent
        )
        diagnostics = {
            "track_slack": track_slack_plan,
            "collision_slack": collision_slack_plan,
            "collision_multipliers": collision_multipliers,
            "max_collision_constraint": float(np.max(raw_collision_constraint)),
            "min_predicted_longitudinal_distance": float(
                np.min(np.abs(state_plan[0, 1:] - opponent_plan[0, 1:]))
            ),
            "min_predicted_lateral_distance": float(
                np.min(np.abs(state_plan[1, 1:] - opponent_plan[1, 1:]))
            ),
            "sensitivity_gradient_norm": float(np.linalg.norm(se_gradient)),
            "sensitivity_loss": float(
                -self.sensitivity_weight
                * np.sum(se_gradient * (state_plan[:2, :] - x_init[:2, :]))
            ),
        }
        return state_plan, control_plan, collision_multipliers, diagnostics

    def solve_pair(
        self, neighbor_x0, own_x0, neighbor_curvature, own_curvature,
        neighbor_speed, own_speed, neighbor_previous_control=None,
        own_previous_control=None,
    ):
        n = self.horizon
        def rollout(x0):
            plan = np.repeat(np.asarray(x0, dtype=float).reshape(4, 1), n + 1, axis=1)
            for k in range(n):
                plan[0, k + 1] = plan[0, k] + max(plan[3, k], 0.0) * self.dt
            return plan

        own_state_plan = rollout(own_x0)
        neighbor_state_plan = rollout(neighbor_x0)
        own_control_plan = np.zeros((2, n))
        neighbor_control_plan = np.zeros((2, n))
        own_multipliers = np.zeros(n)
        neighbor_multipliers = np.zeros(n)
        own_diagnostics = {}
        neighbor_diagnostics = {}

        # Algorithm 1: complete ego/opponent rounds, followed by a final ego
        # solve so the executed command always reflects the latest opponent BR.
        for _ in range(self.iterations):
            previous_own = own_state_plan.copy()
            own_state_plan, own_control_plan, own_multipliers, own_diagnostics = self._solve_best_response(
                own_x0, neighbor_state_plan[:2], own_curvature, own_speed,
                initial_plan=own_state_plan,
                opponent_multipliers=neighbor_multipliers,
                previous_control=own_previous_control,
            )
            neighbor_state_plan, neighbor_control_plan, neighbor_multipliers, neighbor_diagnostics = self._solve_best_response(
                neighbor_x0, own_state_plan[:2], neighbor_curvature, neighbor_speed,
                initial_plan=neighbor_state_plan,
                opponent_multipliers=own_multipliers,
                previous_control=neighbor_previous_control,
            )
            if np.max(np.abs(own_state_plan[:2] - previous_own[:2])) <= self.convergence_tolerance:
                break

        own_state_plan, own_control_plan, own_multipliers, own_diagnostics = self._solve_best_response(
            own_x0, neighbor_state_plan[:2], own_curvature, own_speed,
            initial_plan=own_state_plan,
            opponent_multipliers=neighbor_multipliers,
            previous_control=own_previous_control,
        )
        own_plan = own_state_plan[:2]
        own_u = own_control_plan[:, 0]
        neighbor_u = neighbor_control_plan[:, 0]

        return {
            "action": own_u,
            "state_plan": own_state_plan,
            "control_plan": own_control_plan,
            "frenet_plan": own_plan,
            "neighbor_state_plan": neighbor_state_plan,
            "neighbor_control_plan": neighbor_control_plan,
            "neighbor_action": neighbor_u,
            "collision_multipliers": own_multipliers,
            "diagnostics": own_diagnostics,
            "neighbor_diagnostics": neighbor_diagnostics,
        }


class TrajectoryIBRPolicy(_IBRPolicyMixin, BasePolicy):
    """
    Selectable IBR baseline policy.

    ``local_wang`` follows the general IBR/SE-IBR construction of Wang et al.
    (RSS 2019) while using this project's physical model and MPC constraints.
    Its ``iterations`` counts complete ego/opponent best-response rounds and
    each planning call ends with one additional ego solve. Sensitivity zero
    selects ordinary IBR.

    ``historical_strict`` restores this repository's earlier `_StrictIBRSolver`
    with local vehicle parameters and hard track bounds. It also does not
    support SE and exists only as a historical implementation comparison.
    """

    ibr_level = "trajectory"

    def __init__(
        self,
        track,
        mpc,
        raceline: RaceLine,
        max_speed=5.0,
        car_idx=None,
        policy_name=None,
        iterations=2,
        ibr_horizon=None,
        lateral_half_width=None,
        sensitivity_weight=0.0,
        collision_longitudinal=None,
        collision_lateral=None,
        implementation="local_wang",
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        if raceline is None:
            raise ValueError("IteratedBestResponsePolicy requires a raceline")
        self.raceline = raceline
        self.max_speed = float(max_speed)
        self.last_raceline_s = None
        self.last_raceline_idx = None
        self.implementation = str(implementation).strip().lower()
        if self.implementation not in IBR_IMPLEMENTATIONS:
            raise ValueError(
                f"Unknown IBR implementation {implementation!r}; "
                f"choose one of {IBR_IMPLEMENTATIONS}"
            )
        solver_horizon = self.mpc.horizon if ibr_horizon is None else int(ibr_horizon)
        if lateral_half_width is None:
            lateral_half_width = getattr(self.mpc, "track_constraint_half_width", self.mpc.track_half_width)
        self.iterations = int(iterations)
        if collision_longitudinal is None:
            collision_longitudinal = self.mpc.min_longitudinal_separation
        if collision_lateral is None:
            collision_lateral = self.mpc.min_lateral_separation
        self.sensitivity_weight = max(float(sensitivity_weight), 0.0)
        if self.implementation == "historical_strict":
            if self.sensitivity_weight > 0.0:
                raise ValueError(
                    "SE is not part of the legacy alpha-RACER IBR variants; "
                    "use implementation='local_wang' or set sensitivity_weight=0"
                )
        if self.implementation == "historical_strict":
            self.ibr_solver = _StrictIBRSolver(
                vehicle_params=self.mpc.p,
                horizon=solver_horizon,
                dt=self.mpc.dt,
                iterations=sim_params.IBR_HISTORICAL_ITERATIONS,
                mu=sim_params.IBR_HISTORICAL_FRICTION_COEFFICIENT,
                gravity=GRAVITY,
                use_upstream_collision_expression=True,
                lateral_half_width=float(lateral_half_width),
            )
            self.iterations = self.ibr_solver.iterations
        else:
            self.ibr_solver = _WangIBRSolver(
                vehicle_params=self.mpc.p,
                horizon=solver_horizon,
                dt=self.mpc.dt,
                iterations=self.iterations,
                lateral_half_width=float(lateral_half_width),
                collision_longitudinal=collision_longitudinal,
                collision_lateral=collision_lateral,
                collision_exponent=self.mpc.collision_superellipse_exponent,
                sensitivity_weight=self.sensitivity_weight,
                progress_weight=sim_params.IBR_PROGRESS_WEIGHT,
                lateral_weight=sim_params.IBR_LATERAL_WEIGHT,
                lateral_exponent=sim_params.IBR_LATERAL_EXPONENT,
                heading_weight=sim_params.IBR_HEADING_WEIGHT,
                speed_weight=sim_params.IBR_SPEED_WEIGHT,
                control_weight=sim_params.IBR_CONTROL_WEIGHT,
                control_reference=sim_params.IBR_CONTROL_REFERENCE,
                control_rate_weight=sim_params.IBR_CONTROL_RATE_WEIGHT,
                convergence_tolerance=sim_params.IBR_CONVERGENCE_TOLERANCE,
                track_slack_weight=self.mpc.track_slack_weight,
                collision_slack_weight=self.mpc.collision_slack_weight,
                allow_track_slack=sim_params.IBR_ALLOW_TRACK_SLACK,
                allow_collision_slack=sim_params.IBR_ALLOW_COLLISION_SLACK,
                gravity=GRAVITY,
                mu=self.mpc.friction_coefficient,
                max_steer_delta=self.mpc.max_steer_delta,
                max_speed=self.mpc.max_speed,
                speed_cap_mode=sim_params.IBR_SPEED_CAP_MODE,
                raceline_speed_cap_factor=sim_params.IBR_RACELINE_SPEED_CAP_FACTOR,
                enforce_track_boundary=self.mpc.enforce_track_boundary,
                enforce_collision_avoidance=self.mpc.enforce_collision_avoidance,
                enforce_friction_speed_limit=self.mpc.enforce_friction_speed_limit,
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
        self.last_ibr_diagnostics = {}

    def _report_ibr_failure(self, exc, sim_time=None):
        self._ibr_failure_count += 1
        self._ibr_last_failure = str(exc)
        self.mpc.last_solver_success = False
        self.mpc.last_solver_status = str(exc)
        self.mpc.last_recovery_active = True
        self.mpc.last_recovery_reason = "ibr_hold_last_control"
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
            s = current_s + k * self.ibr_solver.dt * speed
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

    def plan_against_neighbor(
        self, own_state, own_s, own_e, neighbor_state, neighbor_s,
        neighbor_e, sim_time=None, neighbor_previous_control=None,
    ):
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
            neighbor_previous_control=neighbor_previous_control,
            own_previous_control=self.last_u,
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
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        self.mpc.last_control_plan = result["control_plan"]
        self.last_ibr_diagnostics = dict(result.get("diagnostics", {}))

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


class ParameterIBRPolicy(_IBRPolicyMixin, MPCPolicy):
    """Iterated best response in the paper's five-parameter policy space.

    Unlike :class:`TrajectoryIBRPolicy`, this policy never optimizes throttle
    or steering directly.  A best-response step searches
    ``theta=(q, zeta, s1, s2, s3)`` and evaluates every candidate by solving
    the same raceline-reference MPC used by alpha-RACER and the LLM policies.
    The candidate score is independent of the MPC's tunable tracking weight:
    predicted progress minus the MPC's existing track/collision slack costs.
    This makes candidates with different ``q`` comparable.
    """

    ibr_level = "parameter"
    PARAM_BOUNDS = ALPHA_PARAMETER_BOUNDS

    def __init__(
        self,
        *args,
        search_points=3,
        search_passes=1,
        update_interval_seconds=1.0,
        initial_params=None,
        convergence_tolerance=1.0e-3,
        **kwargs,
    ):
        super().__init__(
            *args,
            params=(
                np.asarray(sim_params.ALPHARACER_INITIAL_PARAMS, dtype=float)
                if initial_params is None
                else np.asarray(initial_params, dtype=float)
            ),
            **kwargs,
        )
        if self.params.shape != (5,):
            raise ValueError("ParameterIBRPolicy initial_params must contain q, zeta, s1, s2, s3")
        bounds = np.asarray(self.PARAM_BOUNDS, dtype=float)
        self.params = np.clip(self.params, bounds[:, 0], bounds[:, 1])
        self.search_points = max(2, int(search_points))
        self.search_passes = max(1, int(search_passes))
        self.update_interval_seconds = max(float(update_interval_seconds), 0.0)
        self.convergence_tolerance = max(float(convergence_tolerance), 0.0)
        self.last_parameter_ibr_update_time = None
        self.last_parameter_ibr_score = -np.inf
        self.last_parameter_ibr_diagnostics = {}
        self.predicted_frenet_plan = None
        self.current_mode = "parameter_ibr"

    def update_due(self, sim_time):
        return self._ibr_update_due(
            sim_time,
            self.last_parameter_ibr_update_time,
            self.update_interval_seconds,
        )

    def _world_plan_to_frenet(self, world_plan, initial_s):
        plan = np.zeros((2, world_plan.shape[1]), dtype=float)
        s_hint = float(initial_s)
        for k in range(world_plan.shape[1]):
            wrapped_s, lateral_e = self.track.project(
                float(world_plan[0, k]),
                float(world_plan[1, k]),
                s_hint=s_hint,
                search_radius=8.0,
                psi_hint=float(world_plan[2, k]),
                max_heading_error=math.pi / 2.0,
            )
            s_hint += progress_delta(wrapped_s, s_hint, self.track.total_length)
            plan[:, k] = [s_hint, lateral_e]
        return plan

    def _evaluate_candidate(self, candidate, state, current_s, opponents):
        """Solve the shared MPC once and return an invariant rollout score."""
        self.params = np.asarray(candidate, dtype=float).copy()
        ref_traj = self.get_reference_trajectory(state, current_s, opponents=opponents)
        self.mpc.solve(
            state.to_array(),
            ref_traj,
            self.last_feasible_u,
            q_weight=float(self.params[0]),
            opponents=opponents,
            current_s=current_s,
        )
        world_plan = np.asarray(self.mpc.last_state_plan, dtype=float).copy()
        frenet_plan = self._world_plan_to_frenet(world_plan, current_s)
        predicted_progress = float(frenet_plan[0, -1] - frenet_plan[0, 0])
        track_slack_cost = self.mpc.track_slack_weight * float(
            np.sum(np.square(self.mpc.last_track_slack_plan))
        )
        collision_slack_cost = self.mpc.collision_slack_weight * float(
            np.sum(np.square(self.mpc.last_collision_slack_plan))
        )
        score = predicted_progress - track_slack_cost - collision_slack_cost
        return score, frenet_plan, {
            "predicted_progress": predicted_progress,
            "track_slack_cost": track_slack_cost,
            "collision_slack_cost": collision_slack_cost,
        }

    def best_response(self, state, current_s, opponents):
        """Coordinate search for one agent while opponent plans stay fixed."""
        bounds = np.asarray(self.PARAM_BOUNDS, dtype=float)
        best = np.clip(np.asarray(self.params, dtype=float), bounds[:, 0], bounds[:, 1])
        best_score = -np.inf
        best_plan = None
        best_diagnostics = {}

        # Evaluate the warm start first, then update one parameter at a time.
        try:
            best_score, best_plan, best_diagnostics = self._evaluate_candidate(
                best, state, current_s, opponents
            )
        except Exception:
            pass

        for _ in range(self.search_passes):
            for dim in range(5):
                values = np.unique(np.concatenate((
                    np.linspace(bounds[dim, 0], bounds[dim, 1], self.search_points),
                    np.array([best[dim]]),
                )))
                dimension_best = best.copy()
                dimension_score = best_score
                dimension_plan = best_plan
                dimension_diagnostics = best_diagnostics
                for value in values:
                    candidate = best.copy()
                    candidate[dim] = value
                    try:
                        score, plan, diagnostics = self._evaluate_candidate(
                            candidate, state, current_s, opponents
                        )
                    except Exception:
                        continue
                    if score > dimension_score:
                        dimension_best = candidate
                        dimension_score = score
                        dimension_plan = plan
                        dimension_diagnostics = diagnostics
                best = dimension_best
                best_score = dimension_score
                best_plan = dimension_plan
                best_diagnostics = dimension_diagnostics

        if best_plan is None:
            raise RuntimeError("all parameter-space IBR candidate MPC solves failed")
        self.params = best
        self.predicted_frenet_plan = best_plan
        self.last_parameter_ibr_score = float(best_score)
        self.last_parameter_ibr_diagnostics = {
            **best_diagnostics,
            "score": float(best_score),
            "params": best.copy(),
        }
        return best.copy(), best_plan.copy()


# Backward-compatible public name used by existing scripts and imports.
IteratedBestResponsePolicy = TrajectoryIBRPolicy


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


def _constant_velocity_frenet_plan(policy, state, current_s, current_e):
    horizon = policy.mpc.horizon
    plan = np.zeros((2, horizon + 1), dtype=float)
    plan[0, 0] = float(current_s)
    plan[1, :] = float(current_e)
    for k in range(horizon):
        plan[0, k + 1] = plan[0, k] + max(float(state.vx), 0.0) * policy.mpc.dt
    return plan


def _parameter_ibr_opponents(policy_idx, policies, states, car_s, car_e, plans):
    opponents = []
    for other_idx, state in enumerate(states):
        if other_idx == policy_idx:
            continue
        plan = plans[other_idx]
        opponents.append({
            "car_idx": other_idx,
            "px": float(car_s[other_idx]),
            "py": float(car_e[other_idx]),
            "vx": max(float(state.vx), 0.0),
            "vy": float(state.vy),
            "omega": float(state.omega),
            "psi": float(state.psi),
            "state": state,
            "px_plan": plan[0].copy(),
            "py_plan": plan[1].copy(),
        })
    return opponents


def plan_parameter_ibr_policy_params(policies, states, car_s, car_e, sim_time=None):
    """Alternate best responses in theta while all cars retain the shared MPC."""
    indices = [
        idx for idx, policy in enumerate(policies)
        if isinstance(policy, ParameterIBRPolicy) and policy.update_due(sim_time)
    ]
    if not indices:
        return

    plans = []
    for idx, policy in enumerate(policies):
        cached = getattr(policy, "predicted_frenet_plan", None)
        expected_length = policy.mpc.horizon + 1
        if cached is not None and cached.shape == (2, expected_length):
            plan = np.asarray(cached, dtype=float).copy()
            # Re-anchor stale warm starts at the newly measured Frenet state.
            plan[0] += float(car_s[idx]) - plan[0, 0]
            plan[1] += float(car_e[idx]) - plan[1, 0]
        else:
            plan = _constant_velocity_frenet_plan(
                policy, states[idx], car_s[idx], car_e[idx]
            )
        plans.append(plan)

    rounds = max(1, int(sim_params.PARAMETER_IBR_ROUNDS))
    for _ in range(rounds):
        max_change = 0.0
        for idx in indices:
            policy = policies[idx]
            previous = policy.params.copy()
            opponents = _parameter_ibr_opponents(
                idx, policies, states, car_s, car_e, plans
            )
            params_opt, plan_opt = policy.best_response(
                states[idx], car_s[idx], opponents
            )
            plans[idx] = plan_opt
            span = np.maximum(
                policy.PARAM_BOUNDS[:, 1] - policy.PARAM_BOUNDS[:, 0], 1.0e-12
            )
            max_change = max(
                max_change,
                float(np.max(np.abs(params_opt - previous) / span)),
            )
        if max_change <= min(
            policies[idx].convergence_tolerance for idx in indices
        ):
            break

    for idx in indices:
        policies[idx].predicted_frenet_plan = plans[idx]
        policies[idx].last_parameter_ibr_update_time = sim_time


def plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=None):
    # Parameter-level IBR only updates theta here; its inherited MPCPolicy
    # computes the actual action later in the ordinary per-policy control loop.
    plan_parameter_ibr_policy_params(
        policies, states, car_s, car_e, sim_time=sim_time
    )

    # Trajectory-level IBR directly caches the next low-level action.
    for idx, policy in enumerate(policies):
        if not isinstance(policy, TrajectoryIBRPolicy):
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
                neighbor_previous_control=getattr(
                    policies[neighbor_idx], "last_u", None
                ),
            )
        except Exception as exc:
            policy._report_ibr_failure(exc, sim_time=sim_time)
