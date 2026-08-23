"""Historical alpha-RACER-style IBR retained for reproducibility only."""

import casadi as cs
import numpy as np

from vehicle_model import kinematic_frenet_step


class _HistoricalStrictIBRSolver:
    """Restored locally adapted IBR used in repository history.

    This matches the strict IBR solver present from commit 5ab4f74 through
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
        }
