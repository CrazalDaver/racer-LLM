"""Optional simplified Frenet MPC objective/constraint ablation.

This module is intentionally detachable.  It reuses the shared kinematic
dynamics, solver interface, and diagnostics from :mod:`mpc_controller`, while
building its own small NLP containing only state and control variables.  Track
departures and physical collisions remain simulator outcomes in this profile.

It retains the released controller's small objective and unconstrained layout,
but now shares the project's Frenet state dynamics.  It is therefore not an
exact reproduction of the released world-coordinate numerical controller.
"""

import casadi as cs
import numpy as np

from config import GRAVITY
from mpc_controller import MPCController


class UpstreamSimpleMPCController(MPCController):
    """Small path tracker without safety slacks or safety constraints."""

    formulation = "upstream_simple"

    def __init__(self, *args, **kwargs):
        # These settings keep the inherited parameter/diagnostic interface
        # intact, but the overridden NLP does not allocate safety variables.
        kwargs["max_constrained_opponents"] = 0
        kwargs["enforce_track_boundary"] = False
        kwargs["enforce_collision_avoidance"] = False
        kwargs["velocity_track_weight"] = 0.0
        kwargs["heading_track_weight"] = 0.0
        super().__init__(*args, **kwargs)

    def _build_problem(self):
        track_cost = 0
        action_cost = 0
        zero_cost = cs.SX(0.0)
        g_eq = [self.X[:, 0] - self.x0]
        g_ineq = []

        # Retain the released controller's N reference points, absolute input
        # effort, and absence of terminal and control-rate costs.
        for k in range(self.horizon):
            g_eq.append(
                self.X[:, k + 1]
                - self._vehicle_dynamics(self.X[:, k], self.U[:, k])
            )

            position_error = cs.vertcat(
                self.X[0, k] - self.ref[0, k],
                self.X[1, k] - self.ref[1, k],
            )
            track_cost += self.q_weight * (
                position_error.T @ self.Q @ position_error
            )
            action_cost += self.U[:, k].T @ self.R @ self.U[:, k]

            if self.enforce_friction_speed_limit:
                wheelbase = self.p.LF + self.p.LR
                lateral_proxy = self.X[3, k] ** 2 * cs.tan(self.U[1, k])
                lateral_limit = self.friction_coefficient * GRAVITY * wheelbase
                g_ineq.extend(
                    [
                        lateral_proxy - lateral_limit,
                        -lateral_proxy - lateral_limit,
                    ]
                )

        cost = track_cost + action_cost
        self.g = cs.vertcat(*g_eq, *g_ineq)
        self.x_opt = cs.vertcat(
            cs.reshape(self.X, -1, 1),
            cs.reshape(self.U, -1, 1),
        )
        # Preserve the base parameter layout so policies need no special case.
        self.p_opt = cs.vertcat(
            cs.reshape(self.x0, -1, 1),
            cs.reshape(self.ref, -1, 1),
            cs.reshape(self.u_prev, -1, 1),
            self.q_weight,
            cs.reshape(self.opponent_s, -1, 1),
            cs.reshape(self.opponent_e, -1, 1),
            self.opponent_active,
            self.stall_debt_weight,
        )

        nlp = {"x": self.x_opt, "p": self.p_opt, "f": cost, "g": self.g}
        options = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.max_iter": 500,
        }
        self.solver = cs.nlpsol(
            "upstream_simple_mpc_solver", "ipopt", nlp, options
        )
        self.loss_fn = cs.Function(
            "upstream_simple_mpc_loss_parts",
            [self.x_opt, self.p_opt],
            [
                cost,
                track_cost,
                action_cost,
                zero_cost,  # velocity
                zero_cost,  # heading
                zero_cost,  # combined violation
                zero_cost,  # track slack
                zero_cost,  # collision slack
            ],
        )
        self.terminal_viability_fn = cs.Function(
            "upstream_simple_terminal_viability",
            [self.x_opt, self.p_opt],
            [zero_cost, zero_cost],
        )
        self.stall_debt_fn = cs.Function(
            "upstream_simple_stall_debt",
            [self.x_opt, self.p_opt],
            [zero_cost, zero_cost],
        )

        self.n_x = self.n_states * (self.horizon + 1)
        self.n_u = self.n_controls * self.horizon
        self.n_track_slack = 0
        self.n_collision_slack = 0
        self.n_opt = self.n_x + self.n_u
        self.n_eq = self.n_states * (self.horizon + 1)
        self.n_total = int(self.g.size1())

    def _initial_slack_guess(
        self,
        X_guess,
        ref_traj,
        opponents=None,
        current_s=None,
    ):
        del X_guess, ref_traj, opponents, current_s
        return np.empty(0, dtype=float), np.empty((0, self.horizon), dtype=float)
