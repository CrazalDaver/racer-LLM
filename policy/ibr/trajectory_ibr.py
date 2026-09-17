"""Local trajectory-space IBR baseline and its simulation adapter.

The local implementation follows the *alternating best-response* idea in
Wang et al. (RSS 2019): for a two-car interaction it alternately optimizes
the ego trajectory and a predicted neighbor trajectory, then executes the
first ego command in receding-horizon fashion.  It uses this repository's
Frenet kinematic model and safety conventions, rather than the paper's
piecewise-polynomial SE-IBR formulation.  It is consequently an idea-level
``local_wang`` baseline, not a reproduction of the sensitivity-enhanced
objective or the original numerical implementation.

There are deliberately two layers in this file:

* :class:`LocalWangStyleSolver` owns only the two-car CasADi optimization.
* :class:`TrajectoryIBRPolicy` translates simulator state into solver input,
  stores the resulting control, and adapts plans for common trace fields.
The simulator supplies an immutable race observation to ``get_action``; the
policy performs its own local two-car solve directly from that snapshot.
"""

import math
from dataclasses import dataclass

import casadi as cs
import numpy as np

from config import GRAVITY, params as sim_params
from state import CarAction, CarState
from track import RaceLine
from vehicle_model import (
    kinematic_frenet_step,
    minimum_reachable_speed,
    neutral_throttle,
)

from ..base import BasePolicy
from ..utils import track_curvature_at, track_heading_error
from .historical_ibr import _HistoricalStrictIBRSolver


IBR_IMPLEMENTATIONS = ("local_wang", "historical_strict")


@dataclass(frozen=True)
class TrajectoryIBRSettings:
    """Resolved, immutable inputs of one local trajectory-IBR solver.

    This object is intentionally narrower than the mutable global
    ``config.params``.  Some fields must use the *effective* per-run MPC
    values (for example a CLI velocity/slack override), while objective terms
    remain experiment-level IBR choices.  Constructing this snapshot once
    makes that boundary inspectable and avoids solver code silently reading
    global configuration later.
    """

    horizon: int
    dt: float
    vehicle_params: object
    mu: float
    gravity: float
    lateral_half_width: float
    collision_longitudinal: float
    collision_lateral: float
    collision_exponent: int
    track_slack_weight: float
    collision_slack_weight: float
    allow_track_slack: bool
    allow_collision_slack: bool
    progress_weight: float
    lateral_weight: float
    lateral_exponent: int
    heading_weight: float
    speed_weight: float
    control_weight: float
    control_reference: str
    control_rate_weight: float
    convergence_tolerance: float
    max_steer_delta: float
    max_speed: float
    speed_cap_mode: str
    raceline_speed_cap_factor: float
    enforce_track_boundary: bool
    enforce_collision_avoidance: bool
    enforce_friction_speed_limit: bool
    iterations: int

    @classmethod
    def from_runtime(
        cls,
        *,
        mpc,
        sim_params,
        horizon,
        iterations,
        lateral_half_width,
        collision_longitudinal,
        collision_lateral,
    ):
        """Resolve one solver snapshot from shared MPC and IBR configuration.

        The MPC instance is authoritative for constraints and weights that
        can be overridden per experiment.  ``sim_params`` supplies IBR-only
        objective choices that currently have no per-run CLI override.
        """
        return cls(
            horizon=int(horizon),
            dt=float(mpc.dt),
            vehicle_params=mpc.p,
            mu=float(mpc.friction_coefficient),
            gravity=float(GRAVITY),
            lateral_half_width=float(lateral_half_width),
            collision_longitudinal=float(collision_longitudinal),
            collision_lateral=float(collision_lateral),
            collision_exponent=int(mpc.collision_superellipse_exponent),
            track_slack_weight=float(mpc.track_slack_weight),
            collision_slack_weight=float(mpc.collision_slack_weight),
            allow_track_slack=bool(sim_params.IBR_ALLOW_TRACK_SLACK),
            allow_collision_slack=bool(sim_params.IBR_ALLOW_COLLISION_SLACK),
            progress_weight=float(sim_params.IBR_PROGRESS_WEIGHT),
            lateral_weight=float(sim_params.IBR_LATERAL_WEIGHT),
            lateral_exponent=int(sim_params.IBR_LATERAL_EXPONENT),
            heading_weight=float(sim_params.IBR_HEADING_WEIGHT),
            speed_weight=float(mpc.velocity_track_weight),
            control_weight=float(sim_params.IBR_CONTROL_WEIGHT),
            control_reference=str(sim_params.IBR_CONTROL_REFERENCE),
            control_rate_weight=float(sim_params.IBR_CONTROL_RATE_WEIGHT),
            convergence_tolerance=float(sim_params.IBR_CONVERGENCE_TOLERANCE),
            max_steer_delta=float(mpc.max_steer_delta),
            max_speed=float(mpc.max_speed),
            speed_cap_mode=str(sim_params.IBR_SPEED_CAP_MODE),
            raceline_speed_cap_factor=float(sim_params.IBR_RACELINE_SPEED_CAP_FACTOR),
            enforce_track_boundary=bool(mpc.enforce_track_boundary),
            enforce_collision_avoidance=bool(mpc.enforce_collision_avoidance),
            enforce_friction_speed_limit=bool(mpc.enforce_friction_speed_limit),
            iterations=int(iterations),
        )


class LocalWangStyleSolver:
    """Two-car, finite-horizon alternating best-response optimization.

    Each solve keeps the other car's candidate Frenet ``[s, e]`` plan fixed,
    optimizes one car's dynamically feasible state/control trajectory, then
    alternates players for a bounded number of rounds.  A final ego solve
    reacts to the latest neighbor response.  This is the local approximation
    of the IBR loop; it does *not* contain Wang et al.'s sensitivity-enhanced
    multiplier term.

    The solver is independent of simulator objects after construction.  It
    receives only :class:`TrajectoryIBRSettings`, numerical states, track
    curvature samples, reference-speed samples, and the other trajectory.
    That separation is intentional: policy-level state projection, cache
    management, and multi-car scheduling do not belong in the NLP.
    """

    def __init__(self, settings: TrajectoryIBRSettings):
        self.settings = settings
        self.p = settings.vehicle_params
        self.horizon = settings.horizon
        self.dt = settings.dt
        self.iterations = settings.iterations
        self.mu = settings.mu
        self.gravity = settings.gravity
        self.lateral_half_width = settings.lateral_half_width
        self.collision_longitudinal = settings.collision_longitudinal
        self.collision_lateral = settings.collision_lateral
        self.collision_exponent = settings.collision_exponent
        self.track_slack_weight = settings.track_slack_weight
        self.collision_slack_weight = settings.collision_slack_weight
        self.allow_track_slack = settings.allow_track_slack
        self.allow_collision_slack = settings.allow_collision_slack
        self.progress_weight = settings.progress_weight
        self.lateral_weight = settings.lateral_weight
        self.lateral_exponent = settings.lateral_exponent
        if self.lateral_exponent not in {2, 4}:
            raise ValueError("IBR lateral_exponent must be 2 or 4")
        self.heading_weight = settings.heading_weight
        self.speed_weight = settings.speed_weight
        self.control_weight = settings.control_weight
        self.control_reference = settings.control_reference
        if self.control_reference not in {"neutral", "zero"}:
            raise ValueError("IBR control_reference must be 'neutral' or 'zero'")
        self.control_rate_weight = settings.control_rate_weight
        self.convergence_tolerance = max(settings.convergence_tolerance, 0.0)
        self.max_steer_delta = max(settings.max_steer_delta, 0.0)
        self.max_speed = settings.max_speed
        self.speed_cap_mode = settings.speed_cap_mode
        self.raceline_speed_cap_factor = settings.raceline_speed_cap_factor
        if self.speed_cap_mode not in {"global", "raceline"}:
            raise ValueError("IBR speed_cap_mode must be 'global' or 'raceline'")
        self.enforce_track_boundary = settings.enforce_track_boundary
        self.enforce_collision_avoidance = settings.enforce_collision_avoidance
        self.enforce_friction_speed_limit = settings.enforce_friction_speed_limit
        self._build_solver()

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
        """Build the reusable CasADi NLP once for a fixed settings snapshot.

        Curvature, speed profile, initial state, previous command, and the
        fixed neighbor plan remain numerical NLP parameters.  The variables
        are the focal car's Frenet state/control sequence plus non-negative
        track and collision slack.  Reusing this compiled graph is why these
        quantities are not reconstructed in every policy tick.
        """
        n = self.horizon
        x = cs.MX.sym("x", 4, n + 1)
        u = cs.MX.sym("u", 2, n)
        x0 = cs.MX.sym("x0", 4)
        curvature = cs.MX.sym("curvature", n)
        # Horizon-wise raceline speed profile, matching the shared MPC input.
        target_speed = cs.MX.sym("target_speed", n)
        neutral_control = cs.MX.sym("neutral_control", 2)
        opp_s = cs.MX.sym("opp_s", n + 1)
        opp_e = cs.MX.sym("opp_e", n + 1)
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
            cost += self.speed_weight * (x[3, k] - target_speed[k]) ** 2
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

    def _neutral_throttle(self, speed):
        """Throttle that approximately maintains speed in MPC dynamics."""
        return neutral_throttle(speed, self.p)

    def _solve_best_response(
        self, x0, opponent_plan, curvature, target_speed,
        initial_plan=None, previous_control=None,
    ):
        """Optimize one car while treating ``opponent_plan`` as fixed.

        This is one best-response subproblem, not a joint solve.  Its output
        becomes the other player's fixed trajectory in the next alternating
        step.  Warm-start plans are aids to IPOPT convergence only; all
        dynamic, track, collision, actuation, and slack constraints are
        imposed again by the NLP.
        """
        n = self.horizon
        target_speed = np.asarray(target_speed, dtype=float).reshape(-1)
        if target_speed.size == 1:
            target_speed = np.full(n, float(target_speed[0]))
        if target_speed.size != n:
            raise ValueError(
                f"target_speed must contain {n} samples, got {target_speed.size}"
            )
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
        if previous_control is None:
            previous_control = np.asarray([
                self._neutral_throttle(x0[3]),
                -self.p.Sb / max(self.p.Sa, 1e-6),
            ])
        previous_control = np.asarray(previous_control, dtype=float)
        neutral_control = np.asarray(
            [self._neutral_throttle(target_speed[0]), -self.p.Sb / max(self.p.Sa, 1e-6)],
            dtype=float,
        )
        p_val = np.concatenate(
            [
                np.asarray(x0, dtype=float),
                np.asarray(curvature, dtype=float),
                target_speed,
                neutral_control,
                np.asarray(opponent_plan[0], dtype=float),
                np.asarray(opponent_plan[1], dtype=float),
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
                    self.raceline_speed_cap_factor * float(target_speed[0]),
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
        ds = (state_plan[0, 1:] - opponent_plan[0, 1:]) / self.collision_longitudinal
        de = (state_plan[1, 1:] - opponent_plan[1, 1:]) / self.collision_lateral
        raw_collision_constraint = (
            1.0 - ds ** self.collision_exponent - de ** self.collision_exponent
        )
        diagnostics = {
            "track_slack": track_slack_plan,
            "collision_slack": collision_slack_plan,
            "max_collision_constraint": float(np.max(raw_collision_constraint)),
            "min_predicted_longitudinal_distance": float(
                np.min(np.abs(state_plan[0, 1:] - opponent_plan[0, 1:]))
            ),
            "min_predicted_lateral_distance": float(
                np.min(np.abs(state_plan[1, 1:] - opponent_plan[1, 1:]))
            ),
        }
        return state_plan, control_plan, diagnostics

    def iterate_best_responses(
        self, neighbor_x0, own_x0, neighbor_curvature, own_curvature,
        neighbor_speed, own_speed, neighbor_previous_control=None,
        own_previous_control=None,
    ):
        """Run bounded alternating best responses and return the final ego plan.

        Both initial plans are constant-speed rollouts used solely as first
        guesses.  A round is ego best response followed by neighbor best
        response.  The extra ego solve after the loop mirrors the paper's
        execution convention: the command reacts to the latest neighbor plan.
        Early termination tests only plan change; it is not proof of a Nash
        equilibrium, especially because this local solver omits SE-IBR.
        """
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
        own_diagnostics = {}
        neighbor_diagnostics = {}

        # Algorithm 1: complete ego/opponent rounds, followed by a final ego
        # solve so the executed command always reflects the latest opponent BR.
        for _ in range(self.iterations):
            previous_own = own_state_plan.copy()
            own_state_plan, own_control_plan, own_diagnostics = self._solve_best_response(
                own_x0, neighbor_state_plan[:2], own_curvature, own_speed,
                initial_plan=own_state_plan,
                previous_control=own_previous_control,
            )
            neighbor_state_plan, neighbor_control_plan, neighbor_diagnostics = self._solve_best_response(
                neighbor_x0, own_state_plan[:2], neighbor_curvature, neighbor_speed,
                initial_plan=neighbor_state_plan,
                previous_control=neighbor_previous_control,
            )
            if np.max(np.abs(own_state_plan[:2] - previous_own[:2])) <= self.convergence_tolerance:
                break

        own_state_plan, own_control_plan, own_diagnostics = self._solve_best_response(
            own_x0, neighbor_state_plan[:2], own_curvature, own_speed,
            initial_plan=own_state_plan,
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
            "diagnostics": own_diagnostics,
            "neighbor_diagnostics": neighbor_diagnostics,
        }

class TrajectoryIBRPolicy(BasePolicy):
    """Simulator-facing adapter for a direct trajectory-space IBR planner.

    This policy directly executes the first control from a local two-car IBR
    plan; unlike :class:`MPCPolicy`, it does not give a reference to the
    shared MPC optimizer for control tracking.  The supplied ``mpc`` object
    still defines the effective physical limits and cost scales, and is used
    as the common diagnostics/trace interface expected by the simulator.

    ``get_action`` receives one immutable :class:`RaceObservation`, chooses
    the closest longitudinal opponent, and performs the complete local solve.
    The observation contains previous-tick controls for all cars, so calling
    policies in sequence cannot expose newly updated internal state.

    ``local_wang`` is a local alternating-IBR adaptation, not an SE-IBR
    reproduction.  ``historical_strict`` retains the repository's former
    solver solely for historical comparisons and should not be used as the
    primary Wang-style baseline.
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
        if self.implementation == "historical_strict":
            self.ibr_solver = _HistoricalStrictIBRSolver(
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
            self.ibr_settings = None
        else:
            self.ibr_settings = TrajectoryIBRSettings.from_runtime(
                mpc=self.mpc,
                sim_params=sim_params,
                horizon=solver_horizon,
                iterations=self.iterations,
                lateral_half_width=lateral_half_width,
                collision_longitudinal=collision_longitudinal,
                collision_lateral=collision_lateral,
            )
            self.ibr_solver = LocalWangStyleSolver(self.ibr_settings)
        self._last_ibr_action = None
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
        """Sample centerline curvature along a frozen-speed nominal horizon.

        The Frenet dynamics need a curvature value at every prediction step.
        We use ``s_k = s_0 + k dt v_0`` as an inexpensive external parameter
        schedule, rather than making curvature a symbolic function of the
        optimized progress.  This is a local approximation: it can disagree
        with the eventual plan when the car accelerates or brakes sharply,
        particularly around a hairpin.  It should be audited against the
        solved ``s`` plan before changing baseline behavior.
        """
        curvature = np.zeros(self.ibr_solver.horizon)
        speed = max(float(state.vx), 0.0)
        for k in range(self.ibr_solver.horizon):
            s = current_s + k * self.ibr_solver.dt * speed
            curvature[k] = track_curvature_at(self.track, s, ds=0.5)
        return curvature

    def _reference_speed_profile(self, state, current_s):
        """Sample raceline target speeds for the local solver horizon.

        This is a speed-tracking preference/cap input, not a trajectory given
        to the solver.  The profile advances using its own sampled raceline
        speed; it may therefore differ from both frozen-speed curvature
        lookahead and the optimized progress sequence.
        """
        n = self.ibr_solver.horizon
        if self.raceline is None:
            return np.full(n, max(float(state.vx), 0.5), dtype=float)
        try:
            idx = self.raceline.nearest_index(state.x, state.y)
            self.last_raceline_idx = idx
            self.last_raceline_s = float(self.raceline.s[idx])
            profile = np.empty(n, dtype=float)
            raceline_s = self.last_raceline_s
            for k in range(n):
                sample = self.raceline.interpolate(raceline_s)
                profile[k] = min(float(sample["vx_mps"]), self.max_speed)
                raceline_s += max(profile[k], 0.0) * self.ibr_solver.dt
            return profile
        except Exception:
            return np.full(n, max(float(state.vx), 0.5), dtype=float)

    def _frenet_x0(self, state, current_s, current_e):
        """Project a simulator state into the solver's ``[s,e,psi_err,v]`` state."""
        return np.array(
            [
                float(current_s),
                float(current_e),
                float(track_heading_error(self.track, state.psi, current_s, reference_minus_heading=False)),
                max(float(state.vx), 0.0),
            ],
            dtype=float,
        )

    def _make_ref_from_plan(self, plan, speed_profile):
        """Adapt a direct IBR plan to the shared trace/visualization format.

        The returned reference is diagnostic only: the direct IBR action has
        already been computed by :class:`LocalWangStyleSolver`; it is not fed
        into ``self.mpc.solve``.  Keeping this adapter lets plots and saved
        traces use the same fields for all policy families.
        """
        horizon = self.mpc.horizon
        speed_profile = np.asarray(speed_profile, dtype=float).reshape(-1)
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
            ref[2, k] = speed_profile[min(k, speed_profile.size - 1)]
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = -math.sin(psi)
            ref[7, k] = math.cos(psi)
            ref_world[:, k] = self.track.frenet_to_world(s, e)
        return ref, ref_world

    def prepare_interaction_plan(
        self, own_state, own_s, own_e, neighbor_state, neighbor_s,
        neighbor_e, sim_time=None, neighbor_previous_control=None,
    ):
        """Prepare and cache this policy's first action against one neighbor.

        Both current simulator states are converted to Frenet coordinates.
        The local solver internally produces a predicted plan for each car,
        but only this policy's first command is cached and later executed.
        The neighbor is not commanded by this call; it will prepare its own
        action in the batch coordinator if it is also a trajectory-IBR policy.
        """
        own_x0 = self._frenet_x0(own_state, own_s, own_e)
        neighbor_x0 = self._frenet_x0(neighbor_state, neighbor_s, neighbor_e)
        own_speed = self._reference_speed_profile(own_state, own_s)
        neighbor_speed = self._reference_speed_profile(neighbor_state, neighbor_s)
        # The historical implementation has only a scalar speed cap and no
        # velocity-tracking term; keep its replay interface unchanged.
        if self.implementation == "historical_strict":
            own_speed = float(own_speed[0])
            neighbor_speed = float(neighbor_speed[0])
        result = self.ibr_solver.iterate_best_responses(
            neighbor_x0=neighbor_x0,
            own_x0=own_x0,
            neighbor_curvature=self._curvature_lookahead(neighbor_s, neighbor_state),
            own_curvature=self._curvature_lookahead(own_s, own_state),
            neighbor_speed=neighbor_speed,
            own_speed=own_speed,
            neighbor_previous_control=neighbor_previous_control,
            own_previous_control=self.last_u,
        )
        self._cache_interaction_result(result, own_speed, sim_time)
        self._report_ibr_recovery(sim_time=sim_time)

    def _cache_interaction_result(self, result, ref_speed, sim_time):
        """Store a successful direct-control plan in policy and trace state."""
        action = np.asarray(result["action"], dtype=float)
        action = np.clip(action, -1.0, 1.0)
        self._last_ibr_action = CarAction(throttle=float(action[0]), steer=float(action[1]))
        self.last_u = np.array([self._last_ibr_action.throttle, self._last_ibr_action.steer])
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
            # The shared MPC trace convention is now Frenet
            # [s,e,heading_error,v], matching trajectory IBR directly.
            self.mpc.last_state_plan = np.asarray(
                result["state_plan"], dtype=float
            ).copy()

    def get_action(self, observation) -> CarAction:
        """Solve the local two-car game from ``observation`` and return its first control."""
        if self.car_idx is None:
            raise ValueError("TrajectoryIBRPolicy requires car_idx when used with RaceObservation")
        ego = observation.car(self.car_idx)
        neighbor = observation.nearest_opponent(self.car_idx, self.track.total_length)
        if neighbor is None:
            return CarAction(throttle=float(self.last_u[0]), steer=float(self.last_u[1]))
        try:
            self.prepare_interaction_plan(
                own_state=ego.state,
                own_s=ego.s,
                own_e=ego.e,
                neighbor_state=neighbor.state,
                neighbor_s=neighbor.s,
                neighbor_e=neighbor.e,
                sim_time=observation.sim_time,
                neighbor_previous_control=neighbor.previous_action,
            )
        except Exception as exc:
            self._report_ibr_failure(exc, sim_time=observation.sim_time)
            return CarAction(throttle=float(self.last_u[0]), steer=float(self.last_u[1]))
        return self._last_ibr_action


# Backward-compatible public name used by existing scripts and imports.
IteratedBestResponsePolicy = TrajectoryIBRPolicy
