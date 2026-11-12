"""One round-robin IBR policy with three interchangeable strategy spaces.

The outer game is the same for every strategy space: starting from one frozen
three-car observation, update one modeled player at a time for the configured
number of complete rounds. Only a player's best-response variables change: ``control`` optimizes
throttle/steering directly; ``parameter`` searches theta=(q,zeta,s1,s2,s3);
and ``mode`` chooses an existing parameter preset. The latter two call the
shared reference builder and MPC. This basic IBR does not implement SE-IBR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import casadi as cs
import numpy as np

from config import GRAVITY, params as sim_params
from state import CarAction
from vehicle_model import kinematic_frenet_step

from ..alpharacer.network import ALPHA_PARAMETER_BOUNDS
from ..mpc import MPCPolicy
from ..utils import relative_progress_gain


IBR_STRATEGY_SPACES = ("control", "parameter", "mode")


@dataclass
class BestResponseResult:
    """Output of one modeled player's single best-response optimization."""

    player_idx: int
    state_plan: np.ndarray
    control_plan: np.ndarray
    action: np.ndarray
    utility: float
    params: np.ndarray | None = None
    mode: str | None = None
    diagnostics: dict = field(default_factory=dict)


@dataclass
class RoundRobinResult:
    """Output of all requested IBR rounds from one frozen race observation."""

    plans: list[np.ndarray]
    controls: list[np.ndarray]
    player_results: list[BestResponseResult]
    round_diagnostics: list[dict]


class RoundRobinIBR:
    """Run complete sequential best-response rounds for all modeled players.

    Input is an initial state/control plan for every player plus a callback that
    solves exactly one player's BR against the latest joint plans. Output is
    the final joint plans and the last BR result for each player. The ego is
    updated last in every round, so its executed action responds to the other
    players' newest plans rather than to a vehicle-number ordering accident.
    """

    def __init__(self, rounds: int, convergence_tolerance: float = 0.0):
        self.rounds = int(rounds)
        if self.rounds < 1:
            raise ValueError("IBR rounds must be at least one")
        self.convergence_tolerance = max(float(convergence_tolerance), 0.0)
        self.current_round = None
        self.current_player = None

    def solve(self, initial_plans, initial_controls, ego_idx, solve_best_response):
        plans = [np.asarray(plan, dtype=float).copy() for plan in initial_plans]
        controls = [np.asarray(plan, dtype=float).copy() for plan in initial_controls]
        player_count = len(plans)
        if player_count < 2 or len(controls) != player_count:
            raise ValueError("round-robin IBR requires matching plans for at least two players")
        ego_idx = int(ego_idx)
        if not 0 <= ego_idx < player_count:
            raise IndexError("ego_idx is outside the modeled players")

        order = [idx for idx in range(player_count) if idx != ego_idx] + [ego_idx]
        latest = [None] * player_count
        history = []
        for round_idx in range(self.rounds):
            before = [plan.copy() for plan in plans]
            solve_seconds = 0.0
            utilities = {}
            accepted_max_iter_count = 0
            max_residual = 0.0
            for player_idx in order:
                self.current_round = round_idx + 1
                self.current_player = player_idx
                started = time.perf_counter()
                result = solve_best_response(player_idx, plans, controls)
                solve_seconds += time.perf_counter() - started
                if result.player_idx != player_idx:
                    raise ValueError("best-response result belongs to the wrong player")
                plans[player_idx] = np.asarray(result.state_plan, dtype=float).copy()
                controls[player_idx] = np.asarray(result.control_plan, dtype=float).copy()
                latest[player_idx] = result
                utilities[player_idx] = float(result.utility)
                accepted_max_iter_count += int(
                    bool(result.diagnostics.get("accepted_max_iter", False))
                )
                residual = result.diagnostics.get("residual")
                if residual is not None and np.isfinite(residual):
                    max_residual = max(max_residual, float(residual))
            max_change = max(
                float(np.max(np.abs(plans[idx][:2] - before[idx][:2])))
                for idx in range(player_count)
            )
            history.append({
                "round": round_idx + 1,
                "update_order": tuple(order),
                "max_plan_change": max_change,
                "solve_seconds": solve_seconds,
                "utilities": utilities,
                "accepted_max_iter_count": accepted_max_iter_count,
                "max_residual": max_residual,
            })
            if self.convergence_tolerance > 0.0 and max_change <= self.convergence_tolerance:
                break

        if any(result is None for result in latest):
            raise RuntimeError("round-robin IBR did not update every player")
        return RoundRobinResult(plans, controls, latest, history)


class ControlBestResponseSolver:
    """Solve one player's direct control-space BR against fixed opponent plans.

    Inputs are the focal player's measured Frenet state, previous executed
    command, warm-start plan, and all opponents' fixed ``[s,e]`` plans. Output
    is a dynamically feasible state/control plan. Opponent progress is constant
    during this BR, so maximizing paper relative progress is exactly maximizing
    the focal player's terminal progress.
    """

    _instance_counter = 0

    def __init__(self, mpc, *, horizon, max_opponents, initial_guess_mode="release"):
        self.mpc = mpc
        self.p = mpc.p
        self.dt = float(mpc.dt)
        self.horizon = int(horizon)
        self.max_opponents = int(max_opponents)
        self.initial_guess_mode = str(initial_guess_mode).strip().lower()
        if self.initial_guess_mode not in {"release", "rollout"}:
            raise ValueError("IBR initial guess must be 'release' or 'rollout'")
        self.track_half_width = float(mpc.track_constraint_half_width)
        self.longitudinal_separation = float(mpc.min_longitudinal_separation)
        self.lateral_separation = float(mpc.min_lateral_separation)
        self.collision_exponent = int(mpc.collision_superellipse_exponent)
        self.max_speed = float(mpc.max_speed)
        self.max_steer_delta = float(mpc.max_steer_delta)
        self.mu = float(mpc.friction_coefficient)
        self.track_slack_weight = float(mpc.track_slack_weight)
        self.collision_slack_weight = float(mpc.collision_slack_weight)
        self.enforce_track = bool(mpc.enforce_track_boundary)
        self.enforce_collision = bool(mpc.enforce_collision_avoidance)
        self.enforce_friction = bool(mpc.enforce_friction_speed_limit)
        self.last_diagnostics = {}
        self._build_problem()

    def _build_problem(self):
        """Build the reusable NLP; measured state and joint plans remain inputs."""
        n, m = self.horizon, self.max_opponents
        x = cs.MX.sym("x", 4, n + 1)
        u = cs.MX.sym("u", 2, n)
        x0 = cs.MX.sym("x0", 4)
        u_prev = cs.MX.sym("u_prev", 2)
        curvature = cs.MX.sym("curvature", n)
        opponent_s = cs.MX.sym("opponent_s", m * (n + 1))
        opponent_e = cs.MX.sym("opponent_e", m * (n + 1))
        track_slack = cs.MX.sym("track_slack", n)
        collision_slack = cs.MX.sym("collision_slack", m, n)

        constraints = [x[:, 0] - x0]
        for k in range(n):
            next_state = kinematic_frenet_step(
                x[:, k], u[:, k], curvature[k], self.dt, self.p,
                curvature_mode="geometric", symbolic=True,
            )
            constraints.append(x[:, k + 1] - next_state)

        friction_start = sum(item.numel() for item in constraints)
        for k in range(n):
            constraints.append(x[3, k] ** 2 * cs.tan(u[1, k]))
        speed_start = sum(item.numel() for item in constraints)
        for k in range(n):
            constraints.append(x[3, k + 1])
        track_start = sum(item.numel() for item in constraints)
        for k in range(1, n + 1):
            constraints.extend((
                x[1, k] - self.track_half_width - track_slack[k - 1],
                -x[1, k] - self.track_half_width - track_slack[k - 1],
            ))
        collision_start = sum(item.numel() for item in constraints)
        for opponent_idx in range(m):
            base = opponent_idx * (n + 1)
            for k in range(1, n + 1):
                ds = (x[0, k] - opponent_s[base + k]) / self.longitudinal_separation
                de = (x[1, k] - opponent_e[base + k]) / self.lateral_separation
                constraints.append(
                    1.0 - ds ** self.collision_exponent
                    - de ** self.collision_exponent
                    - collision_slack[opponent_idx, k - 1]
                )
        steer_rate_start = sum(item.numel() for item in constraints)
        for k in range(n):
            previous_steer = u_prev[1] if k == 0 else u[1, k - 1]
            constraints.append(u[1, k] - previous_steer)

        # Opponents are fixed during one BR, so relative progress differs from
        # own progress only by a constant. Release terms favor staying near the
        # road center and avoiding large heading errors and control commands.
        objective = -float(sim_params.IBR_PROGRESS_WEIGHT) * (x[0, n] - x[0, 0])
        for k in range(n):
            objective += float(sim_params.IBR_LATERAL_WEIGHT) * x[1, k] ** 4
            objective += float(sim_params.IBR_HEADING_WEIGHT) * x[2, k] ** 2
            objective += float(sim_params.IBR_SPEED_WEIGHT) * x[3, k] ** 2
            objective += float(sim_params.IBR_CONTROL_WEIGHT) * cs.sumsqr(u[:, k])
            du = u[:, k] - (u_prev if k == 0 else u[:, k - 1])
            objective += float(sim_params.IBR_CONTROL_RATE_WEIGHT) * cs.sumsqr(du)
        # Match the local MPC's cost for track and collision violations.
        objective += self.track_slack_weight * cs.sumsqr(track_slack)
        objective += self.collision_slack_weight * cs.sumsqr(
            cs.reshape(collision_slack, -1, 1)
        )

        variables = cs.vertcat(
            cs.reshape(x, -1, 1), cs.reshape(u, -1, 1),
            track_slack, cs.reshape(collision_slack, -1, 1),
        )
        parameters = cs.vertcat(x0, u_prev, curvature, opponent_s, opponent_e)
        all_constraints = cs.vertcat(*constraints)
        instance_id = ControlBestResponseSolver._instance_counter
        ControlBestResponseSolver._instance_counter += 1
        self.solver = cs.nlpsol(
            f"ibr_control_br_{instance_id}", "ipopt",
            {"f": objective, "x": variables, "g": all_constraints, "p": parameters},
            {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 500},
        )
        self.constraint_fn = cs.Function(
            f"ibr_control_constraints_{instance_id}",
            [variables, parameters], [all_constraints],
        )
        self.n_x = 4 * (n + 1)
        self.n_u = 2 * n
        self.n_track_slack = n
        self.n_collision_slack = m * n
        self.constraint_count = int(all_constraints.numel())
        self.constraint_offsets = {
            "friction": friction_start,
            "speed": speed_start,
            "track": track_start,
            "collision": collision_start,
            "steer_rate": steer_rate_start,
        }

    def solve_best_response(
        self, *, player_idx, x0, previous_control, warm_state_plan,
        warm_control_plan, opponent_plans, curvature,
    ):
        """Return one direct-control BR plan; raise on failed/nonfinite solve."""
        n, m = self.horizon, self.max_opponents
        self.last_diagnostics = {
            "success": False, "status": "not_solved", "solve_seconds": 0.0,
            "iterations": 0, "residual": None, "cost": None,
        }
        x0 = np.asarray(x0, dtype=float)
        previous_control = np.asarray(previous_control, dtype=float)
        state_guess = np.asarray(warm_state_plan, dtype=float).copy()
        control_guess = np.asarray(warm_control_plan, dtype=float).copy()
        opponent_plans = np.asarray(opponent_plans, dtype=float)
        curvature = np.asarray(curvature, dtype=float)
        if state_guess.shape != (4, n + 1) or control_guess.shape != (2, n):
            raise ValueError("control BR warm plans have the wrong shape")
        if opponent_plans.shape != (m, 2, n + 1):
            raise ValueError("control BR requires one [s,e] plan per opponent")
        if self.initial_guess_mode == "release":
            state_guess = np.repeat(x0[:, None], n + 1, axis=1)
            control_guess = np.zeros((2, n))
            track_slack_guess = np.zeros(n)
            collision_slack_guess = np.zeros((m, n))
        else:
            control_guess[0] = np.clip(control_guess[0], 0.0, 1.0)
            control_guess[1] = np.clip(control_guess[1], -1.0, 1.0)
            # Keep the cached controls, but rebuild states for this BR's fixed
            # curvature schedule, matching the main MPC warm-start pattern.
            state_guess[:, 0] = x0
            for k in range(n):
                state_guess[:, k + 1] = kinematic_frenet_step(
                    state_guess[:, k], control_guess[:, k], curvature[k],
                    self.dt, self.p, curvature_mode="geometric",
                )
            if not np.isfinite(state_guess).all():
                raise RuntimeError("IBR control warm-start rollout is nonfinite")
            track_slack_guess = np.maximum(
                np.abs(state_guess[1, 1:]) - self.track_half_width, 0.0
            )
            collision_slack_guess = np.zeros((m, n))
            for opponent_idx in range(m):
                ds = (
                    state_guess[0, 1:] - opponent_plans[opponent_idx, 0, 1:]
                ) / self.longitudinal_separation
                de = (
                    state_guess[1, 1:] - opponent_plans[opponent_idx, 1, 1:]
                ) / self.lateral_separation
                collision_slack_guess[opponent_idx] = np.maximum(
                    1.0 - ds ** self.collision_exponent
                    - de ** self.collision_exponent,
                    0.0,
                )
        initial = np.concatenate((
            state_guess.T.reshape(-1), control_guess.T.reshape(-1),
            track_slack_guess, collision_slack_guess.T.reshape(-1),
        ))
        parameters = np.concatenate((
            x0, previous_control, curvature,
            opponent_plans[:, 0, :].reshape(-1),
            opponent_plans[:, 1, :].reshape(-1),
        ))

        lower_g = np.zeros(self.constraint_count)
        upper_g = np.zeros(self.constraint_count)
        offsets = self.constraint_offsets
        friction_limit = self.mu * GRAVITY * (self.p.LF + self.p.LR)
        if self.enforce_friction:
            lower_g[offsets["friction"]:offsets["speed"]] = -friction_limit
            upper_g[offsets["friction"]:offsets["speed"]] = friction_limit
        else:
            lower_g[offsets["friction"]:offsets["speed"]] = -np.inf
            upper_g[offsets["friction"]:offsets["speed"]] = np.inf
        upper_g[offsets["speed"]:offsets["track"]] = self.max_speed
        lower_g[offsets["track"]:] = -np.inf
        upper_g[offsets["track"]:] = 0.0
        if not self.enforce_track:
            upper_g[offsets["track"]:offsets["collision"]] = np.inf
        if not self.enforce_collision:
            upper_g[offsets["collision"]:offsets["steer_rate"]] = np.inf
        lower_g[offsets["steer_rate"]:] = -self.max_steer_delta
        upper_g[offsets["steer_rate"]:] = self.max_steer_delta

        lower_x = np.full(initial.size, -np.inf)
        upper_x = np.full(initial.size, np.inf)
        for k in range(n):
            start = self.n_x + 2 * k
            lower_x[start:start + 2] = [0.0, -1.0]
            upper_x[start:start + 2] = [1.0, 1.0]
        lower_x[self.n_x + self.n_u:] = 0.0

        started = time.perf_counter()
        self.last_diagnostics = {
            "success": False, "status": "solver_exception", "solve_seconds": 0.0,
            "iterations": 0, "residual": None, "cost": None,
        }
        try:
            solution = self.solver(
                x0=initial, lbx=lower_x, ubx=upper_x,
                lbg=lower_g, ubg=upper_g, p=parameters,
            )
        except Exception as exc:
            self.last_diagnostics.update(
                solve_seconds=time.perf_counter() - started, error=str(exc),
            )
            raise RuntimeError(f"IBR control solver error: {exc}") from exc
        solve_seconds = time.perf_counter() - started
        stats = self.solver.stats()
        self.last_diagnostics.update(
            status=str(stats.get("return_status", "unknown")),
            solve_seconds=solve_seconds,
            iterations=int(stats.get("iter_count", 0)),
        )
        values = solution["x"].full().reshape(-1)
        if not np.isfinite(values).all():
            self.last_diagnostics["status"] = "nonfinite_solution"
            raise RuntimeError("IBR control solve returned nonfinite values")
        constraint_values = np.asarray(
            self.constraint_fn(values, parameters)
        ).reshape(-1)
        if not np.isfinite(constraint_values).all():
            self.last_diagnostics["status"] = "nonfinite_constraints"
            raise RuntimeError("IBR control solve returned nonfinite constraints")
        residual = float(max(
            np.max(np.maximum(lower_g - constraint_values, 0.0)),
            np.max(np.maximum(constraint_values - upper_g, 0.0)),
            np.max(np.maximum(lower_x - values, 0.0)),
            np.max(np.maximum(values - upper_x, 0.0)),
        ))
        solver_success = bool(stats.get("success", False))
        accepted_max_iter = (
            not solver_success
            and self.last_diagnostics["status"] == "Maximum_Iterations_Exceeded"
        )
        if not solver_success and not accepted_max_iter:
            self.last_diagnostics["residual"] = residual
            raise RuntimeError(self.last_diagnostics["status"])
        state_plan = values[:self.n_x].reshape(n + 1, 4).T
        control_plan = values[self.n_x:self.n_x + self.n_u].reshape(n, 2).T
        track_start_x = self.n_x + self.n_u
        collision_start_x = track_start_x + self.n_track_slack
        track_slack_plan = values[track_start_x:collision_start_x]
        collision_slack_plan = values[
            collision_start_x:collision_start_x + self.n_collision_slack
        ].reshape(n, m).T
        collision_values = constraint_values[
            offsets["collision"]:offsets["steer_rate"]
        ]
        diagnostics = {
            "success": True,
            "status": str(stats.get("return_status", "Solve_Succeeded")),
            "solver_success": solver_success,
            "accepted_max_iter": accepted_max_iter,
            "initial_guess_mode": self.initial_guess_mode,
            "solve_seconds": solve_seconds,
            "iterations": int(stats.get("iter_count", 0)),
            "residual": residual,
            "cost": float(solution["f"]),
            "max_collision_constraint": (
                float(np.max(collision_values)) if collision_values.size else np.nan
            ),
            "track_slack": track_slack_plan.copy(),
            "collision_slack": collision_slack_plan.copy(),
            "max_track_slack": float(np.max(track_slack_plan)),
            "max_collision_slack": float(np.max(collision_slack_plan)),
        }
        self.last_diagnostics = diagnostics
        return state_plan, control_plan, diagnostics


class IteratedBestResponsePolicy(MPCPolicy):
    """Run one private three-car IBR game and execute this car's first action."""

    ibr_level = "round_robin"

    def __init__(
        self, *args, strategy_space="control", rounds=3, ibr_horizon=10,
        initial_guess_mode="release",
        parameter_search_points=3, parameter_search_passes=1,
        convergence_tolerance=0.0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strategy_space = str(strategy_space).strip().lower()
        if self.strategy_space not in IBR_STRATEGY_SPACES:
            raise ValueError(
                f"Unknown IBR strategy space {strategy_space!r}; "
                f"choose one of {IBR_STRATEGY_SPACES}"
            )
        if self.mpc.horizon != int(ibr_horizon):
            raise ValueError("IBR requires the shared MPC horizon to match IBR_HORIZON")
        self.round_robin = RoundRobinIBR(rounds, convergence_tolerance)
        self.parameter_search_points = max(3, int(parameter_search_points))
        self.parameter_search_passes = max(1, int(parameter_search_passes))
        self.mode_presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        self.control_solver = (
            ControlBestResponseSolver(
                self.mpc, horizon=self.mpc.horizon, max_opponents=2,
                initial_guess_mode=initial_guess_mode,
            )
            if self.strategy_space == "control" else None
        )
        self.current_mode = f"ibr_{self.strategy_space}"
        self.last_ibr_rounds = []
        self.last_ibr_diagnostics = {}
        self.last_joint_state_plans = None
        self.last_joint_control_plans = None

    def _initial_joint_plans(self, observation):
        """Roll out shifted prior controls from each player's measured state.

        The first control-space tick has no prior plan, so it repeats each
        player's last executed action. Parameter and mode keep that initializer.
        """
        n = self.mpc.horizon
        state_plans, control_plans = [], []
        prior = self.last_joint_control_plans if self.strategy_space == "control" else None
        if prior is not None:
            assert prior.shape == (len(observation.cars), 2, n)
            if not np.isfinite(prior).all():
                raise RuntimeError("IBR warm-start controls contain nonfinite values")
        for player_idx, car in enumerate(observation.cars):
            state_plan = np.repeat(car.kinematic_state[:, None], n + 1, axis=1)
            previous = np.asarray(car.previous_action, dtype=float).copy()
            previous[0] = np.clip(previous[0], 0.0, 1.0)
            previous[1] = np.clip(previous[1], -1.0, 1.0)
            if prior is None:
                control_plan = np.repeat(previous[:, None], n, axis=1)
            else:
                control_plan = np.column_stack((prior[player_idx, :, 1:], prior[player_idx, :, -1]))
            for k in range(n):
                curvature = float(self.mpc.curvature_fn(float(state_plan[0, k] % self.track.total_length)))
                state_plan[:, k + 1] = kinematic_frenet_step(
                    state_plan[:, k], control_plan[:, k], curvature,
                    self.mpc.dt, self.mpc.p, curvature_mode="geometric",
                )
            state_plans.append(state_plan)
            control_plans.append(control_plan)
        return state_plans, control_plans

    def _curvature_schedule(self, state_plan):
        return np.asarray([
            float(self.mpc.curvature_fn(float(s % self.track.total_length)))
            for s in state_plan[0, :-1]
        ])

    def _paper_utility(self, observation, plans, player_idx):
        start = np.asarray([car.s for car in observation.cars], dtype=float)
        final = np.asarray([plan[0, -1] for plan in plans], dtype=float)
        return float(relative_progress_gain(start, final, player_idx))

    def _control_best_response(self, observation, player_idx, plans, controls):
        opponent_indices = [idx for idx in range(len(plans)) if idx != player_idx]
        state_plan, control_plan, diagnostics = self.control_solver.solve_best_response(
            player_idx=player_idx,
            x0=observation.car(player_idx).kinematic_state,
            previous_control=observation.car(player_idx).previous_action,
            warm_state_plan=plans[player_idx],
            warm_control_plan=controls[player_idx],
            opponent_plans=np.stack([plans[idx][:2] for idx in opponent_indices]),
            curvature=self._curvature_schedule(plans[player_idx]),
        )
        candidate_plans = list(plans)
        candidate_plans[player_idx] = state_plan
        return BestResponseResult(
            player_idx, state_plan, control_plan, control_plan[:, 0].copy(),
            self._paper_utility(observation, candidate_plans, player_idx),
            diagnostics=diagnostics,
        )

    def _opponent_records(self, observation, player_idx, plans):
        records = []
        for idx, car in enumerate(observation.cars):
            if idx == player_idx:
                continue
            records.append({
                "car_idx": idx, "px": car.s, "py": car.e,
                "s_dot": car.s_dot, "vx": car.s_dot, "vy": car.vy_body,
                "omega": car.omega, "psi": car.psi, "state": car.state,
                "px_plan": plans[idx][0].copy(), "py_plan": plans[idx][1].copy(),
            })
        return records

    def _evaluate_high_level_candidate(
        self, observation, player_idx, plans, controls, candidate_params,
    ):
        """Map one theta candidate through the existing reference builder and MPC."""
        car = observation.car(player_idx)
        opponents = self._opponent_records(observation, player_idx, plans)
        saved_params = self.params
        saved_control_plan = self.mpc.last_control_plan
        saved_state_plan = self.mpc.last_state_plan
        try:
            self.params = np.asarray(candidate_params, dtype=float).copy()
            self.mpc.last_control_plan = controls[player_idx].copy()
            self.mpc.last_state_plan = plans[player_idx].copy()
            reference = self.get_reference_trajectory(
                car.state, car.s, opponents=opponents, current_e=car.e
            )
            _, action = self.mpc.solve(
                car.state.to_array(), reference, car.previous_action,
                q_weight=float(self.params[0]), opponents=opponents,
                current_s=car.s, frenet_state=car.kinematic_state,
            )
            state_plan = np.asarray(self.mpc.last_state_plan, dtype=float).copy()
            control_plan = np.asarray(self.mpc.last_control_plan, dtype=float).copy()
            diagnostics = dict(self.mpc.last_diagnostics)
            diagnostics["track_slack"] = np.asarray(
                self.mpc.last_track_slack_plan, dtype=float
            ).copy()
            diagnostics["collision_slack"] = np.asarray(
                self.mpc.last_collision_slack_plan, dtype=float
            ).copy()
        finally:
            self.params = saved_params
            self.mpc.last_control_plan = saved_control_plan
            self.mpc.last_state_plan = saved_state_plan
        candidate_plans = list(plans)
        candidate_plans[player_idx] = np.asarray(state_plan, dtype=float)
        utility = self._paper_utility(observation, candidate_plans, player_idx)
        return np.asarray(state_plan), control_plan, np.asarray(action), utility, diagnostics

    def _parameter_candidates(self, center, search_pass, dimension):
        bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
        radius = (bounds[dimension, 1] - bounds[dimension, 0]) * 0.5 ** (search_pass + 1)
        values = center[dimension] + np.linspace(-radius, radius, self.parameter_search_points)
        return np.unique(np.clip(
            np.r_[values, bounds[dimension, 0], center[dimension], bounds[dimension, 1]],
            bounds[dimension, 0], bounds[dimension, 1],
        ))

    def _high_level_best_response(self, observation, player_idx, plans, controls, modeled):
        """Solve one continuous-parameter or discrete-mode best response."""
        candidate_count = 0
        failed_count = 0
        last_error = None
        last_failure = None
        best = None

        def consider(params, mode=None):
            nonlocal best, candidate_count, failed_count, last_error, last_failure
            candidate_count += 1
            prior_diagnostics = self.mpc.last_diagnostics
            try:
                result = self._evaluate_high_level_candidate(
                    observation, player_idx, plans, controls, params
                )
            except Exception as exc:
                failed_count += 1
                last_error = str(exc)
                diagnostics = self.mpc.last_diagnostics
                last_failure = (
                    dict(diagnostics)
                    if diagnostics is not prior_diagnostics and not diagnostics.get("success")
                    else {}
                )
                return
            state_plan, control_plan, action, utility, diagnostics = result
            if best is None or utility > best.utility:
                best = BestResponseResult(
                    player_idx, state_plan, control_plan, action, utility,
                    params=np.asarray(params, dtype=float).copy(), mode=mode,
                    diagnostics=diagnostics,
                )

        if self.strategy_space == "mode":
            for mode, params in self.mode_presets.items():
                consider(params, mode=mode)
        else:
            center = np.asarray(modeled[player_idx]["params"], dtype=float).copy()
            consider(center)
            for search_pass in range(self.parameter_search_passes):
                for dimension in range(5):
                    previous_best = best
                    for value in self._parameter_candidates(center, search_pass, dimension):
                        candidate = center.copy()
                        candidate[dimension] = value
                        consider(candidate)
                    if best is not None and best is not previous_best:
                        center = best.params.copy()

        if best is None:
            self._candidate_failure = {
                **(last_failure or {}), "success": False,
                "status": (last_failure or {}).get("status", "all_candidates_failed"),
                "candidate_count": candidate_count,
                "failed_candidate_count": failed_count,
                "last_candidate_error": last_error,
            }
            raise RuntimeError("all IBR high-level candidates failed")
        best.diagnostics = {
            **best.diagnostics,
            "candidate_count": candidate_count,
            "failed_candidate_count": failed_count,
        }
        modeled[player_idx] = {"params": best.params.copy(), "mode": best.mode}
        return best

    def _modeled_high_level_state(self, observation):
        bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
        modeled = []
        for car in observation.cars:
            params = car.policy_params
            if params is None or np.shape(params) != (5,):
                params = sim_params.ALPHARACER_INITIAL_PARAMS
            params = np.clip(np.asarray(params, dtype=float), bounds[:, 0], bounds[:, 1])
            mode = min(
                self.mode_presets,
                key=lambda name: float(np.linalg.norm(self.mode_presets[name] - params)),
            )
            modeled.append({"params": params.copy(), "mode": mode})
        return modeled

    def get_action(self, observation) -> CarAction:
        """Run all IBR rounds from the frozen scene and return this car's action."""
        if self.car_idx is None:
            raise ValueError("IBRPolicy requires car_idx")
        if len(observation.cars) != 3:
            raise ValueError("IBRPolicy currently requires exactly three cars")
        initial_plans, initial_controls = self._initial_joint_plans(observation)
        modeled = self._modeled_high_level_state(observation)

        def solve_player(player_idx, plans, controls):
            if self.strategy_space == "control":
                return self._control_best_response(observation, player_idx, plans, controls)
            return self._high_level_best_response(
                observation, player_idx, plans, controls, modeled
            )

        self._candidate_failure = None
        self.round_robin.current_round = None
        self.round_robin.current_player = None
        try:
            result = self.round_robin.solve(
                initial_plans, initial_controls, self.car_idx, solve_player
            )
        except RuntimeError as exc:
            diagnostics = (
                self.control_solver.last_diagnostics
                if self.strategy_space == "control" else self._candidate_failure or {}
            )
            status = str(diagnostics.get("status", type(exc).__name__))
            self.mpc.last_solver_success = False
            self.mpc.last_solver_status = status
            self.mpc.last_diagnostics = {
                **diagnostics, "success": False, "status": status, "error": str(exc),
                "ibr_round": self.round_robin.current_round,
                "ibr_player_idx": self.round_robin.current_player,
                "ibr_strategy_space": self.strategy_space,
            }
            self.last_ibr_diagnostics = dict(self.mpc.last_diagnostics)
            self.last_ibr_rounds = []
            self.last_joint_state_plans = None
            self.last_joint_control_plans = None
            self.predicted_frenet_plan = None
            for name in ("last_state_plan", "last_control_plan",
                         "last_track_slack_plan", "last_collision_slack_plan"):
                setattr(self.mpc, name, None)
            raise
        own = result.player_results[self.car_idx]
        if own.params is not None:
            self.params = own.params.copy()
        self.current_mode = own.mode or f"ibr_{self.strategy_space}"
        self.predicted_frenet_plan = own.state_plan[:2].copy()
        self.last_joint_state_plans = np.stack(result.plans)
        self.last_joint_control_plans = np.stack(result.controls)
        self.last_ibr_rounds = result.round_diagnostics
        self.last_ibr_diagnostics = dict(own.diagnostics)
        self.last_ibr_diagnostics.update({
            "strategy_space": self.strategy_space,
            "rounds_completed": len(result.round_diagnostics),
            "paper_utility": own.utility,
            "selected_params": None if own.params is None else own.params.copy(),
            "selected_mode": own.mode,
            "accepted_max_iter_count": sum(
                item["accepted_max_iter_count"] for item in result.round_diagnostics
            ),
            "max_br_residual": max(
                item["max_residual"] for item in result.round_diagnostics
            ),
        })

        action = np.asarray(own.action, dtype=float)
        action[0] = np.clip(action[0], 0.0, 1.0)
        action[1] = np.clip(action[1], -1.0, 1.0)
        self.last_requested_action = action.copy()
        self.last_u = action.copy()
        self.mpc.last_solver_success = True
        self.mpc.last_solver_status = str(own.diagnostics.get("status", "Solve_Succeeded"))
        self.mpc.last_diagnostics = {
            **{key: value for key, value in own.diagnostics.items()
               if key not in ("track_slack", "collision_slack")},
            "success": True,
            "status": self.mpc.last_solver_status,
            "slack": float(np.max(own.diagnostics["track_slack"])),
            "collision_slack": float(np.max(own.diagnostics["collision_slack"])),
        }
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        self.mpc.last_loss = float(own.diagnostics.get("cost", np.nan))
        self.mpc.last_track_loss = np.nan
        self.mpc.last_action_loss = np.nan
        self.mpc.last_velocity_loss = np.nan
        self.mpc.last_heading_loss = np.nan
        track_slack = np.asarray(
            own.diagnostics.get("track_slack", np.zeros(self.mpc.horizon)),
            dtype=float,
        ).copy()
        collision_slack = np.asarray(
            own.diagnostics.get(
                "collision_slack",
                np.zeros((self.mpc.max_constrained_opponents, self.mpc.horizon)),
            ),
            dtype=float,
        ).copy()
        self.mpc.last_track_slack_loss = float(
            self.mpc.track_slack_weight * np.sum(track_slack ** 2)
        )
        self.mpc.last_collision_slack_loss = float(
            self.mpc.collision_slack_weight * np.sum(collision_slack ** 2)
        )
        self.mpc.last_violation_loss = (
            self.mpc.last_track_slack_loss + self.mpc.last_collision_slack_loss
        )
        self.mpc.last_state_plan = own.state_plan.copy()
        self.mpc.last_control_plan = own.control_plan.copy()
        self.mpc.last_track_slack_plan = track_slack
        self.mpc.last_collision_slack_plan = collision_slack
        return CarAction(throttle=float(action[0]), steer=float(action[1]))
