"""Nearest-opponent iterative best response for three-car races.

Each policy models only its closest opponent. The shared alternating scheduler
supports direct controls, five continuous parameters, and discrete modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import casadi as cs
import numpy as np

from config import GRAVITY, params as sim_params
from state import CarAction
from vehicle_model import kinematic_frenet_step, neutral_throttle

from ..alpharacer.network import ALPHA_PARAMETER_BOUNDS
from ..mpc import MPCPolicy
from ..utils import progress_delta, relative_progress_gain


IBR_STRATEGY_SPACES = ("control", "parameter", "mode")


@dataclass(frozen=True)
class ControlBestResponseSettings:
    """Runtime MPC values used by the direct-control BR problem."""

    dynamics_params: object
    dt: float
    horizon: int
    max_opponents: int
    initial_guess_mode: str
    track_half_width: float
    longitudinal_separation: float
    lateral_separation: float
    collision_exponent: int
    max_speed: float
    max_steer_delta: float
    friction_coefficient: float
    track_slack_weight: float
    collision_slack_weight: float
    enforce_track: bool
    enforce_collision: bool
    enforce_friction: bool

    @classmethod
    def from_mpc(
        cls, mpc, *, max_opponents, initial_guess_mode="neutral",
    ):
        """Copy the effective per-run values from an MPCController."""
        return cls(
            dynamics_params=mpc.p,
            dt=float(mpc.dt),
            horizon=int(mpc.horizon),
            max_opponents=int(max_opponents),
            initial_guess_mode=str(initial_guess_mode),
            track_half_width=float(mpc.track_constraint_half_width),
            longitudinal_separation=float(mpc.min_longitudinal_separation),
            lateral_separation=float(mpc.min_lateral_separation),
            collision_exponent=int(mpc.collision_superellipse_exponent),
            max_speed=float(mpc.max_speed),
            max_steer_delta=float(mpc.max_steer_delta),
            friction_coefficient=float(mpc.friction_coefficient),
            track_slack_weight=float(mpc.track_slack_weight),
            collision_slack_weight=float(mpc.collision_slack_weight),
            enforce_track=bool(mpc.enforce_track_boundary),
            enforce_collision=bool(mpc.enforce_collision_avoidance),
            enforce_friction=bool(mpc.enforce_friction_speed_limit),
        )


class ControlBestResponseSolver:
    """Solve one direct-control best response with CasADi.

    Methods:
    - ``solve_best_response``: return one direct-control BR plan; raise on
      failed/nonfinite solve.
    - ``_make_initial_guess``: return a warm-start guess for the NLP optimizer.

    Attributes:
    - ``settings``: fixed runtime values copied from the shared MPC controller.
    - ``solver``: the reusable CasADi NLP solver instance.
    """

    _instance_counter = 0

    def __init__(self, settings):
        self.settings = settings
        self.p = settings.dynamics_params
        self.dt = settings.dt
        self.horizon = settings.horizon
        self.max_opponents = settings.max_opponents
        self.initial_guess_mode = settings.initial_guess_mode.strip().lower()
        if self.initial_guess_mode not in {"release", "rollout", "neutral"}:
            raise ValueError(
                "IBR initial guess must be 'release', 'rollout', or 'neutral'"
            )
        self.track_half_width = settings.track_half_width
        self.longitudinal_separation = settings.longitudinal_separation
        self.lateral_separation = settings.lateral_separation
        self.collision_exponent = settings.collision_exponent
        self.max_speed = settings.max_speed
        self.max_steer_delta = settings.max_steer_delta
        self.mu = settings.friction_coefficient
        self.track_slack_weight = settings.track_slack_weight
        self.collision_slack_weight = settings.collision_slack_weight
        self.enforce_track = settings.enforce_track
        self.enforce_collision = settings.enforce_collision
        self.enforce_friction = settings.enforce_friction
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

    def _make_initial_guess(
        self, x0, previous_control, warm_state_plan, warm_control_plan,
        opponent_plans, curvature,
    ):
        """Build state, control, track-slack, and collision-slack guesses.

        ``release`` repeats the measured state and sets every control/slack to
        zero. ``rollout`` propagates shifted prior controls. ``neutral`` first
        makes a calm speed-holding control seed, then propagates it. Rollout
        and neutral initialize slack from their resulting state trajectory.
        """
        n, m = self.horizon, self.max_opponents
        state_guess = np.asarray(warm_state_plan, dtype=float).copy()
        control_guess = np.asarray(warm_control_plan, dtype=float).copy()
        if self.initial_guess_mode == "release":
            state_guess = np.repeat(x0[:, None], n + 1, axis=1)
            control_guess = np.zeros((2, n))
            track_slack_guess = np.zeros(n)
            collision_slack_guess = np.zeros((m, n))
        else:
            if self.initial_guess_mode == "rollout":
                control_guess[0] = np.clip(control_guess[0], 0.0, 1.0)
                control_guess[1] = np.clip(control_guess[1], -1.0, 1.0)
            else:
                # Start from a calm motion seed: roughly hold speed while
                # steering returns to zero within the existing rate limit.
                steer = float(np.clip(previous_control[1], -1.0, 1.0))
                for k in range(n):
                    steer = float(
                        np.sign(steer)
                        * max(abs(steer) - self.max_steer_delta, 0.0)
                    )
                    control_guess[:, k] = (0.0, steer)

            # Rebuild states for this BR's fixed curvature schedule. This makes
            # the dynamics part of the initial point consistent with the NLP.
            state_guess[:, 0] = x0
            for k in range(n):
                if self.initial_guess_mode == "neutral":
                    control_guess[0, k] = np.clip(
                        neutral_throttle(state_guess[3, k], self.p), 0.0, 1.0
                    )
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
        return (
            state_guess, control_guess, track_slack_guess,
            collision_slack_guess,
        )

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
        warm_state_plan = np.asarray(warm_state_plan, dtype=float)
        warm_control_plan = np.asarray(warm_control_plan, dtype=float)
        opponent_plans = np.asarray(opponent_plans, dtype=float)
        curvature = np.asarray(curvature, dtype=float)
        if warm_state_plan.shape != (4, n + 1) or warm_control_plan.shape != (2, n):
            raise ValueError("control BR warm plans have the wrong shape")
        if opponent_plans.shape != (m, 2, n + 1):
            raise ValueError("control BR requires one [s,e] plan per opponent")
        (
            state_guess, control_guess, track_slack_guess,
            collision_slack_guess,
        ) = self._make_initial_guess(
            x0, previous_control, warm_state_plan, warm_control_plan,
            opponent_plans, curvature,
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
    """Run a nearest-opponent two-car IBR and execute this car's first action."""

    ibr_level = "nearest_pair"

    def __init__(
        self, *args, strategy_space="control", n_rounds=6, ibr_horizon=20,
        initial_guess_mode="neutral", **kwargs,
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
        self.n_rounds = int(n_rounds)
        if self.n_rounds <= 0:
            raise ValueError("IBR n_rounds must be positive")
        self.current_ibr_round = None
        self.current_ibr_player = None
        self.mode_presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        self.control_solver = (
            ControlBestResponseSolver(
                ControlBestResponseSettings.from_mpc(
                    self.mpc,
                    max_opponents=1,
                    initial_guess_mode=initial_guess_mode,
                )
            )
            if self.strategy_space == "control" else None
        )
        self.current_mode = f"ibr_{self.strategy_space}"
        self.last_ibr_rounds = []
        self.last_ibr_diagnostics = {}
        self.last_joint_state_plans = None
        self.last_joint_control_plans = None

    def _initialize_ibr(self, observation):
        """Build initial plans and optional high-level strategy state.

        Control plans shift the previous tick when available. Parameter and
        mode also initialize each player's five parameters and nearest
        named preset. Control-space IBR does not need that strategy state.
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
        strategy_states = None
        if self.strategy_space != "control":
            bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
            strategy_states = []
            for car in observation.cars:
                params = car.policy_params
                if params is None or np.shape(params) != (5,):
                    params = sim_params.ALPHARACER_INITIAL_PARAMS
                params = np.clip(
                    np.asarray(params, dtype=float), bounds[:, 0], bounds[:, 1]
                )
                mode = min(
                    self.mode_presets,
                    key=lambda name: float(
                        np.linalg.norm(self.mode_presets[name] - params)
                    ),
                )
                strategy_states.append({"params": params.copy(), "mode": mode})
        return state_plans, control_plans, strategy_states

    def _nearest_opponent_idx(self, observation):
        """Select the other car with the smallest periodic longitudinal gap."""
        ego_s = observation.car(self.car_idx).s
        return min(
            (idx for idx in range(len(observation.cars)) if idx != self.car_idx),
            key=lambda idx: abs(progress_delta(
                observation.car(idx).s, ego_s, self.track.total_length
            )),
        )

    def _utility(self, observation, plans, player_idx, opponent_idx):
        pair = (player_idx, opponent_idx)
        start = np.asarray([observation.car(idx).s for idx in pair], dtype=float)
        final = np.asarray([plans[idx][0, -1] for idx in pair], dtype=float)
        return float(relative_progress_gain(start, final, 0))

    def _solve_control_best_response(
        self, observation, player_idx, opponent_idx, plans, controls,
    ):
        curvature = np.asarray([
            float(self.mpc.curvature_fn(float(s % self.track.total_length)))
            for s in plans[player_idx][0, :-1]
        ])
        state_plan, control_plan, diagnostics = self.control_solver.solve_best_response(
            player_idx=player_idx,
            x0=observation.car(player_idx).kinematic_state,
            previous_control=observation.car(player_idx).previous_action,
            warm_state_plan=plans[player_idx],
            warm_control_plan=controls[player_idx],
            opponent_plans=np.stack([plans[opponent_idx][:2]]),
            curvature=curvature,
        )
        candidate_plans = list(plans)
        candidate_plans[player_idx] = state_plan
        return {
            "player_idx": player_idx,
            "state_plan": state_plan,
            "control_plan": control_plan,
            "action": control_plan[:, 0].copy(),
            "utility": self._utility(
                observation, candidate_plans, player_idx, opponent_idx
            ),
            "params": None,
            "mode": None,
            "diagnostics": diagnostics,
        }

    def _evaluate_high_level_candidate(
        self, observation, player_idx, opponent_idx, plans, controls,
        candidate_params,
    ):
        """Map one theta candidate through the existing reference builder and MPC."""
        car = observation.car(player_idx)
        opponent = observation.car(opponent_idx)
        opponents = [{
            "car_idx": opponent_idx, "px": opponent.s, "py": opponent.e,
            "s_dot": opponent.s_dot, "vx": opponent.s_dot,
            "vy": opponent.vy_body, "omega": opponent.omega,
            "psi": opponent.psi, "state": opponent.state,
            "px_plan": plans[opponent_idx][0].copy(),
            "py_plan": plans[opponent_idx][1].copy(),
        }]
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
        utility = self._utility(
            observation, candidate_plans, player_idx, opponent_idx
        )
        return np.asarray(state_plan), control_plan, np.asarray(action), utility, diagnostics

    def _solve_high_level_best_response(
        self, observation, player_idx, opponent_idx, plans, controls,
        strategy_states,
    ):
        """Solve one finite-difference parameter or discrete-mode best response."""
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
                    observation, player_idx, opponent_idx, plans, controls, params
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
                return None
            state_plan, control_plan, action, utility, diagnostics = result
            evaluated = {
                "player_idx": player_idx,
                "state_plan": state_plan,
                "control_plan": control_plan,
                "action": action,
                "utility": utility,
                "params": np.asarray(params, dtype=float).copy(),
                "mode": mode,
                "diagnostics": diagnostics,
            }
            if best is None or utility > best["utility"]:
                best = evaluated
            return evaluated

        if self.strategy_space == "mode":
            for mode, params in self.mode_presets.items():
                consider(params, mode=mode)
        else:
            bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
            span = bounds[:, 1] - bounds[:, 0]
            center = np.clip(
                np.asarray(strategy_states[player_idx]["params"], dtype=float),
                bounds[:, 0], bounds[:, 1],
            )
            normalized = (center - bounds[:, 0]) / span
            center_result = consider(center)
            gradient = np.zeros(5)
            difference_step = float(sim_params.IBR_PARAMETER_DIFFERENCE_STEP)
            for dimension in range(5):
                probe = normalized.copy()
                delta = (
                    difference_step
                    if normalized[dimension] <= 1.0 - difference_step
                    else -difference_step
                )
                probe[dimension] += delta
                probe_result = consider(bounds[:, 0] + probe * span)
                if center_result is not None and probe_result is not None:
                    gradient[dimension] = (
                        probe_result["utility"] - center_result["utility"]
                    ) / delta

            scale = float(np.max(np.abs(gradient)))
            if scale > 0.0 and np.isfinite(scale):
                updated = np.clip(
                    normalized
                    + float(sim_params.IBR_PARAMETER_GRADIENT_STEP) * gradient / scale,
                    0.0, 1.0,
                )
                consider(bounds[:, 0] + updated * span)

        if best is None:
            self._candidate_failure = {
                **(last_failure or {}), "success": False,
                "status": (last_failure or {}).get("status", "all_candidates_failed"),
                "candidate_count": candidate_count,
                "failed_candidate_count": failed_count,
                "last_candidate_error": last_error,
            }
            raise RuntimeError("all IBR high-level candidates failed")
        best["diagnostics"] = {
            **best["diagnostics"],
            "candidate_count": candidate_count,
            "failed_candidate_count": failed_count,
            "parameter_optimizer": (
                "finite_difference" if self.strategy_space == "parameter" else None
            ),
        }
        strategy_states[player_idx] = {
            "params": best["params"].copy(), "mode": best["mode"]
        }
        return best

    def _solve_best_response(
        self, observation, player_idx, opponent_idx, plans, controls,
        strategy_states,
    ):
        """Solve one player's BR in the selected strategy space."""
        if self.strategy_space == "control":
            return self._solve_control_best_response(
                observation, player_idx, opponent_idx, plans, controls
            )
        return self._solve_high_level_best_response(
            observation, player_idx, opponent_idx, plans, controls,
            strategy_states,
        )

    def _run_ibr_rounds(
        self, observation, initial_plans, initial_controls, strategy_states,
        opponent_idx, n_rounds,
    ):
        """Run complete rounds that update the opponent and then the ego.

        Inputs are the frozen observation and initial plans for all cars. Only
        the selected pair is updated. Output is final joint plans, the ego's
        last BR result, and one diagnostic entry per completed round.
        """
        n_rounds = int(n_rounds)
        assert n_rounds > 0
        ego_idx = int(self.car_idx)
        opponent_idx = int(opponent_idx)
        assert ego_idx != opponent_idx
        assert 0 <= ego_idx < len(initial_plans)
        assert 0 <= opponent_idx < len(initial_plans)
        assert len(initial_plans) == len(initial_controls)

        plans = [np.asarray(plan, dtype=float).copy() for plan in initial_plans]
        controls = [
            np.asarray(control, dtype=float).copy() for control in initial_controls
        ]
        latest = {}
        history = []
        for round_idx in range(1, n_rounds + 1):
            round_diagnostics = {
                "round": round_idx,
                "update_order": (opponent_idx, ego_idx),
                "solve_seconds": 0.0,
                "max_plan_change": 0.0,
                "utilities": {},
                "accepted_max_iter_count": 0,
                "max_residual": 0.0,
            }
            for player_idx, other_idx in (
                (opponent_idx, ego_idx), (ego_idx, opponent_idx)
            ):
                self.current_ibr_round = round_idx
                self.current_ibr_player = player_idx
                before = plans[player_idx].copy()
                started = time.perf_counter()
                result = self._solve_best_response(
                    observation, player_idx, other_idx, plans, controls,
                    strategy_states,
                )
                round_diagnostics["solve_seconds"] += (
                    time.perf_counter() - started
                )
                if result["player_idx"] != player_idx:
                    raise ValueError("best-response result belongs to the wrong player")
                plans[player_idx] = np.asarray(
                    result["state_plan"], dtype=float
                ).copy()
                controls[player_idx] = np.asarray(
                    result["control_plan"], dtype=float
                ).copy()
                latest[player_idx] = result
                diagnostics = result["diagnostics"]
                residual = diagnostics.get("residual")
                round_diagnostics["max_plan_change"] = max(
                    round_diagnostics["max_plan_change"],
                    float(np.max(np.abs(plans[player_idx][:2] - before[:2]))),
                )
                round_diagnostics["utilities"][player_idx] = float(
                    result["utility"]
                )
                round_diagnostics["accepted_max_iter_count"] += int(
                    bool(diagnostics.get("accepted_max_iter", False))
                )
                if residual is not None and np.isfinite(residual):
                    round_diagnostics["max_residual"] = max(
                        round_diagnostics["max_residual"], float(residual)
                    )
            history.append(round_diagnostics)
        return plans, controls, latest[ego_idx], history

    def _record_ibr_failure(self, exc):
        """Expose the failed BR and clear plans that must not be reused."""
        diagnostics = (
            self.control_solver.last_diagnostics
            if self.strategy_space == "control" else self._candidate_failure or {}
        )
        status = str(diagnostics.get("status", type(exc).__name__))
        self.mpc.last_solver_success = False
        self.mpc.last_solver_status = status
        self.mpc.last_diagnostics = {
            **diagnostics, "success": False, "status": status, "error": str(exc),
            "ibr_round": self.current_ibr_round,
            "ibr_player_idx": self.current_ibr_player,
            "ibr_strategy_space": self.strategy_space,
        }
        self.last_ibr_diagnostics = dict(self.mpc.last_diagnostics)
        self.last_ibr_rounds = []
        self.last_joint_state_plans = None
        self.last_joint_control_plans = None
        self.predicted_frenet_plan = None
        for name in (
            "last_state_plan", "last_control_plan",
            "last_track_slack_plan", "last_collision_slack_plan",
        ):
            setattr(self.mpc, name, None)

    def _accept_ibr_result(
        self, plans, controls, own, history, opponent_idx,
    ):
        """Save one completed IBR decision and return its first ego action."""
        if own["params"] is not None:
            self.params = own["params"].copy()
        self.current_mode = own["mode"] or f"ibr_{self.strategy_space}"
        self.predicted_frenet_plan = own["state_plan"][:2].copy()
        self.last_joint_state_plans = np.stack(plans)
        self.last_joint_control_plans = np.stack(controls)
        self.last_ibr_rounds = history
        self.last_ibr_diagnostics = dict(own["diagnostics"])
        self.last_ibr_diagnostics.update({
            "strategy_space": self.strategy_space,
            "rounds_completed": len(history),
            "br_solve_count": 2 * len(history),
            "selected_opponent_idx": opponent_idx,
            "utility": own["utility"],
            "selected_params": (
                None if own["params"] is None else own["params"].copy()
            ),
            "selected_mode": own["mode"],
            "accepted_max_iter_count": sum(
                item["accepted_max_iter_count"] for item in history
            ),
            "max_br_residual": max(
                item["max_residual"] for item in history
            ),
        })

        action = np.asarray(own["action"], dtype=float)
        action[0] = np.clip(action[0], 0.0, 1.0)
        action[1] = np.clip(action[1], -1.0, 1.0)
        self.last_requested_action = action.copy()
        self.last_u = action.copy()
        self.mpc.last_solver_success = True
        self.mpc.last_solver_status = str(
            own["diagnostics"].get("status", "Solve_Succeeded")
        )
        self.mpc.last_diagnostics = {
            **{key: value for key, value in own["diagnostics"].items()
               if key not in ("track_slack", "collision_slack")},
            "success": True,
            "status": self.mpc.last_solver_status,
            "slack": float(np.max(own["diagnostics"]["track_slack"])),
            "collision_slack": float(
                np.max(own["diagnostics"]["collision_slack"])
            ),
        }
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        self.mpc.last_loss = float(own["diagnostics"].get("cost", np.nan))
        self.mpc.last_track_loss = np.nan
        self.mpc.last_action_loss = np.nan
        self.mpc.last_velocity_loss = np.nan
        self.mpc.last_heading_loss = np.nan
        track_slack = np.asarray(
            own["diagnostics"].get("track_slack", np.zeros(self.mpc.horizon)),
            dtype=float,
        ).copy()
        collision_slack = np.asarray(
            own["diagnostics"].get(
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
        self.mpc.last_state_plan = own["state_plan"].copy()
        self.mpc.last_control_plan = own["control_plan"].copy()
        self.mpc.last_track_slack_plan = track_slack
        self.mpc.last_collision_slack_plan = collision_slack
        return CarAction(throttle=float(action[0]), steer=float(action[1]))

    def get_action(self, observation) -> CarAction:
        """Initialize, select the nearest opponent, and run six IBR rounds."""
        if self.car_idx is None:
            raise ValueError("IBRPolicy requires car_idx")
        if len(observation.cars) != 3:
            raise ValueError("IBRPolicy currently requires exactly three cars")

        plans, controls, strategy_states = self._initialize_ibr(observation)
        opponent_idx = self._nearest_opponent_idx(observation)

        self._candidate_failure = None
        self.current_ibr_round = None
        self.current_ibr_player = None
        try:
            plans, controls, own, history = self._run_ibr_rounds(
                observation, plans, controls, strategy_states, opponent_idx,
                self.n_rounds,
            )
        except RuntimeError as exc:
            self._record_ibr_failure(exc)
            raise
        return self._accept_ibr_result(
            plans, controls, own, history, opponent_idx
        )
