"""Five-parameter IBR evaluated through the shared alpha-RACER MPC."""

import numpy as np

from config import params as sim_params
from ..alpharacer.network import ALPHA_PARAMETER_BOUNDS
from ..mpc import MPCPolicy


class ParameterIBRPolicy(MPCPolicy):
    """Iterated best response in the paper's five-parameter policy space.

    Unlike :class:`TrajectoryIBRPolicy`, this policy never optimizes throttle
    or steering directly.  A best-response step searches
    ``theta=(q, zeta, s1, s2, s3)`` and evaluates every candidate by solving
    the same raceline-reference MPC used by alpha-RACER and the LLM policies.
    Candidates are compared with the same local racing objective as trajectory
    IBR: progress, centerline-relative lateral error, control-rate regularization,
    reference-speed tracking, and the MPC's existing track/collision slack
    costs. The positional MPC tracking term remains excluded so candidates
    with different ``q`` remain comparable.
    """

    ibr_level = "parameter"
    PARAM_BOUNDS = ALPHA_PARAMETER_BOUNDS

    def __init__(
        self,
        *args,
        search_points=5,
        search_passes=2,
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
        self.search_points = max(3, int(search_points))
        self.search_passes = max(1, int(search_passes))
        self.update_interval_seconds = max(float(update_interval_seconds), 0.0)
        self.convergence_tolerance = max(float(convergence_tolerance), 0.0)
        self.last_parameter_ibr_update_time = None
        self.last_parameter_ibr_score = -np.inf
        self.last_parameter_ibr_diagnostics = {}
        self.predicted_frenet_plan = None
        self.current_mode = "parameter_ibr"

    def update_due(self, sim_time):
        if sim_time is None or self.last_parameter_ibr_update_time is None:
            return True
        return (
            sim_time
            >= self.last_parameter_ibr_update_time
            + self.update_interval_seconds
            - 1.0e-9
        )

    def _evaluate_candidate(self, candidate, state, current_s, opponents):
        """Solve MPC once and evaluate the shared local IBR race objective."""
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
        state_plan = np.asarray(self.mpc.last_state_plan, dtype=float)
        frenet_plan = state_plan[:2].copy()
        control_plan = np.asarray(self.mpc.last_control_plan, dtype=float).copy()
        predicted_progress = float(frenet_plan[0, -1] - frenet_plan[0, 0])
        lateral_cost = sim_params.IBR_LATERAL_WEIGHT * float(
            np.sum(
                np.power(
                    frenet_plan[1, :-1],
                    sim_params.IBR_LATERAL_EXPONENT,
                )
            )
        )
        previous_and_plan = np.column_stack((self.last_feasible_u, control_plan))
        control_rate_cost = sim_params.IBR_CONTROL_RATE_WEIGHT * float(
            np.sum(np.square(np.diff(previous_and_plan, axis=1)))
        )
        velocity_cost = self.mpc.velocity_track_weight * float(
            np.sum(np.square(state_plan[3, :-1] - ref_traj[2, :-1]))
        )
        track_slack_cost = self.mpc.track_slack_weight * float(
            np.sum(np.square(self.mpc.last_track_slack_plan))
        )
        collision_slack_cost = self.mpc.collision_slack_weight * float(
            np.sum(np.square(self.mpc.last_collision_slack_plan))
        )
        progress_reward = sim_params.IBR_PROGRESS_WEIGHT * predicted_progress
        # best_response maximizes score, hence score is the negative of the
        # minimization objective used by trajectory IBR.
        score = (
            progress_reward
            - lateral_cost
            - velocity_cost
            - control_rate_cost
            - track_slack_cost
            - collision_slack_cost
        )
        return score, frenet_plan, {
            "predicted_progress": predicted_progress,
            "progress_reward": progress_reward,
            "lateral_cost": lateral_cost,
            "velocity_cost": velocity_cost,
            "control_rate_cost": control_rate_cost,
            "track_slack_cost": track_slack_cost,
            "collision_slack_cost": collision_slack_cost,
        }

    def best_response(self, state, current_s, opponents):
        """Approximate one best response with bounded coarse-to-fine search.

        Each sweep fixes four canonical parameters and searches the fifth in
        a neighborhood of the current warm start.  The neighborhood initially
        spans half of that parameter's allowed range and is halved after every
        sweep.  Explicit lower/upper-bound candidates keep the local search
        able to discover solutions near either end of the canonical range.

        This is still a coordinate best response, but unlike the earlier
        min/mid/max grid it can return progressively finer parameter values.
        """
        bounds = np.asarray(self.PARAM_BOUNDS, dtype=float)
        spans = bounds[:, 1] - bounds[:, 0]
        best = np.clip(np.asarray(self.params, dtype=float), bounds[:, 0], bounds[:, 1])
        best_score = -np.inf
        best_plan = None
        best_diagnostics = {}
        candidate_evaluations = 0
        failed_evaluations = 0

        # Evaluate the warm start first, then update one parameter at a time.
        try:
            best_score, best_plan, best_diagnostics = self._evaluate_candidate(
                best, state, current_s, opponents
            )
        except Exception:
            pass

        for search_pass in range(self.search_passes):
            radius_fraction = 0.5 ** (search_pass + 1)
            for dim in range(5):
                radius = radius_fraction * spans[dim]
                local_values = best[dim] + np.linspace(
                    -radius, radius, self.search_points
                )
                values = np.unique(np.clip(np.concatenate((
                    local_values,
                    np.array([bounds[dim, 0], best[dim], bounds[dim, 1]]),
                )), bounds[dim, 0], bounds[dim, 1]))
                dimension_best = best.copy()
                dimension_score = best_score
                dimension_plan = best_plan
                dimension_diagnostics = best_diagnostics
                for value in values:
                    candidate = best.copy()
                    candidate[dim] = value
                    candidate_evaluations += 1
                    try:
                        score, plan, diagnostics = self._evaluate_candidate(
                            candidate, state, current_s, opponents
                        )
                    except Exception:
                        failed_evaluations += 1
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
            "search_method": "coarse_to_fine_coordinate",
            "search_points": self.search_points,
            "search_passes": self.search_passes,
            "candidate_evaluations": candidate_evaluations,
            "failed_evaluations": failed_evaluations,
        }
        return best.copy(), best_plan.copy()

    def _initial_plan(self, car):
        plan = np.zeros((2, self.mpc.horizon + 1), dtype=float)
        plan[0, 0] = car.s
        plan[1, :] = car.e
        for k in range(self.mpc.horizon):
            plan[0, k + 1] = plan[0, k] + max(float(car.state.vx), 0.0) * self.mpc.dt
        return plan

    def _update_from_observation(self, observation):
        """Run this car's private two-player parameter BR from a frozen scene."""
        ego = observation.car(self.car_idx)
        neighbor = observation.nearest_opponent(self.car_idx, self.track.total_length)
        if neighbor is None:
            return
        bounds = np.asarray(self.PARAM_BOUNDS, dtype=float)
        spans = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
        pair = (ego, neighbor)
        modeled_params, previous_controls, plans = {}, {}, {}
        for car in pair:
            theta = car.policy_params
            if theta is None or theta.shape != (5,):
                theta = sim_params.ALPHARACER_INITIAL_PARAMS
            modeled_params[car.car_idx] = np.clip(theta, bounds[:, 0], bounds[:, 1]).copy()
            previous = car.previous_feasible_action
            if previous is None or previous.shape != (2,):
                previous = car.previous_action
            previous_controls[car.car_idx] = np.asarray(previous, dtype=float).copy()
            cached = car.predicted_frenet_plan
            if cached is not None and cached.shape == (2, self.mpc.horizon + 1):
                plan = np.asarray(cached, dtype=float).copy()
                plan[0] += car.s - plan[0, 0]
                plan[1] += car.e - plan[1, 0]
            else:
                plan = self._initial_plan(car)
            plans[car.car_idx] = plan

        saved = (self.params.copy(), self.last_feasible_u.copy(), self.predicted_frenet_plan,
                 self.last_parameter_ibr_score, self.last_parameter_ibr_diagnostics)
        try:
            for _ in range(max(1, int(sim_params.PARAMETER_IBR_ROUNDS))):
                max_change = 0.0
                for car, other in ((neighbor, ego), (ego, neighbor)):
                    previous = modeled_params[car.car_idx].copy()
                    self.params = previous.copy()
                    self.last_feasible_u = previous_controls[car.car_idx].copy()
                    other_plan = plans[other.car_idx]
                    opponent = {
                        "car_idx": other.car_idx, "px": other.s, "py": other.e,
                        "vx": max(float(other.state.vx), 0.0), "vy": float(other.state.vy),
                        "omega": float(other.state.omega), "psi": float(other.state.psi),
                        "state": other.state, "px_plan": other_plan[0].copy(),
                        "py_plan": other_plan[1].copy(),
                    }
                    theta, plan = self.best_response(car.state, car.s, [opponent])
                    modeled_params[car.car_idx], plans[car.car_idx] = theta, plan
                    max_change = max(max_change, float(np.max(np.abs(theta - previous) / spans)))
                if max_change <= self.convergence_tolerance:
                    break
            diagnostics = {key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in self.last_parameter_ibr_diagnostics.items()}
            result = (modeled_params[ego.car_idx].copy(), plans[ego.car_idx].copy(),
                      float(self.last_parameter_ibr_score), diagnostics)
        finally:
            (self.params, self.last_feasible_u, self.predicted_frenet_plan,
             self.last_parameter_ibr_score, self.last_parameter_ibr_diagnostics) = saved
        self.params, self.predicted_frenet_plan, self.last_parameter_ibr_score, diagnostics = result
        self.last_parameter_ibr_diagnostics = {**diagnostics,
            "opponent_model": "own_parameter_best_response", "modeled_player_count": 2,
            "interaction_neighbor_idx": neighbor.car_idx}
        self.last_parameter_ibr_update_time = observation.sim_time

    def get_action(self, observation):
        if self.car_idx is None:
            raise ValueError("ParameterIBRPolicy requires car_idx when used with RaceObservation")
        if self.update_due(observation.sim_time):
            self._update_from_observation(observation)
        return super().get_action(observation)
