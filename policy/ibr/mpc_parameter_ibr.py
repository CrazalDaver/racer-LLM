"""Five-parameter IBR evaluated through the shared alpha-RACER MPC."""

import math

import numpy as np

from config import params as sim_params
from state import CarState

from ..alpha_network import ALPHA_PARAMETER_BOUNDS
from ..mpc import MPCPolicy
from ..utils import progress_delta


class ParameterIBRPolicy(MPCPolicy):
    """Iterated best response in the paper's five-parameter policy space.

    Unlike :class:`TrajectoryIBRPolicy`, this policy never optimizes throttle
    or steering directly.  A best-response step searches
    ``theta=(q, zeta, s1, s2, s3)`` and evaluates every candidate by solving
    the same raceline-reference MPC used by alpha-RACER and the LLM policies.
    Candidates are compared with the same local racing objective as trajectory
    IBR: progress, centerline-relative lateral error, control-rate regularization,
    and the MPC's existing track/collision slack costs.  This outer score is
    independent of the MPC's tunable tracking weight, so candidates with
    different ``q`` remain comparable.
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
        world_plan = np.asarray(self.mpc.last_state_plan, dtype=float).copy()
        control_plan = np.asarray(self.mpc.last_control_plan, dtype=float).copy()
        frenet_plan = self._world_plan_to_frenet(world_plan, current_s)
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
            - control_rate_cost
            - track_slack_cost
            - collision_slack_cost
        )
        return score, frenet_plan, {
            "predicted_progress": predicted_progress,
            "progress_reward": progress_reward,
            "lateral_cost": lateral_cost,
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


def _constant_velocity_frenet_plan(model_policy, state, current_s, current_e):
    """Initialize one player's internal plan before the first BR update."""
    horizon = model_policy.mpc.horizon
    plan = np.zeros((2, horizon + 1), dtype=float)
    plan[0, 0] = float(current_s)
    plan[1, :] = float(current_e)
    for k in range(horizon):
        plan[0, k + 1] = (
            plan[0, k] + max(float(state.vx), 0.0) * model_policy.mpc.dt
        )
    return plan


def _nearest_interaction_neighbor(policy, car_idx, car_s, states):
    """Match trajectory IBR by selecting one nearest longitudinal neighbor."""
    best_idx = None
    best_abs_delta = np.inf
    for other_idx, _ in enumerate(states):
        if other_idx == car_idx:
            continue
        delta = progress_delta(
            car_s[other_idx], car_s[car_idx], policy.track.total_length
        )
        if abs(delta) < best_abs_delta:
            best_idx = other_idx
            best_abs_delta = abs(delta)
    return best_idx


def _parameter_ibr_opponent(other_idx, states, car_s, car_e, plan):
    """Build the shared-MPC record for one modeled interaction neighbor."""
    state = states[other_idx]
    return {
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
    }


def plan_parameter_ibr_policy_params(policies, states, car_s, car_e, sim_time=None):
    """Run one local two-player parameter-space IBR game per controlled car.

    Each controlled car selects its nearest longitudinal neighbor and uses its
    own :class:`ParameterIBRPolicy` model to alternate the neighbor's
    hypothetical best response and then its own.  The real neighbor policy is
    never invoked.  This mirrors the two-player structure used by the local
    Wang-style trajectory IBR; constant velocity only initializes the game.

    The hypothetical neighbor update remains local.  Only the controlled
    vehicle receives the final optimized parameters and plan.
    """
    controlled_indices = [
        idx for idx, policy in enumerate(policies)
        if isinstance(policy, ParameterIBRPolicy) and policy.update_due(sim_time)
    ]
    if not controlled_indices:
        return

    results = {}
    for ego_idx in controlled_indices:
        model_policy = policies[ego_idx]
        neighbor_idx = _nearest_interaction_neighbor(
            model_policy, ego_idx, car_s, states
        )
        if neighbor_idx is None:
            continue

        bounds = np.asarray(model_policy.PARAM_BOUNDS, dtype=float)
        spans = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
        pair = (ego_idx, neighbor_idx)
        modeled_params = {}
        previous_controls = {}
        plans = {}
        for idx in pair:
            real_policy = policies[idx]
            params = np.asarray(
                getattr(real_policy, "params", sim_params.ALPHARACER_INITIAL_PARAMS),
                dtype=float,
            )
            if params.shape != (5,):
                params = np.asarray(
                    sim_params.ALPHARACER_INITIAL_PARAMS, dtype=float
                )
            modeled_params[idx] = np.clip(
                params, bounds[:, 0], bounds[:, 1]
            ).copy()

            previous_u = getattr(real_policy, "last_feasible_u", None)
            if previous_u is None:
                previous_u = getattr(
                    real_policy, "last_u", model_policy.last_feasible_u
                )
            previous_u = np.asarray(previous_u, dtype=float)
            if previous_u.shape != (2,):
                previous_u = np.asarray(model_policy.last_feasible_u, dtype=float)
            previous_controls[idx] = previous_u.copy()

            cached = getattr(real_policy, "predicted_frenet_plan", None)
            expected_length = model_policy.mpc.horizon + 1
            if cached is not None and cached.shape == (2, expected_length):
                plan = np.asarray(cached, dtype=float).copy()
                plan[0] += float(car_s[idx]) - plan[0, 0]
                plan[1] += float(car_e[idx]) - plan[1, 0]
            else:
                plan = _constant_velocity_frenet_plan(
                    model_policy, states[idx], car_s[idx], car_e[idx]
                )
            plans[idx] = plan

        saved_model_state = (
            model_policy.params.copy(),
            model_policy.last_feasible_u.copy(),
            model_policy.predicted_frenet_plan,
            model_policy.last_parameter_ibr_score,
            model_policy.last_parameter_ibr_diagnostics,
        )
        try:
            rounds = max(1, int(sim_params.PARAMETER_IBR_ROUNDS))
            for _ in range(rounds):
                max_change = 0.0
                # The ego's own BR is deliberately last, so the committed plan
                # responds to the neighbor prediction from this same round.
                for idx, other_idx in (
                    (neighbor_idx, ego_idx),
                    (ego_idx, neighbor_idx),
                ):
                    previous = modeled_params[idx].copy()
                    model_policy.params = previous.copy()
                    model_policy.last_feasible_u = previous_controls[idx].copy()
                    opponent = _parameter_ibr_opponent(
                        other_idx, states, car_s, car_e, plans[other_idx]
                    )
                    params_opt, plan_opt = model_policy.best_response(
                        states[idx], car_s[idx], [opponent]
                    )
                    modeled_params[idx] = params_opt
                    plans[idx] = plan_opt
                    max_change = max(
                        max_change,
                        float(np.max(np.abs(params_opt - previous) / spans)),
                    )
                if max_change <= model_policy.convergence_tolerance:
                    break

            diagnostics = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in model_policy.last_parameter_ibr_diagnostics.items()
            }
            results[ego_idx] = (
                modeled_params[ego_idx].copy(),
                plans[ego_idx].copy(),
                float(model_policy.last_parameter_ibr_score),
                diagnostics,
                neighbor_idx,
            )
        finally:
            (
                model_policy.params,
                model_policy.last_feasible_u,
                model_policy.predicted_frenet_plan,
                model_policy.last_parameter_ibr_score,
                model_policy.last_parameter_ibr_diagnostics,
            ) = saved_model_state

    for ego_idx, result in results.items():
        policy = policies[ego_idx]
        params_opt, plan_opt, score, diagnostics, neighbor_idx = result
        policy.params = params_opt
        policy.predicted_frenet_plan = plan_opt
        policy.last_parameter_ibr_score = score
        policy.last_parameter_ibr_diagnostics = {
            **diagnostics,
            "opponent_model": "own_parameter_best_response",
            "modeled_player_count": 2,
            "interaction_neighbor_idx": neighbor_idx,
        }
        policy.last_parameter_ibr_update_time = sim_time
