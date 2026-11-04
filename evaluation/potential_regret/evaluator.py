"""Detachable paired-rollout diagnostic for learned-potential regret."""

from __future__ import annotations

import time

import numpy as np

from config import duration_to_steps, params
from .optimizer import (
    LearnedPotential,
    PotentialOracleOptimizer,
    observation_cars,
)
from policy.utils import progress_delta, track_curvature_at
from simulation_core.runtime_snapshot import RuntimeSnapshotter


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")


class PotentialRegretEvaluator:
    """Evaluate LLM and potential actions from the same pre-decision snapshot."""

    def __init__(self, race):
        self.race = race
        config = race.config
        self.ego_idx = race.ego_idx
        ego = race.policies[self.ego_idx]
        if race.num_cars != 3:
            raise ValueError("Potential regret currently requires exactly three cars")
        if race.resolved_policy_names[self.ego_idx] != "llm_alpharacer":
            raise ValueError("Potential regret requires the ego policy to be llm_alpharacer")
        if getattr(ego, "response_format", None) != "params":
            raise ValueError("Potential regret requires direct LLM parameter output")
        for idx, policy in enumerate(race.policies):
            theta = np.asarray(getattr(policy, "params", []), dtype=float)
            if theta.shape != (5,):
                raise ValueError(
                    f"Potential regret requires five-parameter policy for car {idx}"
                )

        weight_path = config.potential_regret_weight_path or config.alpha_weight_path
        if weight_path is None:
            weight_path = params.ALPHARACER_WEIGHT_PATH
        self.horizon_seconds = float(config.potential_regret_horizon_seconds)
        self.horizon_steps = duration_to_steps(self.horizon_seconds, params.SIM_DT)
        if not np.isclose(
            self.horizon_steps * params.SIM_DT,
            self.horizon_seconds,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Potential regret horizon must be a multiple of SIM_DT")
        if not np.isclose(self.horizon_seconds, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("The pilot defines an exact one-second branch horizon")

        potential = LearnedPotential(
            race.track,
            params.MPC_DT,
            race.mpc_horizon,
            weight_path,
            device=config.potential_regret_device,
        )
        self.optimizer = PotentialOracleOptimizer(
            potential,
            sobol_points=config.potential_regret_sobol_points,
            max_iterations=config.potential_regret_max_iterations,
            seed=config.potential_regret_optimizer_seed,
        )
        self.snapshotter = RuntimeSnapshotter(race)
        self.decisions = []
        self.snapshot_validation = {
            "performed": False,
            "passed": False,
            "maximum_absolute_difference": None,
        }

    def settings_dict(self):
        return {
            "enabled": True,
            "result_class": "exploratory",
            "horizon_seconds": self.horizon_seconds,
            "horizon_steps": self.horizon_steps,
            "return": "fixed-start-opponent relative-progress gain",
            "comparison_opponent": "highest-progress opponent at branch start",
            "opponent_high_level_parameters": "evaluator-known and held fixed",
            "branch_opponent_response": "closed-loop MPC with fixed high-level theta",
            "snapshot_method": "casadi_pickle_independent_restore",
            "optimizer": self.optimizer.settings_dict(),
            "interaction_rule": {
                "distance_m": 4.0,
                "ttc_s": 2.0,
                "closing_speed_deadband_mps": 0.1,
            },
            "track_context_rule": {
                "curvature_threshold_per_m": 0.05,
                "geometry_probe_m": 2.0,
            },
        }

    @staticmethod
    def _joint_params(policies):
        return np.asarray([policy.params for policy in policies], dtype=float)

    def decision_due(self):
        due = getattr(self.race.policies[self.ego_idx], "llm_decision_due", None)
        return bool(due is not None and due())

    def capture(self):
        return self.snapshotter.capture()

    @staticmethod
    def _hold_policy_params(policy, theta):
        theta = np.asarray(theta, dtype=float).copy()
        policy.params = theta.copy()
        for name in ("last_llm_params", "last_rule_params"):
            if hasattr(policy, name):
                setattr(policy, name, theta.copy())
        if hasattr(policy, "joint_params") and policy.car_idx is not None:
            policy.joint_params[policy.car_idx] = theta.copy()
        freezer = getattr(policy, "freeze_for_oracle_rollout", None)
        if freezer is not None:
            freezer()

    @staticmethod
    def _max_slack(mpc, name):
        values = getattr(mpc, name, None)
        if values is None:
            return 0.0
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else 0.0

    def _simulate_branch(self, snapshot, joint_params, ego_theta, start_step, comparator):
        environment, policies, mpcs = self.snapshotter.restore(snapshot)
        held = np.asarray(joint_params, dtype=float).copy()
        held[self.ego_idx] = np.asarray(ego_theta, dtype=float)
        for policy, theta in zip(policies, held):
            self._hold_policy_params(policy, theta)

        start_progress = np.asarray(environment.progress_s, dtype=float).copy()
        start_relative = float(
            start_progress[self.ego_idx] - start_progress[comparator]
        )
        collision_steps = 0
        boundary_steps = np.zeros(len(policies), dtype=int)
        recovery_steps = np.zeros(len(policies), dtype=int)
        solver_failures = np.zeros(len(policies), dtype=int)
        max_track_slack = np.zeros(len(policies), dtype=float)
        max_collision_slack = np.zeros(len(policies), dtype=float)

        for offset in range(self.horizon_steps):
            environment.begin_step(start_step + offset)
            observation = environment.capture(policies)
            actions = tuple(policy.get_action(observation) for policy in policies)
            transition = environment.step(actions)
            collision_steps += int(transition.contact_happened)
            for idx, policy in enumerate(policies):
                boundary_steps[idx] += int(
                    abs(float(transition.next_e[idx])) > params.TRACK_HALF_WIDTH
                )
                recovery_steps[idx] += int(transition.offroad_active[idx])
                solver_failures[idx] += int(not bool(policy.mpc.last_solver_success))
                max_track_slack[idx] = max(
                    max_track_slack[idx],
                    self._max_slack(policy.mpc, "last_track_slack_plan"),
                )
                max_collision_slack[idx] = max(
                    max_collision_slack[idx],
                    self._max_slack(policy.mpc, "last_collision_slack_plan"),
                )

        final_progress = np.asarray(environment.progress_s, dtype=float).copy()
        final_relative = float(
            final_progress[self.ego_idx] - final_progress[comparator]
        )
        initial_rank = 1 + int(np.sum(start_progress > start_progress[self.ego_idx]))
        final_rank = 1 + int(np.sum(final_progress > final_progress[self.ego_idx]))
        return {
            "start_progress_m": start_progress.tolist(),
            "final_progress_m": final_progress.tolist(),
            "progress_gain_m": (final_progress - start_progress).tolist(),
            "relative_progress_gain_m": float(final_relative - start_relative),
            "comparison_opponent_index": int(comparator),
            "initial_rank": int(initial_rank),
            "final_rank": int(final_rank),
            "rank_change": int(initial_rank - final_rank),
            "physical_collision_seconds": float(collision_steps * params.SIM_DT),
            "track_boundary_violation_seconds_by_car": (
                boundary_steps * params.SIM_DT
            ).tolist(),
            "recovery_active_seconds_by_car": (
                recovery_steps * params.SIM_DT
            ).tolist(),
            "mpc_solver_failures_by_car": solver_failures.tolist(),
            "max_track_slack_by_car": max_track_slack.tolist(),
            "max_collision_slack_by_car": max_collision_slack.tolist(),
        }

    def _context(self, observation):
        ego = observation.car(self.ego_idx)
        opponents = []
        for other in observation.opponents(self.ego_idx):
            delta = progress_delta(other.s, ego.s, self.race.track.total_length)
            closing = (ego.vx - other.vx) if delta > 0.0 else (other.vx - ego.vx)
            ttc = abs(delta) / closing if closing > 0.1 else np.inf
            opponents.append((abs(delta), ttc, delta, other))
        nearest = min(opponents, key=lambda item: item[0])
        distance, ttc, delta, other = nearest
        attacking = delta > 0.0 and ego.vx > other.vx + 0.1
        defending = delta < 0.0 and other.vx > ego.vx + 0.1

        probe = 2.0
        threshold = 0.05
        curvatures = [
            abs(track_curvature_at(self.race.track, ego.s + offset))
            for offset in (-probe, 0.0, probe)
        ]
        if max(curvatures) < threshold:
            geometry = "straight"
        elif curvatures[1] >= max(curvatures[0], curvatures[2]):
            geometry = "apex"
        elif curvatures[2] > curvatures[1]:
            geometry = "corner_entry"
        else:
            geometry = "corner_exit"
        return {
            "interaction": bool(distance <= 4.0 or ttc <= 2.0),
            "attack_defend": "attacking" if attacking else "defending" if defending else "neutral",
            "track_geometry": geometry,
            "nearest_opponent_index": int(other.car_idx),
            "nearest_opponent_relative_progress_m": float(delta),
            "nearest_opponent_side": (
                "left" if other.e > ego.e else "right" if other.e < ego.e else "same"
            ),
            "nearest_opponent_ttc_s": None if not np.isfinite(ttc) else float(ttc),
        }

    def _validate_snapshot(self, snapshot, joint, theta, step, comparator, first):
        second = self._simulate_branch(snapshot, joint, theta, step, comparator)
        keys = ("final_progress_m", "relative_progress_gain_m")
        differences = []
        for key in keys:
            differences.extend(
                np.abs(
                    np.asarray(first[key], dtype=float)
                    - np.asarray(second[key], dtype=float)
                ).reshape(-1).tolist()
            )
        maximum = max(differences, default=0.0)
        self.snapshot_validation = {
            "performed": True,
            "passed": bool(maximum <= 1e-10),
            "maximum_absolute_difference": float(maximum),
        }
        if not self.snapshot_validation["passed"]:
            raise RuntimeError(
                "Potential-regret snapshot replay is not deterministic; result invalid"
            )

    def evaluate(
        self,
        snapshot,
        observation,
        step_index,
        decision_count_before,
        previous_ego_theta,
    ):
        ego = self.race.policies[self.ego_idx]
        base = {
            "status": "ok",
            "step": int(step_index),
            "race_time_s": float(step_index * params.SIM_DT),
            "decision_index": int(len(self.decisions)),
            "start_position_cell": int(self.race.ego_position),
            "track_id": self.race.config.track_name or "configured_default",
            "previous_ego_theta": np.asarray(
                previous_ego_theta, dtype=float
            ).tolist(),
        }
        if ego.policy_decision_count == decision_count_before:
            base.update(status="missing_llm_decision")
            self.decisions.append(base)
            return base
        if ego.last_llm_error is not None:
            base.update(status="llm_error", llm_error=str(ego.last_llm_error))
            self.decisions.append(base)
            return base

        started = time.perf_counter()
        joint = self._joint_params(self.race.policies)
        llm_theta = joint[self.ego_idx].copy()
        cars = observation_cars(observation)
        optimization = self.optimizer.maximize(
            cars, joint, self.ego_idx, llm_theta
        )
        oracle_theta = np.asarray(optimization.theta, dtype=float)
        progress = np.asarray([car.s for car in observation.cars], dtype=float)
        opponent_indices = [idx for idx in range(3) if idx != self.ego_idx]
        # Freeze the identity of the branch-start leading opponent so that the
        # LLM and potential-proposal branches are compared against the same
        # physical car.  This intentionally differs from the paper's one-step
        # utility, which recomputes max_{j != i} p_j at each endpoint.  The
        # resulting signed quantity is therefore a fixed-start-opponent
        # potential-proposal rollout gap, not strict action/Nash regret.
        comparator = max(opponent_indices, key=lambda idx: progress[idx])
        llm_branch = self._simulate_branch(
            snapshot, joint, llm_theta, step_index, comparator
        )
        oracle_branch = self._simulate_branch(
            snapshot, joint, oracle_theta, step_index, comparator
        )
        if not self.snapshot_validation["performed"]:
            self._validate_snapshot(
                snapshot, joint, llm_theta, step_index, comparator, llm_branch
            )

        ego_gap = (
            oracle_branch["progress_gain_m"][self.ego_idx]
            - llm_branch["progress_gain_m"][self.ego_idx]
        )
        opponent_gaps = (
            np.asarray(oracle_branch["progress_gain_m"])
            - np.asarray(llm_branch["progress_gain_m"])
        )
        regret = (
            oracle_branch["relative_progress_gain_m"]
            - llm_branch["relative_progress_gain_m"]
        )
        ego_car = observation.car(self.ego_idx)
        base.update(
            status="ok",
            lap_relative_s_m=float(ego_car.s % self.race.track.total_length),
            unwrapped_progress_m=float(ego_car.s),
            car_states=[
                {
                    "car_index": int(car.car_idx),
                    "s_m": float(car.s),
                    "e_m": float(car.e),
                    "x_m": float(car.state.x),
                    "y_m": float(car.state.y),
                    "heading_rad": float(car.state.psi),
                    "speed_mps": float(car.vx),
                }
                for car in observation.cars
            ],
            llm_theta=llm_theta.tolist(),
            llm_status={
                "fallback_used": False,
                "error": None,
                "reason": str(getattr(ego, "last_llm_reason", "")),
                "basis": str(getattr(ego, "last_llm_basis", "")),
                "requested_mode": str(
                    getattr(ego, "last_llm_requested_mode", "custom")
                ),
            },
            oracle_theta=oracle_theta.tolist(),
            delta_theta=(oracle_theta - llm_theta).tolist(),
            opponent_theta_by_car=joint.tolist(),
            optimizer=optimization.to_dict(),
            rollout_regret_m=float(regret),
            ego_absolute_progress_difference_m=float(ego_gap),
            absolute_progress_difference_by_car_m=opponent_gaps.tolist(),
            llm_branch=llm_branch,
            oracle_branch=oracle_branch,
            context=self._context(observation),
            evaluation_wall_seconds=float(time.perf_counter() - started),
        )
        self.decisions.append(base)
        return base
