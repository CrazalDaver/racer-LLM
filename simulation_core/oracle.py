"""Closed-loop, same-state rollouts for the discrete mode oracle."""

from __future__ import annotations

import pickle
import time

import casadi as cs
import numpy as np

from config import duration_to_steps, params
from policy import progress_delta


def _discard_branch_log(*_args, **_kwargs):
    """Silence expected solver recovery messages inside hypothetical branches."""


class OracleRolloutEvaluator:
    """Evaluate every current mode from one serialized runtime snapshot.

    CasADi solver objects are not isolated by ``copy.deepcopy``.  A snapshot is
    therefore made with CasADi's global pickle context and each candidate is
    restored with the matching unpickle context.  This gives every branch an
    independent solver plus identical environment, policy, MPC warm-start, and
    high-level-policy state without ever stepping the real race.

    The selected candidate is held for the complete prediction horizon.  The
    real policy executes it only for ``commitment_seconds`` before replanning;
    this is the same constant-mode horizon convention used by the repository's
    existing preset audits and avoids inventing a hidden continuation policy.
    """

    def __init__(self, race):
        self.race = race
        config = race.config

        def resolved(value, default):
            return float(default if value is None else value)

        self.horizon_seconds = resolved(
            config.oracle_horizon_seconds, params.ORACLE_HORIZON_SECONDS
        )
        self.progress_weight = resolved(
            config.oracle_progress_weight, params.ORACLE_PROGRESS_WEIGHT
        )
        self.relative_progress_weight = resolved(
            config.oracle_relative_progress_weight,
            params.ORACLE_RELATIVE_PROGRESS_WEIGHT,
        )
        self.collision_penalty = resolved(
            config.oracle_collision_penalty, params.ORACLE_COLLISION_PENALTY
        )
        self.offtrack_penalty_per_second = resolved(
            config.oracle_offtrack_penalty_per_second,
            params.ORACLE_OFFTRACK_PENALTY_PER_SECOND,
        )
        self.tie_tolerance = resolved(
            config.oracle_tie_tolerance, params.ORACLE_TIE_TOLERANCE
        )
        self.horizon_steps = duration_to_steps(
            self.horizon_seconds, params.SIM_DT
        )
        self.effective_horizon_seconds = self.horizon_steps * params.SIM_DT

        oracle_policies = [
            policy
            for policy in race.policies
            if hasattr(policy, "oracle_decision_due")
        ]
        if not oracle_policies:
            raise ValueError("OracleRolloutEvaluator requires an oracle_mode policy")
        maximum_commitment = max(
            policy.commitment_steps * params.SIM_DT
            for policy in oracle_policies
        )
        if self.horizon_seconds <= 0.0:
            raise ValueError("oracle_horizon_seconds must be positive")
        if self.effective_horizon_seconds + 1.0e-12 < maximum_commitment:
            raise ValueError(
                "oracle_horizon_seconds must cover at least one complete "
                "oracle commitment interval"
            )
        for name, value in (
            ("oracle_progress_weight", self.progress_weight),
            ("oracle_relative_progress_weight", self.relative_progress_weight),
            ("oracle_collision_penalty", self.collision_penalty),
            (
                "oracle_offtrack_penalty_per_second",
                self.offtrack_penalty_per_second,
            ),
            ("oracle_tie_tolerance", self.tie_tolerance),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.collision_penalty < 0.0:
            raise ValueError("oracle_collision_penalty must be non-negative")
        if self.offtrack_penalty_per_second < 0.0:
            raise ValueError(
                "oracle_offtrack_penalty_per_second must be non-negative"
            )
        if self.tie_tolerance < 0.0:
            raise ValueError("oracle_tie_tolerance must be non-negative")

    def settings_dict(self) -> dict:
        return {
            "horizon_seconds_requested": self.horizon_seconds,
            "horizon_seconds_effective": self.effective_horizon_seconds,
            "horizon_steps": self.horizon_steps,
            "progress_weight": self.progress_weight,
            "relative_progress_weight": self.relative_progress_weight,
            "collision_penalty_per_event": self.collision_penalty,
            "offtrack_penalty_per_second": self.offtrack_penalty_per_second,
            "tie_tolerance": self.tie_tolerance,
            "tie_break": "prefer_push_within_tolerance_then_preset_order",
            "rollout_mode_duration": "full_horizon",
            "opponent_response": (
                "closed_loop; external LLM/rule high-level updates frozen, "
                "lower-level MPC remains responsive"
            ),
            "snapshot_method": "casadi_pickle_independent_restore",
        }

    def _snapshot_runtime(self) -> bytes:
        # tqdm-bound logger methods are presentation state, not control state,
        # and are not reliably pickleable.  Replace them only while taking the
        # atomic pre-action snapshot, then restore the real policies unchanged.
        saved_loggers = [policy.log_fn for policy in self.race.policies]
        try:
            for policy in self.race.policies:
                policy.log_fn = print
            with cs.global_pickle_context():
                return pickle.dumps(
                    (
                        self.race.environment,
                        self.race.policies,
                        self.race.mpcs,
                    ),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as exc:
            raise RuntimeError(
                "Could not serialize the simulator state for oracle rollouts"
            ) from exc
        finally:
            for policy, logger in zip(self.race.policies, saved_loggers):
                policy.log_fn = logger

    @staticmethod
    def _restore_runtime(snapshot: bytes):
        try:
            with cs.global_unpickle_context():
                environment, policies, mpcs = pickle.loads(snapshot)
        except Exception as exc:
            raise RuntimeError(
                "Could not independently restore an oracle rollout branch"
            ) from exc
        for policy in policies:
            policy.log_fn = _discard_branch_log
        return environment, policies, mpcs

    @staticmethod
    def _relative_to_best_opponent(progress, ego_idx):
        opponents = [
            float(value)
            for idx, value in enumerate(progress)
            if idx != ego_idx
        ]
        if not opponents:
            return 0.0
        return float(progress[ego_idx]) - max(opponents)

    def _simulate_candidate(
        self,
        snapshot: bytes,
        *,
        ego_idx: int,
        mode: str,
        start_step: int,
    ) -> dict:
        environment, policies, _mpcs = self._restore_runtime(snapshot)
        ego_policy = policies[ego_idx]
        if not hasattr(ego_policy, "set_rollout_mode"):
            raise TypeError("Serialized oracle ego does not support fixed rollout mode")
        ego_policy.set_rollout_mode(mode)

        # External language-model calls inside hypothetical opponent futures
        # would be costly and nondeterministic across candidate order.  Freeze
        # only that high-level updater; their copied MPC continues to react to
        # the branch's changing scene.  Deterministic AlphaRacer/IBR policies
        # continue to replan normally.
        for idx, policy in enumerate(policies):
            if idx == ego_idx:
                continue
            freezer = getattr(policy, "freeze_for_oracle_rollout", None)
            if freezer is not None:
                freezer()

        start_progress = np.asarray(environment.progress_s, dtype=float).copy()
        start_relative = self._relative_to_best_opponent(
            start_progress, ego_idx
        )
        collision_count = 0
        collision_steps = 0
        previous_collision = False
        initial_e = float(environment.car_e[ego_idx])
        previous_offtrack = abs(initial_e) > environment.offroad_recovery_threshold
        offtrack_count = 0
        offtrack_steps = 0
        boundary_steps = 0
        solver_failure_count = 0

        for offset in range(self.horizon_steps):
            environment.begin_step(start_step + offset)
            observation = environment.capture(policies)
            actions = tuple(policy.get_action(observation) for policy in policies)
            transition = environment.step(actions)

            collision = bool(transition.contact_happened)
            collision_steps += int(collision)
            collision_count += int(collision and not previous_collision)
            previous_collision = collision

            offtrack = bool(transition.offroad_active[ego_idx])
            offtrack_steps += int(offtrack)
            offtrack_count += int(offtrack and not previous_offtrack)
            previous_offtrack = offtrack
            boundary_steps += int(
                abs(float(transition.next_e[ego_idx])) > params.TRACK_HALF_WIDTH
            )
            solver_failure_count += int(
                not bool(policies[ego_idx].mpc.last_solver_success)
            )

        final_progress = np.asarray(environment.progress_s, dtype=float).copy()
        ego_progress_gain = float(
            final_progress[ego_idx] - start_progress[ego_idx]
        )
        opponent_gains = [
            float(final_progress[idx] - start_progress[idx])
            for idx in range(len(final_progress))
            if idx != ego_idx
        ]
        best_opponent_progress_gain = max(opponent_gains, default=0.0)
        final_relative = self._relative_to_best_opponent(
            final_progress, ego_idx
        )
        relative_progress_gain = final_relative - start_relative
        collision_seconds = collision_steps * params.SIM_DT
        offtrack_seconds = offtrack_steps * params.SIM_DT
        progress_term = self.progress_weight * ego_progress_gain
        relative_term = (
            self.relative_progress_weight * relative_progress_gain
        )
        collision_term = self.collision_penalty * collision_count
        offtrack_term = (
            self.offtrack_penalty_per_second * offtrack_seconds
        )
        score = progress_term + relative_term - collision_term - offtrack_term
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite oracle score for mode {mode!r}")
        finish_position = 1 + sum(
            value > final_progress[ego_idx]
            for idx, value in enumerate(final_progress)
            if idx != ego_idx
        )
        return {
            "mode": mode,
            "score": float(score),
            "ego_progress_gain": ego_progress_gain,
            "best_opponent_progress_gain": best_opponent_progress_gain,
            "relative_progress_gain": float(relative_progress_gain),
            "final_relative_progress": float(final_relative),
            "finish_position": int(finish_position),
            "collision_count": int(collision_count),
            "collision_seconds": float(collision_seconds),
            "offtrack_count": int(offtrack_count),
            "offtrack_seconds": float(offtrack_seconds),
            "track_boundary_seconds": float(boundary_steps * params.SIM_DT),
            "ego_solver_failure_count": int(solver_failure_count),
            "score_components": {
                "progress_term": float(progress_term),
                "relative_progress_term": float(relative_term),
                "collision_penalty": float(collision_term),
                "offtrack_penalty": float(offtrack_term),
            },
        }

    def _select_mode(self, modes, rows):
        scores = {mode: float(rows[mode]["score"]) for mode in modes}
        best_score = max(scores.values())
        tied = [
            mode
            for mode in modes
            if best_score - scores[mode] <= self.tie_tolerance
        ]
        selected = "push" if "push" in tied else tied[0]
        sorted_scores = sorted(scores.values(), reverse=True)
        second = sorted_scores[1] if len(sorted_scores) > 1 else best_score
        return selected, scores, tied, float(best_score - second)

    def _state_summary(self, observation, ego_idx):
        ego = observation.car(ego_idx)
        opponents = []
        for other in observation.opponents(ego_idx):
            opponents.append(
                {
                    "car_index": int(other.car_idx),
                    "relative_progress": float(
                        progress_delta(
                            other.s, ego.s, self.race.track.total_length
                        )
                    ),
                    "lateral_offset": float(other.e),
                    "speed": float(other.vx),
                }
            )
        current_progress = np.asarray(
            [car.s for car in observation.cars], dtype=float
        )
        position = 1 + sum(
            car.s > ego.s for car in observation.cars if car.car_idx != ego_idx
        )
        return {
            "ego": {
                "progress": float(ego.s),
                "lateral_offset": float(ego.e),
                "speed": float(ego.vx),
                "position": int(position),
            },
            "opponents": opponents,
            "current_race_metrics": {
                "position": int(position),
                "relative_progress_to_best_opponent": float(
                    self._relative_to_best_opponent(current_progress, ego_idx)
                ),
            },
        }

    def evaluate(self, policy, observation, step_index: int) -> dict:
        """Return and log one all-mode decision from the pre-action state."""
        modes = tuple(policy.available_modes)
        if modes != tuple(params.LLM_MODE_PRESETS):
            raise ValueError(
                "Oracle policy modes must exactly match config.LLM_MODE_PRESETS"
            )
        started = time.perf_counter()
        snapshot = self._snapshot_runtime()
        rows = {
            mode: self._simulate_candidate(
                snapshot,
                ego_idx=policy.car_idx,
                mode=mode,
                start_step=int(step_index),
            )
            for mode in modes
        }
        selected, scores, tied, best_second_margin = self._select_mode(modes, rows)
        push_score = scores.get("push")
        score_margin_over_push = (
            max(scores.values()) - push_score if push_score is not None else 0.0
        )
        state = self._state_summary(observation, policy.car_idx)
        return {
            "decision_index": int(policy.policy_decision_count),
            "time": float(observation.sim_time),
            "state_summary": state["ego"],
            "opponents": state["opponents"],
            "candidate_scores": scores,
            "candidate_components": rows,
            "selected_mode": selected,
            "tied_best_modes": list(tied),
            "score_margin_over_push": float(score_margin_over_push),
            "best_second_score_margin": float(best_second_margin),
            "current_race_metrics": state["current_race_metrics"],
            "available_modes": list(modes),
            "score_definition": (
                "progress_weight*ego_progress_gain + "
                "relative_progress_weight*relative_progress_gain - "
                "collision_penalty_per_event*collision_count - "
                "offtrack_penalty_per_second*offtrack_seconds"
            ),
            "rollout_runtime_seconds": float(time.perf_counter() - started),
        }
