"""Discrete-mode AlphaRacer policy using the learned potential for selection."""

from __future__ import annotations

import itertools
import time

import numpy as np

from config import duration_to_steps, params as sim_params

from ..mpc import MPCPolicy
from .policy import AlphaRacerPolicy


def _observation_cars(observation):
    if len(observation.cars) < 3:
        raise ValueError("ModeAlphaRacerPolicy requires exactly three race cars")
    return [
        {
            "px": car.s,
            "py": car.e,
            "vx": car.state.vx,
            "vy": car.state.vy,
            "omega": car.state.omega,
            "psi": car.state.psi,
            "state": car.state,
        }
        for car in observation.cars[:3]
    ]


class ModeAlphaRacerPolicy(AlphaRacerPolicy):
    """Choose among the shared four presets by learned-potential score.

    At each high-level update, this policy enumerates all joint assignments of
    the configured modes to the three cars.  It selects the assignment with
    maximum learned potential and executes only this policy's own mode.  The
    other two mode blocks are private predictions, matching the existing
    AlphaRacer policy's joint-optimization/own-block-execution semantics.
    """

    def __init__(
        self,
        *args,
        update_interval_seconds=1.0,
        decision_dt=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mode_presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        if len(self.mode_presets) != 4:
            raise ValueError(
                "ModeAlphaRacerPolicy requires the shared four-mode library"
            )
        self.available_modes = tuple(self.mode_presets)
        self.joint_mode_candidates = tuple(
            itertools.product(self.available_modes, repeat=3)
        )
        self.update_interval_seconds = max(float(update_interval_seconds), 0.0)
        self.decision_dt = (
            float(sim_params.CONTROL_DT)
            if decision_dt is None
            else float(decision_dt)
        )
        self.update_interval_calls = duration_to_steps(
            self.update_interval_seconds, self.decision_dt
        )
        self.control_call_count = 0
        self.policy_decision_count = 0
        self.mode_alpharacer_decision_history = []
        self.current_mode = self.available_modes[0]
        self.params = self.mode_presets[self.current_mode].copy()
        self.last_llm_requested_mode = self.current_mode
        self.last_llm_reason = "potential_not_evaluated"
        self.last_llm_basis = "learned_potential_joint_mode_enumeration"

    def _joint_parameter_batch(self) -> np.ndarray:
        return np.asarray(
            [
                [self.mode_presets[mode] for mode in joint_modes]
                for joint_modes in self.joint_mode_candidates
            ],
            dtype=float,
        )

    def _select_mode(self, cars) -> dict:
        candidates = self._joint_parameter_batch()
        scores = np.asarray(
            self._score_joint_params(cars, candidates), dtype=float
        ).reshape(-1)
        finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
        if not np.isfinite(finite_scores).any():
            raise RuntimeError("learned potential returned no finite mode score")
        best_index = int(np.argmax(finite_scores))
        selected_joint_modes = self.joint_mode_candidates[best_index]
        selected_joint_params = candidates[best_index]
        per_ego_mode_best = {
            mode: float(np.max(finite_scores[
                [
                    idx
                    for idx, joint_modes in enumerate(self.joint_mode_candidates)
                    if joint_modes[self.car_idx] == mode
                ]
            ]))
            for mode in self.available_modes
        }
        ordered = np.sort(finite_scores[np.isfinite(finite_scores)])
        margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
        return {
            "selected_mode": selected_joint_modes[self.car_idx],
            "selected_joint_modes": list(selected_joint_modes),
            "selected_joint_params": selected_joint_params.tolist(),
            "selected_potential": float(finite_scores[best_index]),
            "margin_over_second_best_joint_assignment": margin,
            "best_potential_by_ego_mode": per_ego_mode_best,
            "candidate_count": len(self.joint_mode_candidates),
        }

    def _decision_due(self) -> bool:
        return self.control_call_count % self.update_interval_calls == 0

    def get_action(self, observation):
        if self.car_idx is None or not 0 <= self.car_idx < 3:
            raise ValueError(
                "ModeAlphaRacerPolicy requires the three-car global ordering"
            )
        if not self._oracle_rollout_frozen and self._decision_due():
            started = time.perf_counter()
            decision = self._select_mode(_observation_cars(observation))
            decision["decision_index"] = self.policy_decision_count
            decision["control_call"] = self.control_call_count + 1
            decision["latency_seconds"] = time.perf_counter() - started
            self.current_mode = str(decision["selected_mode"])
            self.params = self.mode_presets[self.current_mode].copy()
            self.joint_params = np.asarray(
                decision["selected_joint_params"], dtype=float
            ).copy()
            self.policy_decision_count += 1
            self.last_llm_requested_mode = self.current_mode
            self.last_llm_reason = (
                f"potential selected {self.current_mode}; "
                f"joint_margin={decision['margin_over_second_best_joint_assignment']:+.6g}"
            )
            self.mode_alpharacer_decision_history.append(decision)
        self.control_call_count += 1
        return MPCPolicy.get_action(self, observation)

