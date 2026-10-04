"""Policies backed by the repository's existing discrete mode presets."""

from __future__ import annotations

import copy

import numpy as np

from config import duration_to_steps, params as sim_params

from .mpc import MPCPolicy


class FixedModePolicy(MPCPolicy):
    """Execute one existing mode preset without changing it online."""

    def __init__(self, *args, mode="push", **kwargs):
        mode_key = str(mode).strip().lower()
        if mode_key not in sim_params.LLM_MODE_PRESETS:
            raise ValueError(f"Unknown fixed mode: {mode!r}")
        kwargs["params"] = np.asarray(
            sim_params.LLM_MODE_PRESETS[mode_key], dtype=float
        ).copy()
        super().__init__(*args, **kwargs)
        self.mode_presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        self.current_mode = mode_key
        self.policy_decision_count = 1


class OracleModePolicy(MPCPolicy):
    """MPC policy whose preset is selected by a race-level rollout evaluator.

    The policy intentionally contains no simulator reference.  :class:`Race`
    asks the rollout evaluator for a decision at the configured high-level
    cadence and applies that result here.  Consequently, serialized rollout
    branches can call the ordinary :class:`MPCPolicy` action path without
    recursively invoking another oracle.
    """

    def __init__(
        self,
        *args,
        commitment_seconds=1.0,
        decision_dt=None,
        **kwargs,
    ):
        presets = {
            name: np.asarray(values, dtype=float).copy()
            for name, values in sim_params.LLM_MODE_PRESETS.items()
        }
        if "push" not in presets:
            raise ValueError("Oracle mode evaluation requires the existing push preset")
        kwargs["params"] = presets["push"].copy()
        super().__init__(*args, **kwargs)
        self.mode_presets = presets
        self.current_mode = "push"
        self.commitment_seconds = float(commitment_seconds)
        self.decision_dt = (
            float(sim_params.CONTROL_DT)
            if decision_dt is None
            else float(decision_dt)
        )
        if self.commitment_seconds <= 0.0:
            raise ValueError("oracle commitment_seconds must be positive")
        self.commitment_steps = duration_to_steps(
            self.commitment_seconds, self.decision_dt
        )
        self.policy_decision_count = 0
        self.oracle_decision_history = []
        self.last_llm_requested_mode = self.current_mode
        self.last_llm_reason = "oracle_not_evaluated"
        self.last_llm_basis = "receding_horizon_rollout"

    @property
    def available_modes(self) -> tuple[str, ...]:
        """Return the exact current preset keys, in their configured order."""
        return tuple(self.mode_presets)

    def oracle_decision_due(self, step_index: int) -> bool:
        return int(step_index) % self.commitment_steps == 0

    def set_rollout_mode(self, mode: str):
        """Force one mode in an isolated candidate branch."""
        mode_key = str(mode)
        if mode_key not in self.mode_presets:
            raise ValueError(f"Unknown oracle rollout mode: {mode!r}")
        self.current_mode = mode_key
        self.params = self.mode_presets[mode_key].copy()
        self.last_llm_requested_mode = mode_key

    def apply_oracle_decision(self, decision: dict):
        """Commit one evaluator result to the real policy state."""
        selected = str(decision["selected_mode"])
        self.set_rollout_mode(selected)
        self.policy_decision_count += 1
        self.last_llm_reason = (
            f"oracle selected {selected}; advantage_over_push="
            f"{float(decision['score_margin_over_push']):+.6g}"
        )
        self.last_llm_basis = "receding_horizon_rollout"
        # Do not retain references to evaluator-owned nested dictionaries.
        self.oracle_decision_history.append(copy.deepcopy(decision))

