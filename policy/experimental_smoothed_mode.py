"""Experimental first-order transition for discrete LLM mode presets.

This module is deliberately detachable.  It changes neither the LLM prompt nor
the four target presets: it only smooths the parameters actually sent to MPC.
Remove this file and the small policy-registry entries if the experiment is not
retained.
"""

import math

import numpy as np

from .llm import LLMAlphaRacerPolicy


class SmoothedModeLLMPolicy(LLMAlphaRacerPolicy):
    """LLM mode policy with a shared first-order parameter transition."""

    def __init__(self, *args, transition_tau_seconds=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        if self.response_format != "mode":
            raise ValueError("SmoothedModeLLMPolicy supports discrete mode output only")
        self.transition_tau_seconds = float(transition_tau_seconds)
        if self.transition_tau_seconds <= 0.0:
            raise ValueError("transition_tau_seconds must be positive")
        self.target_params = self.params.copy()
        self.applied_params = self.params.copy()

    def update_params_from_llm(self, state, current_s, opponents=None):
        # The parent retains the selected preset in last_llm_params.  Restore
        # the applied state before prompting so logged/current parameters remain
        # those actually used by MPC, then capture the newly selected target.
        self.params = self.applied_params.copy()
        super().update_params_from_llm(state, current_s, opponents=opponents)
        self.target_params = np.asarray(self.params, dtype=float).copy()

        alpha = 1.0 - math.exp(-self.decision_dt / self.transition_tau_seconds)
        self.applied_params += alpha * (self.target_params - self.applied_params)
        self.params = self.applied_params.copy()

