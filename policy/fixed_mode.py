"""Baseline that runs one existing discrete mode preset unchanged."""

from __future__ import annotations

import numpy as np

from config import params as sim_params

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
