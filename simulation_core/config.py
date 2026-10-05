"""Simple, flat configuration for one simulation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


PolicyName = Literal[
    "centerline", "static_alpharacer", "fixed_push", "oracle_mode",
    "alpharacer", "ibr", "parameter_ibr",
    "llm_alpharacer", "rule_based", "self_play_rl",
]
CANONICAL_POLICY_NAMES = frozenset(PolicyName.__args__)
LLM_OBSERVATION_VARIANTS = (
    "baseline",
    "all_opponents",
    "no_lap_progress",
    "raceline_speed",
    "decision_feedback",
)


def validate_policy_name(name: str) -> str:
    """Return one canonical policy name or fail with an actionable error."""
    value = str(name).strip()
    if value not in CANONICAL_POLICY_NAMES:
        choices = ", ".join(sorted(CANONICAL_POLICY_NAMES))
        raise ValueError(f"Unknown policy {name!r}. Use one of: {choices}")
    return value


@dataclass(frozen=True)
class SimulationConfig:
    """All user-facing inputs for one race, grouped visually by prefix."""

    # Race
    policies: Sequence[str] = field(default_factory=lambda: ("centerline",))
    duration: float | None = None
    ego_position: int = 1
    random_seed: int | None = None

    # Track and starting grid
    track_name: str | None = None
    track_scale: float | None = None
    raceline: str | None = None
    start_gap: float | None = None
    start_s_offsets: Sequence[float] | None = None
    start_e_offsets: Sequence[float] | None = None
    start_speeds: Sequence[float] | None = None
    initial_speed: float = 0.0

    # Vehicle and MPC
    vehicle_model: str | None = None
    mpc_formulation: str | None = None
    mpc_horizon: int | None = None
    mpc_velocity_weight: float | None = None
    mpc_track_slack: float | None = None
    mpc_collision_slack: float | None = None

    # Static and trained AlphaRacer
    static_params: Sequence[float] | None = None
    static_params_by_car: Sequence[Sequence[float]] | None = None
    alpha_params: Sequence[float] | None = None
    alpha_weight_path: str | None = None
    alpha_weight_paths: Sequence[str | None] | None = None

    # IBR
    ibr_implementation: str | None = None

    # LLM policies
    llm_params: Sequence[float] | None = None
    llm_output_mode: str | None = None
    llm_model: str | None = None
    llm_models: Sequence[str | None] | None = None
    llm_temperature: float | None = None
    llm_temperatures: Sequence[float | None] | None = None
    llm_update_interval: float | None = None
    llm_observation_variant: str = "baseline"
    llm_include_raceline_speed: bool | None = None
    llm_include_current_parameters: bool | None = None
    llm_include_decision_feedback: bool | None = None

    # Discrete receding-horizon oracle.  ``None`` uses the documented defaults
    # in config.params, keeping ordinary simulations unchanged.
    oracle_horizon_seconds: float | None = None
    oracle_commitment_seconds: float | None = None
    oracle_progress_weight: float | None = None
    oracle_relative_progress_weight: float | None = None
    oracle_collision_penalty: float | None = None
    oracle_offtrack_penalty_per_second: float | None = None
    oracle_tie_tolerance: float | None = None

    # Self-play RL
    rl_model_path: str | None = None
    rl_backend: str | None = None
    rl_strict_load: bool | None = None
    rl_action_order: str | None = None

    # Output
    save_trace: bool = True
    trace_output: str | Path | None = None
    show_plot: bool = False
    quiet: bool = False

    def validated_policies(self) -> tuple[str, ...]:
        if not self.policies:
            raise ValueError("SimulationConfig.policies cannot be empty")
        return tuple(validate_policy_name(name) for name in self.policies)
