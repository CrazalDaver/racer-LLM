"""Simple, flat configuration for one simulation run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


PolicyName = Literal[
    "centerline", "static_alpharacer", "alpharacer", "ibr", "parameter_ibr",
    "llm_alpharacer", "llm_raw_balanced", "llm_evidence_first",
    "llm_structured_action_first", "llm_explicit_rules", "llm_smoothed_mode",
    "rule_based", "self_play_rl",
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
