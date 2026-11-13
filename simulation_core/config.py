"""Low-level vehicle parameters and user inputs for one simulation run."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


GRAVITY = 9.81


class DynamicParams:
    """Physical and actuator parameters for one vehicle model."""

    def __init__(self, vehicle_model_type="dbm"):
        model_type = vehicle_model_type.lower()

        # Geometry and mass
        self.LF: float = 0.11
        self.LR: float = 0.23
        self.MASS: float = 4.65
        self.Iz: float = 0.1

        # Integration and tire model
        self.DT: float = 0.05
        self.K_RFY: float = 20.0
        self.K_FFY: float = 20.0
        self.mu: float = 3.0
        self.Cf: float = 1.0
        self.Cr: float = 1.0
        self.Bf: float = 60.0
        self.Br: float = 60.0
        self.hcom: float = 0.0
        self.fr: float = 0.1

        # Throttle and steering actuators
        self.Ta: float = 5.5
        self.Tb: float = -1.0
        self.Sa: float = 0.36
        self.Sb: float = 0.03
        self.delay: int = 4
        self.ros_kinematic = False

        if model_type in {"kinematic", "kinetic", "kinetics"}:
            self._use_kinematic_baseline()
        elif model_type == "dbm":
            self._use_upstream_numerical_dbm()
        else:
            raise ValueError(f"Unknown vehicle model: {vehicle_model_type}")

    def _use_upstream_numerical_dbm(self):
        """Match alpha-RACER's numerical-race DBM instantiation.

        The release overrides the model defaults with Sa=.34, Sb=0, Ta=20,
        Tb=0, mu=.5, and delay=1. The local simulator then integrates this
        plant at SIM_DT rather than the release's .1 s numerical step.
        """
        self.LF = 0.11
        self.LR = 0.23
        self.MASS = 4.65
        self.Iz = 0.1
        self.Ta = 20.0
        self.Tb = 0.0
        self.Sa = 0.34
        self.Sb = 0.0
        self.mu = 0.5
        self.Cf = 1.0
        self.Cr = 1.0
        self.Bf = 60.0
        self.Br = 60.0
        self.hcom = 0.0
        self.fr = 0.1
        self.delay = 1

    def _use_kinematic_baseline(self):
        # The released ROS MPC predictor uses Sa=.34. Earlier local revisions
        # used Sa=.80 as a raceline-tracking workaround.
        self.LF = 0.11
        self.LR = 0.23
        self.Sa = 0.34
        self.Sb = 0.0
        self.Ta = 20.0 / 4.65
        self.Tb = 0.0
        self.fr = 1.0
        self.ros_kinematic = True

    def copy(self):
        return copy.copy(self)


PolicyName = Literal[
    "centerline", "static_alpharacer", "fixed_push", "oracle_mode",
    "alpharacer", "mode_alpharacer", "ibr",
    "llm_alpharacer", "self_play_rl",
]
CANONICAL_POLICY_NAMES = frozenset(PolicyName.__args__)
LLM_OBSERVATION_VARIANTS = (
    "baseline",
    "all_opponents",
    "no_lap_progress",
    "raceline_speed",
    "decision_feedback",
)
LLM_PROMPT_MODULES = ("base", "base_plus_mechanics", "mode_v2")


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
    mpc_formulation: str | None = None  # Replay compatibility; only constrained_local is supported.
    mpc_horizon: int | None = None
    mpc_min_state_speed: float | None = None
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
    ibr_strategy_space: str | None = None

    # LLM policies
    llm_params: Sequence[float] | None = None
    llm_output_mode: str | None = None
    llm_model: str | None = None
    llm_models: Sequence[str | None] | None = None
    llm_temperature: float | None = None
    llm_temperatures: Sequence[float | None] | None = None
    llm_update_interval: float | None = None
    llm_observation_variant: str = "baseline"
    llm_prompt_module: str = "base"
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

    # Optional learned-potential paired-rollout diagnostic. Disabled by
    # default so existing simulations do not load a network or branch state.
    potential_regret_enabled: bool = False
    potential_regret_weight_path: str | Path | None = None
    potential_regret_horizon_seconds: float = 1.0
    potential_regret_sobol_points: int = 8
    potential_regret_max_iterations: int = 40
    potential_regret_optimizer_seed: int = 0
    potential_regret_device: str | None = None

    # Self-play RL
    rl_model_path: str | None = None

    # Output
    save_trace: bool = True
    trace_output: str | Path | None = None
    show_plot: bool = False
    quiet: bool = False

    def validated_policies(self) -> tuple[str, ...]:
        if not self.policies:
            raise ValueError("SimulationConfig.policies cannot be empty")
        return tuple(validate_policy_name(name) for name in self.policies)
