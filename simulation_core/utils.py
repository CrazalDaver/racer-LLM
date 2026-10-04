"""Small construction helpers for vehicles, controllers, and policies.

This is the only module that knows how a canonical policy name maps to a
concrete class. The simulation loop only receives already assembled objects.
"""

import numpy as np

from config import params
from policy import (
    AlphaRacerPolicy, BalancedRawLLMPolicy, CenterLinePolicy,
    EvidenceFirstLLMPolicy, ExplicitRuleLLMPolicy, IteratedBestResponsePolicy,
    FixedModePolicy, LLMAlphaRacerPolicy, OracleModePolicy,
    ParameterIBRPolicy, RuleBasedPolicy,
    SelfPlayRLPolicy, SmoothedModeLLMPolicy, StaticAlphaRacerPolicy,
    StructuredBasisActionFirstLLMPolicy,
)
from .config import validate_policy_name
from vehicle_model import DynamicBicycleModel, KinematicBicycleModel


def resolve_policy_names(policy_type, policy_names, num_cars):
    """Validate the requested roster and expand a single-policy shorthand."""
    if policy_names is not None:
        if len(policy_names) != num_cars:
            raise ValueError(f"policy_names length must equal num_cars ({num_cars}), got {len(policy_names)}")
        return [validate_policy_name(name) for name in policy_names]
    policy_type = validate_policy_name(policy_type)
    if policy_type == "llm_alpharacer" and num_cars > 1:
        return ["llm_alpharacer"] + ["ibr"] * (num_cars - 1)
    return [policy_type] * num_cars


def per_car_option(value, car_idx, default=None):
    """Resolve a scalar or per-car option without leaking list handling to callers."""
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if car_idx >= len(value):
            return default
        item = value[car_idx]
        if item is None or (isinstance(item, str) and item.strip().lower() in {"", "none", "default", "-"}):
            return default
        return item
    return value


def create_vehicle_model(model_type, dt, db_params, max_speed):
    model_type = model_type.lower()
    if model_type in {"kinematic", "kinetic", "kinetics"}:
        return KinematicBicycleModel(dt=dt, db_params=db_params, max_speed=max_speed)
    if model_type == "dbm":
        return DynamicBicycleModel(dt=dt, db_params=db_params)
    raise ValueError(f"Unknown vehicle model: {model_type}")


def create_policy(policy_type, track, mpc, raceline=None, car_idx=0,
                  static_policy_params=None, alpharacer_params=None,
                  alpharacer_weight_path=None, ibr_implementation=None,
                  llm_policy_params=None, llm_output_mode=None,
                  llm_model_name=None, llm_temperature=None,
                  llm_update_interval_seconds=None,
                  llm_observation_variant="baseline",
                  oracle_commitment_seconds=None,
                  self_play_rl_model_path=None, self_play_rl_backend=None,
                  self_play_rl_strict_load=None, self_play_rl_action_order=None):
    """Construct one policy from canonical identity plus relevant overrides."""
    policy_type = validate_policy_name(policy_type)
    common = dict(car_idx=car_idx, policy_name=policy_type)
    if policy_type == "centerline":
        return CenterLinePolicy(track, mpc, step_distance=params.MPC_DT * params.MAX_RACELINE_SPEED, **common)
    if policy_type == "static_alpharacer":
        fixed = static_policy_params if static_policy_params is not None else alpharacer_params
        return StaticAlphaRacerPolicy(track, mpc, params=np.asarray(params.STATIC_POLICY_PARAMS if fixed is None else fixed, dtype=float).copy(), raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, **common)
    if policy_type == "fixed_push":
        return FixedModePolicy(track, mpc, mode="push", raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, **common)
    if policy_type == "oracle_mode":
        return OracleModePolicy(
            track,
            mpc,
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=params.REF_LATERAL_CLIP,
            commitment_seconds=(
                params.ORACLE_COMMITMENT_SECONDS
                if oracle_commitment_seconds is None
                else oracle_commitment_seconds
            ),
            decision_dt=params.CONTROL_DT,
            **common,
        )
    llm_classes = {
        "llm_alpharacer": LLMAlphaRacerPolicy,
        "llm_structured_action_first": StructuredBasisActionFirstLLMPolicy,
        "llm_evidence_first": EvidenceFirstLLMPolicy,
        "llm_raw_balanced": BalancedRawLLMPolicy,
        "llm_explicit_rules": ExplicitRuleLLMPolicy,
        "llm_smoothed_mode": SmoothedModeLLMPolicy,
    }
    if policy_type in llm_classes:
        return llm_classes[policy_type](track, mpc, params=np.asarray(params.LLM_POLICY_PARAMS if llm_policy_params is None else llm_policy_params, dtype=float).copy(), raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, model_name=params.LLM_MODEL if llm_model_name is None else llm_model_name, base_url=params.LLM_BASE_URL, timeout=params.LLM_TIMEOUT, temperature=params.LLM_TEMPERATURE if llm_temperature is None else llm_temperature, update_interval_seconds=params.LLM_UPDATE_INTERVAL_SECONDS if llm_update_interval_seconds is None else llm_update_interval_seconds, decision_dt=params.CONTROL_DT, history_length=params.LLM_RECENT_MODES_MAX_LENGTH, output_mode=params.LLM_OUTPUT_MODE if llm_output_mode is None else llm_output_mode, observation_variant=llm_observation_variant, **common)
    if policy_type == "rule_based":
        return RuleBasedPolicy(track, mpc, params=np.asarray(params.LLM_POLICY_PARAMS if llm_policy_params is None else llm_policy_params, dtype=float).copy(), raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, update_interval_seconds=params.LLM_UPDATE_INTERVAL_SECONDS if llm_update_interval_seconds is None else llm_update_interval_seconds, decision_dt=params.CONTROL_DT, history_length=params.LLM_RECENT_MODES_MAX_LENGTH, **common)
    if policy_type == "alpharacer":
        return AlphaRacerPolicy(track, mpc, raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, weight_path=params.ALPHARACER_WEIGHT_PATH if alpharacer_weight_path is None else alpharacer_weight_path, search_points=params.ALPHARACER_WEIGHT_SEARCH_POINTS, search_iters=params.ALPHARACER_WEIGHT_SEARCH_ITERS, opt_style=params.POTENTIAL_FUNCTION_OPT_STYLE, gradient_rate=params.POTENTIAL_FUNCTION_GRADIENT_RATE, initial_params=params.ALPHARACER_INITIAL_PARAMS, **common)
    if policy_type == "self_play_rl":
        return SelfPlayRLPolicy(track, mpc, params=np.asarray(params.STATIC_POLICY_PARAMS if alpharacer_params is None else alpharacer_params, dtype=float).copy(), raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, model_path=params.SELF_PLAY_RL_MODEL_PATH if self_play_rl_model_path is None else self_play_rl_model_path, backend=params.SELF_PLAY_RL_BACKEND if self_play_rl_backend is None else self_play_rl_backend, strict_load=params.SELF_PLAY_RL_STRICT_LOAD if self_play_rl_strict_load is None else self_play_rl_strict_load, action_order=params.SELF_PLAY_RL_ACTION_ORDER if self_play_rl_action_order is None else self_play_rl_action_order, throttle_bounds=params.SELF_PLAY_RL_THROTTLE_BOUNDS, steer_bounds=params.SELF_PLAY_RL_STEER_BOUNDS, **common)
    if policy_type == "ibr":
        return IteratedBestResponsePolicy(track, mpc, iterations=params.IBR_ITERATIONS, implementation=params.IBR_IMPLEMENTATION if ibr_implementation is None else ibr_implementation, raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, **common)
    if policy_type == "parameter_ibr":
        return ParameterIBRPolicy(track, mpc, raceline=raceline, max_speed=params.MAX_RACELINE_SPEED, ref_lateral_clip=params.REF_LATERAL_CLIP, initial_params=params.ALPHARACER_INITIAL_PARAMS, search_points=params.PARAMETER_IBR_SEARCH_POINTS, search_passes=params.PARAMETER_IBR_SEARCH_PASSES, update_interval_seconds=params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS, convergence_tolerance=params.PARAMETER_IBR_CONVERGENCE_TOLERANCE, **common)
    raise ValueError(f"Unknown policy type: {policy_type}")
