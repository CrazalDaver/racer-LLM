#!/usr/bin/env python3
"""TEMPORARY controlled audit of instantaneous vs smoothed mode application.

No LLM server is called.  Each branch returns a scripted mode through the same
LLM-policy response path, isolating the parameter transition from prompting and
mode-selection quality.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulation
from config import params
from evaluation.mode_preset_audit import (
    build_scenarios,
    summarize_rollout,
    validate_scenario,
)
from policy import LLMAlphaRacerPolicy, SmoothedModeLLMPolicy
from track import Track, configure_track, load_centerline_csv


class _ScriptedResponseMixin:
    scripted_mode = "stable"

    def _query_llm(self, _prompt):
        return {"mode": self.scripted_mode, "reason": "controlled transition audit"}


class _InstantScriptedPolicy(_ScriptedResponseMixin, LLMAlphaRacerPolicy):
    pass


class _SmoothedScriptedPolicy(_ScriptedResponseMixin, SmoothedModeLLMPolicy):
    pass


def _run_branch(args, track, scenario, mode, variant):
    policy_class = _InstantScriptedPolicy if variant == "instant" else _SmoothedScriptedPolicy
    policy_class.scripted_mode = mode
    original_create_policy = simulation.create_policy

    def create_policy(policy_type, *factory_args, **factory_kwargs):
        if policy_type == "llm_alpharacer":
            track_arg, mpc_arg = factory_args
            policy_kwargs = {
                "car_idx": factory_kwargs["car_idx"],
                "policy_name": f"scripted_{variant}",
                "params": np.asarray(params.LLM_MODE_PRESETS["stable"]).copy(),
                "raceline": factory_kwargs["raceline"],
                "max_speed": params.MAX_RACELINE_SPEED,
                "ref_lateral_clip": params.REF_LATERAL_CLIP,
                "model_name": "scripted_transition_audit",
                "base_url": params.LLM_BASE_URL,
                "timeout": params.LLM_TIMEOUT,
                "temperature": 0.0,
                "update_interval_seconds": args.mode_interval,
                "decision_dt": params.CONTROL_DT,
                "history_length": params.LLM_RECENT_MODES_MAX_LENGTH,
                "output_mode": "mode",
                "safeguard_enabled": False,
            }
            if variant == "smooth":
                policy_kwargs["transition_tau_seconds"] = args.tau
            return policy_class(track_arg, mpc_arg, **policy_kwargs)
        return original_create_policy(policy_type, *factory_args, **factory_kwargs)

    simulation.create_policy = create_policy
    try:
        count = len(scenario.s)
        ranks = np.arange(count, dtype=float)
        offsets = np.asarray(scenario.s) + ranks * args.start_gap
        trace = simulation.run_simulation(
            track_name=args.track,
            save_trace=False,
            show_plot=False,
            num_cars=count,
            policy_names=["llm_alpharacer"] + ["static"] * (count - 1),
            static_policy_params_per_car=np.vstack(
                [params.LLM_MODE_PRESETS["stable"]] * count
            ),
            llm_policy_params=np.asarray(params.LLM_MODE_PRESETS["stable"]),
            ego_position=1,
            start_gap=args.start_gap,
            initial_longitudinal_offsets=offsets,
            initial_lateral_offsets=np.asarray(scenario.e),
            initial_speeds=np.asarray(scenario.speed),
            total_time=args.rollout_seconds,
            horizon=args.mpc_horizon,
            velocity_track_weight=args.velocity_track_weight,
            track_slack_weight=args.track_slack_weight,
            quiet=True,
        )
    finally:
        simulation.create_policy = original_create_policy

    row = summarize_rollout(trace, scenario, mode)
    row["variant"] = variant
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporary controlled audit of smoothed LLM mode parameters."
    )
    parser.add_argument("--track", default="berlin")
    parser.add_argument("--rollout-seconds", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--mode-interval", type=float, default=1.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--start-gap", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("result/tmp_03_smoothed_mode_transition_audit"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    by_name = {scene.name: scene for scene in build_scenarios(track)}
    cases = [
        ("clear_curve_approach", "stable"),
        ("clear_straight", "push"),
        ("closing_on_front", "overtake"),
        ("rear_vehicle_closing", "block"),
    ]
    params.DEADLOCK_RESET_ENABLED = False
    params.LOW_SPEED_BOUNDARY_RECOVERY_ENABLED = False
    rows = []
    for scene_name, mode in cases:
        scene = by_name[scene_name]
        validate_scenario(track, scene)
        for variant in ("instant", "smooth"):
            print(f"{scene_name}: {mode}, {variant}", flush=True)
            rows.append(_run_branch(args, track, scene, mode, variant))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "Controlled transition-only audit; scripted modes, no LLM calls.",
        "tau_seconds": args.tau,
        "mode_interval_seconds": args.mode_interval,
        "rollout_seconds": args.rollout_seconds,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    with (args.output_dir / "rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
