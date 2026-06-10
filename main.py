import argparse
import json
import re
from pathlib import Path

import numpy as np

from animate_trace import make_animation
from config import params
from simulation import run_simulation


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def params_snapshot():
    snapshot = {}
    for name in dir(params):
        if name.startswith("_"):
            continue
        value = getattr(params, name)
        if callable(value):
            continue
        snapshot[name] = _jsonable(value)
    return snapshot


def _slug(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-").lower()


def _policy_params_for_name(policy_names, snapshot):
    first_policy = _slug(policy_names[0]) if policy_names else "alpha"
    if first_policy in {"llm", "llm-alpha", "llm-alpharacer", "llm_alpharacer"}:
        return snapshot["LLM_POLICY_PARAMS"]
    return snapshot["ALPHARACER_PARAMS"]


def _format_param(value):
    return f"{float(value):.1f}"


def _policy_param_slug(values):
    q, zeta, s1, s2, s3 = values
    return "-".join(
        [
            f"q{_format_param(q)}",
            f"zeta{_format_param(zeta)}",
            f"s1-{_format_param(s1)}",
            f"s2-{_format_param(s2)}",
            f"s3-{_format_param(s3)}",
        ]
    )


def experiment_name(policy_names, snapshot, custom_name=None):
    if custom_name:
        return _slug(custom_name)

    policy_slug = "-".join(_slug(name) for name in policy_names)
    policy_params = _policy_params_for_name(policy_names, snapshot)
    ego_suffix = ""
    ego_position = int(snapshot.get("EGO_POSITION", 1))
    if ego_position != 1:
        ego_suffix = f"-ego{ego_position}"
    return f"policies-{policy_slug}{ego_suffix}-{_policy_param_slug(policy_params)}"


def save_experiment_config(output_dir, policy_names, snapshot):
    config_path = output_dir / "experiment_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "policy_names": policy_names,
                "params": snapshot,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    return config_path


def _override_snapshot(
    snapshot,
    vehicle_model_type=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_update_interval_steps=None,
    ego_position=None,
):
    snapshot = dict(snapshot)
    if vehicle_model_type is not None:
        snapshot["VEHICLE_MODEL"] = vehicle_model_type
    if alpharacer_params is not None:
        snapshot["ALPHARACER_PARAMS"] = [float(value) for value in alpharacer_params]
    if llm_policy_params is not None:
        snapshot["LLM_POLICY_PARAMS"] = [float(value) for value in llm_policy_params]
    if llm_output_mode is not None:
        snapshot["LLM_OUTPUT_MODE"] = llm_output_mode
    if llm_model_name is not None:
        snapshot["LLM_MODEL"] = llm_model_name
    if llm_update_interval_steps is not None:
        snapshot["LLM_UPDATE_INTERVAL_STEPS"] = int(llm_update_interval_steps)
    if ego_position is not None:
        snapshot["EGO_POSITION"] = int(ego_position)
    return snapshot


def run_experiment(
    policy_names,
    name=None,
    save_animation=True,
    stride=2,
    interval=30,
    dpi=100,
    show_plot=False,
    vehicle_model_type=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_update_interval_steps=None,
    ego_position=1,
):
    snapshot = params_snapshot()
    snapshot = _override_snapshot(
        snapshot,
        vehicle_model_type=vehicle_model_type,
        alpharacer_params=alpharacer_params,
        llm_policy_params=llm_policy_params,
        llm_output_mode=llm_output_mode,
        llm_model_name=llm_model_name,
        llm_update_interval_steps=llm_update_interval_steps,
        ego_position=ego_position,
    )
    exp_name = experiment_name(policy_names, snapshot, custom_name=name)
    output_dir = Path("result") / f"experiment_{exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "simulation_trace.npz"
    animation_path = output_dir / "simulation_trace.gif"
    old_trace_output = params.TRACE_OUTPUT
    params.TRACE_OUTPUT = str(trace_path)

    try:
        config_path = save_experiment_config(output_dir, policy_names, snapshot)
        print(f"Experiment: {exp_name}")
        print(f"Output dir: {output_dir}")
        print(f"Config: {config_path}")
        run_simulation(
            policy_names=policy_names,
            save_trace=True,
            show_plot=show_plot,
            vehicle_model_type=vehicle_model_type,
            alpharacer_params=alpharacer_params,
            llm_policy_params=llm_policy_params,
            llm_output_mode=llm_output_mode,
            llm_model_name=llm_model_name,
            llm_update_interval_steps=llm_update_interval_steps,
            ego_position=ego_position,
        )

        if save_animation:
            make_animation(
                trace_path,
                stride=stride,
                interval=interval,
                save_path=animation_path,
                dpi=dpi,
            )
    finally:
        params.TRACE_OUTPUT = old_trace_output

    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run a racer experiment and save trace/animation under result/.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["alpha", "IBR", "IBR"],
        help='Per-car policy names, e.g. --policies alpha IBR IBR or --policies llm IBR IBR.',
    )
    parser.add_argument("--name", default=None, help="Optional experiment name suffix. Defaults to policy/config name.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation.")
    parser.add_argument("--show-plot", action="store_true", help="Show the static simulation plot.")
    parser.add_argument(
        "--vehicle-model",
        choices=["kinematic", "kinetic", "kinetics", "dbm"],
        default=None,
        help="Simulation vehicle model. Defaults to config.params.VEHICLE_MODEL.",
    )
    parser.add_argument(
        "--alpha-params",
        nargs=5,
        type=float,
        metavar=("Q", "ZETA", "S1", "S2", "S3"),
        default=None,
        help="Temporary ALPHARACER_PARAMS override for this run.",
    )
    parser.add_argument(
        "--llm-params",
        nargs=5,
        type=float,
        metavar=("Q", "ZETA", "S1", "S2", "S3"),
        default=None,
        help="Temporary LLM_POLICY_PARAMS override for this run.",
    )
    parser.add_argument(
        "--llm-output-mode",
        choices=["auto", "mode", "params"],
        default=None,
        help="LLM output style: mode chooses presets; params asks for q/zeta/s1/s2/s3 directly.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Temporary LLM_MODEL override. Names containing 'params' use direct parameter output in auto mode.",
    )
    parser.add_argument(
        "--llm-update-interval-steps",
        type=int,
        default=None,
        help="Temporary LLM_UPDATE_INTERVAL_STEPS override for this run.",
    )
    parser.add_argument(
        "--ego-position",
        type=int,
        default=1,
        help="1-based starting position for policy_names[0] ego; opponents fill the remaining slots.",
    )
    parser.add_argument("--stride", type=int, default=2, help="Animation frame stride.")
    parser.add_argument("--interval", type=int, default=30, help="Animation frame interval in milliseconds.")
    parser.add_argument("--dpi", type=int, default=100, help="Saved animation DPI.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        policy_names=args.policies,
        name=args.name,
        save_animation=not args.no_animation,
        stride=args.stride,
        interval=args.interval,
        dpi=args.dpi,
        show_plot=args.show_plot,
        vehicle_model_type=args.vehicle_model,
        alpharacer_params=args.alpha_params,
        llm_policy_params=args.llm_params,
        llm_output_mode=args.llm_output_mode,
        llm_model_name=args.llm_model,
        llm_update_interval_steps=args.llm_update_interval_steps,
        ego_position=args.ego_position,
    )


if __name__ == "__main__":
    main()
