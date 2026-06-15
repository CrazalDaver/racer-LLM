import argparse
import json
import re
import sys
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
    first_policy = _slug(policy_names[0]) if policy_names else "static"
    if first_policy in {"llm", "llm-alpha", "llm-alpharacer", "llm_alpharacer", "llm_parameter_policy"}:
        return snapshot["LLM_POLICY_PARAMS"]
    return snapshot.get("STATIC_POLICY_PARAMS", snapshot["ALPHARACER_PARAMS"])


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
    model_suffix = ""
    if snapshot.get("LLM_MODEL_NAMES"):
        model_slug = "-".join(_slug(model or "default") for model in snapshot["LLM_MODEL_NAMES"])
        model_suffix = f"-models-{model_slug}"
    return f"policies-{policy_slug}{ego_suffix}-{_policy_param_slug(policy_params)}{model_suffix}"


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


def simulation_kwargs(
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_update_interval_steps=None,
):
    return {
        "vehicle_model_type": vehicle_model_type,
        "static_policy_params": static_policy_params,
        "alpharacer_params": alpharacer_params,
        "llm_policy_params": llm_policy_params,
        "llm_output_mode": llm_output_mode,
        "llm_model_name": llm_model_name,
        "llm_model_names": llm_model_names,
        "llm_update_interval_steps": llm_update_interval_steps,
    }


def _override_snapshot(
    snapshot,
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_update_interval_steps=None,
    ego_position=None,
):
    snapshot = dict(snapshot)
    overrides = simulation_kwargs(
        vehicle_model_type=vehicle_model_type,
        static_policy_params=static_policy_params,
        alpharacer_params=alpharacer_params,
        llm_policy_params=llm_policy_params,
        llm_output_mode=llm_output_mode,
        llm_model_name=llm_model_name,
        llm_model_names=llm_model_names,
        llm_update_interval_steps=llm_update_interval_steps,
    )
    if overrides["vehicle_model_type"] is not None:
        snapshot["VEHICLE_MODEL"] = overrides["vehicle_model_type"]
    fixed_params = overrides["static_policy_params"]
    if fixed_params is None:
        fixed_params = overrides["alpharacer_params"]
    if fixed_params is not None:
        fixed_params = [float(value) for value in fixed_params]
        snapshot["STATIC_POLICY_PARAMS"] = fixed_params
        snapshot["ALPHARACER_PARAMS"] = fixed_params
    if overrides["llm_policy_params"] is not None:
        snapshot["LLM_POLICY_PARAMS"] = [float(value) for value in overrides["llm_policy_params"]]
    if overrides["llm_output_mode"] is not None:
        snapshot["LLM_OUTPUT_MODE"] = overrides["llm_output_mode"]
    if overrides["llm_model_name"] is not None:
        snapshot["LLM_MODEL"] = overrides["llm_model_name"]
    if overrides["llm_model_names"] is not None:
        snapshot["LLM_MODEL_NAMES"] = list(overrides["llm_model_names"])
    if overrides["llm_update_interval_steps"] is not None:
        snapshot["LLM_UPDATE_INTERVAL_STEPS"] = int(overrides["llm_update_interval_steps"])
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
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_model_names=None,
    llm_update_interval_steps=None,
    ego_position=1,
    show_ref_debug=False,
):
    """
    Run a single experiment with specified policies and config overrides,
    save trace and animation under result/.
    """
    sim_kwargs = simulation_kwargs(
        vehicle_model_type=vehicle_model_type,
        static_policy_params=static_policy_params,
        alpharacer_params=alpharacer_params,
        llm_policy_params=llm_policy_params,
        llm_output_mode=llm_output_mode,
        llm_model_name=llm_model_name,
        llm_model_names=llm_model_names,
        llm_update_interval_steps=llm_update_interval_steps,
    )
    snapshot = params_snapshot()
    snapshot = _override_snapshot(snapshot, **sim_kwargs, ego_position=ego_position)
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
            ego_position=ego_position,
            **sim_kwargs,
        )

        if save_animation:
            make_animation(
                trace_path,
                stride=stride,
                interval=interval,
                save_path=animation_path,
                dpi=dpi,
                show_ref_debug=show_ref_debug,
            )
    finally:
        params.TRACE_OUTPUT = old_trace_output

    return output_dir


def _ego_first_from_start_order(values, ego_position):
    ego_index = int(ego_position) - 1
    if ego_index < 0 or ego_index >= len(values):
        raise ValueError(f"ego_position must be between 1 and {len(values)}, got {ego_position}")
    return [values[ego_index], *[value for idx, value in enumerate(values) if idx != ego_index]]


def _as_case_list(cases):
    return [(list(start_order), int(ego_position)) for start_order, ego_position in cases]


def _llm_model_case_for_index(llm_model_cases, case_idx, car_count):
    if llm_model_cases is None:
        return None

    if not llm_model_cases:
        raise ValueError("llm_model_cases cannot be empty.")

    first_item = llm_model_cases[0]
    if isinstance(first_item, str) or first_item is None:
        model_case = llm_model_cases
    else:
        if case_idx >= len(llm_model_cases):
            raise ValueError(
                f"Missing llm_model_cases entry for case {case_idx}; "
                f"got {len(llm_model_cases)} model cases."
            )
        model_case = llm_model_cases[case_idx]

    if len(model_case) != car_count:
        raise ValueError(f"LLM model case length must be {car_count}, got {len(model_case)}.")
    return list(model_case)


def _case_name(name_prefix, start_order, ego_position, start_order_models=None):
    policy_slug = "-".join(_slug(policy) for policy in start_order)
    name = f"{name_prefix}-" if name_prefix else ""
    name = f"{name}startorder-{policy_slug}-ego{ego_position}"
    if start_order_models is not None:
        model_slug = "-".join(_slug(model or "default") for model in start_order_models)
        name = f"{name}-models-{model_slug}"
    return name


def run_experiment_suite(
    start_order_cases,
    llm_model_cases=None,
    name_prefix="suite",
    save_animation=True,
    stride=2,
    interval=30,
    dpi=100,
    show_plot=False,
    show_ref_debug=False,
    vehicle_model_type=None,
    static_policy_params=None,
    alpharacer_params=None,
    llm_policy_params=None,
    llm_output_mode=None,
    llm_model_name=None,
    llm_update_interval_steps=None,
):
    """
    Run experiments described in start-order coordinates.

    start_order_cases is a list of (policies_in_start_order, ego_position).
    llm_model_cases is either one model list reused for every case, or a list
    of model lists aligned with start_order_cases. Model lists are also in
    start-order coordinates. Internally both are converted to ego-first order
    because run_simulation treats car 0 as ego.
    """
    output_dirs = []
    for case_idx, (start_order, ego_position) in enumerate(_as_case_list(start_order_cases)):
        start_order_models = _llm_model_case_for_index(llm_model_cases, case_idx, len(start_order))
        policy_names = _ego_first_from_start_order(start_order, ego_position)
        llm_model_names = (
            _ego_first_from_start_order(start_order_models, ego_position)
            if start_order_models is not None
            else None
        )

        output_dirs.append(
            run_experiment(
                policy_names=policy_names,
                name=_case_name(name_prefix, start_order, ego_position, start_order_models),
                save_animation=save_animation,
                stride=stride,
                interval=interval,
                dpi=dpi,
                show_plot=show_plot,
                show_ref_debug=show_ref_debug,
                vehicle_model_type=vehicle_model_type,
                static_policy_params=static_policy_params,
                alpharacer_params=alpharacer_params,
                llm_policy_params=llm_policy_params,
                llm_output_mode=llm_output_mode,
                llm_model_name=llm_model_name,
                llm_model_names=llm_model_names,
                llm_update_interval_steps=llm_update_interval_steps,
                ego_position=ego_position,
            )
        )
    return output_dirs


def _parse_start_order_cases(start_order_args):
    if not start_order_args:
        return None

    cases = []
    for text in start_order_args:
        parts = [part.strip() for part in str(text).replace(":", ",").split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError("Use --start-order policy1,policy2,...,ego_position")
        cases.append((parts[:-1], int(parts[-1])))
    return cases


def _add_case_args(parser):
    group = parser.add_argument_group("experiment cases")
    group.add_argument("--policies", nargs="+", default=None)
    group.add_argument("--ego-position", type=int, default=1)
    group.add_argument("--start-order", action="append", default=None)
    group.add_argument("--name", default=None)


def _add_sim_args(parser):
    group = parser.add_argument_group("simulation")
    group.add_argument("--vehicle-model", choices=["kinematic", "kinetic", "kinetics", "dbm"], default=None)
    group.add_argument("--static-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--alpha-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--llm-params", nargs=5, type=float, metavar=("Q", "ZETA", "S1", "S2", "S3"), default=None)
    group.add_argument("--llm-output-mode", choices=["auto", "mode", "params"], default=None)
    group.add_argument("--llm-model", default=None)
    group.add_argument("--llm-models", nargs="+", default=None)
    group.add_argument("--llm-update-interval-steps", type=int, default=None)


def _add_output_args(parser):
    group = parser.add_argument_group("output")
    group.add_argument("--no-animation", action="store_true")
    group.add_argument("--show-plot", action="store_true")
    group.add_argument("--show-ref-debug", action="store_true")
    group.add_argument("--stride", type=int, default=2)
    group.add_argument("--interval", type=int, default=30)
    group.add_argument("--dpi", type=int, default=100)


def parse_args():
    parser = argparse.ArgumentParser(description="Run racer experiments.")
    _add_case_args(parser)
    _add_sim_args(parser)
    _add_output_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    common_kwargs = dict(
        save_animation=not args.no_animation,
        stride=args.stride,
        interval=args.interval,
        dpi=args.dpi,
        show_plot=args.show_plot,
        show_ref_debug=args.show_ref_debug,
        vehicle_model_type=args.vehicle_model,
        static_policy_params=args.static_params if args.static_params is not None else args.alpha_params,
        alpharacer_params=args.alpha_params,
        llm_policy_params=args.llm_params,
        llm_output_mode=args.llm_output_mode,
        llm_model_name=args.llm_model,
        llm_update_interval_steps=args.llm_update_interval_steps,
    )

    start_order_cases = _parse_start_order_cases(args.start_order)
    if start_order_cases is not None:
        run_experiment_suite(
            start_order_cases=start_order_cases,
            llm_model_cases=args.llm_models,
            name_prefix=args.name or "suite",
            **common_kwargs,
        )
        return

    if args.policies is None:
        raise ValueError("Pass --policies for one run, or --start-order for a suite.")

    run_experiment(
        policy_names=args.policies,
        name=args.name,
        llm_model_names=args.llm_models,
        ego_position=args.ego_position,
        **common_kwargs,
    )


if __name__ == "__main__":
    # Both lists below are written in start-order coordinates.
    # ego_position says which start-order slot is treated as ego/car0.
    START_ORDER_CASES = [
        (["LLM", "LLM", "LLM"], 3),
        (["LLM", "LLM", "LLM"], 2),
        (["LLM", "LLM", "LLM"], 1),
    ]

    LLM_MODEL_CASES = [
        ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
        ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
        ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
    ]

    if len(sys.argv) == 1: # no args, run the above experiment suite as a demo
        run_experiment_suite(
            start_order_cases=START_ORDER_CASES,
            llm_model_cases=LLM_MODEL_CASES,
            name_prefix="qwen-size-competition",
            save_animation=True,
            vehicle_model_type="kinematic",
            llm_output_mode="mode",
        )
    else:
        main()
