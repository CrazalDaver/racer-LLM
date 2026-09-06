#!/usr/bin/env python3
"""TEMPORARY: compare alpha-potential online gradient step sizes.

This is a disposable diagnostic, intentionally kept outside the policy and
main CLI.  It runs alpha-RACER self-play from all three starting positions for
each requested step size and writes the usual NPZ, JSON, and combined HTML
report under one result directory.

Example:
    .venv/bin/python evaluation/tmp_01_alpharacer_step_size_sweep.py \
        --rates 1e-5 3e-5 1e-4 --total-time 50
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import params
from main import generate_report, run_position_sweep


def _rate_label(rate: float) -> str:
    return f"eta_{rate:.0e}".replace("+", "").replace("-", "m")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporary three-position alpha-RACER gradient-step sweep."
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[1e-5, 3e-5, 1e-4],
        help="Normalized-coordinate gradient step sizes (default: 1e-5 3e-5 1e-4).",
    )
    parser.add_argument("--total-time", type=float, default=50.0, help="Seconds per race.")
    parser.add_argument("--track", default="berlin", help="Track name.")
    parser.add_argument("--output-dir", default="result/tmp_01_alpharacer_step_size_sweep")
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for rate in args.rates:
        if rate <= 0:
            raise ValueError("Every gradient step size must be positive.")
        params.POTENTIAL_FUNCTION_GRADIENT_RATE = float(rate)
        case_dir = output_dir / _rate_label(rate)
        summary_path = case_dir / "summary.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== alpha self-play: gradient rate={rate:g} ===", flush=True)
        run_position_sweep(
            policy_names=["alpharacer", "alpharacer", "alpharacer"],
            ego_positions=(1, 2, 3),
            trials_per_position=1,
            output_path=summary_path,
            keep_npz=True,
            track_name=args.track,
            total_time=args.total_time,
            velocity_track_weight=args.velocity_track_weight,
            track_slack_weight=args.track_slack_weight,
            horizon=args.mpc_horizon,
        )
        summaries.append(summary_path)

    generate_report(
        summaries,
        output=output_dir / "report.html",
        include_animation=False,
    )
    print(f"\nCombined report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
