#!/usr/bin/env python3
"""TEMPORARY controlled audit of the fixed block preset.

This keeps the straight track, initial state, MPC, and scripted 5 m/s rear car
fixed.  Only interpretable block-preset components are changed.  It is an
ablation of parameter semantics, not a search for a globally optimal preset.

Usage:
  .venv/bin/python evaluation/tmp_06_block_preset_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import params
from evaluation.tmp_05_alpha_straight_interaction_probe import (
    run_case,
    summarize,
    write_long_straight,
)


def variants():
    current = np.asarray(params.LLM_MODE_PRESETS["block"], dtype=float)
    cases = {
        "no_block_control": current.copy(),
        "current_block": current.copy(),
        "max_s3": current.copy(),
        "longer_range": current.copy(),
        "longer_range_max_s3": current.copy(),
        "lower_zeta": current.copy(),
    }
    cases["no_block_control"][4] = 0.0
    cases["max_s3"][4] = 1.0
    cases["longer_range"][3] = 0.1
    cases["longer_range_max_s3"][3:] = [0.1, 1.0]
    cases["lower_zeta"][1] = 0.85
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description="Temporary fixed-block semantic audit.")
    parser.add_argument("--total-time", type=float, default=6.0, help="Seconds per fixed-preset rollout.")
    parser.add_argument("--output-dir", type=Path, default=Path("result/tmp_06_block_preset_audit"))
    parser.add_argument("--velocity-track-weight", type=float, default=10.0)
    parser.add_argument("--track-slack-weight", type=float, default=30000.0)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--gif-stride", type=int, default=5, help="Unused; GIF generation is disabled.")
    parser.add_argument("--weight", type=Path, default=Path("train/weights/alpha_racer/potential_local.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.make_gif = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centerline, raceline = write_long_straight(args.output_dir)
    rows = []
    for name, theta in variants().items():
        print(f"Running {name}: {theta.tolist()}", flush=True)
        trace, target_idx, speeds = run_case(
            args, "block", name, centerline, raceline, fixed_params=theta
        )
        row = summarize("block", name, trace, target_idx, speeds)
        row["params"] = theta.tolist()
        rows.append(row)

    payload = {
        "purpose": "Fixed block-preset component ablation on an identical straight-line rear-threat scene.",
        "note": "The scripted rear car never reacts, so pass delay and p_bl activation—not winning—are the primary diagnostics.",
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    fields = [
        "controller", "crossing_times_seconds", "collision_time_seconds",
        "final_lead_m", "max_abs_block_reference_m", "max_abs_lateral_m",
        "progress_m", "params",
    ]
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
