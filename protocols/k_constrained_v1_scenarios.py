#!/usr/bin/env python3
"""Emit the nine frozen K-Constrained-v1 evaluation scenario records."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("K-Constrained-v1.json")


def scenario_matrix() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    spec = manifest["scenario_generation"]
    scenarios = []
    for track in spec["tracks"]:
        for ego_position in spec["ego_start_positions"]:
            scenarios.append(
                {
                    "scenario_id": f"{track}-p{ego_position}",
                    "protocol_version": manifest["protocol_version"],
                    "track_name": track,
                    "ego_position": ego_position,
                    "duration": spec["duration_s"],
                    "random_seed": spec["seed"],
                    "start_gap": spec["start_gap_m"],
                    "start_s_offsets": spec["longitudinal_offsets_m"],
                    "start_e_offsets": spec["lateral_offsets_m"],
                    "start_speeds": spec["initial_speeds_mps"],
                }
            )
    return scenarios


if __name__ == "__main__":
    print(json.dumps(scenario_matrix(), indent=2))
