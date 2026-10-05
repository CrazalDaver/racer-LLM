# Direct-parameter modular input v1

`direct_params_modular_v1` is an experimental LLM-interface schema used on top
of the frozen `K-Constrained-v1` substrate. It does not change the plant, MPC,
track/raceline, timing, parameter bounds, scenarios, or metrics.

The direct-parameter policy must be selected with `--llm-output-mode params`.
The LLM returns exactly `q`, `zeta`, `s1`, `s2`, and `s3`; the MPC remains solely
responsible for throttle and steering.

## Always-present base

```text
ego_state
├── speed_mps
├── lateral_position_m
└── heading_error_rad

traffic_state
└── all_opponents[]
    ├── id
    ├── relative_longitudinal_position_m
    ├── relative_lateral_position_m
    └── relative_speed_mps

road_preview
└── samples_near_to_far[]
    ├── distance_ahead_m
    ├── centerline_curvature_per_m
    └── raceline_offset_m
```

Signs are factual and ego-relative: longitudinal position is positive ahead;
lateral position is positive left; relative speed is opponent minus ego;
heading error is ego heading minus centerline heading; curvature is positive
for a left turn. Every opponent is included and sorted by stable car ID.

## Independently controlled modules

All optional modules default to `false`. Each switch adds only its named
top-level JSON object.

| Module | CLI switch | Contents |
|---|---|---|
| `raceline_speed` | `--llm-include-raceline-speed` | Target speed at each road-preview point. |
| `current_parameters` | `--llm-include-current-parameters` | Currently held `q,zeta,s1,s2,s3`. |
| `decision_feedback` | `--llm-include-decision-feedback` | Previous-interval progress, speed, and per-opponent absolute-gap changes; `null` before an interval is complete. |

Example base arm:

```bash
.venv/bin/python main.py --suite llm-params \
  --llm-output-mode params \
  --no-llm-include-raceline-speed \
  --no-llm-include-current-parameters \
  --no-llm-include-decision-feedback
```

For an ablation, change one optional switch only. `current_parameters` exposes
the controller's current action but no earlier states. `decision_feedback` is
the only temporal-memory module in this schema and should remain off in the
minimal capability baseline.

Saved traces record the schema version, all three switches, exact system
prompt, exact output schema, and raw per-call prompts/responses. Formal arms
must additionally complete `experiment-arm-template.json` and start from a
clean commit.
