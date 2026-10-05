# Minimal direct-parameter LLM experiment

**Status:** implemented as the explicit `llm_param_minimal_v1` experiment arm;
not yet evaluated. The arm is separate from the legacy `llm-params` suite.

## Instructions for an implementing agent

Read, in order:

1. repository-root `AGENTS.md`;
2. `protocols/README.md` and `protocols/K-Constrained-v1.json`;
3. `protocols/direct-params-modular-v1.md`;
4. this document.

Do not change the frozen `K-Constrained-v1` substrate to improve results. Before
editing, audit the current implementation against this brief and identify what
already exists, what is missing, and what would change. Keep implementation
changes small and preserve unrelated work.

The existing `--suite llm-params` is not this minimal experiment: it currently
uses learned alpha-RACER and IBR opponents. Do not silently reinterpret its old
results. Prefer a new named arm/suite or an explicit arm configuration.

## Research question

Can a stateless LLM make useful high-level racing decisions by directly
selecting `(q, zeta, s1, s2, s3)` from the current scene, while the shared MPC
retains sole authority over throttle and steering?

The first experiment should establish whether there is enough signal to justify
more complicated LLM designs. It is not intended to establish final superiority
over alpha-RACER or IBR.

## Fixed substrate

Use `K-Constrained-v1` without modification. In particular, keep the vehicle,
MPC, constraints, recovery/contact behavior, parameter bounds, registered
track/raceline pairs, simulation timestep, and 1 s LLM update interval fixed.

Run the protocol validator before a result-bearing run. Exploratory smoke tests
may use a shorter duration but must be labelled exploratory and must not be
pooled with the frozen 50 s core scan.

## LLM arm: `llm-param-minimal-v1`

Use the `direct_params_modular_v1` interface with these resolved settings:

```text
action interface: direct five-parameter output
output fields: q, zeta, s1, s2, s3 only
update interval: 1.0 simulated seconds
temperature: 0.0
memory/history: none
current_parameters module: false
decision_feedback module: false
raceline_speed module: true
mode vocabulary: none
few-shot examples: none
tactical rules: none
reason/basis output: none
parameter smoothing: none
```

`raceline_speed` is included because `zeta` changes both reference progression
and reference speed. Omitting the local raceline target would make the direct
parameter task unnecessarily under-observed. If its value is later questioned,
test its removal as a separately named one-factor observation ablation rather
than changing this arm in place.

The current observation should contain:

- ego speed, lateral position, and heading error;
- every opponent's ego-relative longitudinal position, lateral position, and
  speed difference;
- fixed-distance road-preview samples containing centerline curvature,
  raceline lateral offset, and raceline target speed;
- factual units and sign conventions;
- the fixed five-parameter definitions and bounds in the system contract.

Do not expose absolute world coordinates, track-specific tactical labels,
oracle rollouts, opponent future plans, previous outcomes, or a hand-written
classification such as `overtake`, `block`, or `push`.

Keep the direct-parameter system prompt fixed across input-module ablations.
It may document the optional module names and meanings, as the current v1
prompt does, provided that it states they may be present and the runtime JSON
contains only enabled modules. This keeps the prompt wording constant while a
comparison changes only the availability of the named data. Disabled-module
values must never be inserted or reconstructed elsewhere in the prompt.

On an invalid response, timeout, or connection failure, retain the last valid
parameters. Before the first valid response, use the frozen neutral/static
vector below. Record the error type and raw response; never silently drop a
failed call.

## Deterministic opponents and comparison arms

Use two identical deterministic opponents in every arm:

```text
oppo1 policy: static_alpharacer
oppo2 policy: static_alpharacer
opponent parameters: [1.3333333333, 0.90, 0.20, 0.30, 0.00]
```

These opponents use the same five-parameter reference/MPC interface, add no
model or solver uncertainty, and are ordinary enough to permit catching and
passing interactions. Keep this opponent roster unchanged across the three ego
arms:

| Arm ID | Ego policy | Purpose |
|---|---|---|
| `static-neutral-v1` | The same fixed neutral/static vector | Ordinary fixed-policy control. |
| `fixed-push-v1` | The frozen `push` preset | Strong simple fixed-policy control. |
| `llm-param-minimal-v1` | Stateless direct-parameter LLM | Candidate adaptive policy. |

The initial/fallback LLM parameters must equal the neutral/static vector so a
failed first call does not grant the LLM a different starting advantage.

This roster isolates the first capability question. Do not add alpha-RACER,
trajectory IBR, parameter IBR, mode selection, or an oracle to this experiment.
Those belong to later, separately named comparisons.

## Execution stages

### Stage 0: integration smoke test

Run one exploratory race:

```text
track: rounded_rectangle
ego start position: 2
duration: 10-20 s
opponents: static_alpharacer, static_alpharacer
ego: llm-param-minimal-v1
```

This stage checks only that:

- the resolved observation contains the intended fields and both opponents;
- disabled modules are absent from the runtime JSON and the shared system
  prompt describes them only as optional;
- the response schema contains exactly five numeric parameters;
- parsing, bounds, fallback, tracing, and MPC execution work;
- the saved resolved configuration matches the requested arm.

Do not treat this result as performance evidence.

### Stage 1: small signal scan

Run all three ego arms from position 2 on each registered track. This is nine
races total:

```text
3 ego arms x 3 tracks x 1 start position = 9 races
```

Use these runs to decide whether the LLM is functional, non-constant, and safe
enough to justify the full core scan.

### Stage 2: complete deterministic core comparison

If Stage 1 passes its checks, run every ego arm at all three start positions on
all three tracks:

```text
3 ego arms x 3 tracks x 3 start positions = 27 races
```

The deterministic controls need no identical repeats. If the nominally
temperature-zero LLM remains nondeterministic, predeclare LLM repeat count and
seeds before expanding the run. Do not choose the repeat count after inspecting
favorable or unfavorable results.

## What to measure

Report every cell before aggregation.

Performance:

- ego progress gain and final unwrapped progress;
- paired progress difference relative to each control in the same cell;
- relative finish position and win rate as secondary summaries.

Safety and reliability:

- physical collision time;
- boundary-violation and recovery-active time;
- maximum track and collision slack;
- MPC solver failures;
- LLM parse, timeout, and connection errors;
- mean/p95 latency and calls exceeding the 1 s decision interval.

Parameter behavior, separately for `q`, `zeta`, `s1`, `s2`, and `s3`:

- mean, standard deviation, minimum, and maximum;
- fraction of decisions at or near each bound;
- mean and maximum change between consecutive decisions;
- total variation over the race;
- variation across tracks and traffic situations.

Save raw decisions and trajectories so scene-conditioned behavior can be
inspected rather than inferred only from aggregate scores.

## Interpretation rules

- If the LLM is worse than the neutral static control, first inspect parse
  failures, parameter saturation, oscillation, and observation/prompt mismatch.
- If it beats neutral but not fixed push, the direct interface may work while a
  simple strong constant remains difficult to beat.
- If it matches fixed push while emitting nearly constant push-like parameters,
  do not claim meaningful adaptive reasoning.
- If it changes parameters by scene but gains no performance, distinguish
  ineffective adaptation from insufficient headroom in the environment.
- If fixed push is already near the best attainable policy in these cells, use
  targeted scenes or a continuous-parameter oracle later to measure headroom;
  do not immediately add prompt complexity.
- Safety regressions must remain visible and must not be hidden inside a single
  performance reward.

## Required run records

Before Stage 1 or Stage 2, create completed arm records rather than editing the
generic template in place. Save both the requested arm and the fully resolved
runtime configuration. The record must include:

- protocol and interface-schema versions;
- policy roster and exact fixed parameter vectors/preset;
- observation modules and rendered system prompt/hash;
- output schema, parser, clipping, timeout, and fallback behavior;
- model identifier, quantization/checkpoint, decoding settings, and update
  interval;
- track, start position, duration, seed, and repeat index;
- Git commit SHA and dirty status;
- raw per-call prompts/responses and the final metric summary.

## Implementation acceptance criteria

An implementing agent should not declare this experiment ready merely because
the code runs. Readiness requires evidence that:

1. protocol validation passes;
2. a resolved arm/config artifact is written before or with every run;
3. the saved config and trace contain the exact selected modules and prompt;
4. a negative test shows that disabled-module values do not leak into the
   runtime observation or any other prompt content;
5. an invalid LLM response exercises and records the specified fallback;
6. all three ego arms use the identical opponent roster and initial conditions;
7. smoke-test outputs are labelled exploratory and cannot be mistaken for the
   50 s core scan.
