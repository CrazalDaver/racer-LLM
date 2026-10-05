# Historical racer-LLM platform code map

> This map predates the current frozen protocol and may contain stale paths.

**Purpose.** This is a local working specification of what the current code
actually executes.  It is not paper text and it is intentionally gitignored:
update it before interpreting a new experiment or changing a baseline.

**Current scope.** The project studies LLM high-level tactical selection in a
shared local multi-car racing platform.  It does **not** claim exact numerical
reproduction of alpha-RACER, Wang IBR, or any ROS-based reference stack.

## 1. Execution map

```text
CLI / experiment shell script
  -> main.py
      records per-run config snapshot and chooses ego start order
  -> simulation.run_simulation()                         [simulation.py]
      loads track + raceline; creates one plant and one MPC per car
      -> create_policy()                                 [simulation.py]
      -> control update (normally every CONTROL_DT)
          obs = RaceObservation.observe(...)              [race_observation.py]
          every policy: get_action(obs)
              reference policies: reference -> private MPC -> action
              trajectory IBR / RL: direct action
      integrates vehicle state, collision/off-road/recovery logic
  -> simulation_trace.npz                         [simulation_core/trace.py]
  -> summary.json / HTML report            [main.py, visualization/generate_html_report.py]
```

### Shared low-level components

- **Plant:** configured in `config.py` and instantiated by
  `simulation.create_vehicle_model()`.  Current primary experiments use the
  local kinematic bicycle plant.
- **Track and raceline:** `track.py`, the selected centerline CSV, and a
  raceline CSV.  The raceline supplies lateral path and reference speed.
- **MPC:** `mpc_controller.py`, instantiated once per car in `simulation.py`.
  The primary formulation is local constrained Frenet MPC.  It tracks the
  high-level reference, has track/collision slack, and emits throttle/steer.
- **Physical events:** the simulator, not a high-level policy, determines
  physical collision, nominal boundary violation, off-road recovery and any
  optional deadlock reset.

### Primary configuration to state explicitly in report commands

The current development configuration is kinematic plant, local Berlin
raceline, `MPC_HORIZON=10`, velocity tracking weight `10`, track slack weight
`1e3`, collision slack weight `1e4`, no low-speed boundary recovery, and no
deadlock reset.  These are development defaults, not permanent paper settings.
Every report-oriented command must still state the values explicitly.

## 2. Common five-parameter action interface

The parameterized policies use canonical paper coordinates

```text
theta = (q, zeta, s1, s2, s3)
```

in `config.py::ALPHARACER_PARAMETER_BOUNDS`.

- `q`: tracking-aggressiveness / lookahead-related parameter.
- `zeta`: reference-speed and reference-horizon progression scale.
- `s1`, `s2`: interaction-driven overtaking lateral-offset magnitude and
  longitudinal decay.
- `s3`: blocking effect for a rear threat.

`policy/mpc.py::MPCPolicy.get_reference_trajectory()` maps `theta`, raceline,
and nearest front/rear opponents to a Frenet reference.  It records separate
raceline, overtaking, blocking, and final lateral-reference components in the
trace.  Thus these policies share the same low-level action execution path;
their main difference is how `theta` is selected.

## 3. Baseline contracts

### 3.1 Fixed static parameters / mode presets

**Implementation.** `StaticAlphaRacerPolicy` in `policy/mpc.py`.  It holds one
five-parameter vector fixed for a whole rollout.  Mode vectors live in
`config.py::LLM_MODE_PRESETS`:

```text
stable   [1/0.75, 1.00, 0.10, 0.50, 0.00]
push     [1/0.85, 1.20, 0.10, 0.50, 0.00]
overtake [1/0.80, 1.20, 0.45, 0.10, 0.00]
block    [1/0.75, 1.00, 0.10, 0.30, 0.70]
```

**Purpose.** Executable fixed-action controls and an action-library audit; not
an optimal tactical policy.

**Evidence.** `evaluation/mode_preset_audit.py` tests fixed branches.  The
responsive block audit is `evaluation/interactive_mode_preset_audit.py`.

**Known limitation.** Preset ranking changes with raceline/track; an
`overtake` vector that works in a short controlled scene is not guaranteed
safe in a tight hairpin.  The existing block audit had unsafe branches and is
not efficacy evidence for blocking.

**Allowed claim.** A disclosed, hand-designed MPC action library.

**Disallowed claim.** Track-invariant tactical skills or globally optimal
five-parameter actions.

### 3.2 Mode LLM tactical selector

**Implementation.** `LLMAlphaRacerPolicy` in `policy/llm.py` plus prompt
variants in `policy/prompts/`; raw-balanced is
`policy/experimental_balanced_raw_llm.py`.

**Input.** A textual observation containing traffic relations, limited road
preview, ego state/context, and short mode history.

**Output.** A discrete `stable`, `push`, `overtake`, or `block` decision at
the LLM update interval.  The mode maps directly to the fixed vector above;
MPC then controls the car.

**Purpose.** Test whether an LLM can select a tactical action from a fixed,
interpretable library.  It is not a continuous racing controller.

**Evidence.** Grounding probes support limited traffic-relation reading.
Raw/balanced prompt, temperature, counterfactual, and opponent-matrix screens
are development evidence only.

**Known limitations.** Inference latency has exceeded the 1 s target.  A mode
does not carry an MPC feasibility certificate.  Mode changes can change a
continuous reference abruptly even when mode switching is not frequent.

**Allowed claim.** High-level tactical selection over a shared fixed MPC
action library, under the stated synchronous simulation protocol.

**Disallowed claim.** Real-time operation, direct vehicle control, or a
general proof of useful tactical racing performance.

### 3.3 Direct-parameter LLM

**Implementation.** Same class as mode LLM with `output_mode=params` and
`policy/prompts/params_system.txt`.

**Purpose.** Action-representation ablation: can an LLM use the canonical
continuous five-parameter interface rather than four presets?

**Current evidence.** It produced drivable trajectories against Parameter-IBR
but strongly concentrated on a few parameter values and made few actual
parameter changes per 50 s race.

**Allowed claim.** A tested continuous-output ablation on this local action
interface.

**Disallowed claim.** Evidence that the LLM can effectively search the full
continuous control space.

### 3.4 Local learned-potential alpha policy

**Implementation.** Offline trainer:
`train/alpha_racer/alpha_potential_training.py`; online policy:
`policy/alpha_racer.py`.

**Offline data.** Current checkpoint: 100 groups, each with a baseline joint
parameter rollout and one unilateral deviation per car; 400 rollouts of 12 s,
48,000 control-sample feature rows.  It uses local kinematic dynamics, Berlin,
and local reward `progress - off-road penalty - collision penalty`.

**Online action.** Each alpha car maximizes its own learned potential over a
15-D joint parameter estimate, keeps a private estimate, and executes only
its own five-parameter block through shared MPC.

**Purpose.** A local learned-potential parameter-policy baseline.

**Compatibility.** Raw upstream weights are adapted only at the network input
boundary; local checkpoints use canonical coordinates with metadata.  This is
interface compatibility, not a guarantee of equal upstream behavior.

**Validation status.** Value models had group-level validation in the original
local run.  The standalone held-out audit
`evaluation/tmp_12_alpha_potential_heldout_audit.py` showed a large potential
train-to-held-out gap for the current small seed-7 dataset.  The checkpoint may
remain an empirical baseline but does not currently support a generalizing
alpha-potential / approximate-Nash claim.

**Allowed claim.** Local learned-potential baseline under the current platform.

**Disallowed claim.** Exact alpha-RACER reproduction, verified alpha-potential
generalization, or verified approximate Nash equilibrium.

### 3.5 Parameter-space IBR

**Implementation.** `policy/ibr/mpc_parameter_ibr.py`.

**Action.** Selects nearest longitudinal neighbor, constructs a local
two-player alternating best-response calculation in the same five-parameter
space, and executes only ego's final parameter/reference plan.

**Important behavior.** The modeled neighbor is hypothetical; its real online
policy is not invoked.  This is a nearest-neighbor, local interaction model,
not a full three-car equilibrium solver.

**Known limitation.** Some collision-induced hairpin states led to repeated
near-neutral first actions and zero-speed receding-horizon procrastination.

**Allowed claim.** Local parameter-space IBR diagnostic baseline.

**Disallowed claim.** Exact published IBR reproduction or complete three-player
Nash computation.

### 3.6 Trajectory IBR (`local_wang`)

**Implementation.** `policy/ibr/trajectory_ibr.py`.

**Action.** Alternates trajectory/control best responses against predicted
opponent trajectories under this repository's local Frenet formulation.

**Purpose.** Online trajectory-planning comparison at a different abstraction
from five-parameter policies.

**Allowed claim.** Local Wang-style trajectory IBR baseline.

**Disallowed claim.** Exact Wang/released-code reproduction or a like-for-like
five-parameter policy comparison.

### 3.7 Rule-based policy

**Implementation.** Rule policy classes in `policy/llm.py`.

**Purpose.** Deterministic control over the same observation/action-library
family as the mode LLM.

**Allowed claim.** Explicit rule selector over the fixed preset library.

**Disallowed claim.** A generic autonomous-racing rule baseline beyond its
documented rules and local platform.

## 4. Evidence categories and interpretation rules

| Evidence | What it can establish | What it cannot establish |
| --- | --- | --- |
| Frozen observation / grounding probe | Reading of selected scene facts | Closed-loop racing skill |
| Fixed-preset audit | Local action/reference effect | Global mode optimality |
| Counterfactual replay | Local conditional action utility | Responsive-opponent causal effect |
| Closed-loop race | Whole deployed system outcome | Fault attribution to LLM alone |
| Solver/slack trace | Numerical/control failure mechanism | Global failure frequency from one trace |
| P1/P2/P3 sweep | Position-stratified development outcome | Independent stochastic replication |

Use the full deployed chain when interpreting an LLM race:

```text
scene understanding != selected-mode utility != preset/MPC feasibility != race result
```

## 5. Required trace and report checks

For any report-oriented run, inspect the saved NPZ and summary for:

- policy names and checkpoint paths/parameterization;
- plant, track/raceline, MPC formulation, horizon, velocity and slack weights;
- initial ranks, gaps, lateral/speed perturbations, reset/recovery settings;
- progress/winner stratified by starting position;
- collision, nominal-boundary, recovery, solver-failure, stall and slack data;
- LLM model, prompt variant, temperature, call count, latency and deadline miss.

Known metadata task: ensure `llm_raw_balanced` is included wherever LLM policy
names are recognized for trace metadata, otherwise its stored temperature is
NaN despite being configured.

## 6. Current decision boundaries

- Treat Berlin P1/P2/P3 results as development evidence, not final
  generalization or statistical-performance results.
- Do not use a single local alpha checkpoint as an expert oracle for judging
  LLM mode correctness.
- Do not infer a mode-selection error solely from an unsafe closed-loop race
  before checking fixed action feasibility and MPC slack.
- Before a real-time claim, test bounded delayed execution or replay using the
  measured LLM latency.
- Before final multi-track comparison, freeze racelines, action library,
  shared MPC configuration and evaluation scenarios.
