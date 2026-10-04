# K-Constrained-v1 (frozen)

`K-Constrained-v1` is the experiment contract for the current project phase.
The machine-readable source of truth is
[`K-Constrained-v1.json`](K-Constrained-v1.json); this page states the rules
that determine whether a result may carry that label.

The primary question is deliberately narrow: **can an LLM make useful
high-level racing decisions by directly selecting `(q,zeta,s1,s2,s3)`, while
the same MPC retains sole authority over throttle and steering?** Empirical
clarity and reproducibility take priority over preserving a mode vocabulary,
prompt architecture, or positive result.

## Frozen substrate

- **Plant:** four-state kinematic bicycle, forward Euler, normalized throttle
  and steering, `Lf=0.11 m`, `Lr=0.23 m`, steering gain `0.34 rad`,
  acceleration `u_t*(20/4.65)-1 m/s²`, speed clipped to `[0,5] m/s`. The
  oriented-rectangle contact footprint is `0.36 m × 0.24 m` (half-length
  `0.18 m`, half-width `0.12 m`).
- **Controller:** local constrained Frenet MPC with state `[s,e,psi_err,v]`,
  geometric Frenet dynamics, `dt=0.1 s`, 10 steps (1.0 s), `Q=diag(1,1)`,
  `R=diag(1,1)`, velocity weight 10 and heading weight 0.
- **Constraints/slack:** point-mass track boundary; nearest front and rear
  opponent; `p=4` collision superellipse with axes `0.428 m` and `0.285 m`;
  non-negative track/collision slacks weighted `1e3`/`1e4`; friction proxy on
  with `mu=1.6`; steering-command rate at most `4/s`.
- **Recovery:** starts at `|e|>0.55 m`; the exact speed, heading, steering and
  throttle recovery constants are in the manifest. Contact remains the
  asymmetric overlap speed decay already implemented by the simulator.
- **Track/raceline:** only the registered Berlin, Modena and rounded-rectangle
  pairs are in-protocol. Their scales, widths and SHA-256 content hashes are
  frozen in the manifest. Re-generating a raceline is a protocol change even
  if its filename stays the same.

## Five parameters

The order is always `theta=(q,zeta,s1,s2,s3)`:

| Parameter | Frozen range | Meaning |
|---|---:|---|
| `q` | `[1, 3.3333333333]` | Position-reference tracking weight in the MPC. |
| `zeta` | `[0.7225, 1.21]` | Multiplier for reference progression and reference speed. |
| `s1` | `[0.10, 0.50]` | Maximum passing/avoidance lateral offset, metres. |
| `s2` | `[0.10, 0.50]` | Gaussian longitudinal interaction decay, `m^-2`; larger is shorter-range. |
| `s3` | `[0, 1]` | Blocking strength toward a faster rear vehicle; zero disables it. |

An LLM may update these five values (or select a preset composed of them) once
per `1.0 s`. Values are held between updates. The LLM never emits actuator
commands.

## Scenario generation

The canonical evaluation matrix is deterministic and paired:

- three cars and a 50 s race;
- all three registered tracks;
- ego start positions 1, 2 and 3, producing 9 cells per policy matchup;
- 6 m longitudinal grid spacing, zero lateral/longitudinal perturbations and
  zero initial speed;
- seed 0; compared policies receive the same track, grid and starting position.

Policy rosters are experiment variables, not part of the plant protocol.
Stochastic policies must record model, sampling settings and seed. Repeats of a
deterministic cell are not independent samples and must not inflate `n`.

The nine deterministic cells are a **core scan**, not broad robustness
evidence. A stochastic evaluation must predeclare its repeat count and seeds,
keep those seeds paired between arms, and report variability. Additional
tracks, traffic perturbations or stressors belong to a declared robustness
extension and must not be silently pooled with the core scan.

## Evaluation metrics

Primary performance is final unwrapped progress, progress gain from the initial
grid, and win rate over the 9 equally weighted cells. A win is greatest final
unwrapped progress; progress differences within `1e-9 m` are ties.

Safety is never folded into a hidden reward. Report physical collision time,
track-boundary violation time, recovery-active time, maximum track and
collision slack, and MPC solver failures. For LLM policies also report decision
and error counts, mean/p95 latency, 1 s deadline misses, token counts and mode
switches. Report every cell before the macro mean.

Failed or missing runs remain part of the result: report them explicitly rather
than dropping them. Parse errors and timeouts are safety/reliability outcomes,
not merely logging noise.

## Controlled comparisons

Every comparison states one scientific factor before running and changes only
that factor. Observation, action space, timing, model, decoding, parser and
fallback behavior remain fixed unless one of them is the named factor. Adding
history, feedback, semantic scaffolding or a mode library requires either an
observed failure motivating it or a predeclared ablation; it is not accumulated
through informal prompt tuning.

Direct five-parameter selection is the primary capability test. Direct
throttle/steering is a diagnostic or a separate action-space study. A discrete
mode library remains a candidate interface and is retained as a contribution
only if it shows measurable headroom and a defensible advantage over direct or
fixed parameters.

For every arm, preserve:

- the stated hypothesis and single changed factor;
- exact prompt, observation schema, history/memory and output schema;
- parser, clipping, retry, timeout and fallback behavior;
- model identifier, checkpoint/quantization and decoding settings;
- update interval, policy roster, scenario IDs, seeds and repeat count;
- observation access, privileged information, action space and replanning rate;
- compute environment, Git commit SHA and dirty status.

## Baselines and claim boundaries

- A locally retrained alpha-RACER model may be the primary learned baseline,
  but its checkpoint and training setting must be recorded. It is not evidence
  that the published paper environment or numbers were reproduced.
- Local IBR is a heterogeneous local baseline and must be described by its
  actual implementation and interface, not presented as a literature
  reproduction without matching code and evidence.
- Fixed push measures a strong constant parameter action; oracle mode estimates
  headroom inside the fixed mode library; direct-parameter LLM tests the less
  restricted interface. They answer different questions.
- Fairness does not require identical internals, but observation access, action
  space, replanning rate, privileged information, compute and fallback behavior
  must be disclosed.

## Result classes and provenance

Results use one of three labels:

| Class | Meaning |
|---|---|
| `exploratory` | Diagnostic or tuning run. A dirty tree is permitted only with this label. |
| `core_scan` | Complete paired nine-cell run from a clean, recoverable commit. |
| `robustness` | Clean-commit, predeclared extension with repeats or additional stressors. |

Formal results require a clean Git commit. Every run records its commit SHA and
dirty status. Negative and null results are preserved. Retuning a prompt,
scenario, preset or metric after inspecting results creates a new exploratory
iteration; it does not overwrite the original result.

Copy and complete
[`experiment-arm-template.json`](experiment-arm-template.json) before a new
comparison. `core_scan` and `robustness` arms must be launched from the clean
commit recorded in that file; incomplete manifests remain exploratory.

## Version rule

Changing any frozen plant, controller, constraint/slack, recovery, track file,
raceline file, timing, parameter definition/range, scenario rule or metric
requires a new manifest and a new protocol ID (for example
`K-Constrained-v2`). DBM and `upstream_simple` are explicitly out of scope.
Policy/prompt/model changes are allowed only when clearly named as experimental
variables and all frozen substrate fields remain unchanged.

A policy, prompt, model or observation change does not by itself create a new
substrate version, but it does require a new named experimental arm. If a
suspected substrate defect is found, diagnose and state whether it invalidates
old results, affects only robustness, or is merely an implementation preference
before changing anything.

Run the guard before launching a result-bearing experiment:

```bash
.venv/bin/python protocols/validate_k_constrained_v1.py
```

To print the exact nine scenario records consumed by an experiment runner:

```bash
.venv/bin/python protocols/k_constrained_v1_scenarios.py
```
