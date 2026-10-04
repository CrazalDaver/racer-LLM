# K-Constrained-v1 (frozen)

`K-Constrained-v1` is the experiment contract for the current project phase.
The machine-readable source of truth is
[`K-Constrained-v1.json`](K-Constrained-v1.json); this page states the rules
that determine whether a result may carry that label.

## Frozen substrate

- **Plant:** four-state kinematic bicycle, forward Euler, normalized throttle
  and steering, `Lf=0.11 m`, `Lr=0.23 m`, steering gain `0.34 rad`,
  acceleration `u_t*(20/4.65)-1 m/s²`, speed clipped to `[0,5] m/s`.
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

## Evaluation metrics

Primary performance is final unwrapped progress, progress gain from the initial
grid, and win rate over the 9 equally weighted cells. A win is greatest final
unwrapped progress; progress differences within `1e-9 m` are ties.

Safety is never folded into a hidden reward. Report physical collision time,
track-boundary violation time, recovery-active time, maximum track and
collision slack, and MPC solver failures. For LLM policies also report decision
and error counts, mean/p95 latency, 1 s deadline misses, token counts and mode
switches. Report every cell before the macro mean.

## Version rule

Changing any frozen plant, controller, constraint/slack, recovery, track file,
raceline file, timing, parameter definition/range, scenario rule or metric
requires a new manifest and a new protocol ID (for example
`K-Constrained-v2`). DBM and `upstream_simple` are explicitly out of scope.
Policy/prompt/model changes are allowed only when clearly named as experimental
variables and all frozen substrate fields remain unchanged.

Run the guard before launching a result-bearing experiment:

```bash
.venv/bin/python protocols/validate_k_constrained_v1.py
```

To print the exact nine scenario records consumed by an experiment runner:

```bash
.venv/bin/python protocols/k_constrained_v1_scenarios.py
```
