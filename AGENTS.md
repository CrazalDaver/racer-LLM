# Research and Experiment Instructions

This repository studies whether an LLM can make useful high-level racing
decisions through the alpha-RACER five-parameter interface. Treat empirical
clarity and reproducibility as more important than preserving any particular
LLM architecture, mode vocabulary, or positive result.

## Read before changing experiments

- Read `protocols/README.md` and `protocols/K-Constrained-v1.json` before
  changing simulation, controller, policy, prompt, or evaluation code.
- Run `protocols/validate_k_constrained_v1.py` before any result-bearing run.
- Distinguish exploratory diagnostics from results intended for a paper.

## Frozen substrate

- `K-Constrained-v1` is the current local experimental world: the kinematic
  plant, constrained Frenet MPC, recovery/contact behavior, registered
  track-raceline pairs, timing, five-parameter definitions and bounds, scenario
  rules, and reported metrics are frozen by the protocol.
- Do not silently tune or repair the frozen substrate to improve a policy's
  result. If a frozen element must change, explain why, create a new protocol
  version, and keep results from different versions separate.
- Do not replace the kinematic model with DBM merely for paper resemblance.
  DBM is an optional robustness study, not a prerequisite for the main study.
- Diagnose suspected substrate problems before modifying them. Report whether
  a finding invalidates old results, changes only robustness, or is merely an
  implementation preference.

## Research sequence

- Start with the simplest clean capability test: an LLM directly selects
  `(q, zeta, s1, s2, s3)` and the shared MPC emits actuator commands.
- Treat direct throttle/steering output as a diagnostic or separate action-space
  study, not as the primary comparison to the five-parameter methods.
- Treat the discrete mode library as a candidate interface, not an assumed
  contribution. Keep it only if experiments show useful headroom and a
  defensible advantage over direct parameters or fixed parameters.
- Add history, memory, decision feedback, semantic scaffolding, or mode
  structure only in response to an observed failure or a predeclared ablation.
  Do not accumulate prompt features through informal tuning.
- Change one scientifically meaningful factor at a time when comparing LLM
  variants. Keep observation, action space, timing, model, decoding, parser,
  and fallback behavior fixed unless that factor is the stated comparison.

## Baselines and claims

- A locally retrained alpha-RACER model in the frozen local setting may be used
  as the primary learned baseline. Record the exact checkpoint and training
  setting; do not imply that it reproduces the original paper's published
  environment or numbers.
- Treat the local IBR implementation as a heterogeneous local baseline. Describe
  its actual algorithm and interface precisely; do not call it a reproduction
  of Wang et al. or another paper without matching code and evidence.
- Fairness does not require every method to have identical internal machinery,
  but comparisons must disclose observation access, action space, replanning
  rate, privileged information, compute, and fallback behavior.
- Fixed push, oracle mode selection, and direct-parameter LLM answer different
  questions. Do not use one as a substitute for another: fixed push measures a
  strong constant action, an oracle estimates headroom within a fixed library,
  and direct parameters test the less restricted interface.

## Experiment hygiene

- For every experimental arm, preserve the exact prompt, observation schema,
  history/memory setting, output schema, parser, clipping, retry/timeout and
  fallback rules, model identifier, checkpoint or quantization, decoding
  settings, update interval, policy roster, scenario list, and seeds.
- Formal results must come from a clean, recoverable Git commit. Record the
  commit SHA and dirty status with the run. Exploratory results from a dirty
  tree must be labelled exploratory.
- Use paired scenarios across compared policies. Report every scenario before
  aggregation, and do not count repeated deterministic runs as independent
  samples.
- For stochastic LLM policies, predeclare the number of repeats and report
  variability. A single deterministic nine-cell grid is a core scan, not broad
  evidence of robustness.
- Report progress and safety separately. Never hide collision, boundary,
  recovery, slack, solver-failure, parse-error, timeout, or latency behavior
  inside an undocumented scalar score.
- Preserve negative and null results. Do not retune scenarios, prompts, mode
  presets, or metrics after seeing results without declaring a new exploratory
  iteration.

## Working practice

- Preserve unrelated user changes in a dirty worktree. Inspect before editing
  and avoid broad cleanup or refactoring during an experiment change.
- Prefer small, reviewable changes with a stated hypothesis and verification
  command. Explain what evidence would support or reject the hypothesis.
- When instructions or code conflict with the frozen protocol, stop and surface
  the conflict instead of choosing whichever setting produces a better result.
