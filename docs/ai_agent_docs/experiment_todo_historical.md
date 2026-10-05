# Historical experiment TODO

> Historical working notes. Check `AGENTS.md` and the frozen protocol before
> treating any item below as current work.

## Immediate follow-up — September 9

- The shared primary MPC now uses `VELOCITY_TRACK_WEIGHT=10`,
  `MPC_TRACK_SLACK_WEIGHT=1e3`, and
  `MPC_COLLISION_SLACK_WEIGHT=1e4` by default.  Keep these values explicit in
  report-oriented commands even though they are defaults.
- Diagnose the remaining zero-speed receding-horizon procrastination without
  changing high-level policies.  First compare the current single 0.1 s Euler
  prediction step with five differentiable 0.02 s substeps on saved deadlock
  states.  This tests numerical model mismatch before adding another loss.
- If substep integration is insufficient, test a cumulative early-progress
  objective, for example
  `-w_s * sum_k (s_k - s_0)`, against the current terminal-progress objective.
  A constant per-step `-1` has no effect in the fixed-length MPC horizon because
  it always sums to the same `-N`.
- Screen direct-parameter LLM against two Parameter-IBR opponents at P1/P2/P3.
  This is the cleanest first comparison because both select the same canonical
  five parameters and use the same low-level MPC.  Add Alpha-Racer afterward
  only if the screen completes reliably; the rule-only opponent is not needed
  for this diagnostic.
- Completed the first temperature screen with the raw mode prompt and two local
  Alpha-Racer opponents at `0`, `0.3`, and `1.0` over P1/P2/P3.  Push accounted
  for 149/150, 149/150, and 148/150 decisions; mean ego progress was 212.42,
  212.58, and 212.42 m.  Temperature alone did not materially change modes or
  outcomes.  Keep temperature zero for reproducibility; only repeat this
  stochastic ablation if a formal null-effect claim is needed.
- Completed the matched balanced-prompt screen.  Temperatures `0`, `0.3`, and
  `1.0` produced push/overtake/stable counts `121/29/0`, `121/29/0`, and
  `117/31/2`, with mean progress 212.59, 212.42, and 211.66 m.  Do not expand
  the temperature grid: wording changed the distribution far more than
  sampling temperature, and the highest temperature did not improve racing.
- Fix trace metadata before any future non-raw LLM run: balanced-policy NPZs
  currently store `llm_temperatures=NaN` because the trace policy-name set omits
  `llm_raw_balanced`, even though the configured temperature is passed to the
  policy and is present in the run command/name.

## Current writing target — September 8

- Future cloud manuscript edits belong to the new Overleaf project
  `LLM-racer-experiment record2`:
  `https://www.overleaf.com/project/6a74ea065860dfb8b4e0c951`.
- The new project's local mirror is `LLM-racer-experiment record 2/`; open its
  one-folder `.code-workspace` before editing so Overleaf Workshop can sync it.
  Repository-root `main.tex` and `LLM-racer-experiment record/main.tex` retain
  the first record and must not be used as the second project's sync target.

## Current experiment priorities — September 8

1. The matched shared-MPC stall regression has been run.  Velocity tracking
   helped ordinary trajectories, but collision-induced P3 deadlocks can still
   occur.  The current defaults are `w_v=10` and track-slack weight `10^3`;
   complete the substep-integration diagnostic above before freezing the MPC.
2. Complete full-lap, single-car validation of the locally generated Modena and
   rounded-rectangle racelines. The three-second audits show higher progress but
   not full-race reliability.
3. Construct a valid responsive passing control before judging `block`: require
   the `s_3=0` branch to complete a safe overtake, then compare the same scene
   with nonzero blocking.
4. Run the final repeated, held-out, multi-track policy comparison only after
   the track assets and action library are frozen. Include inference latency and
   safety/slack outcomes, not only progress.
5. Treat an aggression-only action interface as a later representation
   ablation, not an immediate rewrite. The current labels mix pace
   (`stable/push`) with interaction intent (`overtake/block`); test whether an
   ordered intensity abstraction helps before considering a factorized
   `intent x aggression` controller.

### Claims currently allowed

- The tested LLM can ground a small traffic-relation vocabulary accurately.
- Correct traffic grounding does not by itself establish useful mode selection.
- Mode selection is strongly prompt-sensitive, while controlled two-call
  generation provided no benefit and roughly doubled latency.
- The four presets are a hand-designed tactical action library, not latent
  Alpha-Racer modes and not globally optimal five-parameter actions.

### Claims not yet allowed

- Do not claim that evidence-first reasoning improves race performance or that
  low short-horizon regret proves correct mode selection.
- Do not claim that the current presets transfer unchanged across tracks.
- Do not claim successful blocking from either frozen-opponent or the current
  unsafe responsive-opponent audit.
- Do not claim that the LLM is a full continuous racing controller; it selects
  among fixed high-level actions executed by a shared MPC.

## Paper completion sequence — August 29

1. Freeze the primary platform and record one authoritative configuration:
   kinematic plant, Frenet constrained MPC, raceline, velocity-tracking weight,
   recovery settings, mode presets, LLM model/temperature, and decision interval.
   Resolve the current notes/config mismatch among velocity weights `0`, `0.5`,
   and the explicitly tested `10` before the final experiment grid.
2. Keep raw mode LLM as the direct-selection method. Keep the one-sentence
   no-default-mode version only as a prompt-sensitivity ablation. The controlled
   two-call tests did not improve grounding, progress, or counterfactual utility;
   do not restore them for the primary grid.
3. Run the track-only probe described below, followed by a small interaction-only
   counterfactual probe. These establish whether numerical scene changes are read
   correctly and whether the resulting assessment changes mode selection.
4. Implement a bounded delayed-execution experiment for raw and eight-state
   evidence-first LLM: synchronous oracle timing, fixed delays, and measured-latency
   replay. This is required before making a real-time capability claim.
5. Run one frozen like-for-like final grid for raw LLM, eight-state evidence-first,
   explicit-rule LLM, pure rules, and local Alpha-Racer. Use at least two tracks,
   three start positions, and one additional gap/speed/lateral perturbation. Report
   variation across scenarios rather than treating position permutations as
   independent random trials. Report trajectory IBR separately because it operates
   at a different control abstraction.
6. Reorganize the chronological progress log into introduction/research questions,
   platform, methods, protocol, results, limitations, and conclusion. Move most
   MPC/IBR debugging chronology to a short implementation note or appendix.

### Work that should not delay the paper

- Do not resume SE-IBR, large IBR-objective searches, full DBM grids, released-stack
  reconstruction, new recovery patches, deeper multi-layer CoT, or larger Alpha
  rollout training unless a frozen final check exposes a specific blocker.

## P0 — freeze the September 6 LLM result

- Correct the earlier interpretation: the varied modes in the 11-case offline
  probe were confounded by a shorter, more directive second prompt and balanced
  sampling. The probe shows basis-text sensitivity, not a two-stage benefit.
- Retained controlled result: one-call versus two-call supported basis was
  `37/45` versus `34/45`; mean latency was `3.17 s` versus `6.80 s`; both won
  `1/3`. Two-call policies are removed from the active set.
- Retain raw (`100/100 push` in the P2/P3 50 s screen) and balanced
  (`77 push`, `22 overtake`, `1 stable`) for prompt sensitivity. Do not add
  hand-written curve, gap, overtake, or blocking advice before held-out tests.
- Retained counterfactual result: raw informative acceptance `5/10`; balanced
  `1/7`. Diversity is not evidence of better mode utility.
- Source data: `result/tmp_08_one_vs_two_step_fixed*/`,
  `result/tmp_09_raw_no_default_mode_berlin_50s_p23*/`,
  `result/tmp_10_balanced_counterfactual_bundle/`, and
  `result/tmp_11_balanced_two_step_20s/`.

## Archived P0 — August 28 structured-basis follow-up

- Keep the eight-state traffic-only evidence-first prompt as the current positive
  structured-basis candidate. Preserve raw and structured action-first as controls.
  Do not promote the dual traffic-plus-track prompt: its archived result has
  `62.2%` traffic support, `15.6%` track support, and higher latency. Source data:
  `result/llm_basis_minimal_ab_berlin/audit/` and
  `result/llm_basis_track_ab_berlin/audit/`.
- Build a separate track-only probe before changing the online controller again.
  Automatically select straight, single-hairpin, consecutive-same-direction, and
  direction-changing/S-shaped Berlin windows. For each, generate fixed
  inside/center/outside ego positions with no traffic and a fixed speed.
- Track-only output should separately report near-turn direction, far-turn
  direction, whether direction changes, and ego position relative to the first
  clear turn. Compute the oracle from unwrapped centerline heading changes over
  fixed near/far distance windows, not noisy pointwise curvature signs.
- Run the probe first as a frozen-observation test, then query the raw mode prompt
  on the same observations. This distinguishes failure to read geometry from
  correct geometry assessment that does not affect mode selection. Only after
  that result should a small solo closed-loop hairpin/S-turn screen be considered.
- Implement the probe in three explicitly separated stages:
  (A) geometry-only assessment with no mode output, using identical frozen inputs;
  (B) mode-response queries on the same inputs; and only if useful,
  (C) solo closed-loop segment traversal. Never infer track understanding from a
  closed-loop failure alone because the low-level MPC may be the failing layer.
- For stage A, use a small balanced set: one or two straight controls, single
  hairpins in both directions, consecutive same-direction turns, and genuine
  direction-changing/S windows. Evaluate each at fixed speed and at
  inside/center/outside lateral positions. Target about 12--18 snapshots.
- Define the geometry oracle with unwrapped centerline heading over fixed near and
  far distance intervals. Report near direction, far direction, direction-change
  classification, and inside/center/outside relative to the first clear turn.
  Use explicit angular/lateral deadbands only as reproducible measurement
  definitions; do not map them to `stable` or `push`.
- For stage B, compare raw direct mode output on the frozen snapshots with the
  geometry-only assessment. Measure mode distributions conditional on road class,
  whether paired geometry changes alter the mode, factual explanation conflicts,
  and latency. Do not define one correct mode per road class.
- For optional stage C, start a solo car a fixed distance before selected hairpin
  and S segments and compare raw LLM, eight-state evidence-first, fixed `stable`,
  and fixed `push`. Record segment completion, time, progress, minimum speed,
  maximum lateral displacement, boundary/slack/recovery, and mode sequence.
- Treat heading error and speed-relative-to-local-reference as later continuous
  observation ablations. Do not add `high_speed`, `dangerous`, or other thresholded
  tactical tags while testing track understanding.
- Add an offline paired observation-reduction screen if time permits: full raw,
  raw without multi-sample road preview, traffic-only raw, and full minimal
  evidence-first. This tests whether the traffic basis acts as a useful attention
  bottleneck rather than assuming that the full prompt merely contains too many
  tokens.

## Deferred — interaction-focused scenarios and mode-level lookahead

- Do not expand the scenario taxonomy yet. Adding closing, blocking, squeezed,
  side-by-side, and boundary-recovery cases may require changing what the LLM
  observes and how the prompt describes the scene, so it belongs with a later
  LLM-method improvement stage rather than the current frozen-policy comparison.
- When that stage begins, keep scenario/prompt development separate from the
  held-out evaluation set. Use the added cases to test whether decisions change
  with front gap, rear threat, lateral overlap, curvature, and available space.
- An optional mode-space lookahead policy may enumerate the same four fixed
  presets as the mode LLM and evaluate them with short simulator rollouts. This
  is useful for diagnosing mode selection, but it is not a standard continuous
  trajectory IBR and should not replace the primary IBR baseline.
- The existing coarse parameter-space IBR is also a diagnostic rather than an
  oracle: it uses sparse coordinate search and a short-horizon progress score,
  whereas Alpha-Racer continuously optimizes a learned longer-term potential.
  Do not spend the current project phase tuning its grid or switching penalty.

## Track generalization and local racelines

- Current provenance is mixed: Berlin uses the locally regenerated
  `Sa=0.34` minimum-curvature path/speed profile, while Modena and rounded
  rectangle still use released alpha-RACER profiles. Finish closed-loop
  validation of the generated Modena and rounded-rectangle candidates before
  promoting them or interpreting cross-track policy differences.
- Candidate primary geometry source: TUMFTM `racetrack-database`
  (https://github.com/TUMFTM/racetrack-database). It contains centerlines and
  widths for more than 20 F1/DTM circuits and shares provenance with the TUM
  raceline tools already used locally.
- Candidate convenient 1:10 source: F1TENTH `f1tenth_racetracks`
  (https://github.com/f1tenth/f1tenth_racetracks). It is derived from the TUM
  database, so the two repositories are alternative formats, not independent
  evidence.
- Candidate supplementary source: `MapZoo`
  (https://github.com/zhouhengli/MapZoo), a 32-map MIT-licensed collection with
  centerlines, boundaries, racelines, speed fields, and overtaking-sector
  metadata. Check provenance per map before final use.
- Import external **geometry only** for the primary comparison. Normalize scale
  and usable width, then regenerate every raceline and speed profile with the
  same frozen local vehicle model. Do not mix supplied external speed profiles
  with locally generated ones.
- Proposed split: Berlin for development; locally regenerated Modena and rounded
  rectangle for validation; three untouched real-track geometries for final
  held-out testing. Select the latter by measured curvature/straight statistics
  (fast-flowing, technical, and mixed), not only by track reputation.
- Before multi-agent tests, require every candidate to pass a single-car
  closed-loop screen for completion, boundary violations, stall/reset events,
  solver failures, and actual lap time.

## Deferred — automatic calibration of the four mode presets

- Treat the current mode LLM as a scene-grouping mechanism: collect interaction
  states from local Alpha-Racer self-play and a smaller set of Alpha-Racer/IBR
  races, then record the mode selected by the frozen raw-mode prompt.
- Also attach non-exclusive geometric audit tags such as `front_close`,
  `rear_threat`, `side_overlap`, `overtake_space`, `near_boundary`, and
  `high_curvature`. The LLM label describes the deployed state distribution;
  these tags expose ambiguous or clearly inconsistent assignments rather than
  replacing the LLM with another rule policy.
- For each mode, evaluate candidate five-parameter presets by restarting short
  counterfactual rollouts from its representative states. Use Alpha-Racer and
  IBR only to generate realistic states, not as action or trajectory teachers.
- Select one fixed preset per mode by minimizing aggregate task-specific loss
  across scenes (mean plus a small worst-case term). Do not optimize each scene
  independently and average its optimal parameters, because simulator response
  is nonlinear and the averaged parameter vector need not work in any scene.
- Use mode-specific objectives: stable for safe progress, push for progress,
  overtake for safe relative progress/overtake success, and block for reducing
  the rear car's gain without stalling ego. Apply common collision, off-road,
  recovery, and excessive-slowing penalties.
- Start with a small random/coarse search rather than a full five-dimensional
  grid. Keep calibration scenes disjoint from final evaluation and compare the
  current manual presets with calibrated presets as an ablation.
- Before implementing this, finish the higher-priority prompt-sensitivity and
  delayed-execution experiments below.

## Immediate next sequence (August 15)

1. **Freeze the current prompts and policies.** Treat effect-aware direct params as
   the final direct-parameter ablation. Do not continue tuning either param or mode
   prompts against the six observed races.
2. **Run a small counterfactual prompt-sensitivity test before more racing.** Reuse
   real saved observations and independently vary front gap, rear TTC, lateral
   position, road curvature, and `current_parameters`. Compare raw mode output with
   direct-param output. Measure parameter/mode changes, current-parameter anchoring,
   and explanation-direction consistency. Target 40--60 calls, not another full
   simulation sweep.
3. **Implement delayed/asynchronous execution.** Form a prompt at time `t`, keep the
   old decision active for latency `tau`, and apply the returned decision at
   `t+tau`. Compare synchronous oracle timing, fixed delays (at least 1/3/5 s), and
   measured-latency replay for raw mode LLM. This is the highest-value experiment
   for the real-time research question.
4. **Run the held-out robustness grid only after timing behavior is fixed.** Minimum
   suggested main grid: 3 positions x 2 tracks x 2 start gaps = 12 races per policy
   for raw mode LLM, rule-prompt LLM, rule-only, and local Alpha-Racer. Prefer
   speed/lateral perturbations over duplicate seeds in a deterministic simulator.
5. **Add pace-matched IBR only as a bounded follow-up.** Use the horizon raceline
   speed profile as a soft loss, remove the matched version's absolute
   `u-u_neutral` term, retain `delta_u`, calibrate the speed weight on separate
   non-interactive runs, freeze it, then run three positions. Ordinary IBR remains
   the unrestricted optimization reference.

### Stop conditions

- Do not rerun all tracks for direct-param LLM unless the counterfactual test shows
  meaningful state sensitivity that the six races failed to expose.
- Do not increase temperature merely to make parameter histograms broader;
  diversity is not evidence of correct adaptation.
- Do not tune mode presets or prompts on the final evaluation scenarios. If preset
  calibration becomes necessary, define a disjoint calibration set and objective
  before observing held-out results.
- Do not let further SE/IBR refinement delay the LLM timing and robustness study.

## P0 — freeze a coherent primary platform

- Use the kinematic simulator for primary experiments so the shared four-state
  MPC predictor closely matches the plant and comparisons isolate high-level
  strategy rather than tire-model mismatch.
- Keep one shared low-level MPC for parameterized policies: LLM, LLM+rules,
  pure rule-based, static Alpha-Racer, and locally trained Alpha-Racer.
- Keep the differentiable Frenet superellipse and soft track/collision constraints
  identical across those policies. Report slack activation and solver recovery.
- Keep simulator-level oriented-rectangle contact and off-road recovery identical
  across policies. Distinguish nominal boundary crossing (`|e| > 0.44 m`) from
  released-simulator-style recovery activation (`|e| > 0.55 m`). Treat these as
  two explicit thresholds rather than deriving an undocumented tolerance.
- Freeze `local_wang` IBR as the independent
  interactive trajectory-planning baseline. Do not tune it against the test set.

## P1 — main experiments serving the LLM question

- Compare static Alpha-Racer, local learned Alpha-Racer, pure rules, raw LLM,
  and LLM with explicit prompt rules under identical scenarios.
- Measure progress/rank, collision and overlap, nominal-boundary violation,
  recovery activation, MPC slack, solver success, stopped time, and wall-clock
  decision latency/deadline misses.
- Use multiple tracks, start positions, gaps, speed/lateral perturbations, and
  genuinely independent seeds. Treat homogeneous position permutations as
  symmetry checks rather than independent statistical trials.
- Separate tuning scenarios from held-out evaluation scenarios.

## P2 — focused validation/ablations

- Pause further local DBM development. The primary platform remains kinematic;
  DBM currently introduces an intentional plant/predictor mismatch because MPC
  and IBR use the shared four-state kinematic predictor. Reassess whether the
  incomplete DBM path should be removed after the primary experiments rather
  than extending it into a second fully supported physics stack.
- Run a small hard-vs-soft MPC check: verify slack is essentially zero in normal
  feasible operation and prevents failure cascades only near infeasible states.
- Run a small DBM robustness subset after primary kinematic conclusions are fixed.
  Do not repeat the full grid unless conclusions materially change.
- Verify the local Alpha-Racer checkpoint remains adequate on the final held-out
  scenario set; do not enlarge rollout training merely because more data is possible.

## P3 — optional or appendix-only

- `historical_strict` IBR may remain as a code reference but need not enter the
  primary result table.
## P4 — writing/reproducibility

- Document that the work evaluates high-level strategic decision mechanisms in a
  standardized local platform, not exact reproduction of every external project.
- Audit raceline provenance for every evaluation track. Regenerate with a single
  frozen minimum-curvature/speed-profile pipeline if the existing CSV cannot be
  tied to the evaluated geometry and vehicle limits; otherwise explicitly label
  it as a fixed precomputed reference rather than claiming local optimality.
- Record exact policy/checkpoint, parameterization, vehicle model, initial state,
  track, timing, horizon, constraint settings, recovery thresholds, and seeds.
- Keep the manuscript claims narrower than the tested distribution, especially
  for SE, DBM robustness, and real-time LLM performance.

## Immediate follow-up from the September 7 audit

- Validate each new Modena and rounded-rectangle raceline with full-lap,
  single-car completion/safety tests before using it in policy comparisons.
- Do not retune presets from the current three-second results alone. Treat the
  changed cross-track ordering as a transfer problem to report or calibrate on a
  separate development set.
- Redesign the responsive passing audit before evaluating `block`: sweep initial
  gap, lateral separation, and closing speed, and retain scenes only when the
  `s_3=0` control produces a safe completed pass. Keep mirrored left/right cases.
- After establishing valid controls, compare `block` by pass completion, pass
  time/delay, relative-progress change, collision/off-road rate, and slack.
- Add Alpha-Racer as a secondary responsive opponent only after the controlled
  same-MPC audit is valid. Do not use trajectory/parameter IBR for the first
  block-mechanism test because its different controller would confound the
  result.
