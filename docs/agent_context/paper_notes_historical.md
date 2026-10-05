# Historical paper notes

> Working notes only. Verify all claims against current code and results.

This is a writing checklist, not manuscript-ready prose. Distinguish established
implementation facts, current empirical observations, and claims that still need
additional experiments.

## 1. Scope and research question

- The project studies how an LLM-based policy behaves in a real-time,
  multi-agent racing task. Alpha-Racer, IBR/SE-IBR, static/rule-based policies,
  and possibly self-play RL are baselines rather than identical reproductions
  of their source papers.
- Say explicitly that this is a different simulator/project. Extracted algorithmic
  ideas are integrated with the local vehicle model, track representation, timing,
  and constraints; avoid claiming exact reproduction unless that version is
  separately implemented and verified.
- Report wall-clock/control latency and missed real-time deadlines for the LLM,
  not only racing performance. Real-time behavior is part of the research question.

## 2. Vehicle and prediction models

- The simulator exposes a six-state vehicle interface
  `(x, y, psi, vx, vy, omega)`.
- MPC predicts a reduced four-state global kinematic model `(x, y, psi, v)`;
  IBR predicts a four-state Frenet model `(s, e, heading_error, v)`.
- With the default `kinematic` simulator, `vy=0` and `omega` is reconstructed
  from the kinematic yaw rate. Thus the six-state interface does not represent
  six independently evolving dynamic states. With `dbm`, lateral velocity,
  tire forces, and yaw rate are genuinely dynamic, creating greater planner–plant
  model mismatch.
- A reduced-order planner inside a six-state environment is intentional and common,
  but the planner/simulator mismatch and selected simulator model must be disclosed.
- Simulation, MPC, and both IBR variants now share the same command mappings and
  local parameters (`Sa`, `Sb`, `Ta`, `Tb`, `LF`, `LR`, friction semantics).
  The current local steering scale is `Sa=0.34`, with `Sb=0`, `LF=0.11`, and
  `LR=0.23` (wheelbase `0.34 m`). An earlier experiment accidentally retained
  `Sa=0.80`; it is not the current baseline. The released project is itself
  internally mixed: its numerical DBM is instantiated with `Sa=0.34` and
  wheelbase `0.36 m`, while its standalone IBR helper defaults to `Sa=0.36`.
- The friction-derived speed/steering constraint is a local consistency/safety
  addition based on the available lateral acceleration. Identify it as a local
  modeling choice rather than attributing it verbatim to Wang et al.

## 3. Alpha-Racer baseline and local training

- The repository-facing policy coordinates are the paper's canonical five
  parameters `(q, zeta, s1, s2, s3)`. Official checkpoints were trained with
  upstream transformed coordinates (lookahead factor, speed factor, blocking
  terms), so runtime conversion is applied only at the network boundary.
- Both official and local checkpoints optimize the same normalized canonical
  paper-coordinate search space `[0,1]^15`; this prevents checkpoint-dependent
  online search spaces. The official network input is transformed automatically,
  while locally trained checkpoints consume canonical inputs directly.
- Explain that the upstream transformed parameter ranges were converted into
  canonical bounds; the bounds are inherited/calibrated design ranges, not
  theoretically unique limits.
- Local potential training has three offline stages: rollout collection,
  Monte-Carlo value-function fitting for the three agents, and potential-function
  fitting. It is MC rather than TD because complete trajectories are generated
  before value-network updates.
- Each rollout group samples one joint baseline parameter vector and three
  unilateral deviations (one changed agent at a time), producing four simulated
  races. State the group count, duration, track, vehicle model, discount/reward,
  random seed, and checkpoint-selection procedure in the experimental section.
- Current checkpoint evaluation found the local checkpoint roughly 10% better in
  progress than the released checkpoint across Berlin, Modena, rounded rectangle,
  and tested perturbations, with mostly safe runs. Treat this as evidence within
  the tested local distribution—not proof of universal superiority. Mention the
  frequent saturation of `zeta` if it remains present in final experiments.
- The local checkpoint is reasonable as the project baseline only if the final
  comparison uses held-out tracks/initial conditions and identical evaluation
  budgets. Increasing training data is optional unless broader robustness tests
  expose instability; do not justify adequacy solely from training loss decreasing.

## 4. IBR

- Wang et al.'s central idea is retained: players repeatedly solve finite-horizon
  best-response optimal-control problems against the other player's latest plan.
- `local_wang` is Wang-inspired, not a literal reproduction: it uses local Frenet
  trajectory states, shared local command dynamics, local track/collision/slack
  constraints, local objectives, receding-horizon timing, complete alternating
  rounds, and a final ego solve.
- `historical_strict` is the locally adapted Alpha-Racer-style IBR baseline restored
  from repository history. It retains the upstream objective, simplified Frenet
  curvature rate `yaw_rate - v*kappa`, three alternating solves, local vehicle
  parameters, hard local track bounds, and the historical collision expression.
  It is not source-faithful upstream code.
- `local_wang` instead uses the geometric Frenet heading correction
  `v*cos(heading_error)*kappa/(1-e*kappa)`. This is a concrete model difference
  that should accompany solver comparisons.
- Earlier source-faithful/upstream-legacy IBR performed very poorly locally and
  was removed. If mentioned, frame it as evidence of simulator/model incompatibility,
  not as evidence that IBR itself is faulty.
- Homogeneous three-position sweeps are permutations of the same physical race,
  not three statistically independent trials. Use multiple seeds, asymmetric
  policies/speeds/gaps, tracks, and initial perturbations for uncertainty estimates.
- Historical-strict's observed no-offroad/no-collision run supports stability only
  for that tested scenario. Do not generalize it without the same robustness sweep.

## 5. Constraints, collision, and recovery

- Optimization uses smooth Frenet collision constraints suitable for IPOPT,
  whereas physical simulation contact uses oriented vehicle rectangles. State
  this distinction because constraint satisfaction and simulated collision are
  not identical metrics.
- Track and collision constraints may use non-negative slack with penalties;
  report maximum slack and solver success/status, not only collision/offroad time.
- Off-track recovery is simulator-level logic applied to every high-level policy:
  speed decay, heading realignment, inward steering, and low-speed throttle boost.
  It is not an ability learned by the LLM or Alpha-Racer policy. Report whether it
  is enabled and keep it identical across policies.
- The safeguard/rule mechanism must be separated from raw LLM performance.
  Experiments should distinguish LLM without safeguard, LLM with explicit prompt
  rules, pure rule-based policy, and any later safeguard-enabled configuration.

## 6. Experimental fairness and reporting checklist

- Use the same track geometry, vehicle model, initial states, start gaps, control
  period, MPC horizon, speed limits, constraints, recovery/contact rules, and
  simulation duration across policies unless an ablation intentionally changes one.
- Report per-car policy and checkpoint, checkpoint parameterization, random seed,
  ego start position, initial longitudinal/lateral/speed perturbations, solver
  success, slack, collision/offroad duration, progress, rank, speed, stopped time,
  and computation latency.
- Compare local and official Alpha-Racer checkpoints directly before involving the
  LLM. Compare IBR variants separately; do not use an unstable or unvalidated IBR
  result as the sole reference for claiming LLM improvement.
- Separate tuning tracks/scenarios from held-out evaluation. Robustness tests should
  include all available track CSVs plus gap, lateral-offset, speed, and seed sweeps.
- Position 1/2/3 runs with homogeneous policies mainly test start-order symmetry;
  they are useful but insufficient as independent samples.
- Preserve exact CLI/config values and weight hashes or paths for reproducibility.
  NPZ traces already record most policy, model, initialization, and solver metadata.

## 7. Claims still requiring evidence

- Whether the LLM meets real-time requirements and whether delayed decisions change
  outcomes.
- Whether prompt rules help because of better reasoning or merely because they act
  as a deterministic safety controller; include the pure-rule baseline.
- Whether conclusions persist under the dynamic bicycle simulator (`dbm`) rather
  than only the default kinematic model.
- Statistical uncertainty: final claims need repeated independent trials/seeds,
  not only three start-position permutations.

## 8. Recommended final system design

- The manuscript's purpose is to study high-level LLM strategy in a real-time
  multi-agent task. Prefer a stable,
  standardized platform over source-faithful low-level differences between
  baselines.
- Primary environment: kinematic simulation. Its dynamics closely match the
  shared four-state MPC predictor, reducing planner–plant mismatch. Use the DBM
  only for a small robustness ablation after primary conclusions are fixed.
- Shared parameterized-policy stack: static Alpha-Racer, learned Alpha-Racer,
  pure rules, raw LLM, and LLM with explicit prompt rules should differ only in
  how they select `(q, zeta, s1, s2, s3)` and otherwise share MPC, constraints,
  vehicle parameters, collision handling, and recovery.
- Locally adapted IBR (`local_wang`) may serve as an independent
  trajectory-planning baseline. It is not at exactly the same abstraction level
  as the five-parameter policies and should be described separately.

## 9. Collision geometry, slack, and boundary metrics

- Simulator contact uses oriented rectangles. The center displacement is rotated
  into each car's body frame. Rectangles overlap only when both longitudinal and
  lateral projections overlap; both body frames are checked because headings may
  differ. A positive overlap area triggers the contact speed penalty.
- In axis-aligned Frenet coordinates, rectangular overlap is approximately
  `|delta_s| < 2*half_length AND |delta_e| < 2*half_width`. Collision-free motion
  is its logical negation:
  `|delta_s| >= 2*half_length OR |delta_e| >= 2*half_width`. This disjunction is
  awkward in a smooth IPOPT NLP.
- The planner therefore uses the differentiable exclusion constraint
  `(delta_s/a)^p + (delta_e/b)^p >= 1`, with `p=4`, as a conservative smooth
  approximation of the rectangular overlap region. Physical oriented-rectangle
  collision and predicted superellipse violation must be reported separately.
- Soft track/collision constraints are a shared feasibility mechanism, not a
  learned policy capability or safety guarantee. Report maximum slack, activation
  fraction, and solver recovery for every policy.
- For the local Berlin configuration, crossing `|e| > 0.44 m` is recorded as a
  nominal boundary violation. The released alpha-RACER competition simulator
  independently activates recovery at `|e| > 0.55 m`; the paper itself states
  only the symbolic boundary `|p_y| > w_max/2` and does not report `0.55`.
  Therefore describe `0.55 m` as a released-simulator implementation threshold,
  not a paper-derived tolerance or a newly inferred physical track width.
- Record both nominal-boundary violation time and recovery-active time. The former
  is the primary geometric track-compliance metric; the latter exposes how often
  simulator assistance was required.

## 10. Raceline provenance

- Alpha-RACER Appendix B constructs the single-agent reference by minimizing
  summed squared curvature over lateral offsets, constrained by local track width
  and vehicle width, then creates a speed profile from longitudinal/lateral
  acceleration limits. Cite this as the provenance of an upstream raceline.
- A raceline and speed profile are environment assets, not learned strategic-policy
  outputs. Recompute them when track geometry/width, coordinate scaling, vehicle
  footprint, friction/acceleration limits, or the target track changes materially.
- Exact Alpha-RACER replication may retain the released Berlin raceline. For local
  or additional tracks, either regenerate each raceline using one frozen optimizer
  configuration or label supplied CSVs as fixed precomputed references and avoid
  claiming that they are locally optimal. All compared policies must share the
  same track-specific reference.

## 11. August 13--15: initial LLM and action-interface results

### Shared-MPC policy comparison

- The initial three-position comparison used the same four hand-designed tactical
  modes (`stable`, `push`, `overtake`, `block`) and the same low-level MPC for raw
  LLM, explicit-rule-prompt LLM, and deterministic rule-only policies.
- Against two locally trained Alpha-Racer cars, raw mode-output LLM won 2/3 and
  reached mean progress `193.2 m`, close to Alpha-Racer self-play (`193.0 m`).
  Rule-prompt LLM and rule-only won 0/3 and reached `111.3 m` and `134.8 m`.
- The explicit rules tended to select `stable` repeatedly. `stable` is not a
  recovery controller, so more rules made the tested policies conservative without
  reliably improving safety. Do not generalize beyond the current rule set.
- Raw mode inference took roughly 3--4 s and the longer rule prompt roughly 4--5 s,
  missing the one-second high-level deadline on every call. The synchronous
  simulator pauses while waiting, so these races measure decision quality under an
  oracle timing assumption, not physical performance under stale decisions.

### Direct five-parameter LLM ablation

- Direct-param LLM removes the four mode presets and outputs canonical
  `(q, zeta, s1, s2, s3)` inside the same official-derived bounds. This is an
  action-interface ablation, not a separately trained controller.
- Initial direct-param prompt: 0/3 wins against Alpha and IBR; mean progress was
  `179.5 m` vs Alpha and `183.7 m` vs IBR, below raw mode LLM (`193.2/193.1 m`).
  Across 300 calls, choices collapsed near
  `(2.80, 1.12, 0.405, 0.276, 0.015)`. `s1` was `0.4` in 271/300 calls and `s3`
  stayed at `0.015/0.016`. Several textual reasons reversed the actual effects of
  `q`, `s2`, or `s3`.
- One principled prompt correction was tested: re-evaluate every parameter, state
  the easy-to-confuse monotonic directions, permit exact bounds, and use `s3=0`
  when blocking is disabled. No scene-to-number rules or preset values were added.
- Effect-aware direct-param result: still 0/3 against both opponents; mean progress
  fell to `177.0 m` vs Alpha and `177.4 m` vs IBR. Exact-bound compliance improved:
  `s3=0` appeared in 156/300 calls, and a mild `s3~=0.056` appeared in one IBR
  race. This did not produce better overall performance.
- The main remaining behavior was race-level locking rather than continuous
  adaptation. In IBR position 3, `zeta=1.21, s1=0.3, s2=0.2, s3=0` persisted for
  all 50 calls. In IBR position 2, low `zeta~=0.902` persisted for 46/50 calls and
  `s3~=0.056` for 47/50, even after the rear car disappeared; that run reached only
  `153.1 m`.
- Direct-param inference averaged roughly 6--8 s per call. Six races require 300
  serial calls, about 35 minutes of LLM waiting alone, before MPC/IBR and report
  generation time.
- Supported claim: under the tested Qwen2.5-7B model and prompts, direct parameter
  selection used a small set of persistent configurations, was less state-adaptive,
  slower to infer, and lower-performing than the semantic mode interface.
- Do not claim that all LLMs cannot output continuous controls. Plausible mechanisms
  still to test include indirect/coupled parameter effects, lack of rollout-derived
  value information, weak numerical calibration, and anchoring to current
  parameters. The five parameters modify an MPC reference rather than immediate
  throttle/steering, while Alpha-Racer learns their consequences from rollout data.
- The hand-designed mode presets remain a disclosed action abstraction. Their use
  means mode-output results evaluate LLM selection within a fixed tactical library,
  not unconstrained end-to-end control. Direct-param underperformance shows the
  abstraction is useful for this model, but does not prove the current preset values
  are optimal.

### IBR fairness interpretation

- Alpha/LLM MPC has explicit velocity-loss weight zero, but raceline speed still
  enters implicitly through the time-parameterized longitudinal reference:
  `s_ref[k+1] = s_ref[k] + zeta*v_race[k]*dt`. Tracking reference position therefore
  induces a nominal pace.
- Ordinary IBR directly rewards terminal progress and can use the global `5 m/s`
  state limit. It achieved `237.3 m`, `4.74 m/s`, and 3/3 wins against Alpha. A
  physically reachable raceline-speed hard-cap diagnostic fell to `183.0 m`,
  `3.61 m/s`, and 0/3. Thus unrestricted IBR's lead cannot be attributed solely to
  better interaction reasoning.
- Keep ordinary IBR as an unrestricted full-trajectory optimization reference, not
  a like-for-like high-level-policy ranking. If needed, add a separately named
  pace-matched IBR with a soft horizon speed-profile loss calibrated on
  non-interactive runs; do not replace ordinary IBR or use the hard-cap diagnostic
  as the primary baseline.

## 12. August 28: structured-basis and generation-order ablations

### Initial evidence-first basis

- The raw mode policy generates `mode -> free-text basis -> reason`. It selected
  `push` in 44/45 decisions in the three-position 15 s Berlin screen, so its race
  performance alone did not establish situation-dependent tactical selection.
- A detachable evidence-first variant generated one structured observable basis
  before its mode. The first 13-label version changed the distribution to 24
  `push`, 16 `overtake`, and 5 `stable`, while raw remained almost entirely
  `push`. Race outcomes were effectively unchanged: both won 1/3, mean progress
  was `61.24 m` (raw) versus `61.19 m` (evidence-first), and neither had a
  collision, boundary violation, or off-road recovery.
- Only 25/38 non-ambiguous evidence-first claims were supported (`65.8%`). Ten
  instances labelled a rear vehicle as closing when its signed closing speed was
  negative. Road-left labels were also produced for negative/right-turn preview
  values. The structured output therefore exposed factual errors that free text
  could not be scored reliably.
- Data: `result/llm_basis_ab_berlin/audit/summary.json`,
  `result/llm_basis_ab_berlin/audit/decisions.csv`,
  `result/llm_basis_ab_berlin/audit/report.html`, and race summaries
  `result/llm_basis_ab_berlin_raw/summary.json` and
  `result/llm_basis_ab_berlin_evidence_first/summary.json`. Per-position NPZ files
  are under each summary's `summary_traces/` directory.

### Controlled action-first versus evidence-first order

- To separate vocabulary priming from autoregressive order, the non-output parts
  of the prompts were made identical and three variants were run: raw free text,
  structured action-first (`mode -> basis`), and structured evidence-first
  (`basis -> mode`). Action-first and evidence-first used exactly the same labels.
- Raw and structured action-first produced the same mode for all 45 decisions and
  identical trajectories. Thus merely listing the structured vocabulary after
  the mode did not explain the changed behavior in this screen.
- Evidence-first produced 33 `push` and 12 `overtake`. Its non-ambiguous factual
  support was `78.4%`, compared with `55.6%` for action-first. This initially
  appeared to support an effect of generation order. Later controlled tests did
  not reproduce a two-call benefit and showed strong sensitivity to small prompt
  changes. Treat the early mode diversity as a prompt-confounded observation,
  not evidence that a basis-first reasoning path is better.
- Data: `result/llm_basis_order_ab_berlin/audit/summary.json`,
  `result/llm_basis_order_ab_berlin/audit/decisions.csv`,
  `result/llm_basis_order_ab_berlin/audit/report.html`, and the three race files
  `result/llm_basis_order_ab_berlin_{raw,structured_action_first,evidence_first}/summary.json`.

### Minimal traffic-only basis

- Ego-side and road-direction labels were removed from the intermediate basis.
  The retained eight states describe whether a front/rear vehicle is absent or
  whether its gap is closing, opening, or approximately constant. The full ego
  and road observation remained available to mode selection; only the auditable
  bottleneck was narrowed.
- Minimal evidence-first produced 37 `push` and 8 `overtake`. Its factual support
  rose to `88.6%` (31 supported, 10 ambiguous, 4 contradicted), while mean progress
  (`61.19 m`), speed (`4.45 m/s`), wins (1/3), and safety were unchanged. This is
  the cleanest current structured-basis variant.
- Minimal action-first selected `push` 45/45 and reached `60.0%` non-ambiguous
  support. Its frequent false `front_vehicle_pulling_away` statements after
  choosing `push` are consistent with post-decision rationalization.
- The supported interpretation is that a small task-relevant traffic bottleneck
  improved extraction and auditability and reduced mode collapse, not that it
  improved race performance.
- Data: `result/llm_basis_minimal_ab_berlin/audit/summary.json`,
  `result/llm_basis_minimal_ab_berlin/audit/decisions.csv`,
  `result/llm_basis_minimal_ab_berlin/audit/report.html`, and
  `result/llm_basis_minimal_ab_berlin_{raw,structured_action_first,evidence_first}/summary.json`.

### Dual traffic plus track basis

- A second five-state track basis attempted to combine previewed turn direction
  with ego lateral position (`inside`, `outside`, `near center`, direction change,
  or no clear turn). This was an output-layer change only; no heading or speed
  observation was added.
- The dual evidence-first version degraded traffic support from `88.6%` to
  `62.2%` and achieved only `15.6%` track support. It selected
  `upcoming_turn_direction_changes` in 37/45 decisions, including many previews
  with large, consistent single-direction turn angles and curvatures. Evidence
  suggests a new label collapse rather than successful track abstraction.
- It also increased mean latency from `2.98 s` to `3.55 s`, shifted modes back
  toward `push` (40 `push`, 5 `overtake`), and did not change race outcome or
  safety. The dual version should remain a negative ablation, not replace the
  minimal traffic-only version.
- Data: `result/llm_basis_track_ab_berlin/audit/summary.json`,
  `result/llm_basis_track_ab_berlin/audit/decisions.csv`,
  `result/llm_basis_track_ab_berlin/audit/report.html`, and
  `result/llm_basis_track_ab_berlin_{raw,structured_action_first,evidence_first}/summary.json`.

### Interpretation limits

- `supported/ambiguous/contradicted` measures whether the emitted observable
  relation matches the numerical input. It is not mode accuracy, optimal-action
  accuracy, or overall scene-understanding accuracy.
- The closing-speed audit uses a `0.1 m/s` numerical deadband; lateral and preview
  checks use `5%` half-width and `1 degree` measurement deadbands. These are
  audit definitions, not mode-selection rules.
- Sign interpretation remains an observed weakness rather than a universally
  reversed convention. In the minimal evidence-first screen,
  `closing_on_front_vehicle` was correct 15/15 and
  `rear_vehicle_falling_behind` was supported 9 times, but three decisions still
  reversed the sign of closing speed (one rear-closing and two front-pulling-away
  claims). A fourth contradiction claimed no front vehicle when one was present.
- The earlier left/right errors have not been shown to be fixed. The minimal
  traffic version does not emit a left/right road label, and the dual track version
  failed mostly through `direction_changes` collapse. A separate track-only probe
  is required before claiming that the LLM understands signed road geometry.

## 13. September 5--6: corrected basis interpretation and prompt sensitivity

### Correction to the early two-stage interpretation

- The 11-case offline basis intervention used a shorter, more directive second
  prompt and traffic-balanced independent samples. Its varied mode distribution
  cannot be attributed to two-stage reasoning.
- It remains valid only as a sensitivity result: reversing the supplied traffic
  relation changed `7/11` paired outputs. It did not provide a correct-mode label
  or execute the selected action.
- A controlled 15 s online comparison found no two-call benefit. One-call versus
  two-call results were: supported basis `37/45` vs `34/45`, `push` `39` vs `43`,
  `overtake` `6` vs `2`, and mean latency `3.17 s` vs `6.80 s`. Both won `1/3`.
- A balanced two-call follow-up had `32/40` supported, `7/40` ambiguous, and
  `1/40` contradicted basis labels. Its informative counterfactual acceptance was
  `0/3`. Online two-call policies were removed after the result was committed.
- Data: `result/tmp_08_one_vs_two_step_fixed*/` and
  `result/tmp_11_balanced_two_step_20s/`.

### Minimal no-default-mode prompt intervention

- Raw and balanced differ by one sentence saying that no mode is the default.
  It adds no threshold, quota, or condition-to-mode mapping.
- In matched P2/P3 50 s Berlin races, raw selected `push` `100/100`. Balanced
  selected `push` `77/100`, `overtake` `22/100`, and `stable` `1/100`; neither
  selected `block`. Mode switches increased from `0` to `46`.
- Mean P2/P3 progress increased from `191.26 m` to `202.61 m`. The gains were
  `8.12 m` at P2 and `14.59 m` at P3. Safety effects were mixed.
- This shows strong prompt sensitivity. It does not establish better tactical
  reasoning. Two balanced decisions described overtaking a rear vehicle.
- Data: `result/tmp_09_raw_no_default_mode_berlin_50s_p23*/`.

### Short-horizon counterfactual mode screen

- Each sampled decision is restored and replayed with all four fixed presets.
  Opponents follow their recorded paths. Branches run for `2 s`, extending to
  `5 s` when needed.
- Score: change in ego lead over the most advanced recorded opponent. A safe mode
  is accepted within a 20% regret allowance. A scene is informative when at least
  one mode is rejected.
- Stratified P2/P3 results: raw accepted `27/32` overall and `5/10` informative
  choices; balanced accepted `26/32` overall and `1/7` informative choices.
- Most ordinary states accepted all four modes. Several balanced `push` or
  `overtake` choices near sharp turns lost relative progress to `stable` or
  `block`. Added diversity therefore did not imply higher local utility.
- Frozen opponents make this a diagnostic, not optimal-mode ground truth.
- Data: `result/tmp_10_balanced_counterfactual_bundle/`.

### Retained claim and policy decision

- The LLM usually extracts a simple traffic relationship correctly. The mapping
  from factual grounding to useful action remains unreliable.
- Retain raw as the simple policy. Retain balanced only as a prompt-sensitivity
  ablation. Do not retain online two-call policies.
- Treat Berlin as development data. Pause further prompt tuning before held-out
  track evaluation.

## 14. September 6: paired real-scene regret benchmark

- A common development bank was formed from 12 dynamically reachable states in
  the 50 s Berlin P2/P3 raw and balanced traces. It contains three samples each
  for ordinary running, high upcoming curvature, closing front traffic, and
  closing rear traffic. Selection uses only observed state/traffic/geometry,
  suppresses near duplicates, and never uses the recorded mode.
- Every prompt received the same saved JSON observation. Each of the four mode
  presets was replayed from the identical state with frozen recorded opponents.
  The primary horizon was `1 s`, matching one online high-level decision
  interval; `3 s` fixed-mode persistence was retained as a diagnostic.
- Primary constrained regret was
  `best safe relative-progress gain - chosen safe relative-progress gain`.
  Unsafe choices were counted separately rather than assigned an arbitrary
  metre-valued crash penalty.
- One-second results (`12` paired scenes): raw selected `push 12`; balanced
  selected `push 7 / overtake 5`; evidence-first selected
  `push 10 / overtake 2`. All choices were safe. Mean safe-choice regret was
  `0.024 m`, `0.013 m`, and `0.011 m`, respectively. All three were within
  `0.05 m` of best in `10/12` scenes.
- Only `6/12` scenes produced at least `0.05 m` spread between mode outcomes.
  On these effect-bearing scenes, mean regret was `0.047 m` (raw), `0.026 m`
  (balanced), and `0.021 m` (evidence-first). These small development-set
  differences are insufficient for a superiority claim.
- Evidence-first basis grounding was `9 supported / 3 ambiguous / 0
  contradicted`. The three ambiguous cases called an approximately constant
  front gap `pulling away`.
- At the non-primary `3 s` persistence horizon, mean regret increased to
  `0.225 m`, `0.173 m`, and `0.168 m`. This suggests mode effects need time to
  accumulate, but holding one mode for three decision periods is not the online
  policy being evaluated.
- No prompt selected `block`. Frozen opponent replay cannot measure the causal
  effect of blocking because the rear car does not react. Rear-closing scenes
  therefore test prompt response and ego motion, not blocking effectiveness.
- Data and report: `result/common_scene_mode_dev_v1/summary.json`,
  `result/common_scene_mode_dev_v1/decisions.csv`, and
  `result/common_scene_mode_dev_v1/report.html`.
- Code: `evaluation/common_scene_mode_benchmark.py` and
  `run_common_scene_mode_benchmark.sh`. Literature notes are in
  `REFERENCES.local.md`.

## 15. September 7: local racelines and responsive-opponent audit

- Sources: `result/local_raceline_and_interactive_audits/`. The preset audits
  used the new local Modena and rounded-rectangle racelines, with 24 three-second
  branches per track.
- Both local profiles increased clear-road progress relative to the previous
  profiles. However, they also changed the ordering of the four fixed presets.
  On Modena, `stable` slightly outperformed `push` on the clear straight
  (`11.50 m` versus `11.23 m`). On rounded rectangle the difference was only
  `0.07 m`. Interaction rankings also changed substantially. This is evidence
  that the current presets are tied to the track/profile/controller and should
  not yet be interpreted as invariant racing skills.
- The new Modena audit had `1/24` unsafe branch and rounded rectangle had
  `0/24`; neither had solver failures. Full-lap safety and completion have not
  yet been established, so these profiles remain candidates rather than frozen
  evaluation references.
- The responsive-opponent block audit used the same plant and constrained MPC
  for ego and challenger. It covered three closing conditions, two mirrored
  passing sides, and four ego presets (`24` branches). Mirrored branches were
  numerically symmetric.
- That block scene was not a valid efficacy test: `18/24` branches were unsafe,
  and even the `stable` and `s_3=0` controls collided in every passing case.
  `block` did alter the lateral reference (`max |p_{bl}|=0.136--0.153 m`) and
  sometimes delayed the challenger, but it was unsafe in all six branches and
  did not help consistently. Zero solver failures together with large collision
  and track slack show feasible soft-constrained solutions, not a numerical
  crash.
- The next interactive audit must first screen for scenes in which the
  `s_3=0` challenger completes a safe overtake. Only those valid controls can be
  used to ask whether nonzero blocking safely delays or prevents the pass.

## 16. September 8: synthesis after repairing the shared baseline

### What was corrected or experimentally tested before interpreting the LLM experiments

- The steering map was restored to `S_a=0.34`, and the erroneous
  Cartesian/Frenet prediction projection was replaced with an internally Frenet
  four-state MPC. Reference-speed tracking was tested as a separate objective
  ablation; it is **not** currently enabled in the default configuration.
- In one controlled Position-3 hairpin, the result changed from a stopped car at
  `64.2 m` with `w_v=0` to successful traversal at `98.0 m` with `w_v=10`,
  without solver failure or recovery. A three-position `w_v=10` regression also
  removed the observed stall for local Alpha-Racer and rule-based control, while
  raw LLM and Parameter-IBR retained one failure each.
- This evidence is case-specific. It does not establish that the common stall is
  generally solved, and it does not describe the current default, which remains
  `VELOCITY_TRACK_WEIGHT=0.0`. A matched `w_v=0` versus `w_v=10` regression is
  required before selecting the frozen primary MPC configuration. Results before
  the `S_a` and projection corrections should not be mixed numerically with the
  final comparison.

### What the traffic-basis experiments establish

- The retained traffic basis describes one observable relationship: absence of
  a front/rear vehicle or whether its gap is closing, opening, or approximately
  constant. It is an auditable bridge from numerical observation to action, not
  an expert label for the correct mode.
- In the 45-decision screen, `35/36` non-ambiguous evidence-first labels were
  supported. In the longer 150-decision screen, evidence-first had `96`
  supported, `53` ambiguous, and `1` contradicted label. The tested LLM can
  therefore extract these simple traffic relations reliably.
- Adding a five-state track basis failed: only `5/45` labels were supported and
  the output collapsed toward `upcoming_turn_direction_changes`. Track
  understanding has not been established and should be tested separately from
  traffic grounding.
- Correct basis extraction did not imply useful action selection. The long raw,
  action-first, and evidence-first runs all selected `push` in `148--150/150`
  decisions and produced almost identical progress. The unresolved step is
  `traffic basis -> useful tactical action`.

### What generation order and prompt changes establish

- The early 11-scene supplied-basis probe was confounded by a shorter, more
  directive second prompt. Its larger mode diversity demonstrates text
  sensitivity, not a two-stage reasoning benefit.
- In the controlled online comparison, one call versus two calls produced
  `37/45` versus `34/45` supported labels, essentially unchanged progress, and
  mean latency of `3.17 s` versus `6.80 s`. The two-call implementation was
  removed from the active policy set.
- A single neutral sentence stating that no mode was the default changed the
  P2/P3 distribution from raw `push 100/100` to balanced `push 77`,
  `overtake 22`, `stable 1`, with 46 switches. Mean progress increased from
  `191.26 m` to `202.61 m`, but informative counterfactual acceptance fell from
  `5/10` to `1/7`. More varied actions and greater progress are not, by
  themselves, evidence of more accurate tactical reasoning.

### What the counterfactual rollout can and cannot measure

- The evaluator restores a real recorded state, executes every fixed preset,
  replays recorded opponents, and compares safe change in ego progress relative
  to the leading opponent. It avoids assigning a manually chosen best mode.
- In the original 12-scene common bank, only `6/12` scenes separated mode
  outcomes by at least `0.05 m` over one second. Mean regret on these scenes was
  small (`0.047/0.026/0.021 m` for raw/balanced/evidence-first), insufficient
  for a method ranking.
- The expanded 60-scene bank reinforced the limitation. Raw, balanced, and
  evidence-first selected `push` in `60/60`, `55/60`, and `58/60` cases. Their
  mean one-second regret was nearly identical
  (`0.01035/0.01021/0.01028 m`), and `58/60` choices for every method were
  within `0.05 m` of best. One-second mode effects are often too weakly
  separated; longer fixed-mode rollout changes the deployed decision problem.
- Frozen opponents cannot react to blocking, so this evaluator measures local
  ego utility rather than a complete interactive best response or ground-truth
  mode accuracy.

### What Alpha-Racer and preset audits say about the action abstraction

- Local Alpha-Racer updates all five parameters continuously. In the 50-second
  traces, post-hoc `K=5` clusters formed long phases and gradual parameter
  movement; they did not reveal four naturally discrete
  `stable/push/overtake/block` states. Cluster IDs were also fitted independently
  per race and cannot be assigned fixed tactical meanings.
- The LLM modes should therefore be described as a hand-designed executable
  action library. They intentionally reduce the numerical burden on the LLM but
  introduce quantization, abrupt target changes, within-mode rigidity, and a
  preset-dependent performance ceiling. Direct-parameter LLM remains the action
  representation ablation; its tested outputs were persistent and less
  competitive, but this does not make the manual presets optimal.
- Controlled target scenarios showed that `stable`, `push`, and `overtake` can
  create useful directional effects. `block` remains opponent- and timing-
  dependent. The new Modena and rounded-rectangle audits further showed that
  changing the raceline/profile changes preset ranking, so mode names are not
  guaranteed to transfer unchanged across tracks.
- The current vocabulary mixes two dimensions: `stable/push` mainly describe
  pace or aggression, while `overtake/block` describe interaction intent. A
  later aggression-only ablation (`cautious/balanced/aggressive`) could test a
  cleaner ordered interface. A factorized `intent x aggression` policy is a
  possible longer-term design, but it is not justified as an immediate rewrite.

### Current defensible result

- The LLM reliably extracts a limited observable traffic relation and is highly
  sensitive to prompt wording. Neither structured basis generation, two-call
  decomposition, nor increased mode diversity has yet demonstrated better
  tactical action utility.
- The main method is therefore an **LLM tactical selector over a fixed MPC action
  library**, not a reproduction of Alpha-Racer's continuous optimizer. The next
  evidence should come from frozen multi-track evaluation, valid responsive
  interaction scenes, and delayed-execution tests rather than further Berlin
  prompt tuning.

### Manuscript synchronization

- Future cloud edits target the new Overleaf project
  `LLM-racer-experiment record2`:
  `https://www.overleaf.com/project/6a74ea065860dfb8b4e0c951`.
- The second record is stored only under `LLM-racer-experiment record 2/`, whose
  `.overleaf/settings.json` points to the new project ID. Repository-root
  `main.tex` and `LLM-racer-experiment record/main.tex` are identical copies of
  the first record. The old directory is deliberately not bound to the new
  project.
