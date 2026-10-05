# Reference log

This file records literature that may support the project design. It is a
working note, not a claim that the current code exactly reproduces every cited
method.

## LLM high-level decisions with a conventional low-level controller

- Sha et al., **LanguageMPC: Large Language Models as Decision Makers for
  Autonomous Driving** (2023), https://arxiv.org/abs/2310.03026
  - Relevant precedent: an LLM handles high-level driving decisions and its
    output is translated into parameters/actions used by a lower-level control
    system.
  - Relation to this project: close architectural motivation for
    observation -> LLM tactical choice -> MPC, but not the same racing task,
    modes, or evaluation.
- Ichter et al., **Do As I Can, Not As I Say: Grounding Language in Robotic
  Affordances** (CoRL 2022/2023),
  https://proceedings.mlr.press/v205/ichter23a.html
  - Relevant precedent: separate semantic high-level choice from physically
    executable skills and ground the choice with skill values/affordances.
  - Relation: the four alpha-RACER presets are analogous to a small executable
    skill library. Our rollout score is closer to an after-the-fact local
    affordance audit than to SayCan's learned affordance model.

## Structured `basis` between observation and mode

- Koh et al., **Concept Bottleneck Models** (ICML 2020),
  https://proceedings.mlr.press/v119/koh20a.html
  - Relevant precedent: predict human-readable concepts before a downstream
    prediction, and intervene on those concepts to test their effect.
  - Important qualification: our `basis` is prompt-generated, not a supervised
    learned concept bottleneck. Therefore the paper motivates the separation
    and intervention idea, but does not validate our eight traffic labels.
- Wei et al., **Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models** (NeurIPS 2022),
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html
  - Relevant precedent: eliciting an intermediate reasoning step can change
    task performance.
  - Qualification: the short categorical `basis` is not a private or complete
    chain of thought. It is a deliberately verifiable public intermediate.
- Turpin et al., **Language Models Don't Always Say What They Think:
  Unfaithful Explanations in Chain-of-Thought Prompting** (NeurIPS 2023),
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html
  - Relevant warning: a plausible explanation can rationalize an answer and
    need not reveal the cause of the prediction.
  - Consequence here: `reason` is qualitative evidence only. Factual basis
    grounding, controlled basis interventions, and rollout outcomes are kept
    separate from explanation quality.

There is no identified paper that exactly uses our pipeline
`numeric racing observation -> one traffic basis label -> alpha-RACER mode`.
The defensible description is **concept-bottleneck-inspired structured
reasoning**, not a reproduction of Concept Bottleneck Models or CoT.

## Scenario-set and behavioural evaluation design

- Ribeiro et al., **Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList** (ACL 2020), https://aclanthology.org/2020.acl-main.442/
  - Relevant precedent: organize black-box evaluation by capabilities and use
    minimum-functionality, invariance, and directional tests rather than one
    aggregate accuracy number.
  - Relation: ordinary racing, upcoming curvature, front interaction, and rear
    interaction can be treated as explicit capability slices. Synthetic target
    scenarios are diagnostic tests rather than a natural-frequency benchmark.
- Fremont et al., **Scenic: A Language for Scenario Specification and Scene
  Generation** (PLDI 2019),
  https://people.eecs.berkeley.edu/~sseshia/pubdir/pldi19-scenic.pdf
  - Relevant precedent: declaratively generate constrained scenes to cover
    rare autonomous-system conditions and generalize known failures.
- Althoff, Koschi, and Manzinger, **CommonRoad: Composable Benchmarks for
  Motion Planning on Roads** (IV 2017),
  https://commonroad.in.tum.de/static/media/LA17.8f9d9dc3.pdf
  - Relevant precedent: a reproducible benchmark specifies the scenario,
  dynamics, vehicle parameters, cost, and planning problem. CommonRoad mixes
  recorded and hand-crafted scenarios.
- Caesar et al., **Towards Learning-Based Planning: The nuPlan Benchmark for
  Real-World Autonomous Driving** (2024),
  https://arxiv.org/abs/2403.04133
  - Relevant precedent: mine and taxonomize common and rare driving situations,
    then report scenario-specific closed-loop planning metrics rather than only
    an aggregate trajectory error.
  - Relation: supports reporting separate clear-track, front-interaction,
    rear-interaction, and cornering capability slices alongside full races.
- Jia et al., **Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop
  End-To-End Autonomous Driving** (NeurIPS 2024),
  https://arxiv.org/abs/2406.03877
  - Relevant precedent: evaluates 44 interactive scenario categories over
    multiple locations and separates abilities such as overtaking and cut-in
    handling.
  - Relation: supports the two-level design of short controlled mode probes
    plus longer races. It does not imply that our four modes have externally
    defined ground-truth labels.
- Bak et al., **Stress Testing Autonomous Racing Overtake Maneuvers with RRT**
  (2021), https://arxiv.org/abs/2110.01095
  - Relevant precedent: vary opponent behaviour in otherwise controlled racing
    scenarios and search for failures in overtaking logic.
  - Relation: motivates responsive/adversarial opponent variants for testing
    `overtake` and `block`; a constant-velocity dummy alone cannot establish a
    causal blocking effect.
- Oegretmen et al., **Trajectory Planning Using Reinforcement Learning for
  Interactive Overtaking Maneuvers in Autonomous Racing Scenarios** (2024),
  https://arxiv.org/abs/2404.10658
  - Relevant precedent: evaluates interactive racing planners under different
    blocking aggressiveness and reports both success and collision behaviour.
  - Relation: supports treating interaction strength as an experimental factor
    instead of reducing all mode quality to short-term progress.

Design consequence for this project:

1. Use a frozen **real-trace scene bank** for the primary local-regret score.
   These states are dynamically reachable under the actual simulator.
2. Use **synthetic target scenes** only to guarantee coverage of rare semantic
   capabilities such as a fast rear vehicle. Report them by capability; do not
   mix them into the primary score as if they represented race frequency.
3. Freeze scene IDs and prompt variants before final evaluation. Use other
   tracks and repeated races as a held-out confirmation set.

## Counterfactual rollout and regret

- Bottou et al., **Counterfactual Reasoning and Learning Systems: The Example
  of Computational Advertising** (JMLR 2013),
  https://www.jmlr.org/papers/v14/bottou13a.html
  - Relevant principle: compare a system change by estimating what would have
    happened under the changed decision while making causal/model assumptions
    explicit.

For scene (x), let (U(x,m)) be the simulated change in ego's signed lead
over the leading opponent after holding mode (m) for one high-level decision
interval. Among safe branches,

`regret(x, chosen) = max_m U(x,m) - U(x, chosen)`.

Regret is reported in metres and is zero for a best safe branch. If the chosen
branch is unsafe, no invented metre-equivalent crash penalty is added: unsafe
choice rate is reported separately. The older "within 80%" label remains only
as a threshold-sensitivity diagnostic.

Limitations that must accompany results:

- Opponents replay recorded motion and do not react to the counterfactual ego.
- Regret is local to the rollout horizon and the four preset action library.
- A simulator counterfactual is evidence under that simulator, not causal
  identification from real-world observational data.
- Scenes used while editing prompts form a development set, not a final held-out
  test set.

## Project-specific primary methods

- Wang et al., **Game-Theoretic Planning for Self-Driving Cars in Competitive
  Scenarios** (2019), https://msl.stanford.edu/papers/wang_game_2019.pdf
  - Source for iterative best response and sensitivity-aware game planning.
- Zheng et al., **alpha-RACER** (2024/2025),
  https://arxiv.org/abs/2412.08855
  - Source for the five-parameter racing policy, value/potential learning, and
    released baseline implementation.
  - Released controller code:
    https://github.com/sastry-group/alpha-RACER/blob/main/car_ros2/car_ros2/mpc_controller.py

## Candidate track datasets and raceline tools

- TUMFTM, **racetrack-database**,
  https://github.com/TUMFTM/racetrack-database
  - More than 20 F1/DTM circuits with centerline coordinates, left/right track
    widths, and minimum-curvature racelines; LGPL-3.0.
  - The geometry originates from smoothed OpenStreetMap GPS centerlines and
    satellite-derived widths. The authors explicitly warn that source quality
    varies by circuit.
  - Best primary source for adding held-out real-track geometries. Import the
    centerline/bounds and regenerate the speed profile with this project's
    frozen local vehicle model.
- F1TENTH, **f1tenth_racetracks**,
  https://github.com/f1tenth/f1tenth_racetracks
  - F1TENTH-ready 1:10 versions of over 20 F1/DTM circuits, including
    centerlines, widths, occupancy maps, and optimized racelines; GPL-3.0.
  - Closest plug-in format for the present small-car simulator, but its supplied
    velocity profiles still reflect a different generation setup. Prefer
    regenerating them locally for controlled comparisons.
- Zhou, **MapZoo: A Collection of 1:10-Scale Racing Maps**,
  https://github.com/zhouhengli/MapZoo
  - Curated MIT-licensed collection of 32 maps with centerlines, bounds,
    racelines, velocity fields, and overtaking-sector metadata.
  - Useful secondary source for F1TENTH/ForzaETH-style tracks and deliberately
    interaction-oriented maps. Provenance should be checked per map before use
    in a final benchmark because the collection aggregates multiple sources.
- TUMFTM, **global_racetrajectory_optimization**,
  https://github.com/TUMFTM/global_racetrajectory_optimization
  - Provides shortest-path, minimum-curvature, iterative minimum-curvature, and
    minimum-time trajectory optimization; LGPL-3.0.
  - The local `train/raceline/optimize_raceline.py` reuses its geometry pipeline
    but constructs speed profiles from the local kinematic vehicle parameters.

Recommended use for generalization experiments:

1. Treat Berlin as the development track on which prompts and controller
   settings were inspected.
2. Regenerate Modena and rounded-rectangle racelines using the same local
   optimizer before calling them controlled validation tracks.
3. Select a small frozen held-out set from the TUM/F1TENTH geometry collection,
   covering a fast-flowing layout, a technical layout, and a mixed layout.
4. Run single-car feasibility checks first, then the identical multi-agent
   policy suite. Never mix supplied external speed profiles with locally
   generated profiles in the primary cross-track comparison.
