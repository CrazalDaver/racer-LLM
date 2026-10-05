# Local versus released alpha-RACER low-level implementation

> Historical comparison; `K-Constrained-v1` is authoritative for the current
> frozen substrate.

Last checked: August 31, 2026

This is a local, gitignored fact log for later debugging and manuscript writing.
It compares the current repository with the released alpha-RACER numerical race
code in `../alpha-RACER`. It is not a claim that either implementation is the
unique formulation described by the paper. The released code contains multiple
controllers and parameter sets, so every comparison below names the relevant
code path.

## Source map

- Local shared parameters: `config.py`
- Local vehicle steps: `vehicle_model.py`
- Local simulator and recovery/contact handling: `simulation_core/environment.py`
- Local shared MPC: `mpc_controller.py`
- Local trajectory IBR: `policy/ibr/trajectory_ibr.py`
- Released numerical race harness:
  `../alpha-RACER/car_ros2/car_ros2/multi_car_comp_blocking.py`
- Released numerical plant:
  `../alpha-RACER/car_dynamics/car_dynamics/models_jax/dbm.py`
- Released environment action clipping:
  `../alpha-RACER/car_dynamics/car_dynamics/envs/car3.py`
- Released numerical simulator parameters:
  `../alpha-RACER/simulators/params-num.yaml`

## Confirmed differences and correspondences

| Item | Current local implementation | Released numerical project | Consequence / how to report it |
|---|---|---|---|
| Executed plant | Default is a kinematic bicycle wrapped in the six-field state `(x,y,psi,vx,vy,omega)`. Here `vy=0` and `omega` is reconstructed. A DBM is available only as an ablation. | The numerical race instantiates the six-state JAX `DynamicBicycleModel` and advances it through `OffroadCar`. | The main local experiments deliberately use a reduced kinematic plant, not the released DBM. This improves agreement with the four-state planner but is not a source-faithful plant reproduction. |
| Planning state | Shared MPC and trajectory IBR use four-state Frenet dynamics `(s,e,psi_error,v)`. | The released standalone IBR helper also uses four Frenet states. The released race plant remains six-state DBM. | Both stacks contain planner--plant abstraction, but the local default plant is much closer to its planner. |
| MPC coordinate handling | The current MPC projects the measured Cartesian state once, then predicts and evaluates errors directly in Frenet coordinates. Before August 24 it predicted Cartesian states but measured each prediction against a preselected future reference cross-section; this was an inaccurate local projection and has been removed. | The released IBR is directly Frenet; other released tracking paths use their own waypoint/controller implementation. | The old projection was a real local formulation bug, not a paper or upstream feature. Its removal improved ordinary Alpha screens but did not by itself eliminate Parameter-IBR deadlock. |
| Timing | Plant/contact, policy, and MPC all advance every `0.1 s`; LLM policies may retain high-level parameters for `1 s`. | Numerical race uses `DT=0.1 s`; the IBR helper also uses `dt_ibr=0.1 s`. | The primary local timing now matches the released numerical step. The slower LLM decision interval is a policy-level schedule, not a second physics rate. |
| Geometry | `LF=0.11`, `LR=0.23`, wheelbase `0.34 m`. Vehicle half-length/half-width are `0.18/0.12 m`. | Numerical harness declares `LF=0.12`, `LR=0.24`, wheelbase `0.36 m`; its rectangle vertices use these dimensions and width approximately `L/3=0.12 m`. | Width is effectively aligned, while wheelbase differs by `0.02 m`. The local wheelbase must be used when regenerating curvature-limited racelines. |
| Steering map | Current baseline: `delta=0.34*u_delta`, with `Sb=0`. | Numerical DBM is instantiated with `Sa=0.34`, `Sb=0`; the standalone IBR helper defaults to `Sa=0.36`, `L=0.36`. | The local plant matches the released numerical DBM steering scale, but not every released helper. Historical local `Sa=0.80` runs are invalid as current calibration evidence. |
| Longitudinal kinematic model | `v_next = v + (20/4.65*u_a - 1)*dt`. Normalized throttle can be negative in the kinematic experiment. | Released IBR uses the same displayed kinematic equation. The numerical DBM instead receives throttle clipped by `OffroadCar` to `[0,1]` and uses its force model. | Local kinematic MPC/IBR acceleration matches the released IBR helper, not the executed released DBM in every physical detail. |
| DBM parameters | Optional local DBM uses `Ta=20`, `Tb=0`, `Sa=0.34`, `Sb=0`, `mu=0.5`, delay metadata `1`, and the local six-state force equations. | The numerical harness instantiates its DBM with the same listed values. | Parameter values were aligned for the DBM ablation, but equivalence still requires checking integration and delay semantics; do not call it bitwise reproduction. |
| Frenet heading dynamics | Local main trajectory solver uses the geometric term `kappa*v*cos(psi_error)/(1-kappa*e)` and protects the singular denominator smoothly. | Released IBR uses the approximation `kappa*v` in heading error, although its longitudinal update retains `(1-kappa*e)^{-1}`. | This is an intentional local modeling correction and can alter difficult boundary/curvature behavior. `historical_ibr.py` retains the released approximation for comparison. |
| Track width and thresholds | Nominal half-width is `0.44 m`; planner boundary is `|e|<=0.44`. Recovery begins at `|e|>0.55 m`. | `params-num.yaml` says total track width `1.0 m`; the race harness has `track_width=1.0` and separately triggers recovery at `|e|>0.55`. The released IBR helper does not visibly impose the same local `0.44 m` boundary. | Local optimization is stricter than released recovery. Report nominal violation (`0.44`) separately from recovery activation (`0.55`); the `0.11 m` gap is not a paper-derived tolerance. |
| Track constraint | Local MPC/trajectory IBR enforce lateral containment and soften it with non-negative quadratic slack. | The released IBR NLP shown in the race harness has no explicit lateral track-boundary constraint. The released race relies partly on reference following and simulator recovery. | Slack is a local feasibility mechanism necessitated by stricter optimization constraints. Report slack activation; do not attribute it to Wang or alpha-RACER. |
| Collision constraint | Local planners use a differentiable Frenet superellipse with optional slack and constraints on nearby opponents. | Released IBR uses a quadratic expression in opponent progress fields; physical contact is tested separately using oriented rectangles. | Local smooth collision avoidance is not a literal reproduction. In both projects, physical contact and the planner's collision surrogate are different objects. |
| Physical contact | Oriented rectangles are tested in the simulator, followed by asymmetric front/rear speed decay once per `0.1 s` step. | The numerical harness constructs oriented vehicle polygons and applies collision effects once per `0.1 s` update. | The update frequency and decay cadence now align; exact collision response should still be treated as locally adapted. |
| Off-road recovery | Reproduces the released `0.55 m` speed decay, heading realignment, inward steering, and large-heading throttle boost once per `0.1 s` step. | The numerical harness modifies state and steering after `|e|>0.55`, and adds `0.2` throttle for heading error over one radian. | Recovery is simulator assistance shared by all policies, not learned behavior. |
| Friction use | MPC/IBR optionally enforce the released-style proxy `|v^2 tan(u_delta)| <= mu*g*L` with `mu=1.6`. The physical DBM ablation uses tire friction `mu=0.5`. | Released IBR uses the same proxy and default `mu=1.6`; the executed DBM is instantiated with `mu=0.5`; track speed-generation scripts also contain `mu=0.8`. | The symbol `mu` has different roles across subsystems. Always state which friction parameter is meant; there is no single upstream friction number. |
| Speed limit/reference | Local planners have a global state cap of `5 m/s`. The shared MPC's explicit velocity-profile cost is optional and currently has default weight `0`; raceline points still define its positional reference. | Released waypoint/controller stack supplies raceline and speed information, while released IBR applies a speed bound tied to another car in its helper. | “Raceline-following” does not automatically mean identical speed objectives. Fairness discussions must distinguish positional reference, velocity cost, and hard speed cap. |
| Raceline | The active Berlin raceline was regenerated locally after restoring `Sa=0.34`; the `Sa=0.80` candidate is archived as history. | Released project supplies its own Berlin reference and optimization utilities under `global_racetrajectory_optimization`. | A raceline is an environment/controller asset tied to geometry and acceleration assumptions. All policies in a local comparison should use the same active file. |

## Important internal inconsistencies in the released project

These are reasons to identify a concrete released code path instead of writing
“the original parameters” as though there were only one set.

1. The numerical DBM uses `Sa=0.34`, while the standalone IBR dynamics default
   to `Sa=0.36`.
2. The numerical geometry uses wheelbase `0.36 m`, while the local inherited
   geometry is `0.34 m`.
3. Friction appears as at least `0.5` in the DBM, `1.6` in the IBR lateral proxy,
   and `0.8` in raceline speed helper scripts. These values serve different
   models and are not interchangeable.
4. The released physical plant is DBM, while its IBR prediction is a reduced
   four-state kinematic Frenet model.
5. The released `0.55 m` recovery trigger is an implementation constant; it is
   not documented in the paper as an extra geometric tolerance.

## Historical local changes that must not be confused with the current baseline

- `Sa=0.80` was a historical workaround/error. The active baseline is `0.34`.
- The raceline generated under `Sa=0.80` is retained only as a diagnostic asset.
- Source-faithful/simple-MPC and DBM experiments were temporary audits, not the
  main system unless a future experiment explicitly selects them.
- Low-speed boundary recovery, terminal-viability penalties, and stall-debt
  experiments are not enabled in the current default configuration.
- Velocity and heading tracking terms exist as ablation controls; both currently
  default to zero in `config.py`. Quote the actual CLI/configuration of a run.

## Manuscript-safe summary

The clean description is: the project retains alpha-RACER's high-level policy
interface and a related reduced-order racing controller, but evaluates them in a
local simulator with a kinematic default plant, locally regenerated reference,
explicit smooth safety constraints, and finer plant integration. Released
checkpoints and controller ideas are adapted at their interfaces; they are not a
bit-for-bit reproduction of the released ROS/numerical stack. Trajectory IBR is
an additional Wang-inspired local baseline and should be presented separately
from policies that share the five-parameter MPC interface.

## Checks still worth doing before final submission

- Record the exact active raceline hash/path and all low-level config values in
  each final result bundle.
- Verify the local optional DBM equation and delay handling line by line against
  `models_jax/dbm.py` before making any DBM robustness claim.
- Run final comparisons with one frozen plant, timing, recovery, contact, MPC,
  and raceline configuration.
- Report planner slack, nominal boundary violation, physical recovery, and
  oriented-rectangle contact as separate measurements.
- If the released project is discussed quantitatively, distinguish reproduction
  results from local-adaptation results and name the released controller path.
