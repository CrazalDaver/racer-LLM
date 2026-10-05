# MPC low-speed stall investigation

> Historical diagnostic log; not current protocol evidence.

Date: August 22--23, 2026

Purpose: preserve what was tried while investigating the local MPC/IBR low-speed
failure. This is a diagnostic log, not manuscript-ready evidence. Several runs
were single screens with different durations, so their raw numbers must not be
treated as a controlled benchmark. The corresponding `result/` directories are
still present and contain `summary.json`, NPZ traces, and HTML reports.

## 1. Starting observation

After switching to the locally recomputed raceline, policies using the shared MPC
(especially Parameter-IBR) could enter a persistent low-speed state near the track
boundary or after an interaction. The repeated applied command was often close to
neutral throttle and zero steering. Some trajectory-IBR and Alpha-Racer runs also
showed related low-speed behavior, but Parameter-IBR exposed it most reliably.

The local stack at the time used:

- a four-state kinematic MPC prediction model;
- `SIM_DT = 0.02 s`, `CONTROL_DT = MPC_DT = 0.1 s`;
- soft track and collision constraints with separate slacks;
- a time/progress-parameterized raceline reference;
- simulator contact using oriented rectangles and MPC collision avoidance using a
  differentiable Frenet superellipse;
- nominal track violation at `|e| > 0.44 m` and simulator off-road recovery at
  `|e| > 0.55 m`.

## 1.1 How the investigation arrived here (chronology)

This chronology matters because the current stall was not the original research
question. It appeared after several earlier confounds were removed.

1. **A local trajectory-level IBR was implemented.** Its ordinary formulation
   directly optimized throttle/steering and terminal progress. In the initial
   three-position comparison it was much faster than every policy using the shared
   Alpha-Racer MPC: the retained manuscript note reports mean progress `237.3 m`,
   mean speed `4.74 m/s`, and 3/3 wins against local Alpha-Racer, compared with
   approximately `193.0 m` and `3.91 m/s` for Alpha-Racer. Related evidence is in
   `result/ibr_baseline_freeze/position_sweep_summary.json`, the objective/slack
   variants under `result/ibr_recommended_audit_20260817/`,
   `result/ibr_speed_cap_suite/raceline_capped_ibr.json`, and the earlier
   `result/ibr_comparison/runner.log`.
   These directories contain multiple IBR formulations/ablations, so consult each
   summary or runner log before quoting a number.

2. **Parameter-IBR was introduced to separate abstraction level from interaction
   reasoning.** Instead of directly choosing throttle and steering, it iteratively
   searched the same five high-level parameters `(q, zeta, s1, s2, s3)` and then
   passed them through the same MPC used by Alpha-Racer and the LLM. It was much
   slower than trajectory IBR and generally closer to the other MPC-based methods.
   This suggested that much of trajectory IBR's speed advantage might come from
   direct trajectory/control optimization rather than IBR reasoning alone.
   Useful early screens are:

   - `result/llm_vs_parameter_ibr_screen/`
   - `result/alpharacer_vs_parameter_ibr_screen/`
   - `result/llm_vs_refined_parameter_ibr/`
   - `result/local_alpha_vs_refined_parameter_ibr/`
   - `result/alpha_vs_parameter_ibr_shared_loss/`

   For example, the 15 s Alpha-versus-Parameter-IBR screen recorded progress in
   the approximate `37--63 m` range, far below the scale of the earlier 50 s
   unrestricted trajectory-IBR comparison. Durations differ, so those raw values
   must not be compared directly; the important controlled observation was that
   methods sharing MPC behaved more similarly to one another.

3. **Attention therefore shifted to the common MPC/raceline stack.** The upstream
   raceline was tied to different geometry and vehicle assumptions, so a local
   minimum-curvature path and speed profile were generated. Initial 15 s static
   screens showed higher progress with the candidate raceline:

   - `result/raceline_old_screen/`: `[54.12, 48.02, 40.85] m`
   - `result/raceline_new_screen/`: `[64.09, 58.55, 52.62] m`

   This initially made the MPC methods look substantially better and reinforced
   the suspicion that the earlier reference asset was limiting them.

4. **A historical vehicle-parameter error was then found.** The parameter was
   `Sa`, not `Sb`:

   - `Sa` is the steering-command scale in
     `steering_angle = Sa * steering_command + Sb`;
   - `Sb` is an additive steering bias;
   - an earlier local workaround had left `Sa=0.80`, more than twice the released
     numerical setting;
   - the current kinematic baseline uses `Sa=0.34`, `Sb=0.0`, with
     `LF+LR=0.34 m`.

   The first locally optimized raceline had therefore been generated under
   unrealistically strong steering authority. It is retained only as
   `track_csv/berlin_2018_raceline_with_speeds_local_sa080_history.csv`.
   After restoring `Sa=0.34`, the raceline was recomputed with the corresponding
   curvature bound. Evidence:

   - `result/sa034_old_raceline_screen/`: old reference, progress
     `[54.12, 47.99, 40.84] m`
   - `result/sa034_new_raceline_screen/`: `Sa=0.34` candidate, progress
     `[62.43, 56.91, 51.09] m`
   - raceline generation provenance: `track_csv/tools/README.md` and
     `track_csv/generated/`

   The corrected raceline still improved the short static screen, so the entire
   gain was not solely an `Sa=0.80` artifact. However, the correction reduced
   available turning authority and made constrained recovery maneuvers harder.

5. **The corrected setup exposed the present low-speed failure.** A 36-race suite
   on the promoted corrected raceline showed that some cars progressed normally
   while others stopped near approximately `65 m`, especially in Parameter-IBR
   and some shared-MPC interactions. The main evidence map is:

   - runner: `result/new_raceline_full_suite/runner.log`
   - Alpha self-play: `result/new_raceline_suite_local_alpha_selfplay/`
   - Parameter-IBR self-play:
     `result/new_raceline_suite_parameter_ibr_selfplay/`
   - trajectory-IBR self-play:
     `result/new_raceline_suite_trajectory_ibr_selfplay/`
   - Alpha vs Parameter-IBR:
     `result/new_raceline_suite_local_alpha_vs_parameter_ibr/`
   - Alpha vs trajectory IBR:
     `result/new_raceline_suite_local_alpha_vs_trajectory_ibr/`
   - Parameter-IBR vs trajectory IBR:
     `result/new_raceline_suite_parameter_ibr_vs_trajectory_ibr/`
   - LLM/rule comparisons: the other
     `result/new_raceline_suite_*` directories.

   Concrete examples from the 40 s homogeneous runs:

   - Alpha self-play contained two cars near `65 m` in one ordering and one car
     near `65 m` in the others;
   - Parameter-IBR self-play always had one car near `167 m` and two near `65 m`;
   - trajectory-IBR self-play also contained a near-`64 m` car, although the other
     two reached roughly `146--156 m`.

   These results are why the investigation moved from “trajectory IBR is simply
   faster” to “what common low-level mechanism causes some optimizers to settle or
   repeatedly replan into a stopped state?” The detailed deadlock, speed-tracking,
   recovery, DBM, and simplified-MPC tests below followed from this point.

This ordering should be preserved in a later write-up. Otherwise it is easy to
mistake the current MPC debugging as arbitrary baseline tuning. In reality, each
step removed a confound: abstraction level, raceline provenance, and then the
historical steering calibration. The remaining stall is the behavior that emerged
under the corrected local stack.

## 2. Working explanation developed from the saved horizon diagnostics

The current explanation has two connected parts.

First, the kinematic yaw rate is proportional to speed,

```text
psi_dot = v / L * tan(delta).
```

At nearly zero speed, steering alone cannot rotate the car. Returning toward the
reference therefore requires acceleration and steering together. In a difficult
boundary/collision configuration, that maneuver can increase predicted lateral
error and track/collision slack before it produces useful progress. Neutral
throttle approximately balances drag, while zero steering keeps the immediate
predicted state nearly unchanged, so the first stationary action can have lower
short-horizon cost.

Second, this can become receding-horizon procrastination. A solved horizon may
contain a neutral first action and acceleration later in the plan. Only the first
action is executed. At the next MPC update the problem is solved again and the
acceleration can again be postponed. Thus the saved prediction need not be
stationary over its entire horizon even though the executed car remains stopped.

For Parameter-IBR there is an additional suspected reference pathology. Its chosen
five parameters move the Frenet reference. If the vehicle stalls while the
time-indexed/reference horizon continues advancing, a local Frenet/tangent
linearization can predict rapidly increasing lateral error or slack. This makes
the delayed recovery maneuver look even more expensive. This mechanism is
supported by trace inspection, but has not yet been isolated by a clean reference
construction ablation; it should still be described as the leading hypothesis,
not a proved theorem.

## 3. Diagnostic runs and what they do (and do not) show

### 3.1 Detailed deadlock trace

- Evidence: `result/parameter_ibr_deadlock_diagnostic/`
- Setup recorded in the summary: three Parameter-IBR cars, kinematic plant,
  `SIM_DT=0.02`, velocity weight `0.5`, about 25 s, ego start position 3.
- Final progresses were approximately `[64.65, 65.32, 66.18] m`.
- The run contained substantial boundary/recovery/collision activity, including
  `3.08 s` of physical collision and up to `7.68 s` nominal-boundary violation for
  one car.
- Horizon logging showed the important distinction between planned future
  acceleration and the repeatedly executed near-neutral first action.
- Interpretation: this is the main evidence for the procrastination explanation.
  It is one difficult run, not a failure-rate estimate.

### 3.2 Velocity-reference term

The shared MPC was given the standard quadratic speed tracking term

```text
w_v * sum_k (v_k - v_ref,k)^2,
v_ref,k = zeta * v_raceline,k.
```

Relevant screens:

- `result/mpc_speed_tracking_alpha_w0/`: Alpha self-play, `w_v=0`, about 30 s,
  final progress `[117.86, 130.76, 119.20] m`.
- `result/mpc_speed_tracking_alpha_w05/`: same screen with `w_v=0.5`, final
  progress `[120.52, 131.06, 122.10] m`. Off-road recovery time fell from `0.34 s`
  for one car to zero, while nominal-boundary time changed but did not disappear.
- `result/mpc_speed_tracking_parameter_ibr_w05/`: Parameter-IBR remained in the
  same poor/deadlock-like outcome as the detailed diagnostic.

Interpretation: `w_v=0.5` mildly improved the ordinary shared-MPC Alpha screen and
removed one recovery activation, but it did not resolve the Parameter-IBR-specific
case. Do not claim statistical improvement from these single runs. The term remains
enabled because it removes the otherwise weak incentive to maintain raceline pace
and has a conventional MPC interpretation.

### 3.3 Sustained-low-speed boundary recovery

An action-only simulator supervisor was tried: after the car remained slow near
the boundary for a configured duration, it imposed extra throttle and inward
steering until an exit condition was reached.

- Evidence: `result/low_speed_recovery_alpha_on/` and
  `result/low_speed_recovery_traj_ibr/` (about 20 s screens).
- The recorded Alpha screen reported no actual low-speed recovery trigger.
- The trajectory-IBR screen still had boundary recovery, collision, and uneven
  progress; it did not demonstrate a robust cure.

Interpretation: this supervisor changes executed actions without repairing the
optimization/reference that produced the bad plan. It is therefore difficult to
attribute and could hide policy failures. The code may remain for diagnosis, but
`LOW_SPEED_BOUNDARY_RECOVERY_ENABLED` is currently `False` and it should not be a
primary-method component without new evidence.

### 3.4 Dynamic bicycle model trial

- Evidence: `result/parameter_ibr_dbm_w0_diagnostic/` and
  `result/parameter_ibr_dbm_upstream_calibrated_w0/`, compared with
  `result/parameter_ibr_constrained_local_w0_control/`.
- All were about 25 s, position 3, with velocity tracking disabled.
- DBM changed the outcome but produced large nominal-boundary and recovery times;
  the upstream-calibrated variant had roughly `6.3--8.14 s` recovery per car.

Interpretation: changing the plant can move or remove a particular local optimum,
but these runs introduced predictor/plant and actuator-semantic differences and
did not yield a clean, safer controller. They do not show that DBM is a direct fix.
Use DBM later as a robustness ablation rather than changing the primary platform
in response to this failure.

### 3.5 Simplified-MPC constraint ablation

`upstream_simple` removed the local track/collision constraints, slacks, and speed
tracking and retained a much simpler reference/input objective.

- Evidence: `result/alpharacer_upstream_simple_w0/` and
  `result/parameter_ibr_upstream_simple_w0/` (about 25 s screens).
- The cars continued making progress, but boundary/recovery exposure increased.
  Parameter-IBR had approximately `5.08--9.36 s` nominal-boundary violation,
  `3.44--7.40 s` recovery, and `4.14 s` physical collision.

Interpretation: removing the feasibility/safety structure can avoid the same
stationary compromise, but does so by accepting unsafe trajectories. This supports
the claim that the interaction between constraints, slacks, and the reference is
part of the stall mechanism. It does not justify adopting the simplified MPC as
the local primary controller. The small `upstream_simple` implementation is kept
only as a detachable ablation.

### 3.6 Stricter released-stack reproduction audit (now removed from runtime code)

Temporary code reproduced more of the released numerical stack: simplified MPC,
DBM-like dynamics, released projection/recovery/contact behavior, released starts,
and a `0.1 s` simulator step. The scaffolding was removed after the audit to avoid
maintaining a second simulator inside the local project. Result files were retained.

- `result/upstream_exact_alpha_confirm/`: three 10 s Alpha self-play screens at
  `SIM_DT=0.02`; every car accumulated roughly `2.0--2.5 s` nominal-boundary time.
- `result/upstream_exact_runtime_dt01/`: three 10 s screens with the released-style
  runtime and `SIM_DT=0.1`; nominal-boundary time remained about `1.6--2.5 s`.
- `result/upstream_alpha_vs_released_ibr_three_positions_v2/`: three 50 s races,
  released-style Alpha against two released-style IBR cars. Alpha won none, but all
  cars accumulated substantial boundary and recovery time; Alpha had approximately
  `13.9--16.7 s` nominal-boundary violation across the three positions.

Interpretation: closer released-code behavior did not eliminate excursions; its
simulator recovery is materially involved in completing races. This is useful
context when explaining why source-code faithfulness is not automatically a safe
or clean local baseline. It is not an exact ROS/competition reproduction, and it
does not validate the local formulation. The local project has therefore returned
to one standardized simulator rather than keeping this reproduction layer.

## 4. Idea discussed but not implemented

An exponentially increasing penalty for predicted persistent stagnation was
proposed, roughly making a sequence of low predicted speeds increasingly costly.
It was **not implemented or tested**. At present it is not recommended because:

- it would be a new locally invented objective component;
- its scale and trigger would require more tuning;
- it could force unsafe acceleration in genuinely blocked situations;
- it might conceal the reference/linearization defect instead of resolving it.

If later needed, a conventional terminal progress, terminal speed, or viability
condition has clearer MPC precedent than an exponential stagnation penalty, but
even those should follow the reference-horizon audit.

## 5. Current retained local configuration

- Primary plant: kinematic.
- Primary MPC: `constrained_local` with explicit track/collision slacks.
- `SIM_DT=0.02 s`; control and MPC updates remain `0.1 s`.
- Velocity tracking weight remains `0.5`.
- Low-speed boundary recovery remains disabled.
- Ordinary off-road recovery still activates only past `|e|=0.55 m`.
- `upstream_simple` remains only as a small constraint ablation.
- Strict released runtime/MPC/IBR/start-profile branches have been deleted.
- Existing `result/upstream_exact_*` and
  `result/upstream_alpha_vs_released_ibr_*` directories remain as audit evidence,
  although their deleted runtime modes cannot now be rerun without recovering the
  implementation from version history/local backup.

## 6. Recommended next investigation

Do not add another recovery mechanism or tune a new penalty first. Isolate the
reference construction while holding dynamics, constraints, weights, and scenario
fixed:

1. Re-run one saved Parameter-IBR deadlock scenario with the current reference.
2. Add a diagnostic variant whose reference advancement is tied to predicted/local
   vehicle progress rather than continuing to move away from a stalled vehicle.
3. Compare first applied action, later planned actions, predicted `e`, predicted
   speed, track slack, collision slack, and solver status at the onset of stopping.
4. If this removes repeated deferral without increasing boundary/collision events,
   retain it as a reference-consistency fix.
5. Only if procrastination remains after that controlled test, evaluate one
   conventional terminal progress or terminal-speed term.

The stopping criterion for baseline engineering should be practical: the shared
MPC should run the frozen evaluation scenarios without systematic artificial
stalling, and safety/slack metrics should be measurable and comparable. It does
not need to reproduce every released engineering detail before returning to the
main research question about LLM high-level decisions.

## 7. Short answers if asked in a meeting

**Was neutral throttle explicitly programmed as recovery?** No. It emerged as the
first action of a short-horizon constrained optimization near a difficult state.
Neutral approximately cancels resistance, while accelerating and steering can
immediately increase predicted slack.

**Did the MPC predict zero speed forever?** Not necessarily. In the diagnosed case
it often planned acceleration later, but replanning repeatedly postponed that
action. This is why inspecting only the final applied command was misleading.

**Did velocity tracking solve it?** It mildly helped the ordinary Alpha screen but
did not solve the Parameter-IBR case. Current evidence is diagnostic and based on
single runs, not repeated statistics.

**Why not keep the manual low-speed recovery?** It intervenes after optimization
and makes attribution harder. It also did not show a clear benefit in the small
screen. True off-road recovery at `|e|>0.55 m` remains separate.

**Would DBM solve it?** The tested DBM variants changed the behavior but introduced
large off-road/recovery exposure. They are not evidence of a clean fix.

**Why keep slack?** Without slack, a temporarily infeasible nonlinear problem can
fail entirely. The simplified no-safety screen kept moving partly by accepting
large excursions and collisions. Slack should remain explicit and reported, while
its activation and interaction with the reference are audited.

**Why did the stricter released-code screen also leave the track?** The released
engineering stack uses simpler optimization constraints and relies materially on
simulator recovery. Closer reproduction therefore traded some optimization
deadlock for more excursions; it did not establish a universally better local
controller.

**What is the strongest current conclusion?** The stall is best understood as an
interaction among low-speed kinematic controllability, moving/reference
linearization, finite-horizon replanning, and soft safety costs. It is not yet
correct to attribute it to one weight, to kinematic dynamics alone, or to IBR
strategic reasoning.

## 8. Four-state Frenet MPC rewrite (August 24, 2026)

The main `constrained_local` MPC has now been rewritten so that its complete
internal prediction state is

\[
z_k=[s_k,e_k,\Delta\psi_k,v_k]^T,
\]

while the simulator continues to use Cartesian vehicle states. The Cartesian
state is projected into Frenet coordinates only once at the beginning of each
MPC solve. Within the NLP, the state is propagated using centerline curvature:

\[
\dot s=\frac{v\cos(\Delta\psi+\beta)}{1-\kappa(s)e},\qquad
\dot e=v\sin(\Delta\psi+\beta),
\]

\[
\dot{\Delta\psi}=\frac{v}{l_r}\sin\beta-\kappa(s)\dot s,
\qquad \dot v=a(v,u_a).
\]

This removes the old approximation that evaluated a symbolic Cartesian
prediction relative to the preselected future reference cross-section. Tracking,
track-boundary, and collision terms now use predicted `s` and `e` directly. The
same predicted Frenet plan is also reused by Parameter-IBR instead of being
projected back from Cartesian a second time.

Centerline curvature is computed periodically from the existing polyline heading
over a `0.25 m` centered window. A smooth minimum denominator of `0.05` is used
only to keep recovery iterates numerically defined when a soft-constraint iterate
temporarily leaves the valid Frenet tube. This is numerical regularization rather
than an added racing objective.

The default velocity-profile weight is returned to `0.0`. Thus the retained core
objective advances through longitudinal/lateral position references, as in the
paper-level formulation; velocity tracking remains an explicit CLI ablation.

### Verification results

- `result/frenet_mpc_alpha_25s_w0`: three ordinary Alpha policies, P3, 25 s.
  Progress was `106.12`, `109.69`, and `106.27 m`; all cars finished at about
  `4.87--5.00 m/s`. There was no low-speed time after the first second, no
  boundary violation, no recovery, no collision, and every MPC solve succeeded.
- `result/frenet_mpc_alpha_vs_parameter_ibr_5s`: one Alpha and two Parameter-IBR
  policies completed the short screen without boundary/recovery/collision events
  and with all solves successful.
- `result/frenet_mpc_parameter_ibr_deadlock_recheck_w05`: three Parameter-IBR
  policies still stopped near `s=64 m` in the tight hairpin during a 25 s run,
  despite all NLP solves succeeding and without actual off-road events.
- A single Alpha car also crossed this hairpin and reached approximately
  `107.20 m` in 25 s, so the corrected path and `S_b=0.34` are individually
  feasible.

The revised conclusion is narrower and more useful. The previous coordinate
approximation was a real formulation error and is now removed. It was not,
however, the sole cause of the saved Parameter-IBR deadlock. In that run,
Parameter-IBR's interaction-dependent offsets brought the three vehicles into the
tight hairpin with large heading errors; the finite-horizon MPC then repeatedly
postponed acceleration from a locally difficult configuration. Ordinary Alpha
policies do not reproduce that failure in the matching 25 s screen. Further work
should therefore examine Parameter-IBR candidate feasibility/selection rather
than adding another global MPC recovery patch.

## 9. Saved-state viability audit (September 8, 2026)

The matched 50 s `w_v=0`/`w_v=10` regression showed that speed tracking reduces
nominal-boundary exposure but does not eliminate the hairpin stall.  A detachable
audit, `evaluation/tmp_08_deadlock_state_feasibility.py`, restored stalled states
from the `w_v=10`, track-slack-weight `10^5` traces.  It removed the tracking
objective and asked whether the same kinematic dynamics, control/rate/friction
bounds, and hard `|e|<=0.44 m` boundary admit 0.25 m or 1.0 m of forward progress.
IPOPT used 24 initial guesses at 2 s and 3 s horizons.  This is strong numerical
evidence, not a formal global infeasibility certificate.

- The stalled Rule P3 ego (`e=0.209`, heading error `0.328 rad`) could advance
  1 m without track violation at `N>=20` when considered alone.  With its saved
  opponents and hard collision avoidance, no safe progress solution was found.
- Parameter-IBR P2 (`e=0.385`, heading error `0.693 rad`) needed about `0.20 m`
  extra half-width even to advance 0.25 m, including at `N=30` and with opponents
  removed.
- Parameter-IBR P3 (`e=0.016`, heading error `1.297 rad`) needed about `0.25 m`
  extra half-width to advance 0.25 m after the stall was established.  At stall
  onset it could still advance 0.25 m safely, but not 1 m; the subsequent executed
  motion carried it out of the forward-safe viability region.
- The stationary Alpha immediately ahead of the Rule ego was itself outside the
  strict forward-safe region: 0.25 m progress required about `0.041 m` extra
  half-width.  It therefore created a secondary traffic blockage rather than an
  independently recoverable two-car waiting equilibrium.

The current soft-slack NLP remains numerically solvable, but several saved states
have no zero-slack forward continuation under the modeled bounds.  This explains
why changing the objective after stopping is ineffective.  The next root-oriented
step should be prevention: preserve a known feasible continuation or reject a
high-level reference before it drives the car outside the forward-safe viability
region.  Switching only the simulator plant to DBM is not supported as a cure by
the existing DBM audits.

## 10. Lower slack penalty plus velocity tracking (September 8, 2026)

The 50 s trace `1000_slack_velocity_track_test.npz` used the latest local Berlin
raceline, `N=10`, track-slack weight `1e3`, and reference-speed weight `10`.  The
P3 raw-LLM car and both local Alpha cars completed the run without a sustained
low-speed stall, and every MPC solve succeeded.  Progress gains were `206.6`,
`217.1`, and `218.6 m`, respectively.

This combination is a useful empirical mitigation of the hairpin deadlock.  The
speed term discourages repeatedly postponing acceleration, while the lower slack
penalty permits a temporarily imperfect recovery trajectory.  It is not yet a
complete solution: the LLM car crossed the nominal boundary for `1.92 s`, was
actually off-road (`|e|>0.55 m`) for `1.30 s`, reached `max |e|=0.710 m`, and
had `0.26 s` of collision contact.  Its off-road recovery returned it to the
track, which is more plausible than a permanent stationary equilibrium but
introduces a safety--liveness trade-off.  The LLM also selected `push` in all 50
decisions, so this run evaluates low-level liveness rather than diverse LLM
tactical reasoning.  Treat `rho_t=1e3, w_v=10` as a calibration candidate, not
as a proven optimum; compare nearby slack weights before freezing the shared MPC.
