from config import GRAVITY
from vehicle_model import kinematic_frenet_step

import numpy as np
import casadi as cs


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class MPCController:
    """
    Shared four-state Frenet kinematic MPC tracker.

    The simulator remains Cartesian, but the optimization state is
    ``[s, e, psi_error, v]``.  This directly represents the paper's
    longitudinal/lateral tracking and track constraint instead of projecting
    symbolic Cartesian predictions through a future reference cross-section.
    """

    _instance_counter = 0

    def __init__(
        self,
        dt,
        horizon,
        Q,
        R,
        max_vel_rate,
        max_steer_delta,
        db_params,
        track_width,
        track_boundary_slack=0.0,
        track_constraint_half_width=None,
        max_state_speed=5.0,
        velocity_track_weight=0.0,
        heading_track_weight=0.0,
        track_length=None,
        min_longitudinal_separation=0.36,
        min_lateral_separation=0.20,
        max_constrained_opponents=2,
        collision_superellipse_exponent=4,
        enforce_track_boundary=True,
        enforce_collision_avoidance=True,
        track_slack_weight=1.0e3,
        collision_slack_weight=1.0e4,
        enforce_friction_speed_limit=True,
        friction_coefficient=1.6,
        track=None,
    ):
        self.dt = dt
        self.horizon = horizon
        self.Q = Q[:2, :2] if Q.shape[0] > 2 else Q
        self.R = R
        self.max_vel_rate = max_vel_rate
        self.max_steer_delta = max_steer_delta
        self.p = db_params
        self.track_half_width = track_width / 2.0
        if track_constraint_half_width is None:
            self.track_constraint_half_width = self.track_half_width + float(track_boundary_slack)
        else:
            self.track_constraint_half_width = float(track_constraint_half_width)
        self.track_boundary_slack = self.track_constraint_half_width - self.track_half_width
        self.max_speed = max_state_speed
        self.velocity_track_weight = velocity_track_weight
        self.heading_track_weight = heading_track_weight
        self.track_length = None if track_length is None else float(track_length)
        self.min_longitudinal_separation = float(min_longitudinal_separation)
        self.min_lateral_separation = float(min_lateral_separation)
        self.max_constrained_opponents = max(0, int(max_constrained_opponents))
        self.collision_superellipse_exponent = int(collision_superellipse_exponent)
        self.enforce_track_boundary = bool(enforce_track_boundary)
        self.enforce_collision_avoidance = bool(enforce_collision_avoidance)
        self.track_slack_weight = float(track_slack_weight)
        self.collision_slack_weight = float(collision_slack_weight)
        self.enforce_friction_speed_limit = bool(enforce_friction_speed_limit)
        self.friction_coefficient = float(friction_coefficient)
        if track is None:
            raise ValueError("Frenet MPC requires the project Track instance")
        self.track = track
        if self.min_longitudinal_separation <= 0.0 or self.min_lateral_separation <= 0.0:
            raise ValueError("MPC collision separations must be positive")
        if self.collision_superellipse_exponent < 2 or self.collision_superellipse_exponent % 2:
            raise ValueError("collision_superellipse_exponent must be an even integer >= 2")
        if self.track_slack_weight < 0.0 or self.collision_slack_weight < 0.0:
            raise ValueError("MPC slack weights must be non-negative")
        if self.friction_coefficient <= 0.0:
            raise ValueError("MPC friction coefficient must be positive")
        self.last_loss = np.nan
        self.last_track_loss = np.nan
        self.last_action_loss = np.nan
        self.last_velocity_loss = np.nan
        self.last_heading_loss = np.nan
        self.last_violation_loss = np.nan
        self.last_track_slack_loss = np.nan
        self.last_collision_slack_loss = np.nan
        self.last_track_slack_plan = None
        self.last_collision_slack_plan = None
        self.last_solver_success = True
        self.last_solver_status = "not_run"
        self.last_recovery_active = False
        self.last_recovery_reason = ""
        self.last_state_plan = None
        self.last_control_plan = None
        self.last_opponent_s_plan = None
        self.last_opponent_e_plan = None
        self.last_opponent_active = None
        self.n_states = 4
        self.n_controls = 2
        curvature_s, curvature = self.track.curvature_profile()
        self.max_abs_curvature = float(np.max(np.abs(curvature)))
        self.minimum_nominal_frenet_denominator = float(
            1.0 - self.max_abs_curvature * self.track_constraint_half_width
        )
        length = float(self.track.total_length)
        # Three periodic copies let a short unwrapped MPC horizon cross the
        # start/finish seam without a symbolic modulo operation.
        extended_s = np.concatenate((
            curvature_s[:-1] - length,
            curvature_s,
            curvature_s[1:] + length,
        ))
        extended_curvature = np.concatenate((
            curvature[:-1], curvature, curvature[1:],
        ))
        instance_id = MPCController._instance_counter
        MPCController._instance_counter += 1
        self._curvature_grid = extended_s
        self._curvature_values = extended_curvature
        self.curvature_fn = cs.interpolant(
            f"centerline_curvature_{instance_id}",
            "bspline",
            [extended_s],
            extended_curvature,
        )

        # Define optimization variables and parameters
        self.X = cs.SX.sym(
            "X", self.n_states, self.horizon + 1
        )  # Frenet state trajectory: (s, e, heading_error, v)
        self.U = cs.SX.sym("U", self.n_controls, self.horizon) # control trajectory
        # Non-negative violation variables soften only track and collision
        # safety constraints. Dynamics, state bounds and control limits remain
        # hard constraints.
        self.track_slack = cs.SX.sym("track_slack", self.horizon)
        self.collision_slack = cs.SX.sym(
            "collision_slack",
            self.max_constrained_opponents,
            self.horizon,
        )
        self.x0 = cs.SX.sym("x0", self.n_states, 1) # initial state
        # Reference rows remain width-eight for trace/backward compatibility;
        # the Frenet solver uses [s_ref, e_ref, v_ref].
        self.ref = cs.SX.sym("ref", 8, self.horizon + 1)    
        self.u_prev = cs.SX.sym("u_prev", self.n_controls, 1) # previous control input for rate penalty
        self.q_weight = cs.SX.sym("q_weight", 1) 
        # Paper (1g)-(1h): predicted Frenet positions of the nearest vehicle
        # ahead and behind. Following the paper, opponents keep constant
        # longitudinal velocity and zero lateral velocity over the horizon.
        self.opponent_s = cs.SX.sym(
            "opponent_s",
            self.max_constrained_opponents,
            self.horizon + 1,
        )
        self.opponent_e = cs.SX.sym(
            "opponent_e",
            self.max_constrained_opponents,
            self.horizon + 1,
        )
        self.opponent_active = cs.SX.sym(
            "opponent_active",
            self.max_constrained_opponents,
        )
        self._build_problem()

    def _vehicle_dynamics(self, xk, uk):
        return kinematic_frenet_step(
            xk,
            uk,
            self.curvature_fn(xk[0]),
            self.dt,
            self.p,
            curvature_mode="geometric",
            symbolic=True,
        )

    def _vehicle_dynamics_np(self, xk, uk):
        curvature = float(np.interp(
            float(xk[0]), self._curvature_grid, self._curvature_values
        ))
        return kinematic_frenet_step(
            xk,
            uk,
            curvature,
            self.dt,
            self.p,
            curvature_mode="geometric",
        )

    def _make_x0_mpc(self, x0, current_s):
        current_s = float(current_s)
        _, lateral_e = self.track.project(
            float(x0[0]),
            float(x0[1]),
            s_hint=current_s,
            search_radius=2.0,
            psi_hint=float(x0[2]),
            max_heading_error=np.pi / 2.0,
        )
        _, _, center_heading = self.track.interpolate(current_s)
        heading_error = wrap_angle(float(x0[2]) - center_heading)
        return np.array(
            [
                current_s % self.track.total_length,
                lateral_e,
                heading_error,
                max(float(x0[3]), 0.0),
            ],
            dtype=float,
        )

    def _localize_reference(self, ref_traj, current_s):
        """Align an unwrapped policy reference with the MPC's local lap."""
        localized = np.asarray(ref_traj, dtype=float).copy()
        anchor = float(current_s) % self.track.total_length
        for k in range(localized.shape[1]):
            localized[0, k] = anchor + self._progress_delta(
                float(ref_traj[0, k]), float(current_s)
            )
        return localized

    def _progress_delta(self, s_a, s_b):
        if self.track_length is None:
            return float(s_a) - float(s_b)
        return np.arctan2(
            np.sin((float(s_a) - float(s_b)) * 2.0 * np.pi / self.track_length),
            np.cos((float(s_a) - float(s_b)) * 2.0 * np.pi / self.track_length),
        ) * self.track_length / (2.0 * np.pi)

    def _select_interaction_opponents(self, opponents, current_s):
        if not opponents or self.max_constrained_opponents <= 0:
            return []
        enriched = []
        for opponent in opponents:
            delta = self._progress_delta(opponent["px"], current_s)
            enriched.append((delta, opponent))
        front = [item for item in enriched if item[0] > 0.0]
        behind = [item for item in enriched if item[0] < 0.0]
        selected = []
        if front:
            selected.append(min(front, key=lambda item: item[0])[1])
        if behind and len(selected) < self.max_constrained_opponents:
            selected.append(max(behind, key=lambda item: item[0])[1])
        if len(selected) < self.max_constrained_opponents:
            selected_ids = {id(opponent) for opponent in selected}
            remaining = [
                item for item in sorted(enriched, key=lambda item: abs(item[0]))
                if id(item[1]) not in selected_ids
            ]
            selected.extend(item[1] for item in remaining[: self.max_constrained_opponents - len(selected)])
        return selected

    def _prepare_opponent_predictions(self, ref_traj, opponents=None, current_s=None):
        n = self.max_constrained_opponents
        opponent_s = np.zeros((n, self.horizon + 1), dtype=float)
        opponent_e = np.zeros((n, self.horizon + 1), dtype=float)
        active = np.zeros(n, dtype=float)
        if n == 0:
            return opponent_s, opponent_e, active

        if current_s is None:
            current_s = float(ref_traj[0, 0])
        selected = self._select_interaction_opponents(opponents, current_s)
        for opponent_idx, opponent in enumerate(selected):
            active[opponent_idx] = 1.0
            for k in range(self.horizon + 1):
                if "px_plan" in opponent and len(opponent["px_plan"]):
                    plan_idx = min(k, len(opponent["px_plan"]) - 1)
                    predicted_s = float(opponent["px_plan"][plan_idx])
                else:
                    predicted_s = float(opponent["px"]) + k * self.dt * float(opponent.get("vx", 0.0))
                if "py_plan" in opponent and len(opponent["py_plan"]):
                    plan_idx = min(k, len(opponent["py_plan"]) - 1)
                    predicted_e = float(opponent["py_plan"][plan_idx])
                else:
                    predicted_e = float(opponent.get("py", 0.0))

                # Align unwrapped simulator progress with the local wrapped
                # Frenet coordinate used in the MPC reference at this step.
                reference_s = float(ref_traj[0, k])
                opponent_s[opponent_idx, k] = reference_s + self._progress_delta(predicted_s, reference_s)
                opponent_e[opponent_idx, k] = predicted_e
        return opponent_s, opponent_e, active

    def _make_p_val(
        self,
        x0_mpc,
        ref_traj,
        u_prev,
        q_weight,
        opponents=None,
        current_s=None,
    ):
        opponent_s, opponent_e, opponent_active = self._prepare_opponent_predictions(
            ref_traj,
            opponents=opponents,
            current_s=current_s,
        )
        self.last_opponent_s_plan = opponent_s.copy()
        self.last_opponent_e_plan = opponent_e.copy()
        self.last_opponent_active = opponent_active.copy()
        return np.concatenate(
            [
                x0_mpc.reshape(-1, 1), # initial state
                ref_traj.T.reshape(-1, 1), # reference trajectory
                u_prev.reshape(-1, 1), # previous control actions
                np.array([[float(q_weight)]]), # Q weight
                opponent_s.T.reshape(-1, 1),
                opponent_e.T.reshape(-1, 1),
                opponent_active.reshape(-1, 1),
            ]
        )

    def _store_loss_parts(self, x_opt, p_val):
        (
            loss,
            track_loss,
            action_loss,
            velocity_loss,
            heading_loss,
            violation_loss,
            track_slack_loss,
            collision_slack_loss,
        ) = self.loss_fn(x_opt, p_val)
        self.last_loss = float(loss)
        self.last_track_loss = float(track_loss)
        self.last_action_loss = float(action_loss)
        self.last_velocity_loss = float(velocity_loss)
        self.last_heading_loss = float(heading_loss)
        self.last_violation_loss = float(violation_loss)
        self.last_track_slack_loss = float(track_slack_loss)
        self.last_collision_slack_loss = float(collision_slack_loss)

    def evaluate_fallback_loss(
        self,
        x0,
        ref_traj,
        u_prev,
        q_weight=1.0,
        opponents=None,
        current_s=None,
    ):
        if current_s is None:
            raise ValueError("current_s is required by the Frenet MPC")
        ref_traj = self._localize_reference(ref_traj, current_s)
        x0_mpc = self._make_x0_mpc(x0, current_s)
        if (
            self.last_control_plan is not None
            and np.shape(self.last_control_plan) == (self.n_controls, self.horizon)
        ):
            U_guess = np.column_stack((
                self.last_control_plan[:, 1:],
                self.last_control_plan[:, -1],
            ))
        else:
            U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        X_guess = np.zeros((self.n_states, self.horizon + 1))
        X_guess[:, 0] = x0_mpc
        for k in range(self.horizon):
            X_guess[:, k + 1] = self._vehicle_dynamics_np(X_guess[:, k], U_guess[:, k])

        track_slack_guess, collision_slack_guess = self._initial_slack_guess(
            X_guess,
            ref_traj,
            opponents=opponents,
            current_s=current_s,
        )
        self.last_track_slack_plan = track_slack_guess.copy()
        self.last_collision_slack_plan = collision_slack_guess.copy()
        self.last_state_plan = X_guess.copy()
        self.last_state_plan[0] += float(current_s) - (
            float(current_s) % self.track.total_length
        )
        self.last_control_plan = U_guess.copy()
        x_eval = np.concatenate(
            [
                X_guess.T.reshape(-1),
                U_guess.T.reshape(-1),
                track_slack_guess,
                collision_slack_guess.T.reshape(-1),
            ]
        )
        p_val = self._make_p_val(
            x0_mpc,
            ref_traj,
            u_prev,
            q_weight,
            opponents=opponents,
            current_s=current_s,
        )
        self._store_loss_parts(x_eval, p_val)
        return self.last_loss

    def _build_problem(self):
        # The cost includes a quadratic penalty on the position error to the reference trajectory,
        # and a quadratic penalty on the change in control inputs.
        track_cost = 0
        action_cost = 0
        velocity_cost = 0
        heading_cost = 0
        track_slack_cost = self.track_slack_weight * cs.sumsqr(self.track_slack)
        collision_slack_cost = (
            self.collision_slack_weight * cs.sumsqr(self.collision_slack)
        )
        violation_cost = track_slack_cost + collision_slack_cost

        # initial state constraint
        g_eq = [self.X[:, 0] - self.x0]

        # dynamics constraints： x_{k+1} = f(x_k, u_k) (1b)
        for k in range(self.horizon):
            x_next = self._vehicle_dynamics(self.X[:, k], self.U[:, k])
            g_eq.append(self.X[:, k + 1] - x_next)

        # inequality constraints for track boundaries and control rate limits
        g_ineq = []
        for k in range(self.horizon + 1):
            # X=[s,e,psi_error,v], so the paper's Frenet position error is
            # represented directly; no Cartesian projection approximation is
            # involved in either tracking or safety constraints.
            car_px = self.X[0, k]
            car_py = self.X[1, k]
            pos_err = cs.vertcat(
                car_px - self.ref[0, k],
                car_py - self.ref[1, k],
            )
            track_cost += self.q_weight * (pos_err.T @ self.Q @ pos_err)
            speed_err = self.X[3, k] - self.ref[2, k]
            velocity_cost += self.velocity_track_weight * speed_err * speed_err
            heading_err = cs.atan2(cs.sin(self.X[2, k]), cs.cos(self.X[2, k]))
            heading_cost += self.heading_track_weight * heading_err * heading_err

            # Track boundary constraints for predicted future states. The
            # measured k=0 state is fixed by equality constraints and may
            # already be slightly outside the MPC boundary; constraining it
            # would make recovery problems immediately infeasible.
            if k >= 1:
                if self.enforce_track_boundary:
                    lateral_e = self.X[1, k]

                    # Enforce the MPC track boundary constraint (1f). The
                    # supplied half-width is the geometric point-mass bound.
                    slack = self.track_slack[k - 1]
                    g_ineq.append(lateral_e - self.track_constraint_half_width - slack)
                    g_ineq.append(-lateral_e - self.track_constraint_half_width - slack)

                # Paper (1g)-(1h) describe separate minimum lateral and
                # longitudinal separations. Enforcing both inequalities with
                # logical AND would prohibit ordinary same-lane following and
                # side-by-side overtaking. Their collision-free meaning is the
                # disjunction |ds| >= px_min OR |de| >= py_min. A smooth
                # superellipse is the standard NLP representation of that
                # rectangular exclusion region and approaches the two paper
                # bounds more closely than a circular distance constraint.
                if self.enforce_collision_avoidance:
                    for opponent_idx in range(self.max_constrained_opponents):
                        delta_s = car_px - self.opponent_s[opponent_idx, k]
                        delta_e = car_py - self.opponent_e[opponent_idx, k]
                        collision_metric = (
                            (delta_s / self.min_longitudinal_separation)
                            ** self.collision_superellipse_exponent
                            + (delta_e / self.min_lateral_separation)
                            ** self.collision_superellipse_exponent
                        )
                        slack = self.collision_slack[opponent_idx, k - 1]
                        g_ineq.append(
                            self.opponent_active[opponent_idx] * (1.0 - collision_metric)
                            - slack
                        )

        for k in range(self.horizon):
            if k == 0:
                du = self.U[:, k] - self.u_prev
            else:
                du = self.U[:, k] - self.U[:, k - 1]
            action_cost += du.T @ self.R @ du

            # Control rate constraints (1d)-(1e)
            g_ineq.append(du[1] - self.max_steer_delta)
            g_ineq.append(-du[1] - self.max_steer_delta)

            if self.enforce_friction_speed_limit:
                # Match the released alpha-RACER MPC: the raw normalized
                # steering command appears inside tan(), rather than Sa*steer.
                wheelbase = self.p.LF + self.p.LR
                lateral_proxy = self.X[3, k] ** 2 * cs.tan(self.U[1, k])
                lateral_limit = self.friction_coefficient * GRAVITY * wheelbase
                g_ineq.append(lateral_proxy - lateral_limit)
                g_ineq.append(-lateral_proxy - lateral_limit)

        cost = (
            track_cost
            + action_cost
            + velocity_cost
            + heading_cost
            + violation_cost
        )
        self.g = cs.vertcat(*g_eq, *g_ineq)
        self.x_opt = cs.vertcat(
            cs.reshape(self.X, -1, 1),
            cs.reshape(self.U, -1, 1),
            self.track_slack,
            cs.reshape(self.collision_slack, -1, 1),
        )
        self.p_opt = cs.vertcat(
            cs.reshape(self.x0, -1, 1),
            cs.reshape(self.ref, -1, 1),
            cs.reshape(self.u_prev, -1, 1),
            self.q_weight,
            cs.reshape(self.opponent_s, -1, 1),
            cs.reshape(self.opponent_e, -1, 1),
            self.opponent_active,
        )

        nlp = {"x": self.x_opt, "p": self.p_opt, "f": cost, "g": self.g}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500}
        self.solver = cs.nlpsol("mpc_solver", "ipopt", nlp, opts)
        self.loss_fn = cs.Function(
            "mpc_loss_parts",
            [self.x_opt, self.p_opt],
            [
                cost,
                track_cost,
                action_cost,
                velocity_cost,
                heading_cost,
                violation_cost,
                track_slack_cost,
                collision_slack_cost,
            ],
        )
        self.n_x = self.n_states * (self.horizon + 1)
        self.n_u = self.n_controls * self.horizon
        self.n_track_slack = self.horizon
        self.n_collision_slack = self.max_constrained_opponents * self.horizon
        self.n_opt = (
            self.n_x
            + self.n_u
            + self.n_track_slack
            + self.n_collision_slack
        )
        self.n_eq = len(g_eq) * self.n_states
        self.n_total = self.g.size1()

    def _initial_slack_guess(
        self,
        X_guess,
        ref_traj,
        opponents=None,
        current_s=None,
    ):
        """Seed slacks with the violations in the initial state guess."""
        track_slack = np.zeros(self.horizon, dtype=float)
        collision_slack = np.zeros(
            (self.max_constrained_opponents, self.horizon),
            dtype=float,
        )
        opponent_s, opponent_e, active = self._prepare_opponent_predictions(
            ref_traj,
            opponents=opponents,
            current_s=current_s,
        )
        for k in range(1, self.horizon + 1):
            car_s = X_guess[0, k]
            car_e = X_guess[1, k]
            if self.enforce_track_boundary:
                track_slack[k - 1] = max(
                    abs(car_e) - self.track_constraint_half_width,
                    0.0,
                )
            if self.enforce_collision_avoidance:
                for opponent_idx in range(self.max_constrained_opponents):
                    if active[opponent_idx] <= 0.0:
                        continue
                    delta_s = car_s - opponent_s[opponent_idx, k]
                    delta_e = car_e - opponent_e[opponent_idx, k]
                    metric = (
                        (delta_s / self.min_longitudinal_separation)
                        ** self.collision_superellipse_exponent
                        + (delta_e / self.min_lateral_separation)
                        ** self.collision_superellipse_exponent
                    )
                    collision_slack[opponent_idx, k - 1] = max(1.0 - metric, 0.0)
        return track_slack, collision_slack

    def solve(
        self,
        x0,
        ref_traj,
        u_prev,
        q_weight=1.0,
        opponents=None,
        current_s=None,
    ):
        if current_s is None:
            raise ValueError("current_s is required by the Frenet MPC")
        ref_traj = self._localize_reference(ref_traj, current_s)
        x0_mpc = self._make_x0_mpc(x0, current_s)
        if (
            self.last_control_plan is not None
            and np.shape(self.last_control_plan) == (self.n_controls, self.horizon)
        ):
            U_guess = np.column_stack((
                self.last_control_plan[:, 1:],
                self.last_control_plan[:, -1],
            ))
        else:
            U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        X_guess = np.zeros((self.n_states, self.horizon + 1))
        X_guess[:, 0] = x0_mpc
        for k in range(self.horizon):
            X_guess[:, k + 1] = self._vehicle_dynamics_np(
                X_guess[:, k], U_guess[:, k]
            )
        p_val = self._make_p_val(
            x0_mpc,
            ref_traj,
            u_prev,
            q_weight,
            opponents=opponents,
            current_s=current_s,
        )
        track_slack_guess, collision_slack_guess = self._initial_slack_guess(
            X_guess,
            ref_traj,
            opponents=opponents,
            current_s=current_s,
        )
        x_init = np.concatenate(
            [
                X_guess.T.reshape(-1),
                U_guess.T.reshape(-1),
                track_slack_guess,
                collision_slack_guess.T.reshape(-1),
            ]
        )

        # constrains for states
        lbx = np.full(self.n_opt, -np.inf)
        ubx = np.full(self.n_opt, np.inf)

        # State bounds: x, y unbounded; psi unbounded; v in [0, max_speed] for predicted states.
        # Do not bound k=0 here: it is fixed by the measured initial condition, and a DBM
        # rollout can briefly exceed the MPC speed bound, which would make the problem
        # infeasible before the controller has a chance to brake.
        for k in range(1, self.horizon + 1):
            lbx[k * self.n_states + 3] = 0.0
            ubx[k * self.n_states + 3] = self.max_speed

        # Control bounds: longitudinal command in [-1, 1], steering command in [-1, 1].
        # Negative longitudinal commands act as braking, matching the alpha-RACER convention.
        u_start = self.n_x
        for k in range(self.horizon):
            lbx[u_start + k * 2] = -1.0
            ubx[u_start + k * 2] = 1.0
            lbx[u_start + k * 2 + 1] = -1.0
            ubx[u_start + k * 2 + 1] = 1.0

        # Safety violation variables are non-negative and otherwise unbounded:
        # a large violation remains expensive but cannot make the NLP
        # structurally infeasible.
        slack_start = self.n_x + self.n_u
        lbx[slack_start:] = 0.0

        # g <= 0 for inequality constraints, g = 0 for equality constraints
        lbg = np.zeros(self.n_total)
        ubg = np.zeros(self.n_total)
        lbg[self.n_eq:] = -np.inf
        try:
            sol = self.solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        except Exception as exc:
            stats = self.solver.stats()
            self.last_solver_success = False
            self.last_solver_status = stats.get("return_status", type(exc).__name__)
            raise RuntimeError(self.last_solver_status) from exc
        stats = self.solver.stats()
        self.last_solver_success = bool(stats.get("success", False))
        self.last_solver_status = stats.get("return_status", "unknown")
        if not self.last_solver_success:
            raise RuntimeError(f"{self.last_solver_status}")

        x_opt = sol["x"].full().flatten()
        self._store_loss_parts(x_opt, p_val)
        self.last_state_plan = x_opt[:self.n_x].reshape(self.horizon + 1, self.n_states).T
        # Diagnostics and Parameter-IBR use continuous race progress, whereas
        # the NLP is localized to the current lap for periodic interpolation.
        self.last_state_plan[0] += float(current_s) - (float(current_s) % self.track.total_length)
        u_end = self.n_x + self.n_u
        u_opt = x_opt[self.n_x:u_end].reshape(self.horizon, self.n_controls).T
        track_slack_end = u_end + self.n_track_slack
        self.last_track_slack_plan = x_opt[u_end:track_slack_end].copy()
        self.last_collision_slack_plan = x_opt[track_slack_end:].reshape(
            self.horizon,
            self.max_constrained_opponents,
        ).T
        self.last_control_plan = u_opt
        return u_opt, u_opt[:, 0]
