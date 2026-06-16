from config import GRAVITY

import numpy as np
import casadi as cs


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class MPCController:
    """
    Minimal alpha-RACER style MPC tracker.

    The MPC prediction model is kinematic [x, y, psi, v], matching the
    controller in car_ros2/mpc_controller.py. The simulation model remains DBM.
    """

    def __init__(
        self,
        dt,
        horizon,
        Q,
        R,
        max_vel_rate,
        max_steer_rate,
        db_params,
        track_width,
        track_boundary_slack=0.0,
        max_state_speed=5.0,
        velocity_track_weight=0.0,
        heading_track_weight=0.0,
    ):
        self.dt = dt
        self.horizon = horizon
        self.Q = Q[:2, :2] if Q.shape[0] > 2 else Q
        self.R = R
        self.max_vel_rate = max_vel_rate
        self.max_steer_rate = max_steer_rate
        self.p = db_params
        self.track_half_width = track_width / 2.0
        self.track_boundary_slack = float(track_boundary_slack)
        self.track_constraint_half_width = self.track_half_width + self.track_boundary_slack
        self.max_speed = max_state_speed
        self.velocity_track_weight = velocity_track_weight
        self.heading_track_weight = heading_track_weight
        self.last_loss = np.nan
        self.last_track_loss = np.nan
        self.last_action_loss = np.nan
        self.last_velocity_loss = np.nan
        self.last_heading_loss = np.nan
        self.last_solver_success = True
        self.last_solver_status = "not_run"
        self.last_state_plan = None
        self.last_control_plan = None
        self.n_states = 4
        self.n_controls = 2

        # Define optimization variables and parameters
        self.X = cs.SX.sym("X", self.n_states, self.horizon + 1) # state trajectory: (x, y, psi, v) for each timestep
        self.U = cs.SX.sym("U", self.n_controls, self.horizon) # control trajectory
        self.x0 = cs.SX.sym("x0", self.n_states, 1) # initial state
        # reference trajectory 
        # (x_ref, y_ref, cos(psi_ref), sin(psi_ref), x_center, y_center, tangent_x, tangent_y)
        self.ref = cs.SX.sym("ref", 8, self.horizon + 1)    
        self.u_prev = cs.SX.sym("u_prev", self.n_controls, 1) # previous control input for rate penalty
        self.q_weight = cs.SX.sym("q_weight", 1) 

        self._build_problem()

    def _vehicle_dynamics(self, xk, uk):
        x, y, psi, v = xk[0], xk[1], xk[2], xk[3]
        target_vel = uk[0]
        target_steer = uk[1]
        wheelbase = self.p.LF + self.p.LR
        actual_steer = target_steer * self.p.Sa + self.p.Sb
        if getattr(self.p, "ros_kinematic", False):
            acceleration = target_vel * self.p.Ta - self.p.fr
        else:
            throttle_force = target_vel * self.p.Ta - target_vel * self.p.Tb * v
            acceleration = throttle_force / self.p.MASS - self.p.fr * GRAVITY

        return cs.vertcat(
            x + v * cs.cos(psi) * self.dt,
            y + v * cs.sin(psi) * self.dt,
            psi + (v / wheelbase) * cs.tan(actual_steer) * self.dt,
            v + acceleration * self.dt,
        )

    def _vehicle_dynamics_np(self, xk, uk):
        x, y, psi, v = xk[0], xk[1], xk[2], xk[3]
        target_vel = uk[0]
        target_steer = uk[1]
        wheelbase = self.p.LF + self.p.LR
        actual_steer = target_steer * self.p.Sa + self.p.Sb
        if getattr(self.p, "ros_kinematic", False):
            acceleration = target_vel * self.p.Ta - self.p.fr
        else:
            throttle_force = target_vel * self.p.Ta - target_vel * self.p.Tb * v
            acceleration = throttle_force / self.p.MASS - self.p.fr * GRAVITY

        return np.array(
            [
                x + v * np.cos(psi) * self.dt,
                y + v * np.sin(psi) * self.dt,
                psi + (v / wheelbase) * np.tan(actual_steer) * self.dt,
                v + acceleration * self.dt,
            ],
            dtype=float,
        )

    def _make_x0_mpc(self, x0):
        return np.array([x0[0], x0[1], wrap_angle(x0[2]), max(x0[3], 0.0)], dtype=float)

    def _make_p_val(self, x0_mpc, ref_traj, u_prev, q_weight):
        return np.concatenate(
            [
                x0_mpc.reshape(-1, 1), # initial state
                ref_traj.T.reshape(-1, 1), # reference trajectory
                u_prev.reshape(-1, 1), # previous control actions
                np.array([[float(q_weight)]]), # Q weight
            ]
        )

    def _store_loss_parts(self, x_opt, p_val):
        loss, track_loss, action_loss, velocity_loss, heading_loss = self.loss_fn(x_opt, p_val)
        self.last_loss = float(loss)
        self.last_track_loss = float(track_loss)
        self.last_action_loss = float(action_loss)
        self.last_velocity_loss = float(velocity_loss)
        self.last_heading_loss = float(heading_loss)

    def evaluate_fallback_loss(self, x0, ref_traj, u_prev, q_weight=1.0):
        x0_mpc = self._make_x0_mpc(x0)
        U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        X_guess = np.zeros((self.n_states, self.horizon + 1))
        X_guess[:, 0] = x0_mpc
        for k in range(self.horizon):
            X_guess[:, k + 1] = self._vehicle_dynamics_np(X_guess[:, k], U_guess[:, k])

        x_eval = np.concatenate([X_guess.T.reshape(-1), U_guess.T.reshape(-1)])
        p_val = self._make_p_val(x0_mpc, ref_traj, u_prev, q_weight)
        self._store_loss_parts(x_eval, p_val)
        return self.last_loss

    def _build_problem(self):
        # The cost includes a quadratic penalty on the position error to the reference trajectory,
        # and a quadratic penalty on the change in control inputs.
        track_cost = 0
        action_cost = 0
        velocity_cost = 0
        heading_cost = 0
        # initial state constraint
        g_eq = [self.X[:, 0] - self.x0]

        # dynamics constraints： x_{k+1} = f(x_k, u_k)
        for k in range(self.horizon):
            x_next = self._vehicle_dynamics(self.X[:, k], self.U[:, k])
            g_eq.append(self.X[:, k + 1] - x_next)

        # inequality constraints for track boundaries and control rate limits
        g_ineq = []
        for k in range(self.horizon + 1):
            # ref: 0:p_ref_x, 1:p_ref_y, 2:ref_speed, 3:psi_ref, 4:center_x, 5:center_y, 6:-sin psi(norm_x), 7:cos psi (norm_y)
            dx = self.X[0, k] - self.ref[4, k]
            dy = self.X[1, k] - self.ref[5, k]
            tangent_x = self.ref[7, k]
            tangent_y = -self.ref[6, k]
            car_px = self.ref[0, k] + dx * tangent_x + dy * tangent_y
            car_py = dx * self.ref[6, k] + dy * self.ref[7, k]
            pos_err = cs.vertcat(car_px - self.ref[0, k], car_py - self.ref[1, k])
            track_cost += self.q_weight * (pos_err.T @ self.Q @ pos_err)
            speed_err = self.X[3, k] - self.ref[2, k]
            velocity_cost += self.velocity_track_weight * speed_err * speed_err
            heading_err = cs.atan2(cs.sin(self.X[2, k] - self.ref[3, k]), cs.cos(self.X[2, k] - self.ref[3, k]))
            heading_cost += self.heading_track_weight * heading_err * heading_err

            # track boundary constraints
            if k >= 0:
                lateral_e = (
                    (self.X[0, k] - self.ref[4, k]) * self.ref[6, k]
                    + (self.X[1, k] - self.ref[5, k]) * self.ref[7, k]
                )
                # References are still clipped to the true track half-width. The
                # MPC boundary gets a small slack so projection/model mismatch
                # does not immediately make the optimization infeasible.
                g_ineq.append(lateral_e - self.track_constraint_half_width)
                g_ineq.append(-lateral_e - self.track_constraint_half_width)

        for k in range(self.horizon):
            if k == 0:
                du = self.U[:, k] - self.u_prev
            else:
                du = self.U[:, k] - self.U[:, k - 1]
            action_cost += du.T @ self.R @ du
            g_ineq.append(du[1] - self.max_steer_rate)
            g_ineq.append(-du[1] - self.max_steer_rate)

        cost = track_cost + action_cost + velocity_cost + heading_cost
        self.g = cs.vertcat(*g_eq, *g_ineq)
        self.x_opt = cs.vertcat(cs.reshape(self.X, -1, 1), cs.reshape(self.U, -1, 1))
        self.p_opt = cs.vertcat(
            cs.reshape(self.x0, -1, 1),
            cs.reshape(self.ref, -1, 1),
            cs.reshape(self.u_prev, -1, 1),
            self.q_weight,
        )

        nlp = {"x": self.x_opt, "p": self.p_opt, "f": cost, "g": self.g}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500}
        self.solver = cs.nlpsol("mpc_solver", "ipopt", nlp, opts)
        self.loss_fn = cs.Function(
            "mpc_loss_parts",
            [self.x_opt, self.p_opt],
            [cost, track_cost, action_cost, velocity_cost, heading_cost],
        )
        
        self.n_x = self.n_states * (self.horizon + 1)
        self.n_u = self.n_controls * self.horizon
        self.n_eq = len(g_eq) * self.n_states
        self.n_total = self.g.size1()

    def solve(self, x0, ref_traj, u_prev, q_weight=1.0):
        x0_mpc = self._make_x0_mpc(x0)
        X_guess = np.tile(x0_mpc.reshape(-1, 1), (1, self.horizon + 1))
        U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        x_init = np.concatenate([X_guess.T.reshape(-1), U_guess.T.reshape(-1)])
        p_val = self._make_p_val(x0_mpc, ref_traj, u_prev, q_weight)

        # constrains for states
        lbx = np.full(self.n_x + self.n_u, -np.inf)
        ubx = np.full(self.n_x + self.n_u, np.inf)

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

        # g <= 0 for inequality constraints, g = 0 for equality constraints
        lbg = np.zeros(self.n_total)
        ubg = np.zeros(self.n_total)
        lbg[self.n_eq:] = -np.inf
        sol = self.solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        stats = self.solver.stats()
        self.last_solver_success = bool(stats.get("success", False))
        self.last_solver_status = stats.get("return_status", "unknown")
        if not self.last_solver_success:
            raise RuntimeError(f"{self.last_solver_status}")

        x_opt = sol["x"].full().flatten()
        self._store_loss_parts(x_opt, p_val)
        self.last_state_plan = x_opt[:self.n_x].reshape(self.horizon + 1, self.n_states).T
        u_opt = x_opt[self.n_x:].reshape(self.horizon, self.n_controls).T
        self.last_control_plan = u_opt
        return u_opt, u_opt[:, 0]
