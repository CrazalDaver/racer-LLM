import subprocess
import sys

from config import GRAVITY

import numpy as np

try:
    import casadi as cs
except ImportError:
    print("CasADi not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "casadi"])
    import casadi as cs


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
        max_state_speed=5.0,
    ):
        self.dt = dt
        self.horizon = horizon
        self.Q = Q[:2, :2] if Q.shape[0] > 2 else Q
        self.R = R
        self.max_vel_rate = max_vel_rate
        self.max_steer_rate = max_steer_rate
        self.p = db_params
        self.track_half_width = track_width / 2.0
        self.max_speed = max_state_speed
        self.last_loss = np.nan
        self.last_track_loss = np.nan
        self.last_action_loss = np.nan
        self.n_states = 4
        self.n_controls = 2

        self.X = cs.SX.sym("X", self.n_states, self.horizon + 1)
        self.U = cs.SX.sym("U", self.n_controls, self.horizon)
        self.x0 = cs.SX.sym("x0", self.n_states, 1)
        self.ref = cs.SX.sym("ref", 8, self.horizon + 1)
        self.u_prev = cs.SX.sym("u_prev", self.n_controls, 1)
        self.q_weight = cs.SX.sym("q_weight", 1)

        self._build_problem()

    def _vehicle_dynamics(self, xk, uk):
        x, y, psi, v = xk[0], xk[1], xk[2], xk[3]
        target_vel = uk[0]
        target_steer = uk[1]
        wheelbase = self.p.LF + self.p.LR
        actual_steer = target_steer * self.p.Sa + self.p.Sb
        throttle_force = target_vel * self.p.Ta - target_vel * self.p.Tb * v
        acceleration = throttle_force / self.p.MASS - self.p.fr * GRAVITY

        return cs.vertcat(
            x + v * cs.cos(psi) * self.dt,
            y + v * cs.sin(psi) * self.dt,
            psi + (v / wheelbase) * cs.tan(actual_steer) * self.dt,
            v + acceleration * self.dt,
        )

    def _build_problem(self):
        track_cost = 0
        action_cost = 0
        g_eq = [self.X[:, 0] - self.x0]

        for k in range(self.horizon):
            x_next = self._vehicle_dynamics(self.X[:, k], self.U[:, k])
            g_eq.append(self.X[:, k + 1] - x_next)

        g_ineq = []
        for k in range(self.horizon + 1):
            dx = self.X[0, k] - self.ref[4, k]
            dy = self.X[1, k] - self.ref[5, k]
            tangent_x = self.ref[7, k]
            tangent_y = -self.ref[6, k]
            car_px = self.ref[0, k] + dx * tangent_x + dy * tangent_y
            car_py = dx * self.ref[6, k] + dy * self.ref[7, k]
            pos_err = cs.vertcat(car_px - self.ref[0, k], car_py - self.ref[1, k])
            track_cost += self.q_weight * (pos_err.T @ self.Q @ pos_err)

            if k > 0:
                lateral_e = (
                    (self.X[0, k] - self.ref[4, k]) * self.ref[6, k]
                    + (self.X[1, k] - self.ref[5, k]) * self.ref[7, k]
                )
                g_ineq.append(lateral_e - self.track_half_width)
                g_ineq.append(-lateral_e - self.track_half_width)

        for k in range(self.horizon):
            if k == 0:
                du = self.U[:, k] - self.u_prev
            else:
                du = self.U[:, k] - self.U[:, k - 1]
            action_cost += du.T @ self.R @ du

            vel_rate = du[0] / self.dt
            steer_rate = du[1] / self.dt
            g_ineq.append(vel_rate - self.max_vel_rate)
            g_ineq.append(-vel_rate - self.max_vel_rate)
            g_ineq.append(steer_rate - self.max_steer_rate)
            g_ineq.append(-steer_rate - self.max_steer_rate)

        cost = track_cost + action_cost
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
            [cost, track_cost, action_cost],
        )

        self.n_x = self.n_states * (self.horizon + 1)
        self.n_u = self.n_controls * self.horizon
        self.n_eq = len(g_eq) * self.n_states
        self.n_total = self.g.size1()

    def solve(self, x0, ref_traj, u_prev, q_weight=1.0):
        x0_mpc = np.array([x0[0], x0[1], x0[2], max(x0[3], 0.0)])
        X_guess = np.tile(x0_mpc.reshape(-1, 1), (1, self.horizon + 1))
        U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        x_init = np.concatenate([X_guess.T.reshape(-1), U_guess.T.reshape(-1)])
        p_val = np.concatenate(
            [
                x0_mpc.reshape(-1, 1),
                ref_traj.T.reshape(-1, 1),
                u_prev.reshape(-1, 1),
                np.array([[float(q_weight)]]),
            ]
        )

        lbx = np.full(self.n_x + self.n_u, -np.inf)
        ubx = np.full(self.n_x + self.n_u, np.inf)
        for k in range(self.horizon + 1):
            lbx[k * self.n_states + 3] = 0.0
            ubx[k * self.n_states + 3] = self.max_speed

        u_start = self.n_x
        for k in range(self.horizon):
            lbx[u_start + k * 2] = 0.0
            ubx[u_start + k * 2] = 1.0
            lbx[u_start + k * 2 + 1] = -1.0
            ubx[u_start + k * 2 + 1] = 1.0

        lbg = np.zeros(self.n_total)
        ubg = np.zeros(self.n_total)
        lbg[self.n_eq:] = -np.inf
        sol = self.solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        x_opt = sol["x"].full().flatten()
        loss, track_loss, action_loss = self.loss_fn(x_opt, p_val)
        self.last_loss = float(loss)
        self.last_track_loss = float(track_loss)
        self.last_action_loss = float(action_loss)
        u_opt = x_opt[self.n_x:].reshape(self.horizon, self.n_controls).T
        return u_opt, u_opt[:, 0]
