import subprocess
import sys

import numpy as np

from config import GRAVITY

try:
    import casadi as cs
except ImportError:
    print("CasADi not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "casadi"])
    import casadi as cs


class MPCController:
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
        max_state_vel_rate,
        max_state_speed,
        max_lateral_speed,
        max_yaw_rate,
    ):
        self.dt = dt
        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.max_vel_rate = max_vel_rate
        self.max_steer_rate = max_steer_rate
        self.track_half_width = track_width / 2.0
        self.max_state_vel_rate = max_state_vel_rate
        self.max_state_speed = max_state_speed
        self.max_lateral_speed = max_lateral_speed
        self.max_yaw_rate = max_yaw_rate
        self.p = db_params
        self.n_states = 6
        self.n_controls = 2

        self.X = cs.SX.sym("X", self.n_states, self.horizon + 1)
        self.U = cs.SX.sym("U", self.n_controls, self.horizon)
        self.x0 = cs.SX.sym("x0", self.n_states, 1)
        self.ref = cs.SX.sym("ref", 8, self.horizon + 1)
        self.u_prev = cs.SX.sym("u_prev", self.n_controls, 1)

        self._build_problem()

    def _dbm_dxdt_casadi(self, state, u):
        x, y, vx, vy, psi, omega = state[0], state[1], state[2], state[3], state[4], state[5]
        target_vel = u[0]
        target_steer = u[1]
        steer = target_steer * self.p.Sa + self.p.Sb
        prev_vel = cs.sqrt(vx**2 + vy**2 + 1e-6)
        throttle = target_vel * self.p.Ta - target_vel * self.p.Tb * prev_vel
        dx = vx * cs.cos(psi) - vy * cs.sin(psi)
        dy = vx * cs.sin(psi) + vy * cs.cos(psi)
        dpsi = omega
        safe_vx = cs.fmax(vx, 0.5)
        alpha_f = steer - cs.atan((self.p.LF * omega + vy) / safe_vx)
        alpha_r = cs.atan((self.p.LR * omega - vy) / safe_vx)
        sign_vx = cs.if_else(vx > 0, 1, cs.if_else(vx < 0, -1, 0))
        F_rx = throttle - self.p.fr * self.p.MASS * GRAVITY * sign_vx
        F_fz = (
            0.5 * self.p.MASS * GRAVITY * self.p.LR / (self.p.LF + self.p.LR)
            - 0.5 * self.p.hcom / (self.p.LF + self.p.LR) * F_rx
        )
        F_rz = (
            0.5 * self.p.MASS * GRAVITY * self.p.LF / (self.p.LF + self.p.LR)
            + 0.5 * self.p.hcom / (self.p.LF + self.p.LR) * F_rx
        )
        F_fy = 2 * self.p.mu * F_fz * cs.sin(self.p.Cf * cs.arctan(self.p.Bf * alpha_f))
        F_ry = 2 * self.p.mu * F_rz * cs.sin(self.p.Cr * cs.arctan(self.p.Br * alpha_r))
        dvx = (F_rx - F_fy * cs.sin(steer) + vy * omega * self.p.MASS) / self.p.MASS
        dvy = (F_ry + F_fy * cs.cos(steer) - vx * omega * self.p.MASS) / self.p.MASS
        domega = (F_fy * self.p.LF * cs.cos(steer) - F_ry * self.p.LR) / self.p.Iz
        return cs.vertcat(dx, dy, dvx, dvy, dpsi, domega)

    def _build_problem(self):
        cost = 0
        g_eq = [self.X[:, 0] - self.x0]
        for k in range(self.horizon):
            dxdt = self._dbm_dxdt_casadi(self.X[:, k], self.U[:, k])
            x_next = self.X[:, k] + self.dt * dxdt
            g_eq.append(self.X[:, k + 1] - x_next)

        for k in range(self.horizon + 1):
            psi_err = cs.atan2(cs.sin(self.X[4, k] - self.ref[3, k]), cs.cos(self.X[4, k] - self.ref[3, k]))
            err = cs.vertcat(
                self.X[0, k] - self.ref[0, k],
                self.X[1, k] - self.ref[1, k],
                self.X[2, k] - self.ref[2, k],
                psi_err,
            )
            cost += err.T @ self.Q @ err

        g_ineq = []
        for k in range(self.horizon):
            if k == 0:
                du = self.U[:, k] - self.u_prev
            else:
                du = self.U[:, k] - self.U[:, k - 1]
            cost += du.T @ self.R @ du
            vel_rate = du[0] / self.dt
            steer_rate = du[1] / self.dt
            g_ineq.append(vel_rate - self.max_vel_rate)
            g_ineq.append(-vel_rate - self.max_vel_rate)
            g_ineq.append(steer_rate - self.max_steer_rate)
            g_ineq.append(-steer_rate - self.max_steer_rate)

        for k in range(1, self.horizon + 1):
            state_vel_rate = (self.X[2, k] - self.X[2, k - 1]) / self.dt
            g_ineq.append(state_vel_rate - self.max_state_vel_rate)
            g_ineq.append(-state_vel_rate - self.max_state_vel_rate)

        for k in range(1, self.horizon + 1):
            lateral_e = (
                (self.X[0, k] - self.ref[4, k]) * self.ref[6, k]
                + (self.X[1, k] - self.ref[5, k]) * self.ref[7, k]
            )
            g_ineq.append(lateral_e - self.track_half_width)
            g_ineq.append(-lateral_e - self.track_half_width)

        self.g = cs.vertcat(*g_eq, *g_ineq)
        self.x_opt = cs.vertcat(cs.reshape(self.X, -1, 1), cs.reshape(self.U, -1, 1))
        self.p_opt = cs.vertcat(
            cs.reshape(self.x0, -1, 1),
            cs.reshape(self.ref, -1, 1),
            cs.reshape(self.u_prev, -1, 1),
        )

        nlp = {"x": self.x_opt, "p": self.p_opt, "f": cost, "g": self.g}
        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 100}
        self.solver = cs.nlpsol("mpc_solver", "ipopt", nlp, opts)

        self.n_x = self.n_states * (self.horizon + 1)
        self.n_u = self.n_controls * self.horizon
        self.n_eq = len(g_eq) * 6
        self.n_total = self.g.size1()

    def solve(self, x0, ref_traj, u_prev):
        X_guess = np.tile(x0.reshape(-1, 1), (1, self.horizon + 1))
        U_guess = np.tile(u_prev.reshape(-1, 1), (1, self.horizon))
        x_init = np.concatenate([X_guess.T.reshape(-1), U_guess.T.reshape(-1)])
        p_val = np.concatenate([x0.reshape(-1, 1), ref_traj.T.reshape(-1, 1), u_prev.reshape(-1, 1)])
        lbx = np.full(self.n_x + self.n_u, -np.inf)
        ubx = np.full(self.n_x + self.n_u, np.inf)
        vel_start = self.n_x
        for k in range(self.horizon):
            lbx[vel_start + k * 2] = 0.0
            ubx[vel_start + k * 2] = 1.0
            lbx[vel_start + k * 2 + 1] = -1.0
            ubx[vel_start + k * 2 + 1] = 1.0
        for k in range(self.horizon + 1):
            lbx[k * self.n_states + 2] = 0.0
            ubx[k * self.n_states + 2] = self.max_state_speed
            lbx[k * self.n_states + 3] = -self.max_lateral_speed
            ubx[k * self.n_states + 3] = self.max_lateral_speed
            lbx[k * self.n_states + 5] = -self.max_yaw_rate
            ubx[k * self.n_states + 5] = self.max_yaw_rate
        lbg = np.zeros(self.n_total)
        ubg = np.zeros(self.n_total)
        lbg[self.n_eq:] = -np.inf
        sol = self.solver(x0=x_init, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
        x_opt = sol["x"].full().flatten()
        u_opt = x_opt[self.n_x:].reshape(self.n_controls, self.horizon)
        return u_opt, u_opt[:, 0]
