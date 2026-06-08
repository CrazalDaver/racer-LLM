import numpy as np

from config import GRAVITY


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class DynamicBicycleModel:
    def __init__(self, dt, db_params):
        self.dt = dt
        self.p = db_params

    def _dbm_dxdt(self, state, target_vel, target_steer):
        x, y, psi, vx, vy, omega = state
        steer = target_steer * self.p.Sa + self.p.Sb
        prev_vel = max(np.sqrt(vx**2 + vy**2), 0.1)
        throttle = target_vel * self.p.Ta - target_vel * self.p.Tb * prev_vel
        dx = vx * np.cos(psi) - vy * np.sin(psi)
        dy = vx * np.sin(psi) + vy * np.cos(psi)
        dpsi = omega
        safe_vx = max(vx, 0.5)
        alpha_f = steer - np.arctan((self.p.LF * omega + vy) / safe_vx)
        alpha_r = np.arctan((self.p.LR * omega - vy) / safe_vx)
        F_rx = throttle - self.p.fr * self.p.MASS * GRAVITY * np.sign(vx)
        F_fz = (
            0.5 * self.p.MASS * GRAVITY * self.p.LR / (self.p.LF + self.p.LR)
            - 0.5 * self.p.hcom / (self.p.LF + self.p.LR) * F_rx
        )
        F_rz = (
            0.5 * self.p.MASS * GRAVITY * self.p.LF / (self.p.LF + self.p.LR)
            + 0.5 * self.p.hcom / (self.p.LF + self.p.LR) * F_rx
        )
        F_fy = 2 * self.p.mu * F_fz * np.sin(self.p.Cf * np.arctan(self.p.Bf * alpha_f))
        F_ry = 2 * self.p.mu * F_rz * np.sin(self.p.Cr * np.arctan(self.p.Br * alpha_r))
        dvx = (F_rx - F_fy * np.sin(steer) + vy * omega * self.p.MASS) / self.p.MASS
        dvy = (F_ry + F_fy * np.cos(steer) - vx * omega * self.p.MASS) / self.p.MASS
        domega = (F_fy * self.p.LF * np.cos(steer) - F_ry * self.p.LR) / self.p.Iz
        return np.array([dx, dy, dpsi, dvx, dvy, domega])

    def step(self, state, target_vel, target_steer):
        def f(s, tv, ts):
            return self._dbm_dxdt(s, tv, ts)

        k1 = f(state, target_vel, target_steer)
        k2 = f(state + 0.5 * self.dt * k1, target_vel, target_steer)
        k3 = f(state + 0.5 * self.dt * k2, target_vel, target_steer)
        k4 = f(state + self.dt * k3, target_vel, target_steer)
        next_state = state + self.dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state[2] = wrap_angle(next_state[2])
        return next_state


class KinematicBicycleModel:
    """
    Six-state wrapper around the same kinematic model used by the MPC.

    The simulation state remains [x, y, psi, vx, vy, omega], but vy is kept at
    zero and omega is derived from the kinematic yaw-rate equation.
    """

    def __init__(self, dt, db_params, max_speed=None):
        self.dt = dt
        self.p = db_params
        self.max_speed = max_speed

    def step(self, state, target_vel, target_steer):
        x, y, psi, vx, _, _ = state
        v = max(vx, 0.0)
        wheelbase = self.p.LF + self.p.LR
        actual_steer = target_steer * self.p.Sa + self.p.Sb
        if getattr(self.p, "ros_kinematic", False):
            acceleration = target_vel * self.p.Ta - self.p.fr
        else:
            throttle_force = target_vel * self.p.Ta - target_vel * self.p.Tb * v
            acceleration = throttle_force / self.p.MASS - self.p.fr * GRAVITY

        omega = (v / wheelbase) * np.tan(actual_steer)
        next_x = x + v * np.cos(psi) * self.dt
        next_y = y + v * np.sin(psi) * self.dt
        next_psi = wrap_angle(psi + omega * self.dt)
        next_vx = max(v + acceleration * self.dt, 0.0)
        if self.max_speed is not None:
            next_vx = min(next_vx, self.max_speed)
        next_omega = (next_vx / wheelbase) * np.tan(actual_steer)
        return np.array([next_x, next_y, next_psi, next_vx, 0.0, next_omega])
