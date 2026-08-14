import casadi as cs
import numpy as np

from config import GRAVITY


def steering_angle(steering_command, vehicle_params):
    """Map a normalized steering command to the physical front-wheel angle."""
    return steering_command * vehicle_params.Sa + vehicle_params.Sb


def longitudinal_drive_force(throttle_command, speed, vehicle_params):
    """Map a normalized throttle command to drive force."""
    return throttle_command * vehicle_params.Ta - throttle_command * vehicle_params.Tb * speed


def longitudinal_acceleration(throttle_command, speed, vehicle_params):
    """Longitudinal acceleration shared by simulation, MPC, and IBR."""
    if getattr(vehicle_params, "ros_kinematic", False):
        return throttle_command * vehicle_params.Ta - vehicle_params.fr
    drive_force = longitudinal_drive_force(throttle_command, speed, vehicle_params)
    return drive_force / vehicle_params.MASS - vehicle_params.fr * GRAVITY


def neutral_throttle(speed, vehicle_params):
    """Command that approximately produces zero longitudinal acceleration."""
    if getattr(vehicle_params, "ros_kinematic", False):
        return vehicle_params.fr / max(float(vehicle_params.Ta), 1.0e-6)
    gain = vehicle_params.Ta - vehicle_params.Tb * max(float(speed), 0.0)
    return vehicle_params.fr * GRAVITY * vehicle_params.MASS / max(float(gain), 1.0e-6)


def minimum_reachable_speed(speed, dt, vehicle_params):
    """Lowest one-step speed allowed by the shared normalized throttle model."""
    acceleration = longitudinal_acceleration(-1.0, float(speed), vehicle_params)
    return max(float(speed) + float(acceleration) * float(dt), 0.0)


def _vector(values, symbolic):
    return cs.vertcat(*values) if symbolic else np.asarray(values, dtype=float)


def kinematic_global_step(state, control, dt, vehicle_params, *, symbolic=False):
    """Euler step for global state ``[x, y, psi, v]`` and normalized control."""
    trig = cs if symbolic else np
    x, y, psi, speed = state[0], state[1], state[2], state[3]
    throttle, steering_command = control[0], control[1]
    wheelbase = vehicle_params.LF + vehicle_params.LR
    steer = steering_angle(steering_command, vehicle_params)
    acceleration = longitudinal_acceleration(throttle, speed, vehicle_params)
    return _vector(
        [
            x + speed * trig.cos(psi) * dt,
            y + speed * trig.sin(psi) * dt,
            psi + speed / wheelbase * trig.tan(steer) * dt,
            speed + acceleration * dt,
        ],
        symbolic,
    )


def kinematic_frenet_step(
    state,
    control,
    curvature,
    dt,
    vehicle_params,
    *,
    curvature_mode="geometric",
    symbolic=False,
):
    """Euler step for Frenet state ``[s, e, psi_error, v]``.

    ``upstream_approx`` retains alpha-RACER's simplified curvature term;
    ``geometric`` uses the standard Frenet-coordinate correction.
    """
    if curvature_mode not in {"geometric", "upstream_approx"}:
        raise ValueError(f"Unknown Frenet curvature mode: {curvature_mode}")
    trig = cs if symbolic else np
    s, e, psi_error, speed = state[0], state[1], state[2], state[3]
    throttle, steering_command = control[0], control[1]
    wheelbase = vehicle_params.LF + vehicle_params.LR
    steer = steering_angle(steering_command, vehicle_params)
    acceleration = longitudinal_acceleration(throttle, speed, vehicle_params)
    denominator = 1.0 - e * curvature
    path_heading_rate = (
        speed * curvature
        if curvature_mode == "upstream_approx"
        else speed * trig.cos(psi_error) * curvature / denominator
    )
    return _vector(
        [
            s + speed * trig.cos(psi_error) / denominator * dt,
            e + speed * trig.sin(psi_error) * dt,
            psi_error
            + (speed / wheelbase * trig.tan(steer) - path_heading_rate) * dt,
            speed + acceleration * dt,
        ],
        symbolic,
    )


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class DynamicBicycleModel:
    """
    Dynamic Bicycle Model(DBM) with 6 states: [x, y, psi, vx, vy, omega]
    where x, y are the global position, psi is the heading angle, vx and vy are
    the longitudinal and lateral velocities, and omega is the yaw rate.
    """

    def __init__(self, dt, db_params):
        self.dt = dt
        self.p = db_params

    def _dbm_dxdt(self, state, target_vel, target_steer):
        x, y, psi, vx, vy, omega = state
        steer = steering_angle(target_steer, self.p)
        prev_vel = max(np.sqrt(vx**2 + vy**2), 0.1)
        throttle = longitudinal_drive_force(target_vel, prev_vel, self.p)
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
        prediction = kinematic_global_step(
            np.array([x, y, psi, v], dtype=float),
            np.array([target_vel, target_steer], dtype=float),
            self.dt,
            self.p,
        )
        next_x, next_y = prediction[0], prediction[1]
        next_psi = wrap_angle(prediction[2])
        next_vx = max(prediction[3], 0.0)
        if self.max_speed is not None:
            next_vx = min(next_vx, self.max_speed)
        actual_steer = steering_angle(target_steer, self.p)
        next_omega = (next_vx / (self.p.LF + self.p.LR)) * np.tan(actual_steer)
        return np.array([next_x, next_y, next_psi, next_vx, 0.0, next_omega])
