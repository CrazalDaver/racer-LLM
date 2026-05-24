import abc
import math

import numpy as np

from state import CarAction, CarState
from track import RaceLine, Track


class BasePolicy(abc.ABC):
    """
    Policy base class that embeds an MPC solver.
    Subclasses must implement get_reference_trajectory.
    """

    def __init__(self, track: Track, mpc, step_distance: float = 0.1):
        self.track = track
        self.mpc = mpc
        self.step_distance = step_distance
        self.last_u = np.array([0.5, 0.0])

    @abc.abstractmethod
    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        pass

    def get_action(self, state: CarState, current_s: float) -> CarAction:
        ref_traj = self.get_reference_trajectory(state, current_s)
        state_arr = state.to_array()
        try:
            _, u_first = self.mpc.solve(state_arr, ref_traj, self.last_u)
        except Exception as e:
            print(f"MPC solve failed: {e}, using last control")
            u_first = self.last_u

        target_vel = np.clip(u_first[0], 0.0, 1.0)
        target_steer = np.clip(u_first[1], -1.0, 1.0)
        self.last_u = np.array([target_vel, target_steer])
        return CarAction(throttle=target_vel, steer=target_steer)


class CenterLinePolicy(BasePolicy):
    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        for k in range(horizon + 1):
            s = current_s + k * self.step_distance
            x, y, psi = self.track.interpolate(s)
            ref[0, k] = x
            ref[1, k] = y
            ref[2, k] = state.vx
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = -math.sin(psi)
            ref[7, k] = math.cos(psi)
        return ref


class AlphaRacerPolicy(BasePolicy):
    """
    Five-parameter policy: q, zeta, s1, s2, s3.
    The current example offsets the reference path from the centerline.
    """

    def __init__(self, track, mpc, step_distance=1.2, params=None, raceline: RaceLine = None, max_speed=5.0):
        super().__init__(track, mpc, step_distance)
        self.params = params if params is not None else np.array([1.0, 0.0, 0.5, 0.2, 0.1])
        self.raceline = raceline
        self.max_speed = max_speed
        self.last_raceline_s = None

    def get_action(self, state: CarState, current_s: float) -> CarAction:
        action = super().get_action(state, current_s)
        _, center_error = self.track.project(state.x, state.y, s_hint=current_s, search_radius=4.0)

        if abs(center_error) < 0.65:
            return action

        lookahead_s = current_s + max(0.8, 0.6 * max(state.vx, 0.5))
        target_x, target_y, _ = self.track.interpolate(lookahead_s)
        dx = target_x - state.x
        dy = target_y - state.y
        local_x = dx * math.cos(state.psi) + dy * math.sin(state.psi)
        local_y = dy * math.cos(state.psi) - dx * math.sin(state.psi)
        dist2 = max(local_x * local_x + local_y * local_y, 1e-4)

        wheelbase = self.mpc.p.LF + self.mpc.p.LR
        desired_actual_steer = np.clip(2.0 * wheelbase * local_y / dist2, -self.mpc.p.Sa, self.mpc.p.Sa)
        target_steer = (desired_actual_steer - self.mpc.p.Sb) / self.mpc.p.Sa

        action.steer = np.clip(target_steer, -1.0, 1.0)
        action.throttle = min(action.throttle, 0.35)
        self.last_u = np.array([action.throttle, action.steer])
        return action

    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        horizon = self.mpc.horizon
        q, zeta, s1, s2, s3 = self.params
        ref = np.zeros((8, horizon + 1))
        raceline_s = None
        base_point = None
        if self.raceline is not None:
            raceline_s, _ = self.raceline.project(
                state.x,
                state.y,
                s_hint=self.last_raceline_s,
                search_radius=4.0,
            )
            self.last_raceline_s = raceline_s
            base_point = self.raceline.interpolate(raceline_s)
        for k in range(horizon + 1):
            if self.raceline is not None:
                s = raceline_s + k * self.mpc.dt * max(base_point["vx_mps"], 0.5)
                point = self.raceline.interpolate(s)
                xc, yc, psi_center = self.track.interpolate(s)
                ref[0, k] = point["x_m"]
                ref[1, k] = point["y_m"]
                ref[2, k] = min(point["vx_mps"], self.max_speed)
                ref[3, k] = point["psi_rad"]
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = -math.sin(psi_center)
                ref[7, k] = math.cos(psi_center)
            else:
                s = current_s + k * self.step_distance
                xc, yc, psi = self.track.interpolate(s)
                nx = -math.sin(psi)
                ny = math.cos(psi)
                offset = s1 * math.sin(s3 * s) * math.exp(-s2 * s)
                offset *= q
                ref[0, k] = xc + offset * nx
                ref[1, k] = yc + offset * ny
                ref[2, k] = state.vx
                ref[3, k] = psi
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = nx
                ref[7, k] = ny
        return ref
