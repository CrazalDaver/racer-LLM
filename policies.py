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
        self.last_u = np.array([0.5, self._neutral_steer_command()])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None

    def _neutral_steer_command(self) -> float:
        return np.clip(-self.mpc.p.Sb / self.mpc.p.Sa, -1.0, 1.0)

    @abc.abstractmethod
    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        pass

    def get_q_weight(self) -> float:
        return 1.0

    def get_action(self, state: CarState, current_s: float) -> CarAction:
        ref_traj = self.get_reference_trajectory(state, current_s)
        self.last_ref_traj = ref_traj
        state_arr = state.to_array()
        try:
            _, u_first = self.mpc.solve(state_arr, ref_traj, self.last_u, q_weight=self.get_q_weight())
        except Exception as e:
            print(f"MPC solve failed: {e}, using last control")
            self.mpc.last_loss = np.nan
            self.mpc.last_track_loss = np.nan
            self.mpc.last_action_loss = np.nan
            u_first = self.last_u

        target_vel = np.clip(u_first[0], 0.0, 1.0)
        target_steer = np.clip(u_first[1], -1.0, 1.0)
        self.last_u = np.array([target_vel, target_steer])
        return CarAction(throttle=target_vel, steer=target_steer)


class CenterLinePolicy(BasePolicy):
    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        horizon = self.mpc.horizon
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        for k in range(horizon + 1):
            s = current_s + k * self.step_distance
            x, y, psi = self.track.interpolate(s)
            ref[0, k] = s % self.track.total_length
            ref[1, k] = 0.0
            ref[2, k] = state.vx
            ref[3, k] = psi
            ref[4, k] = x
            ref[5, k] = y
            ref[6, k] = -math.sin(psi)
            ref[7, k] = math.cos(psi)
            ref_world[:, k] = [x, y]
        self.last_ref_world_horizon = ref_world
        return ref


class AlphaRacerPolicy(BasePolicy):
    """
    Five-parameter alpha-RACER policy: q, zeta, s1, s2, s3.
    q scales the MPC track loss as Q_eff = q * I when config.Q is identity.
    """

    def __init__(
        self,
        track,
        mpc,
        step_distance=1.2,
        params=None,
        raceline: RaceLine = None,
        max_speed=5.0,
        search_back_steps=20,
        search_forward_steps=200,
    ):
        super().__init__(track, mpc, step_distance)
        self.params = params if params is not None else np.array([1.0, 1.0, 0.5, 0.2, 0.1])
        self.raceline = raceline
        self.max_speed = max_speed
        self.last_raceline_s = None
        self.last_raceline_idx = None
        self.search_back_steps = search_back_steps
        self.search_forward_steps = search_forward_steps

    def get_q_weight(self) -> float:
        return float(self.params[0])

    def _nearest_raceline_s(self, state: CarState) -> float:
        self.last_raceline_idx = self.raceline.nearest_index(
            state.x,
            state.y,
            last_idx=self.last_raceline_idx,
            back_steps=self.search_back_steps,
            forward_steps=self.search_forward_steps,
        )
        return float(self.raceline.s[self.last_raceline_idx])

    def _raceline_frenet_state(self, raceline_s: float, center_hint: float = None) -> dict:
        point = self.raceline.interpolate(raceline_s)
        center_s, lateral_e = self.track.project(
            point["x_m"],
            point["y_m"],
            s_hint=center_hint,
            search_radius=8.0,
            psi_hint=point["psi_rad"],
            max_heading_error=math.pi / 2.0,
        )
        xc, yc, psi_center = self.track.interpolate(center_s)
        return {
            "track_s": center_s,
            "lateral_e": lateral_e,
            "speed": min(point["vx_mps"], self.max_speed),
            "center_x": xc,
            "center_y": yc,
            "psi_center": psi_center,
            "normal_x": -math.sin(psi_center),
            "normal_y": math.cos(psi_center),
        }

    def get_reference_trajectory(self, state: CarState, current_s: float) -> np.ndarray:
        horizon = self.mpc.horizon
        q, zeta, s1, s2, s3 = self.params
        ref = np.zeros((8, horizon + 1))
        ref_world = np.zeros((2, horizon + 1))
        raceline_world = np.full((2, horizon + 1), np.nan)
        if self.raceline is not None:
            raceline_s = self._nearest_raceline_s(state)
            self.last_raceline_s = raceline_s
            frenet = self._raceline_frenet_state(raceline_s, center_hint=current_s)
            source_speed = max(frenet["speed"], 0.0)
            p_pert_x = frenet["track_s"]
            p_pert_y = frenet["lateral_e"]
        for k in range(horizon + 1):
            if self.raceline is not None:
                source_s = raceline_s + k * source_speed * self.mpc.dt
                point = self.raceline.interpolate(source_s)
                p_ref_x = p_pert_x % self.track.total_length
                p_ref_y = float(np.clip(p_pert_y, -self.mpc.track_half_width, self.mpc.track_half_width))
                xc, yc, psi_center = self.track.interpolate(p_ref_x)
                ref_world[:, k] = self.track.frenet_to_world(p_ref_x, p_ref_y)
                raceline_world[:, k] = [point["x_m"], point["y_m"]]
                ref[0, k] = p_ref_x
                ref[1, k] = p_ref_y
                ref[2, k] = source_speed
                ref[3, k] = psi_center
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = -math.sin(psi_center)
                ref[7, k] = math.cos(psi_center)

                p_pert_x += zeta * source_speed * self.mpc.dt
            else:
                s = current_s + k * self.step_distance
                xc, yc, psi = self.track.interpolate(s)
                offset = np.clip(
                    s1 * math.sin(s3 * s) * math.exp(-s2 * s),
                    -self.mpc.track_half_width,
                    self.mpc.track_half_width,
                )
                ref[0, k] = s % self.track.total_length
                ref[1, k] = offset
                ref[2, k] = state.vx
                ref[3, k] = psi
                ref[4, k] = xc
                ref[5, k] = yc
                ref[6, k] = -math.sin(psi)
                ref[7, k] = math.cos(psi)
                ref_world[:, k] = self.track.frenet_to_world(ref[0, k], ref[1, k])
        self.last_ref_world_horizon = ref_world
        self.last_raceline_world_horizon = raceline_world
        return ref
