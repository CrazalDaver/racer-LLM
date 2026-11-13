"""Immutable per-control-tick race observations shared by all policies.

The simulator owns physical state integration.  Before policies choose an
action, it captures the complete current race scene here.  Policies may update
their own private cache while deciding, but they only observe these copied
previous-tick fields for every other car; action enumeration order therefore
cannot leak newly written parameters or controls into another policy's input.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from state import CarState
from track import Track

def _progress_delta(s_a, s_b, total_length):
    return math.atan2(
        math.sin((s_a - s_b) * 2.0 * math.pi / total_length),
        math.cos((s_a - s_b) * 2.0 * math.pi / total_length),
    ) * total_length / (2.0 * math.pi)


def _readonly_copy(value):
    if value is None:
        return None
    copied = np.asarray(value, dtype=float).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class CarObservation:
    """One car's complete, read-only snapshot at a control tick."""

    car_idx: int
    x: float
    y: float
    vx: float
    vy: float
    psi: float
    omega: float
    s: float
    e: float
    previous_action: np.ndarray
    previous_feasible_action: Optional[np.ndarray]
    policy_params: Optional[np.ndarray]
    predicted_frenet_plan: Optional[np.ndarray]

    # Cached derived values for this tick; vx/vy above remain body components.
    track_heading: Optional[float] = None
    heading_error: Optional[float] = None
    curvature: Optional[float] = None
    vx_world: Optional[float] = None
    vy_world: Optional[float] = None
    speed_mps: Optional[float] = None
    s_dot: Optional[float] = None
    e_dot: Optional[float] = None

    @property
    def vx_body(self):
        return self.vx

    @property
    def vy_body(self):
        return self.vy

    @property
    def kinematic_state(self):
        """Forward-only MPC initial state [unwrapped s,e,chi,body vx]."""
        if self.heading_error is None:
            raise ValueError("A projected observation is required for MPC initialization")
        return np.array([self.s,self.e,self.heading_error,max(self.vx_body,0.)])

    @property
    def state(self):
        """Return an independent legacy ``CarState`` for MPC/model code."""
        return CarState(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            psi=self.psi,
            omega=self.omega,
        )


@dataclass(frozen=True)
class RaceObservation:
    """Complete, immutable race scene supplied to one policy decision."""

    sim_time: float
    cars: tuple[CarObservation, ...]
    _track: object = None
    _curvature: object = None

    @classmethod
    def from_states(
        cls, states, car_s, car_e, sim_time, *, track, curvature_fn, policies=None
    ):
        """Capture measured states, optionally including policy-owned metadata."""
        if not (len(states) == len(car_s) == len(car_e)):
            raise ValueError("states, car_s, and car_e must have equal length")
        if policies is not None and len(policies) != len(states):
            raise ValueError("policies and states must have equal length")
        cars = []
        for idx, state in enumerate(states):
            policy = None if policies is None else policies[idx]
            params = getattr(policy, "params", None) if policy is not None else None
            if params is not None and np.asarray(params).shape != (5,):
                params = None
            plan = (
                getattr(policy, "predicted_frenet_plan", None)
                if policy is not None else None
            )
            frame = track.state_kinematics(state, car_s[idx], car_e[idx],
                float(curvature_fn(car_s[idx] % track.total_length)))
            cars.append(
                CarObservation(
                    car_idx=idx,
                    x=float(state.x),
                    y=float(state.y),
                    vx=float(state.vx),
                    vy=float(state.vy),
                    psi=float(state.psi),
                    omega=float(state.omega),
                    s=float(car_s[idx]),
                    e=float(car_e[idx]),
                    previous_action=_readonly_copy(
                        getattr(policy, "last_u", np.zeros(2))
                        if policy is not None else np.zeros(2)
                    ),
                    previous_feasible_action=_readonly_copy(
                        getattr(policy, "last_feasible_u", None)
                        if policy is not None else None
                    ),
                    policy_params=_readonly_copy(params),
                    predicted_frenet_plan=_readonly_copy(plan),
                    **frame,
                )
            )
        return cls(sim_time=float(sim_time), cars=tuple(cars),
                   _track=track, _curvature=curvature_fn)

    @classmethod
    def from_runtime(cls, policies, states, car_s, car_e, sim_time, *, track=None, curvature_fn=None):
        """Capture state and previous policy outputs before any new action is chosen."""
        if not policies:
            raise ValueError("from_runtime requires at least one policy")
        track = policies[0].track if track is None else track
        curvature_fn = policies[0].mpc.curvature_fn if curvature_fn is None else curvature_fn
        return cls.from_states(
            states, car_s, car_e, sim_time, track=track,
            curvature_fn=curvature_fn, policies=policies,
        )

    def car(self, car_idx):
        return self.cars[int(car_idx)]

    def opponents(self, ego_idx):
        return tuple(car for car in self.cars if car.car_idx != int(ego_idx))

    def nearest_opponent(self, ego_idx, track_length):
        ego = self.car(ego_idx)
        opponents = self.opponents(ego_idx)
        if not opponents:
            return None
        return min(
            opponents,
            key=lambda other: abs(_progress_delta(other.s, ego.s, track_length)),
        )

    def self_play_rl_input(self, ego_idx, lookahead_seconds=0.9):
        """Return the shared 15-value input for the direct-action PPO policy.

        The ordering and relative-position/heading signs follow the released
        evaluation code.  Velocity components use the paper's Frenet-frame
        state definition for both cars.
        """
        if self._track is None or self._curvature is None:
            raise ValueError("Self-play RL input requires track and curvature data")
        if lookahead_seconds < 0.0:
            raise ValueError("lookahead_seconds must be non-negative")
        ego = self.car(ego_idx)
        opponent = self.nearest_opponent(ego_idx, self._track.total_length)
        if opponent is None:
            raise ValueError("Self-play RL input requires at least two cars")
        required = (
            ego.heading_error, ego.s_dot, ego.e_dot, ego.curvature,
            opponent.heading_error, opponent.s_dot, opponent.e_dot,
            opponent.curvature,
        )
        if any(value is None for value in required):
            raise ValueError("Self-play RL input requires projected car kinematics")

        ego_lookahead_s = ego.s + max(float(ego.s_dot), 0.0) * lookahead_seconds
        opponent_lookahead_s = (
            opponent.s + max(float(opponent.s_dot), 0.0) * lookahead_seconds
        )
        values = np.asarray(
            [
                _progress_delta(ego.s, opponent.s, self._track.total_length),
                ego.e,
                opponent.e,
                -ego.heading_error,
                ego.s_dot,  # vx in Frenet
                ego.e_dot,  # vy in Frenet
                ego.omega,
                -opponent.heading_error,
                opponent.s_dot,  # vx in Frenet
                opponent.e_dot,  # vy in Frenet
                opponent.omega,
                ego.curvature,
                opponent.curvature,
                float(self._curvature(ego_lookahead_s % self._track.total_length)),
                float(self._curvature(
                    opponent_lookahead_s % self._track.total_length
                )),
            ],
            dtype=np.float32,
        )
        if values.shape != (15,) or not np.all(np.isfinite(values)):
            raise ValueError("Self-play RL input must contain 15 finite values")
        return values

    def _progress_rate(self, car):
        if car.s_dot is not None:
            return car.s_dot
        # Compatibility for manually constructed observations in offline callers.
        return Track.state_kinematics(self._track, car.state, car.s, car.e,
            float(self._curvature(car.s % self._track.total_length)))["s_dot"]

    def mpc_opponents(self, ego_idx):
        """Return the legacy opponent records consumed by shared MPC/reference code."""
        opponents = []
        for car in self.opponents(ego_idx):
            state = car.state
            opponents.append({
                "car_idx": car.car_idx,
                "px": car.s,
                "py": car.e,
                "s_dot": self._progress_rate(car),
                "e_dot": car.e_dot,
                "vx_body": car.vx_body,
                "vy_body": car.vy_body,
                "vx": self._progress_rate(car),  # legacy alias for forecast ds/dt
                "vy": car.vy,
                "omega": car.omega,
                "psi": car.psi,
                "state": state,
            })
        return opponents
