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

    @classmethod
    def from_runtime(cls, policies, states, car_s, car_e, sim_time):
        """Capture state and previous policy outputs before any new action is chosen."""
        if not (len(policies) == len(states) == len(car_s) == len(car_e)):
            raise ValueError("policies, states, car_s, and car_e must have equal length")
        cars = []
        for idx, (policy, state) in enumerate(zip(policies, states)):
            params = getattr(policy, "params", None)
            if params is not None and np.asarray(params).shape != (5,):
                params = None
            plan = getattr(policy, "predicted_frenet_plan", None)
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
                    previous_action=_readonly_copy(getattr(policy, "last_u", np.zeros(2))),
                    previous_feasible_action=_readonly_copy(
                        getattr(policy, "last_feasible_u", None)
                    ),
                    policy_params=_readonly_copy(params),
                    predicted_frenet_plan=_readonly_copy(plan),
                )
            )
        return cls(sim_time=float(sim_time), cars=tuple(cars))

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

    def mpc_opponents(self, ego_idx):
        """Return the legacy opponent records consumed by shared MPC/reference code."""
        opponents = []
        for car in self.opponents(ego_idx):
            state = car.state
            opponents.append({
                "car_idx": car.car_idx,
                "px": car.s,
                "py": car.e,
                "vx": max(car.vx, 0.0),
                "vy": car.vy,
                "omega": car.omega,
                "psi": car.psi,
                "state": state,
            })
        return opponents
