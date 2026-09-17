"""Shared mutable vehicle state and control-command value objects."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    psi: float = 0.0
    omega: float = 0.0

    def to_array(self):
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.omega])

    @classmethod
    def from_array(cls, arr):
        return cls(x=arr[0], y=arr[1], psi=arr[2], vx=arr[3], vy=arr[4], omega=arr[5])


@dataclass
class CarAction:
    throttle: float = 0.0
    steer: float = 0.0
