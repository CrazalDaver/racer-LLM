import numpy as np


class CarState:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, psi=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.psi = psi
        self.omega = omega

    def to_array(self):
        return np.array([self.x, self.y, self.vx, self.vy, self.psi, self.omega])

    @classmethod
    def from_array(cls, arr):
        return cls(x=arr[0], y=arr[1], vx=arr[2], vy=arr[3], psi=arr[4], omega=arr[5])


class CarAction:
    def __init__(self, throttle=0.0, steer=0.0):
        self.throttle = throttle
        self.steer = steer
