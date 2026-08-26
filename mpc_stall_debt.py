"""Detachable history-dependent anti-stagnation state for the shared MPC."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StallDebtSettings:
    max_weight: float = 0.0
    base_weight: float = 10.0
    activation_seconds: float = 0.5
    exponential_time_scale: float = 0.5
    speed_threshold: float = 0.10
    progress_rate_threshold: float = 0.05
    recovery_decay_rate: float = 2.0
    minimum_horizon_progress: float = 0.50


class StallDebtTracker:
    """Accumulate debt only during sustained low-speed, low-progress motion."""

    def __init__(self, settings: StallDebtSettings):
        self.settings = settings
        self.seconds = 0.0
        self.weight = 0.0
        self.active = False
        self._last_time = None
        self._last_progress = None

    @property
    def enabled(self):
        return self.settings.max_weight > 0.0

    def update(self, progress, speed, sim_time):
        progress, speed = float(progress), abs(float(speed))
        if not self.enabled or sim_time is None:
            self.weight, self.active = 0.0, False
            return self.weight
        sim_time = float(sim_time)
        if self._last_time is None:
            self._last_time, self._last_progress = sim_time, progress
            return self.weight
        if sim_time <= self._last_time + 1.0e-9:
            return self.weight

        dt = sim_time - self._last_time
        progress_rate = abs(progress - self._last_progress) / max(dt, 1.0e-9)
        stalled = (
            speed < self.settings.speed_threshold
            and progress_rate < self.settings.progress_rate_threshold
        )
        if stalled:
            self.seconds += dt
        else:
            self.seconds = max(
                0.0, self.seconds - self.settings.recovery_decay_rate * dt
            )
        excess = max(0.0, self.seconds - self.settings.activation_seconds)
        raw = (
            self.settings.base_weight
            * (math.exp(excess / self.settings.exponential_time_scale) - 1.0)
            if excess > 0.0
            else 0.0
        )
        self.weight = min(self.settings.max_weight, raw)
        self.active = self.weight > 0.0
        self._last_time, self._last_progress = sim_time, progress
        return self.weight
