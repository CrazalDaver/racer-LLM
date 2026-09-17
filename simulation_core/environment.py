"""Physical race environment advanced once per simulator tick.

Policies never integrate vehicle state directly.  They observe a frozen scene,
choose commands, and hand those commands to :class:`RaceEnvironment`, which
owns recovery rules, vehicle dynamics, Frenet projection, and contact response.
"""

from dataclasses import dataclass

import numpy as np

from collision import oriented_rectangle_overlap_cost
from config import params
from policy import progress_delta, track_heading_error
from .observation import RaceObservation
from state import CarAction, CarState
from vehicle_model import neutral_throttle


@dataclass(frozen=True)
class StepResult:
    """Flat snapshot produced by one call to :meth:`RaceEnvironment.step`."""

    actions: tuple[CarAction, ...]
    trace_states: tuple[CarState, ...]
    previous_s: tuple[float, ...]
    previous_e: tuple[float, ...]
    next_s: tuple[float, ...]
    next_e: tuple[float, ...]
    offroad_active: tuple[bool, ...]
    offroad_heading_errors: tuple[float, ...]
    contact_happened: bool
    contact_max_overlap: float


class RaceEnvironment:
    """Own mutable physical state and advance it through a small step API."""

    def __init__(self, track, states, vehicle_models, initial_progress):
        self.track = track
        self.states = states
        self.vehicle_models = vehicle_models
        self.progress_s = np.asarray(initial_progress, dtype=float)
        self.offroad_recovery_threshold = params.OFFROAD_RECOVERY_THRESHOLD
        self.step_index = None
        self.car_s = None
        self.car_e = None

    def project_current(self):
        """Return current unwrapped progress and lateral error for every car."""
        car_s, car_e = [], []
        for idx, state in enumerate(self.states):
            s, e = self.track.project(
                state.x,
                state.y,
                s_hint=self.progress_s[idx],
                search_radius=params.PROJECTION_SEARCH_RADIUS,
                psi_hint=state.psi,
                max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
            )
            self.progress_s[idx] = s
            car_s.append(float(s))
            car_e.append(float(e))
        return car_s, car_e

    def begin_step(self, step_index):
        """Cache the state shared by observation, recovery, and trace for a tick."""
        self.step_index = int(step_index)
        self.car_s, self.car_e = self.project_current()

    @property
    def sim_time(self):
        if self.step_index is None:
            raise RuntimeError("begin_step() must be called before reading sim_time")
        return self.step_index * params.SIM_DT

    def capture(self, policies):
        """Capture the complete policy observation for the current tick."""
        self._require_active_step()
        return RaceObservation.from_runtime(
            policies, self.states, self.car_s, self.car_e, self.sim_time
        )

    def step(self, actions):
        """Apply commands, recovery, dynamics, projection, and contact response."""
        self._require_active_step()
        car_s, car_e = self.car_s, self.car_e
        actions, offroad_active, heading_errors = self._apply_recovery(
            actions, car_s, car_e
        )
        # Trace semantics historically capture the state after recovery but
        # immediately before vehicle integration.
        trace_states = tuple(
            CarState.from_array(state.to_array()) for state in self.states
        )
        next_s, next_e, contact_happened, max_overlap = (
            self._integrate_and_resolve_contacts(actions)
        )
        return StepResult(
            actions=tuple(actions),
            trace_states=trace_states,
            previous_s=tuple(car_s),
            previous_e=tuple(car_e),
            next_s=tuple(next_s),
            next_e=tuple(next_e),
            offroad_active=tuple(offroad_active),
            offroad_heading_errors=tuple(heading_errors),
            contact_happened=contact_happened,
            contact_max_overlap=max_overlap,
        )

    def _require_active_step(self):
        if self.step_index is None or self.car_s is None or self.car_e is None:
            raise RuntimeError("begin_step() must be called before capture() or step()")

    def _apply_recovery(self, requested_actions, car_s, car_e):
        actions = [
            CarAction(throttle=action.throttle, steer=action.steer)
            for action in requested_actions
        ]
        offroad_active, heading_errors = [], []
        for idx, state in enumerate(self.states):
            e = car_e[idx]
            heading = track_heading_error(self.track, state.psi, car_s[idx])
            offroad = abs(e) > self.offroad_recovery_threshold
            offroad_active.append(offroad)
            heading_errors.append(heading if offroad else 0.0)
            if offroad:
                overshoot = abs(e) - self.offroad_recovery_threshold
                speed_scale = np.exp(-params.OFFROAD_STRENGTH * overshoot)
                state.vx *= speed_scale
                state.vy *= speed_scale
                blend = 1.0 - np.exp(
                    -params.OFFROAD_HEADING_REALIGN_STRENGTH
                    * overshoot
                )
                state.psi = np.arctan2(
                    np.sin(state.psi + blend * heading),
                    np.cos(state.psi + blend * heading),
                )
                steer_blend = 1.0 - speed_scale
                inward_steer = -np.sign(e) * params.OFFROAD_INWARD_STEER
                actions[idx].steer += (
                    inward_steer - actions[idx].steer
                ) * steer_blend
                if abs(heading) > params.OFFROAD_LARGE_HEADING_ERROR:
                    actions[idx].throttle += params.OFFROAD_THROTTLE_BOOST
                if state.vx < params.OFFROAD_LOW_SPEED_THRESHOLD:
                    neutral = neutral_throttle(
                        state.vx, self.vehicle_models[idx].p
                    )
                    actions[idx].throttle = max(
                        actions[idx].throttle,
                        neutral + params.OFFROAD_THROTTLE_BOOST,
                    )

            actions[idx].throttle = float(
                np.clip(actions[idx].throttle, -1.0, 1.0)
            )
            actions[idx].steer = float(np.clip(actions[idx].steer, -1.0, 1.0))

        return actions, offroad_active, heading_errors

    def _integrate_and_resolve_contacts(self, actions):
        for idx, state in enumerate(self.states):
            self.states[idx] = CarState.from_array(
                self.vehicle_models[idx].step(
                    state.to_array(), actions[idx].throttle, actions[idx].steer
                )
            )

        next_s, next_e = self.project_current()
        scales = np.ones(len(self.states))
        max_overlap = 0.0
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                overlap = oriented_rectangle_overlap_cost(
                    self.states[i],
                    self.states[j],
                    params.VEHICLE_HALF_LENGTH,
                    params.VEHICLE_HALF_WIDTH,
                )
                if overlap <= 0.0:
                    continue
                max_overlap = max(max_overlap, overlap)
                delta = progress_delta(
                    next_s[i], next_s[j], self.track.total_length
                )
                if delta == 0.0:
                    continue
                front, rear = (i, j) if delta > 0.0 else (j, i)
                scales[front] *= np.exp(
                    -params.COLLISION_FRONT_DECAY
                    * overlap
                )
                scales[rear] *= np.exp(
                    -params.COLLISION_REAR_DECAY
                    * overlap
                )

        happened = bool(np.any(scales < 1.0))
        for idx, scale in enumerate(scales):
            if scale < 1.0:
                self.states[idx].vx *= scale
                self.states[idx].vy *= scale
        return next_s, next_e, happened, float(max_overlap)
