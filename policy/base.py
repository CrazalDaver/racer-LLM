import abc

import numpy as np

from state import CarAction, CarState
from track import Track


class BasePolicy(abc.ABC):
    """
    Common interface and observable state shared by every driving policy.
    get_action: Return the control action selected by this policy (throttle, steering).
    """

    def __init__(self, track: Track, mpc, car_idx=None, policy_name=None):
        self.track = track
        self.mpc = mpc
        self.car_idx = car_idx
        self.policy_name = policy_name
        car_label = "car?" if car_idx is None else f"car{car_idx}"
        # for logging and warning messages, include the policy name if provided
        self.warning_label = car_label if policy_name is None else f"{car_label}:{policy_name}"

        self.last_u = np.zeros(2, dtype=float)
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        # TEMPORARY DIAGNOSTIC: per-horizon reference-trajectory components.
        # Remove together with the matching simulation trace fields after the
        # current competitive-offset investigation.
        self.last_ref_components = None
        self.log_fn = print

    @abc.abstractmethod
    def get_action(self, state: CarState, current_s: float, opponents=None, sim_time=None) -> CarAction:
        """Return the control action selected by this policy."""
        raise NotImplementedError

    def _log(self, message: str):
        self.log_fn(message)

    def flush_solver_warnings(self, sim_time=None):
        """Flush policy-specific solver warnings, if the policy uses a solver."""

    def flush_mpc_warnings(self, sim_time=None):
        """Backward-compatible wrapper for older simulation callers."""
        self.flush_solver_warnings(sim_time=sim_time)

