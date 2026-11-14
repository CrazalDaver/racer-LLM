"""Runtime policy for native Torch self-play checkpoints."""

import numpy as np
import torch

from state import CarAction
from track import RaceLine

from ..base import BasePolicy
from .network import OBSERVATION_SIZE, load_actor_checkpoint


class SelfPlayRLPolicy(BasePolicy):
    """
    Map the shared 15-value observation directly to throttle and steering.

    """

    def __init__(
        self,
        track,
        mpc,
        raceline: RaceLine = None,
        max_speed=5.0,
        car_idx=None,
        policy_name=None,
        model=None,
        model_path=None,
        device=None,
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        if car_idx is None:
            raise ValueError("SelfPlayRLPolicy requires car_idx")
        self.raceline = raceline
        self.max_speed = float(max_speed)
        self.last_raceline_s = None
        self.last_raceline_idx = None
        self.current_mode = "self_play_rl"
        self.last_observation = np.full(OBSERVATION_SIZE, np.nan)
        self.last_raw_action = np.full(2, np.nan)
        self.last_ref_traj = None
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if model is not None:
            self.model = model
            self.model_path = None
            self.model_metadata = {}
            self.model_kind = "object"
        else:
            if model_path in {None, ""}:
                raise ValueError("Self-play RL requires a native Torch checkpoint")
            self.model_path, self.model, self.model_metadata = load_actor_checkpoint(
                model_path, self.device
            )
            self.model_kind = "torch"

    def _predict(self, observation):
        """
        Main inference function for self-play RL policies.
        Call model.predict() to predict the raw two-value action from the 15-value observation.
        Return the raw two-value action predicted by the self-play RL model.
        """
        values = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(values, deterministic=True)
        self.last_raw_action = np.asarray(action, dtype=float).reshape(-1)[:2].copy()
        return self.last_raw_action

    def _construct_input(self, observation):
        """Build the shared input and update reference fields for recording."""
        ego = observation.car(self.car_idx)
        if self.raceline is not None:
            self.last_raceline_idx = self.raceline.nearest_index(ego.x, ego.y)
            self.last_raceline_s = float(self.raceline.s[self.last_raceline_idx])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        model_input = observation.self_play_rl_input(self.car_idx)
        self.last_observation = model_input.copy()
        return model_input

    @staticmethod
    def _bounded_action(raw_action):
        """
        Clip the raw action to the valid range for throttle and steering.
        throttle: [0.0, 1.0]
        steer: [-1.0, 1.0]
        Return a CarAction object with the bounded throttle and steering values.
        """
        raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
        if raw_action.size < 2 or not np.all(np.isfinite(raw_action[:2])):
            raise RuntimeError("Self-play actor returned a nonfinite two-value action")
        return CarAction(
            throttle=float(np.clip(raw_action[0], 0.0, 1.0)),
            steer=float(np.clip(raw_action[1], -1.0, 1.0)),
        )

    def _record_direct_action(self, action):
        """Save the selected action and the recorder's common MPC fields."""
        self.last_u = np.array([action.throttle, action.steer])
        # Direct policies do not invoke the MPC solver, but the recorder expects
        # the same diagnostic surface from every policy.
        self.mpc.last_solver_success = True
        self.mpc.last_solver_status = f"direct_{self.model_kind}"
        self.mpc.last_recovery_active = False
        self.mpc.last_recovery_reason = ""
        self.mpc.last_loss = np.nan
        self.mpc.last_track_loss = np.nan
        self.mpc.last_action_loss = np.nan
        self.mpc.last_velocity_loss = np.nan
        self.mpc.last_heading_loss = np.nan
        self.mpc.last_violation_loss = np.nan
        self.mpc.last_track_slack_plan = None
        self.mpc.last_collision_slack_plan = None
        self.mpc.last_control_plan = np.repeat(
            self.last_u.reshape(2, 1), self.mpc.horizon, axis=1
        )
        self.mpc.last_state_plan = None

    def get_action(self, observation) -> CarAction:
        """Build the input, predict one action, and record the decision."""
        model_input = self._construct_input(observation)
        raw_action = self._predict(model_input)
        action = self._bounded_action(raw_action)
        self._record_direct_action(action)
        return action
