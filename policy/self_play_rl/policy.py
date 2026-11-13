"""Runtime policy for native Torch self-play checkpoints."""

import numpy as np
import torch

from state import CarAction
from track import RaceLine

from ..base import BasePolicy
from .network import OBSERVATION_SIZE, load_actor_checkpoint


class SelfPlayRLPolicy(BasePolicy):
    """Map the shared 15-value observation directly to throttle and steering."""

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
                model_path, car_idx, self.device
            )
            self.model_kind = "torch"

    def _predict(self, observation):
        values = np.asarray(observation, dtype=np.float32)
        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(values, deterministic=True)
            return np.asarray(action, dtype=float)
        if callable(self.model):
            return np.asarray(self.model(values), dtype=float)
        raise TypeError(f"Unsupported self-play model object: {type(self.model)!r}")

    @staticmethod
    def _bounded_action(raw_action):
        raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
        if raw_action.size < 2 or not np.all(np.isfinite(raw_action[:2])):
            raise RuntimeError("Self-play actor returned a nonfinite two-value action")
        return CarAction(
            throttle=float(np.clip(raw_action[0], 0.0, 1.0)),
            steer=float(np.clip(raw_action[1], -1.0, 1.0)),
        )

    def get_action(self, observation) -> CarAction:
        ego = observation.car(self.car_idx)
        if self.raceline is not None:
            self.last_raceline_idx = self.raceline.nearest_index(ego.x, ego.y)
            self.last_raceline_s = float(self.raceline.s[self.last_raceline_idx])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        model_input = observation.self_play_rl_input(self.car_idx)
        self.last_observation = model_input.copy()
        self.last_raw_action = self._predict(model_input).reshape(-1)[:2].copy()
        action = self._bounded_action(self.last_raw_action)
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
        return action
