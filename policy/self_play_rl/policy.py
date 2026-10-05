import json
import math
from pathlib import Path

import numpy as np

from state import CarAction, CarState
from track import RaceLine

from ..base import BasePolicy
from ..utils import progress_delta, track_curvature_at, track_heading_error

class SelfPlayRLPolicy(BasePolicy):
    """
    Direct-action self-play RL baseline compatible with alpha-RACER's RL branch.

    The upstream comparison loads ``PPO.load("trained_policy9")`` and calls
    ``predict`` on a 15-value observation. The local training module follows
    the same Stable-Baselines3 checkpoint and inference interface.
    """

    OBSERVATION_SIZE = 15

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
        backend="auto",
        strict_load=False,
        deterministic=True,
        action_order="throttle_steer",
        throttle_bounds=(0.0, 1.0),
        steer_bounds=(-1.0, 1.0),
        obs_mean=None,
        obs_std=None,
        action_scale=None,
        action_bias=None,
        fallback_gain=(0.4, 1.5),
    ):
        super().__init__(track, mpc, car_idx=car_idx, policy_name=policy_name)
        self.raceline = raceline
        self.max_speed = float(max_speed)
        self.last_raceline_s = None
        self.last_raceline_idx = None
        self.current_mode = "self_play_rl"
        self.backend = str(backend or "auto").strip().lower()
        self.strict_load = bool(strict_load)
        self.deterministic = bool(deterministic)
        self.action_order = str(action_order or "throttle_steer").strip().lower()
        self.throttle_bounds = tuple(float(v) for v in throttle_bounds)
        self.steer_bounds = tuple(float(v) for v in steer_bounds)
        self.obs_mean = None if obs_mean is None else np.asarray(obs_mean, dtype=float)
        self.obs_std = None if obs_std is None else np.asarray(obs_std, dtype=float)
        self.action_scale = None if action_scale is None else np.asarray(action_scale, dtype=float)
        self.action_bias = None if action_bias is None else np.asarray(action_bias, dtype=float)
        self.fallback_gain = np.asarray(fallback_gain, dtype=float)
        self.model_path = None if model_path in {None, ""} else Path(model_path)
        self.model = model
        self.model_kind = "object" if model is not None else "fallback"
        self.last_observation = np.full(self.OBSERVATION_SIZE, np.nan)
        self.last_raw_action = np.full(2, np.nan)
        self.last_ref_traj = None
        self._warned_model_load = False
        if self.model is None and self.model_path is not None:
            self._load_model_from_path(self.model_path)

    def _warn_load_failure(self, exc):
        if self.strict_load:
            raise exc
        if not self._warned_model_load:
            self._log(
                f"[self-play RL][{self.warning_label}] model unavailable; "
                f"using fallback direct policy. reason={exc}"
            )
            self._warned_model_load = True

    def _load_model_from_path(self, model_path: Path):
        path = Path(model_path)
        if not path.exists():
            self._warn_load_failure(FileNotFoundError(path))
            return

        suffix = path.suffix.lower()
        try:
            if self.backend in {"auto", "npz"} and suffix == ".npz":
                self.model = self._load_npz_policy(path)
                self.model_kind = "npz"
                return
            if self.backend in {"auto", "json"} and suffix == ".json":
                self.model = self._load_json_policy(path)
                self.model_kind = "json"
                return
            if self.backend in {"auto", "npy"} and suffix == ".npy":
                self.model = {"type": "linear", "weights": np.load(path), "bias": np.zeros(2)}
                self.model_kind = "npy"
                return
            if suffix in {".pth", ".pt"}:
                raise ValueError(
                    "Native Torch self-play checkpoints are unsupported; "
                    "train with `python -m policy.self_play_rl.train` and use its .zip output."
                )
            if self.backend in {"auto", "sb3", "ppo"} and suffix == ".zip":
                self.model = self._load_sb3_policy(path)
                self.model_kind = "sb3"
                return
            raise ValueError(f"Unsupported self-play RL policy file: {path}")
        except Exception as exc:
            self.model = None
            self.model_kind = "fallback"
            self._warn_load_failure(exc)

    def _load_sb3_policy(self, path: Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable_baselines3 is only required for SB3/PPO self-play RL models. "
                "Use a .npz/.json exported policy or install stable-baselines3 to load this file."
            ) from exc
        return PPO.load(str(path))

    def _load_npz_policy(self, path: Path):
        data = np.load(path, allow_pickle=False)
        policy = {"type": "linear"}
        if "W" in data:
            policy["weights"] = np.asarray(data["W"], dtype=float)
            policy["bias"] = np.asarray(data["b"], dtype=float) if "b" in data else np.zeros(2)
        elif "weights" in data:
            policy["weights"] = np.asarray(data["weights"], dtype=float)
            policy["bias"] = np.asarray(data["bias"], dtype=float) if "bias" in data else np.zeros(2)
        else:
            layers = []
            idx = 0
            while f"W{idx}" in data:
                bias_key = f"b{idx}"
                layers.append(
                    (
                        np.asarray(data[f"W{idx}"], dtype=float),
                        np.asarray(data[bias_key], dtype=float) if bias_key in data else None,
                    )
                )
                idx += 1
            if not layers:
                raise ValueError(f"{path} must contain W/b, weights/bias, or W0/b0... arrays")
            policy = {"type": "mlp", "layers": layers}

        self._read_optional_npz_arrays(data)
        return policy

    def _read_optional_npz_arrays(self, data):
        if "obs_mean" in data:
            self.obs_mean = np.asarray(data["obs_mean"], dtype=float)
        if "obs_std" in data:
            self.obs_std = np.asarray(data["obs_std"], dtype=float)
        if "action_scale" in data:
            self.action_scale = np.asarray(data["action_scale"], dtype=float)
        if "action_bias" in data:
            self.action_bias = np.asarray(data["action_bias"], dtype=float)

    def _load_json_policy(self, path: Path):
        with path.open("r", encoding="utf-8") as file:
            spec = json.load(file)
        if "obs_mean" in spec:
            self.obs_mean = np.asarray(spec["obs_mean"], dtype=float)
        if "obs_std" in spec:
            self.obs_std = np.asarray(spec["obs_std"], dtype=float)
        if "action_scale" in spec:
            self.action_scale = np.asarray(spec["action_scale"], dtype=float)
        if "action_bias" in spec:
            self.action_bias = np.asarray(spec["action_bias"], dtype=float)
        if "action_order" in spec:
            self.action_order = str(spec["action_order"]).strip().lower()

        if "layers" in spec:
            return {
                "type": "mlp",
                "layers": [
                    (
                        np.asarray(layer["weight"], dtype=float),
                        np.asarray(layer.get("bias", 0.0), dtype=float),
                    )
                    for layer in spec["layers"]
                ],
            }
        return {
            "type": "linear",
            "weights": np.asarray(spec["weights"], dtype=float),
            "bias": np.asarray(spec.get("bias", [0.0, 0.0]), dtype=float),
        }

    def _normalize_observation(self, observation):
        obs = np.asarray(observation, dtype=float)
        obs[~np.isfinite(obs)] = 0.0
        if self.obs_mean is not None:
            obs = obs - self.obs_mean
        if self.obs_std is not None:
            obs = obs / np.maximum(self.obs_std, 1e-6)
        return obs

    def _apply_action_transform(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)[:2]
        if self.action_scale is not None:
            action = action * self.action_scale[:2]
        if self.action_bias is not None:
            action = action + self.action_bias[:2]
        if self.action_order in {"steer_throttle", "steering_throttle"}:
            action = np.asarray([action[1], action[0]], dtype=float)
        throttle = float(np.clip(action[0], self.throttle_bounds[0], self.throttle_bounds[1]))
        steer = float(np.clip(action[1], self.steer_bounds[0], self.steer_bounds[1]))
        return CarAction(throttle=throttle, steer=steer)

    def _predict_array_policy(self, policy, observation):
        x = self._normalize_observation(observation)
        if policy["type"] == "linear":
            weights = np.asarray(policy["weights"], dtype=float)
            if weights.shape[0] == x.shape[0] and weights.ndim == 2:
                weights = weights.T
            y = weights @ x + np.asarray(policy["bias"], dtype=float)
            return y

        y = x
        for idx, (weights, bias) in enumerate(policy["layers"]):
            weights = np.asarray(weights, dtype=float)
            if weights.shape[0] == y.shape[0] and weights.ndim == 2:
                weights = weights.T
            y = np.asarray(weights, dtype=float) @ y
            if bias is not None:
                y = y + np.asarray(bias, dtype=float)
            if idx < len(policy["layers"]) - 1:
                y = np.tanh(y)
        return y

    def _predict_model(self, observation):
        if self.model is None:
            return self._fallback_action(observation)
        if isinstance(self.model, dict):
            return self._predict_array_policy(self.model, observation)
        if hasattr(self.model, "predict"):
            action, _ = self.model.predict(np.asarray(observation, dtype=float), deterministic=self.deterministic)
            return action
        if callable(self.model):
            return self.model(np.asarray(observation, dtype=float))
        raise TypeError(f"Unsupported self-play RL model object: {type(self.model)!r}")

    def _fallback_action(self, observation):
        obs = np.asarray(observation, dtype=float)
        lateral_error = obs[1]
        heading_error = obs[3]
        current_speed = max(obs[4], 0.0)
        target_speed = self.max_speed
        if self.raceline is not None and self.last_raceline_s is not None:
            try:
                target_speed = min(self.raceline.interpolate(self.last_raceline_s)["vx_mps"], self.max_speed)
            except Exception:
                target_speed = self.max_speed
        speed_gain = self.fallback_gain[0] if self.fallback_gain.size > 0 else 0.4
        steer_gain = self.fallback_gain[1] if self.fallback_gain.size > 1 else 1.5
        throttle = speed_gain * (target_speed - current_speed)
        steer = -steer_gain * heading_error - 0.8 * lateral_error
        return np.asarray([throttle, steer], dtype=float)

    def _nearest_opponent(self, current_s, opponents):
        if not opponents:
            return None
        return min(
            opponents,
            key=lambda opponent: abs(progress_delta(float(opponent["px"]), float(current_s), self.track.total_length)),
        )

    def _build_original_observation(self, state: CarState, current_s: float, opponents):
        # 15-dim observation: 
        # [delta_s, e, opp_e, own_heading_error, vx, vy, omega, opp_heading_error, opp_vx, opp_vy, opp_omega, kappa_own, kappa_opp, kappa_own_lookahead, kappa_opp_lookahead]
        _, current_e = self.track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=8.0,
            psi_hint=state.psi,
            max_heading_error=math.pi / 2.0,
        )
        opponent = self._nearest_opponent(current_s, opponents)
        if opponent is None:
            opponent = {
                "px": current_s,
                "py": current_e,
                "vx": state.vx,
                "vy": state.vy,
                "omega": state.omega,
                "psi": state.psi,
            }

        own_heading_error = track_heading_error(self.track, state.psi, current_s, reference_minus_heading=False)
        opp_heading_error = track_heading_error(
            self.track,
            float(opponent.get("psi", state.psi)),
            float(opponent["px"]),
            reference_minus_heading=False,
        )
        lookahead_time = self.mpc.dt * self.mpc.horizon
        own_lookahead_s = current_s + max(float(state.vx), 0.0) * lookahead_time
        opp_lookahead_s = float(opponent["px"]) + max(float(opponent.get("vx", 0.0)), 0.0) * lookahead_time
        observation = np.asarray(
            [
                progress_delta(float(opponent["px"]), current_s, self.track.total_length), # opponent minus ego, as upstream
                current_e,                                              # self lateral error
                float(opponent.get("py", 0.0)),                         # self lateral position
                own_heading_error,                                      # self heading error
                float(state.vx),                                        # self longitudinal velocity
                float(state.vy),                                        # self lateral velocity
                float(state.omega),                                     # self angular velocity
                opp_heading_error,                                      # opponent heading error
                float(opponent.get("vx", 0.0)),                         # opponent longitudinal velocity
                float(opponent.get("vy", 0.0)),                         # opponent lateral velocity
                float(opponent.get("omega", 0.0)),                      # opponent angular velocity
                track_curvature_at(self.track, current_s),                    # self track curvature
                track_curvature_at(self.track, float(opponent["px"])),        # opponent track curvature
                track_curvature_at(self.track, own_lookahead_s),              # self lookahead track curvature
                track_curvature_at(self.track, opp_lookahead_s),              # opponent lookahead track curvature
            ],
            dtype=float,
        )
        observation[~np.isfinite(observation)] = 0.0
        return observation

    def get_action(self, observation) -> CarAction:
        if self.car_idx is None:
            raise ValueError("SelfPlayRLPolicy requires car_idx when used with RaceObservation")
        ego = observation.car(self.car_idx)
        state = ego.state
        current_s = ego.s
        opponents = observation.mpc_opponents(self.car_idx)
        if self.raceline is not None:
            self.last_raceline_idx = self.raceline.nearest_index(state.x, state.y)
            self.last_raceline_s = float(self.raceline.s[self.last_raceline_idx])
        self.last_ref_traj = None
        self.last_ref_world_horizon = None
        self.last_raceline_world_horizon = None
        observation = self._build_original_observation(state, current_s, opponents or [])
        self.last_observation = observation.copy()
        raw_action = self._predict_model(observation)
        self.last_raw_action = np.asarray(raw_action, dtype=float).reshape(-1)[:2].copy()
        action = self._apply_action_transform(raw_action)
        self.last_u = np.array([action.throttle, action.steer])
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
        self.mpc.last_control_plan = np.repeat(self.last_u.reshape(2, 1), self.mpc.horizon, axis=1)
        self.mpc.last_state_plan = None
        return action
