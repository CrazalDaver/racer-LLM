"""Train the alpha-RACER self-play RL baseline with Stable-Baselines3 PPO.

The resulting ``.zip`` file is loaded directly by ``SelfPlayRLPolicy`` through
``stable_baselines3.PPO.load``.  Training is offline: run this module once,
then reuse the saved model for simulation experiments.

Paper-style training (long running)::

    python -m train.self_play_rl --output self_play_rl_policy.zip

Quick pipeline check::

    python -m train.self_play_rl --iterations 1 --steps-per-iteration 256 \
        --num-envs 1 --episode-steps 50 --output /tmp/self_play_smoke.zip

The paper appendix specifies 99 round-robin phases, 100K time steps per phase,
three cars, and eight parallel environments.  Its numerical-results section
also summarizes the baseline as 2M steps; see ``--paper-total-steps`` for that
alternative budget.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from config import make_vehicle_params, params
from state import CarState
from track import Track, load_centerline_csv
from vehicle_model import (
    DynamicBicycleModel,
    KinematicBicycleModel,
    oriented_rectangle_overlap_cost,
)


OBSERVATION_SIZE = 15
ACTION_SIZE = 2


def progress_delta(s_a, s_b, total_length):
    """Signed shortest progress from ``s_b`` to ``s_a`` on a closed track."""
    return math.atan2(
        math.sin((s_a - s_b) * 2.0 * math.pi / total_length),
        math.cos((s_a - s_b) * 2.0 * math.pi / total_length),
    ) * total_length / (2.0 * math.pi)


def angle_delta(angle_a, angle_b):
    return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))


def track_curvature_at(track, s, ds=0.25):
    _, _, psi_prev = track.interpolate(s - ds)
    _, _, psi_next = track.interpolate(s + ds)
    return angle_delta(psi_next, psi_prev) / (2.0 * ds)


def heading_error(track, psi, s):
    _, _, reference_heading = track.interpolate(s)
    return angle_delta(psi, reference_heading)


@dataclass
class SelfPlayEnvConfig:
    num_cars: int = 3
    episode_steps: int = 500
    dt: float = 0.1
    vehicle_model: str = "kinematic"
    initial_speed: float = 2.0
    start_gap: float = 3.0
    start_jitter: float = 0.15
    track_half_width: float = 0.5


class FrozenOpponentPolicy:
    """Inference-only copy of the learner used by opponents during one phase."""

    def __init__(self, model):
        self.policy = model.policy_class(
            model.observation_space,
            model.action_space,
            lr_schedule=lambda _: 0.0,
            **model.policy_kwargs,
        ).to("cpu")
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in model.policy.state_dict().items()
        }
        self.policy.load_state_dict(state_dict)
        self.policy.set_training_mode(False)

    def predict(self, observation):
        action, _ = self.policy.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
        return np.asarray(action, dtype=np.float32)


class SelfPlayRacingEnv(gym.Env):
    """Three-car racing environment exposing one car as the learning agent.

    One shared PPO policy is trained in round-robin phases.  At phase ``k``,
    car ``k % num_cars`` is controlled by the learner while the other cars use
    a frozen snapshot from the preceding phase.  This matches the switching
    scheme described in the alpha-RACER appendix while producing one reusable
    SB3 policy.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg=None, seed=0):
        super().__init__()
        self.cfg = cfg or SelfPlayEnvConfig()
        if self.cfg.num_cars < 2:
            raise ValueError("Self-play requires at least two cars")
        self.rng = np.random.default_rng(seed)
        self.track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
        self.action_space = spaces.Box(
            low=np.asarray([0.0, -1.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        vehicle_params = make_vehicle_params(self.cfg.vehicle_model)
        if self.cfg.vehicle_model.lower() in {"kinematic", "kinetic", "kinetics"}:
            factory = lambda: KinematicBicycleModel(
                dt=self.cfg.dt,
                db_params=vehicle_params,
                max_speed=params.MAX_STATE_SPEED,
            )
        else:
            factory = lambda: DynamicBicycleModel(dt=self.cfg.dt, db_params=vehicle_params)
        self.vehicle_models = [factory() for _ in range(self.cfg.num_cars)]
        self.learning_car = 0
        self.opponent_policy = None
        self.states = []
        self.progress_s = np.zeros(self.cfg.num_cars, dtype=float)
        self.lateral_e = np.zeros(self.cfg.num_cars, dtype=float)
        self.step_count = 0

    def set_self_play_phase(self, learning_car, opponent_policy):
        self.learning_car = int(learning_car) % self.cfg.num_cars
        self.opponent_policy = opponent_policy

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.states = []

        # Randomize the portion of track while retaining the paper's three
        # ordered starting regions.  Rotation avoids tying a car identity to a
        # permanent advantage and creates varied curvature interactions.
        base_s = float(self.rng.uniform(0.0, self.track.total_length))
        slots = self.rng.permutation(self.cfg.num_cars)
        for car_idx in range(self.cfg.num_cars):
            start_s = (
                base_s
                - float(slots[car_idx]) * self.cfg.start_gap
                + float(self.rng.uniform(-self.cfg.start_jitter, self.cfg.start_jitter))
            )
            start_e = float(self.rng.uniform(-0.05, 0.05))
            x, y = self.track.frenet_to_world(start_s, start_e)
            _, _, psi = self.track.interpolate(start_s)
            self.states.append(
                CarState(
                    x=x,
                    y=y,
                    vx=self.cfg.initial_speed,
                    vy=0.0,
                    psi=psi,
                    omega=0.0,
                )
            )
            self.progress_s[car_idx] = start_s
            self.lateral_e[car_idx] = start_e

        self._project_all()
        return self._observation_for(self.learning_car), self._info()

    def _project_all(self):
        for idx, state in enumerate(self.states):
            projected_s, e = self.track.project(
                state.x,
                state.y,
                s_hint=self.progress_s[idx],
                search_radius=max(params.PROJECTION_SEARCH_RADIUS, 2.0),
                psi_hint=state.psi,
                max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
            )
            self.progress_s[idx] += progress_delta(
                projected_s,
                self.progress_s[idx],
                self.track.total_length,
            )
            self.lateral_e[idx] = e

    def _nearest_opponent(self, car_idx):
        return min(
            (idx for idx in range(self.cfg.num_cars) if idx != car_idx),
            key=lambda idx: abs(
                progress_delta(
                    self.progress_s[idx],
                    self.progress_s[car_idx],
                    self.track.total_length,
                )
            ),
        )

    def _observation_for(self, car_idx):
        opponent_idx = self._nearest_opponent(car_idx)
        own = self.states[car_idx]
        opponent = self.states[opponent_idx]
        own_s = self.progress_s[car_idx]
        opponent_s = self.progress_s[opponent_idx]
        lookahead_time = params.MPC_DT * params.MPC_HORIZON
        own_lookahead = own_s + max(float(own.vx), 0.0) * lookahead_time
        opponent_lookahead = opponent_s + max(float(opponent.vx), 0.0) * lookahead_time

        observation = np.asarray(
            [
                # Upstream multi_car_comp_blocking.py uses opponent - ego.
                progress_delta(opponent_s, own_s, self.track.total_length),
                self.lateral_e[car_idx],
                self.lateral_e[opponent_idx],
                heading_error(self.track, own.psi, own_s),
                own.vx,
                own.vy,
                own.omega,
                heading_error(self.track, opponent.psi, opponent_s),
                opponent.vx,
                opponent.vy,
                opponent.omega,
                track_curvature_at(self.track, own_s),
                track_curvature_at(self.track, opponent_s),
                track_curvature_at(self.track, own_lookahead),
                track_curvature_at(self.track, opponent_lookahead),
            ],
            dtype=np.float32,
        )
        observation[~np.isfinite(observation)] = 0.0
        return observation

    def _baseline_action(self, car_idx):
        state = self.states[car_idx]
        speed_error = params.MAX_RACELINE_SPEED - max(float(state.vx), 0.0)
        steer = (
            -1.5 * heading_error(self.track, state.psi, self.progress_s[car_idx])
            - 0.8 * self.lateral_e[car_idx]
        )
        return np.asarray(
            [np.clip(0.35 * speed_error, 0.0, 1.0), np.clip(steer, -1.0, 1.0)],
            dtype=np.float32,
        )

    def _opponent_action(self, car_idx):
        if self.opponent_policy is None:
            return self._baseline_action(car_idx)
        return self.opponent_policy.predict(self._observation_for(car_idx))

    def _relative_lead(self, car_idx):
        opponents = [self.progress_s[idx] for idx in range(self.cfg.num_cars) if idx != car_idx]
        return float(self.progress_s[car_idx] - max(opponents))

    def _apply_racing_incidents(self):
        offroad = np.abs(self.lateral_e) > self.cfg.track_half_width
        for idx in np.flatnonzero(offroad):
            self.states[idx].vx *= 0.5
            _, _, psi = self.track.interpolate(self.progress_s[idx])
            self.states[idx].psi = psi

        collisions = 0
        collision_scales = np.ones(self.cfg.num_cars, dtype=float)
        for i in range(self.cfg.num_cars):
            for j in range(i + 1, self.cfg.num_cars):
                overlap_cost = oriented_rectangle_overlap_cost(
                    self.states[i],
                    self.states[j],
                    params.VEHICLE_HALF_LENGTH,
                    params.VEHICLE_HALF_WIDTH,
                )
                if overlap_cost <= 0.0:
                    continue
                collisions += 1
                if self.progress_s[i] >= self.progress_s[j]:
                    front, rear = i, j
                else:
                    front, rear = j, i
                collision_scales[front] *= math.exp(
                    -params.COLLISION_FRONT_DECAY * overlap_cost
                )
                collision_scales[rear] *= math.exp(
                    -params.COLLISION_REAR_DECAY * overlap_cost
                )
        for idx, scale in enumerate(collision_scales):
            if scale < 1.0:
                self.states[idx].vx *= scale
                self.states[idx].vy *= scale
        return int(np.sum(offroad)), collisions

    def step(self, action):
        actions = np.zeros((self.cfg.num_cars, ACTION_SIZE), dtype=np.float32)
        actions[self.learning_car] = np.clip(action, self.action_space.low, self.action_space.high)
        for idx in range(self.cfg.num_cars):
            if idx != self.learning_car:
                actions[idx] = self._opponent_action(idx)

        previous_lead = self._relative_lead(self.learning_car)
        for idx, state in enumerate(self.states):
            next_state = self.vehicle_models[idx].step(
                state.to_array(),
                float(actions[idx, 0]),
                float(actions[idx, 1]),
            )
            self.states[idx] = CarState.from_array(next_state)
        self._project_all()
        offroad, collisions = self._apply_racing_incidents()

        # Paper utility: change in ego's lead over the furthest-ahead opponent.
        reward = self._relative_lead(self.learning_car) - previous_lead
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.cfg.episode_steps
        info = self._info()
        info.update({"offroad": offroad, "collisions": collisions})
        return self._observation_for(self.learning_car), float(reward), terminated, truncated, info

    def _info(self):
        return {
            "learning_car": self.learning_car,
            "relative_lead": self._relative_lead(self.learning_car) if self.states else 0.0,
            "mean_abs_lateral_error": float(np.mean(np.abs(self.lateral_e))),
        }


def train(args):
    if args.paper_total_steps:
        args.iterations = max(1, math.ceil(2_000_000 / args.steps_per_iteration))

    cfg = SelfPlayEnvConfig(
        num_cars=args.num_cars,
        episode_steps=args.episode_steps,
        dt=args.dt,
        vehicle_model=args.vehicle_model,
        initial_speed=args.initial_speed,
        start_gap=args.start_gap,
    )
    env = make_vec_env(
        lambda: SelfPlayRacingEnv(cfg=cfg),
        n_envs=args.num_envs,
        seed=args.seed,
    )
    rollout_batch_size = args.rollout_steps * args.num_envs
    batch_size = min(args.batch_size, rollout_batch_size)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_steps,
        batch_size=batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.entropy_coef,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    checkpoint = None
    if args.checkpoint_interval > 0:
        checkpoint_dir = Path(args.output).with_suffix("").parent / "self_play_checkpoints"
        checkpoint = CheckpointCallback(
            save_freq=max(args.checkpoint_interval // args.num_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="self_play_rl",
        )

    frozen_opponents = None
    for phase in range(args.iterations):
        learning_car = phase % args.num_cars
        env.env_method("set_self_play_phase", learning_car, frozen_opponents)
        print(
            f"self-play phase {phase + 1}/{args.iterations}: "
            f"training car {learning_car} for {args.steps_per_iteration} steps"
        )
        model.learn(
            total_timesteps=args.steps_per_iteration,
            reset_num_timesteps=False,
            callback=checkpoint,
            progress_bar=args.progress_bar,
        )
        frozen_opponents = FrozenOpponentPolicy(model)

    output = Path(args.output)
    if output.suffix != ".zip":
        output = Path(f"{output}.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    print(f"saved SB3 self-play policy to {output}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the alpha-RACER self-play PPO baseline.")
    parser.add_argument("--output", default="self_play_rl_policy.zip")
    parser.add_argument("--iterations", type=int, default=99)
    parser.add_argument("--steps-per-iteration", type=int, default=100_000)
    parser.add_argument(
        "--paper-total-steps",
        action="store_true",
        help="Use the 2M total-step budget stated in the paper's numerical-results section.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-cars", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--vehicle-model", choices=["kinematic", "dbm"], default="dbm")
    parser.add_argument("--initial-speed", type=float, default=2.0)
    parser.add_argument("--start-gap", type=float, default=3.0)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
