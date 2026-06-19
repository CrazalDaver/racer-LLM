"""
Self-play PPO baseline for the local alpha-RACER-style simulator.

This file is intentionally standalone: it defines the policy network, a small
multi-car racing environment, the PPO training loop, and checkpoint load/save
helpers.  A trained checkpoint can be loaded by policies.SelfPlayRLPolicy.

Example:
    python self_play_rl.py --total-steps 200000 --num-cars 3 --output self_play_rl_policy.pth
"""

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - this module is only useful with torch.
    raise ImportError("self_play_rl.py requires PyTorch for training and inference.") from exc

from config import make_vehicle_params, params
from state import CarState
from track import Track, load_centerline_csv, load_raceline_csv
from vehicle_model import DynamicBicycleModel, KinematicBicycleModel


OBSERVATION_SIZE = 15
ACTION_SIZE = 2


def progress_delta(s_a, s_b, total_length):
    return math.atan2(
        math.sin((s_a - s_b) * 2.0 * math.pi / total_length),
        math.cos((s_a - s_b) * 2.0 * math.pi / total_length),
    ) * total_length / (2.0 * math.pi)


def angle_delta(angle_a, angle_b):
    return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for idx in range(len(sizes) - 1):
        act = activation if idx < len(sizes) - 2 else output_activation
        layers.extend([nn.Linear(sizes[idx], sizes[idx + 1]), act()])
    return nn.Sequential(*layers)


class RunningMeanStd:  # computes mean and std online for normalizing observations
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.var, 1e-6))


class SelfPlayActorCritic(nn.Module):
    """
    Gaussian actor-critic with alpha-RACER-compatible direct actions.

    Observations are the 15-value local interaction features used by
    SelfPlayRLPolicy.  Actions are ordered as [throttle, steer], with inference
    mapped to throttle in [0, 1] and steer in [-1, 1].
    """

    def __init__(self, obs_dim=OBSERVATION_SIZE, hidden_sizes=(128, 128), action_dim=ACTION_SIZE):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        hidden_sizes = tuple(int(v) for v in hidden_sizes)
        self.actor = mlp((self.obs_dim, *hidden_sizes, self.action_dim))
        self.critic = mlp((self.obs_dim, *hidden_sizes, 1))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))
        self.register_buffer("obs_mean", torch.zeros(self.obs_dim))
        self.register_buffer("obs_std", torch.ones(self.obs_dim))

    def set_obs_stats(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=self.obs_mean.device))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32, device=self.obs_std.device).clamp_min(1e-6))

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std.clamp_min(1e-6)

    def distribution(self, obs):
        z = self.normalize_obs(obs)
        mean = self.actor(z)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def value(self, obs):
        z = self.normalize_obs(obs)
        return self.critic(z).squeeze(-1)

    def squash_action(self, raw_action):
        throttle = torch.sigmoid(raw_action[..., 0])
        steer = torch.tanh(raw_action[..., 1])
        return torch.stack([throttle, steer], dim=-1)

    def act(self, obs, deterministic=False):
        dist = self.distribution(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return self.squash_action(raw_action), log_prob, self.value(obs), raw_action

    def evaluate_raw_actions(self, obs, raw_action):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(obs)

    def predict(self, observation, deterministic=True):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            action, _, _, _ = self.act(obs, deterministic=deterministic)
        if was_training:
            self.train()
        action_np = action.cpu().numpy()
        if np.asarray(observation).ndim == 1:
            return action_np[0], None
        return action_np, None


def track_curvature_at(track, s, ds=0.5):
    _, _, psi_prev = track.interpolate(s - ds)
    _, _, psi_next = track.interpolate(s + ds)
    return angle_delta(psi_next, psi_prev) / (2.0 * ds)


def heading_error(track, psi, s):
    _, _, ref_psi = track.interpolate(s)
    return angle_delta(psi, ref_psi)


@dataclass
class SelfPlayEnvConfig:
    num_cars: int = 3
    episode_steps: int = 500
    vehicle_model: str = "kinematic"
    v0: float = 2.0
    car_start_gap: float = 3.0
    unsafe_distance: float = 0.4
    offroad_tolerance: float = 0.05
    progress_reward: float = 1.0
    rank_reward: float = 0.25
    offroad_penalty: float = 5.0
    collision_penalty: float = 4.0
    control_penalty: float = 0.01


class SelfPlayRacingEnv:
    """Small numerical self-play environment matching this repository's simulator conventions."""

    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or SelfPlayEnvConfig()
        self.rng = np.random.default_rng(seed)
        waypoints = load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE)
        self.raceline = load_raceline_csv(params.RACELINE_CSV, scale=params.TRACK_SCALE)
        self.track = Track(waypoints)
        vehicle_params = make_vehicle_params(self.cfg.vehicle_model)
        if self.cfg.vehicle_model.lower() in {"kinematic", "kinetic", "kinetics"}:
            model_cls = lambda: KinematicBicycleModel(params.DT, vehicle_params, max_speed=params.MAX_STATE_SPEED)
        else:
            model_cls = lambda: DynamicBicycleModel(params.DT, vehicle_params)
        self.vehicle_models = [model_cls() for _ in range(self.cfg.num_cars)]
        self.states = []
        self.progress_s = []
        self.lateral_e = []
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.states = []
        self.progress_s = []
        self.lateral_e = []
        jitter = self.rng.normal(0.0, 0.05, size=self.cfg.num_cars)
        for car_idx in range(self.cfg.num_cars):
            start_s = -car_idx * self.cfg.car_start_gap + float(jitter[car_idx])
            start_e = float(self.rng.uniform(-0.05, 0.05))
            x, y = self.track.frenet_to_world(start_s, start_e)
            _, _, psi = self.track.interpolate(start_s)
            self.states.append(CarState(x=x, y=y, vx=self.cfg.v0, vy=0.0, psi=psi, omega=0.0))
            self.progress_s.append(start_s)
            self.lateral_e.append(start_e)
        return self._observations()

    def _project_all(self):
        for idx, state in enumerate(self.states):
            s_wrapped, e = self.track.project(
                state.x,
                state.y,
                s_hint=self.progress_s[idx],
                search_radius=params.PROJECTION_SEARCH_RADIUS,
                psi_hint=state.psi,
                max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
            )
            self.progress_s[idx] = s_wrapped
            self.lateral_e[idx] = e

    def _nearest_opponent_idx(self, car_idx):
        best_idx = None
        best_dist = np.inf
        for other_idx in range(self.cfg.num_cars):
            if other_idx == car_idx:
                continue
            dist = abs(progress_delta(self.progress_s[other_idx], self.progress_s[car_idx], self.track.total_length))
            if dist < best_dist:
                best_dist = dist
                best_idx = other_idx
        return best_idx

    def _observation_for(self, car_idx):
        state = self.states[car_idx]
        current_s = self.progress_s[car_idx]
        current_e = self.lateral_e[car_idx]
        opp_idx = self._nearest_opponent_idx(car_idx)
        if opp_idx is None:
            opp_s = current_s
            opp_e = current_e
            opp = state
        else:
            opp_s = self.progress_s[opp_idx]
            opp_e = self.lateral_e[opp_idx]
            opp = self.states[opp_idx]
        lookahead_time = params.MPC_DT * params.MPC_HORIZON
        own_lookahead_s = current_s + max(float(state.vx), 0.0) * lookahead_time
        opp_lookahead_s = opp_s + max(float(opp.vx), 0.0) * lookahead_time
        obs = np.asarray(
            [
                progress_delta(current_s, opp_s, self.track.total_length),
                current_e,
                opp_e,
                heading_error(self.track, state.psi, current_s),
                state.vx,
                state.vy,
                state.omega,
                heading_error(self.track, opp.psi, opp_s),
                opp.vx,
                opp.vy,
                opp.omega,
                track_curvature_at(self.track, current_s),
                track_curvature_at(self.track, opp_s),
                track_curvature_at(self.track, own_lookahead_s),
                track_curvature_at(self.track, opp_lookahead_s),
            ],
            dtype=np.float32,
        )
        obs[~np.isfinite(obs)] = 0.0
        return obs

    def _observations(self):
        self._project_all()
        return np.stack([self._observation_for(idx) for idx in range(self.cfg.num_cars)], axis=0)

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        prev_s = np.asarray(self.progress_s, dtype=float)
        prev_e = np.asarray(self.lateral_e, dtype=float)
        for idx, state in enumerate(self.states):
            throttle = float(np.clip(actions[idx, 0], 0.0, 1.0))
            steer = float(np.clip(actions[idx, 1], -1.0, 1.0))
            next_state = self.vehicle_models[idx].step(state.to_array(), throttle, steer)
            self.states[idx] = CarState.from_array(next_state)

        self._project_all()
        current_s = np.asarray(self.progress_s, dtype=float)
        current_e = np.asarray(self.lateral_e, dtype=float)
        progress = np.asarray(
            [progress_delta(current_s[idx], prev_s[idx], self.track.total_length) for idx in range(self.cfg.num_cars)]
        )
        mean_other_progress = np.asarray(
            [
                np.mean([progress[j] for j in range(self.cfg.num_cars) if j != idx])
                for idx in range(self.cfg.num_cars)
            ]
        )
        rewards = self.cfg.progress_reward * (progress - mean_other_progress)
        rewards += self.cfg.rank_reward * (
            current_s - np.mean(current_s)
        ) / max(self.track.total_length, 1.0)
        rewards -= self.cfg.control_penalty * (actions[:, 0] ** 2 + actions[:, 1] ** 2)

        offroad_half_width = params.TRACK_HALF_WIDTH + self.cfg.offroad_tolerance
        offroad = np.abs(current_e) > offroad_half_width
        rewards -= self.cfg.offroad_penalty * offroad.astype(float)
        for idx, is_offroad in enumerate(offroad):
            if is_offroad:
                self.states[idx].vx *= 0.5
                _, _, psi = self.track.interpolate(self.progress_s[idx])
                self.states[idx].psi = psi

        collision = np.zeros(self.cfg.num_cars, dtype=bool)
        for i in range(self.cfg.num_cars):
            for j in range(i + 1, self.cfg.num_cars):
                dist = math.hypot(self.states[i].x - self.states[j].x, self.states[i].y - self.states[j].y)
                if dist <= self.cfg.unsafe_distance:
                    collision[i] = True
                    collision[j] = True
                    if progress_delta(current_s[i], current_s[j], self.track.total_length) >= 0.0:
                        front_idx, rear_idx = i, j
                    else:
                        front_idx, rear_idx = j, i
                    self.states[front_idx].vx *= 0.5
                    self.states[rear_idx].vx *= 1.0 / 3.0
        rewards -= self.cfg.collision_penalty * collision.astype(float)

        self.step_count += 1
        done = self.step_count >= self.cfg.episode_steps
        obs = self._observations()
        info = {
            "mean_progress": float(np.mean(progress)),
            "collisions": int(np.sum(collision) // 2),
            "offroad": int(np.sum(offroad)),
            "mean_lateral_error": float(np.mean(np.abs(current_e))),
            "prev_lateral_error": prev_e,
        }
        return obs, rewards.astype(np.float32), done, info


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.raw_actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, raw_action, log_prob, reward, done, value):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.raw_actions.append(np.asarray(raw_action, dtype=np.float32))
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32))
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.dones.append(np.asarray(done, dtype=np.float32))
        self.values.append(np.asarray(value, dtype=np.float32))

    def as_arrays(self):
        return {
            "obs": np.concatenate(self.obs, axis=0),
            "raw_actions": np.concatenate(self.raw_actions, axis=0),
            "log_probs": np.concatenate(self.log_probs, axis=0),
            "rewards": np.stack(self.rewards, axis=0),
            "dones": np.stack(self.dones, axis=0),
            "values": np.stack(self.values, axis=0),
        }


def compute_gae(rewards, dones, values, last_values, gamma, gae_lambda):
    steps, num_agents = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(num_agents, dtype=np.float32)
    for step in reversed(range(steps)):
        next_values = last_values if step == steps - 1 else values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)


def save_checkpoint(path, model, obs_rms, args):
    checkpoint = {
        "format": "self_play_rl_actor_critic_v1",
        "model_state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "hidden_sizes": [module.out_features for module in model.actor if isinstance(module, nn.Linear)][:-1],
        "obs_mean": obs_rms.mean.astype(np.float32),
        "obs_std": obs_rms.std.astype(np.float32),
        "action_order": "throttle_steer",
        "throttle_bounds": np.asarray([0.0, 1.0], dtype=np.float32),
        "steer_bounds": np.asarray([-1.0, 1.0], dtype=np.float32),
        "train_args": vars(args),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_policy_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device)
    hidden_sizes = tuple(checkpoint.get("hidden_sizes", (128, 128)))
    model = SelfPlayActorCritic(
        obs_dim=int(checkpoint.get("obs_dim", OBSERVATION_SIZE)),
        hidden_sizes=hidden_sizes,
        action_dim=int(checkpoint.get("action_dim", ACTION_SIZE)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        model.set_obs_stats(checkpoint["obs_mean"], checkpoint["obs_std"])
    model.to(device)
    model.eval()
    metadata = {
        "action_order": checkpoint.get("action_order", "throttle_steer"),
        "throttle_bounds": checkpoint.get("throttle_bounds", np.asarray([0.0, 1.0])),
        "steer_bounds": checkpoint.get("steer_bounds", np.asarray([-1.0, 1.0])),
    }
    return model, metadata


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    cfg = SelfPlayEnvConfig(
        num_cars=args.num_cars,
        episode_steps=args.episode_steps,
        vehicle_model=args.vehicle_model,
        v0=args.v0,
    )
    env = SelfPlayRacingEnv(cfg=cfg, seed=args.seed)
    model = SelfPlayActorCritic(hidden_sizes=tuple(args.hidden_sizes)).to(device)
    old_model = copy.deepcopy(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_rms = RunningMeanStd((OBSERVATION_SIZE,))

    obs = env.reset()
    global_step = 0
    update = 0
    while global_step < args.total_steps:
        buffer = RolloutBuffer()
        episode_returns = []
        for _ in range(args.rollout_steps):
            obs_rms.update(obs)
            model.set_obs_stats(obs_rms.mean, obs_rms.std)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_t, log_prob_t, value_t, raw_action_t = old_model.act(obs_t, deterministic=False)
            actions = action_t.cpu().numpy()
            next_obs, rewards, done, info = env.step(actions)
            dones = np.full(args.num_cars, float(done), dtype=np.float32)
            buffer.add(
                obs,
                raw_action_t.cpu().numpy(),
                log_prob_t.cpu().numpy(),
                rewards,
                dones,
                value_t.cpu().numpy(),
            )
            global_step += args.num_cars
            episode_returns.append(float(np.mean(rewards)))
            obs = env.reset() if done else next_obs

        model.set_obs_stats(obs_rms.mean, obs_rms.std)
        old_model.set_obs_stats(obs_rms.mean, obs_rms.std)
        arrays = buffer.as_arrays()
        with torch.no_grad():
            last_values = old_model.value(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        advantages, returns = compute_gae(
            arrays["rewards"],
            arrays["dones"],
            arrays["values"],
            last_values,
            args.gamma,
            args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.as_tensor(arrays["obs"], dtype=torch.float32, device=device)
        raw_action_tensor = torch.as_tensor(arrays["raw_actions"], dtype=torch.float32, device=device)
        old_log_prob_tensor = torch.as_tensor(arrays["log_probs"], dtype=torch.float32, device=device)
        adv_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        batch_size = obs_tensor.shape[0]
        indices = np.arange(batch_size)

        for _ in range(args.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, args.batch_size):
                mb_idx = indices[start : start + args.batch_size]
                new_log_prob, entropy, value = model.evaluate_raw_actions(
                    obs_tensor[mb_idx],
                    raw_action_tensor[mb_idx],
                )
                ratio = torch.exp(new_log_prob - old_log_prob_tensor[mb_idx])
                unclipped = ratio * adv_tensor[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * adv_tensor[mb_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (return_tensor[mb_idx] - value).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        old_model.load_state_dict(model.state_dict())
        update += 1
        if update % args.log_interval == 0:
            print(
                f"update={update} steps={global_step} "
                f"mean_reward={np.mean(episode_returns):.4f} obs_std={np.mean(obs_rms.std):.3f}"
            )
        if args.save_interval > 0 and update % args.save_interval == 0:
            save_checkpoint(args.output, model, obs_rms, args)

    save_checkpoint(args.output, model, obs_rms, args)
    print(f"saved self-play RL policy to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a self-play PPO racing baseline.")
    parser.add_argument("--output", default="self_play_rl_policy.pth")
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--num-cars", type=int, default=3)
    parser.add_argument("--vehicle-model", default="kinematic", choices=["kinematic", "dbm"])
    parser.add_argument("--v0", type=float, default=2.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
