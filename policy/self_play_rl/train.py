"""Pure-Torch round-robin PPO self-play training.

Alpha-RACER supplies the PPO hyperparameters and three-agent 100K-step
switching schedule.  The actor/critic architecture and opponent snapshot
schedule follow the referenced Thakkar self-play implementation.
"""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import multiprocessing
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from config import make_vehicle_params, params
from policy.utils import relative_progress, track_curvature_at
from simulation_core.environment import RaceEnvironment
from simulation_core.observation import RaceObservation
from state import CarAction, CarState
from track import Track, configure_track, load_centerline_csv
from vehicle_model import DynamicBicycleModel

from .network import (
    CHECKPOINT_FORMAT,
    HIDDEN_SIZES,
    PPOActorCritic,
    actor_state,
)


@dataclass(frozen=True)
class PPOSettings:
    """Fixed PPO contract; unspecified values follow standard SB3 defaults."""

    learning_rate: float = 5e-4
    batch_size: int = 1024
    epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5


@dataclass(frozen=True)
class SelfPlaySettings:
    """Paper/Thakkar self-play and race settings."""

    num_agents: int = 3
    agent_switches: int = 99
    steps_per_phase: int = 100_000
    rollout_steps: int = 256
    episode_steps: int = 500
    dt: float = 0.1
    initial_speed: float = 2.0
    start_gap: float = 3.0
    start_jitter: float = 0.15
    snapshot_steps: int = 20_000
    opponent_swap_steps: int = 10_000
    opponent_window: int = 10
    latest_opponent_probability: float = 0.5


def _cpu_policy_state(model):
    return {
        # NumPy avoids torch's shared-memory transport, which is unnecessary
        # for these small frozen worker copies and unavailable on some servers.
        "actor_state_dict": {
            name: value.detach().cpu().numpy().copy()
            for name, value in model.actor.state_dict().items()
        },
        "log_std": model.log_std.detach().cpu().numpy().copy(),
    }


def _copy_actor_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


class TrainingRace:
    """One worker-owned DBM race with the production environment semantics."""

    def __init__(self, setup, seed):
        self.setup = setup
        self.rng = np.random.default_rng(seed)
        waypoints = load_centerline_csv(setup["track_csv"], scale=setup["track_scale"])
        self.track = Track(waypoints)
        self.curvature_fn = lambda s: track_curvature_at(self.track, float(s))
        self.step_count = 0
        self.environment = None
        self.reset()

    def reset(self):
        settings = self.setup["self_play"]
        num_agents = settings["num_agents"]
        base_s = float(self.rng.uniform(0.0, self.track.total_length))
        slots = self.rng.permutation(num_agents)
        states, progress = [], []
        for car_idx in range(num_agents):
            start_s = (
                base_s
                - float(slots[car_idx]) * settings["start_gap"]
                + float(self.rng.uniform(-settings["start_jitter"], settings["start_jitter"]))
            )
            start_e = float(self.rng.uniform(-0.05, 0.05))
            x, y = self.track.frenet_to_world(start_s, start_e)
            _, _, psi = self.track.interpolate(start_s)
            states.append(CarState(
                x=x, y=y, psi=psi, vx=settings["initial_speed"], vy=0.0, omega=0.0
            ))
            progress.append(start_s)
        vehicle_params = make_vehicle_params("dbm")
        models = [
            DynamicBicycleModel(dt=settings["dt"], db_params=vehicle_params)
            for _ in range(num_agents)
        ]
        self.environment = RaceEnvironment(
            self.track, states, models, progress, curvature_fn=self.curvature_fn
        )
        self.step_count = 0
        self.environment.begin_step(0)

    def scene(self):
        """Capture the three-car scene once for all policy inputs this tick."""
        return RaceObservation.from_states(
            self.environment.states,
            self.environment.car_s,
            self.environment.car_e,
            self.step_count * self.setup["self_play"]["dt"],
            track=self.track,
            curvature_fn=self.curvature_fn,
        )

    def observation(self, car_idx):
        return self.scene().self_play_rl_input(car_idx)

    def step(self, actions, learning_car):
        previous = relative_progress(self.environment.car_s, learning_car)
        result = self.environment.step(tuple(actions))
        self.environment.car_s = list(result.next_s)
        self.environment.car_e = list(result.next_e)
        self.step_count += 1
        reward = relative_progress(result.next_s, learning_car) - previous
        next_observation = self.observation(learning_car)
        episode_end = self.step_count >= self.setup["self_play"]["episode_steps"]
        metrics = {
            "offroad": int(sum(result.offroad_active)),
            "contact": int(result.contact_happened),
        }
        if episode_end:
            self.reset()
        else:
            self.environment.begin_step(self.step_count)
        return next_observation, float(reward), episode_end, metrics


def _load_worker_policies(policy_states):
    models = []
    for state in policy_states:
        model = PPOActorCritic().cpu().eval()
        model.actor.load_state_dict({
            name: torch.from_numpy(value)
            for name, value in state["actor_state_dict"].items()
        })
        model.log_std.data.copy_(torch.from_numpy(state["log_std"]))
        models.append(model)
    return models


def _worker_loop(connection, setup, worker_id):
    torch.set_num_threads(1)
    worker_seed = setup["seed"] + 10_000 * worker_id
    torch.manual_seed(worker_seed)
    race = TrainingRace(setup, worker_seed)
    try:
        while True:
            command = connection.recv()
            if command["kind"] == "close":
                return
            if command["kind"] != "collect":
                raise ValueError(f"Unknown worker command: {command['kind']}")
            if command["reset"]:
                race.reset()
            learning_car = command["learning_car"]
            models = _load_worker_policies(command["policies"])
            observations, actions, rewards = [], [], []
            next_observations, episode_ends = [], []
            offroad = contact = 0
            for _ in range(command["steps"]):
                joint_actions = []
                scene = race.scene()
                policy_observations = [
                    scene.self_play_rl_input(car_idx)
                    for car_idx in range(len(models))
                ]
                learning_observation = policy_observations[learning_car]
                raw_learning_action = None
                for car_idx, model in enumerate(models):
                    observation = policy_observations[car_idx]
                    tensor = torch.from_numpy(observation).float().unsqueeze(0)
                    with torch.no_grad():
                        if car_idx == learning_car:
                            raw_action, _, _ = model.sample(tensor)
                            raw_learning_action = raw_action.squeeze(0).numpy()
                        else:
                            raw_action = model.actor(tensor)
                        bounded = raw_action.squeeze(0).numpy()
                    joint_actions.append(CarAction(
                        throttle=float(np.clip(bounded[0], 0.0, 1.0)),
                        steer=float(np.clip(bounded[1], -1.0, 1.0)),
                    ))
                next_observation, reward, episode_end, metrics = race.step(
                    joint_actions, learning_car
                )
                observations.append(learning_observation)
                actions.append(raw_learning_action)
                rewards.append(reward)
                next_observations.append(next_observation)
                episode_ends.append(episode_end)
                offroad += metrics["offroad"]
                contact += metrics["contact"]
            connection.send({
                "observations": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "next_observations": np.asarray(next_observations, dtype=np.float32),
                "episode_ends": np.asarray(episode_ends, dtype=bool),
                "offroad": offroad,
                "contact": contact,
            })
    finally:
        connection.close()


class RolloutWorkers:
    """Persistent CPU workers; policy updates remain in the GPU main process."""

    def __init__(self, count, setup):
        context = multiprocessing.get_context("spawn")
        self.connections, self.processes = [], []
        for worker_id in range(count):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_loop, args=(child, setup, worker_id), daemon=True
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def collect(self, learning_car, policies, steps, reset):
        for connection, worker_policies, worker_steps in zip(
            self.connections, policies, steps
        ):
            connection.send({
                "kind": "collect",
                "learning_car": learning_car,
                "policies": worker_policies,
                "steps": worker_steps,
                "reset": reset,
            })
        return [connection.recv() for connection in self.connections]

    def close(self):
        for connection in self.connections:
            try:
                connection.send({"kind": "close"})
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
        for connection in self.connections:
            connection.close()


class PPOUpdater:
    """Compute GAE and update one actor/critic from an on-policy rollout."""

    def __init__(self, settings):
        self.settings = settings

    def update(self, model, optimizer, rollouts, device, rng):
        """Consume current-policy worker trajectories and return mean losses."""
        all_observations, all_actions, all_advantages, all_returns = [], [], [], []
        all_old_log_probabilities = []
        model.eval()
        for rollout in rollouts:
            observations = torch.from_numpy(rollout["observations"]).to(device)
            actions = torch.from_numpy(rollout["actions"]).to(device)
            next_observations = torch.from_numpy(rollout["next_observations"]).to(device)
            with torch.no_grad():
                old_distribution = model.distribution(observations)
                old_log_probabilities = old_distribution.log_prob(actions).sum(dim=-1)
                values = model.critic(observations).squeeze(-1)
                next_values = model.critic(next_observations).squeeze(-1)
            rewards = torch.from_numpy(rollout["rewards"]).to(device)
            episode_ends = torch.from_numpy(rollout["episode_ends"]).to(device)
            deltas = rewards + self.settings.gamma * next_values - values
            advantages = torch.zeros_like(rewards)
            running = torch.zeros((), device=device)
            for index in range(len(rewards) - 1, -1, -1):
                continuation = 1.0 - episode_ends[index].float()
                running = deltas[index] + (
                    self.settings.gamma * self.settings.gae_lambda
                    * continuation * running
                )
                advantages[index] = running
            all_observations.append(observations)
            all_actions.append(actions)
            all_old_log_probabilities.append(old_log_probabilities)
            all_advantages.append(advantages)
            all_returns.append(advantages + values)

        observations = torch.cat(all_observations)
        actions = torch.cat(all_actions)
        old_log_probabilities = torch.cat(all_old_log_probabilities)
        advantages = torch.cat(all_advantages)
        returns = torch.cat(all_returns)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
        sample_count = len(observations)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        model.train()
        for _ in range(self.settings.epochs):
            order = rng.permutation(sample_count)
            for start in range(0, sample_count, self.settings.batch_size):
                indices = torch.as_tensor(
                    order[start:start + self.settings.batch_size], device=device
                )
                log_probability, entropy, value = model.evaluate(
                    observations[indices], actions[indices]
                )
                ratio = torch.exp(log_probability - old_log_probabilities[indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.settings.clip_ratio,
                    1.0 + self.settings.clip_ratio,
                ) * advantages[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(value, returns[indices])
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.settings.value_coefficient * value_loss
                    - self.settings.entropy_coefficient * entropy_mean
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.settings.max_gradient_norm
                )
                optimizer.step()
                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy_mean.detach())
                metrics["updates"] += 1
        for key in ("policy_loss", "value_loss", "entropy"):
            metrics[key] /= max(metrics["updates"], 1)
        return metrics


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _git_metadata():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
    }


def _deployment_checkpoint(models, metadata):
    return {
        "format": CHECKPOINT_FORMAT,
        "agents": [
            {"actor_state_dict": actor_state(model)} for model in models
        ],
        "metadata": metadata,
    }


def _select_opponent_state(models, pools, agent_idx, rng, latest_probability):
    if not pools[agent_idx] or rng.random() < latest_probability:
        return _cpu_policy_state(models[agent_idx])
    pool = list(pools[agent_idx])
    selected = pool[int(rng.integers(0, len(pool)))]
    return {
        "actor_state_dict": {
            name: value.detach().cpu().numpy().copy()
            for name, value in selected.items()
        },
        "log_std": models[agent_idx].log_std.detach().cpu().numpy().copy(),
    }


def _worker_steps(total_steps, workers, maximum_per_worker):
    total_steps = min(total_steps, workers * maximum_per_worker)
    base, remainder = divmod(total_steps, workers)
    return [base + (worker < remainder) for worker in range(workers)]


def train(args):
    """Run or resume the fixed three-agent self-play schedule."""
    ppo = PPOSettings()
    self_play = SelfPlaySettings()
    if args.smoke:
        self_play = SelfPlaySettings(
            agent_switches=3,
            steps_per_phase=256,
            rollout_steps=32,
            episode_steps=64,
            snapshot_steps=128,
            opponent_swap_steps=64,
        )
    shortest_boundary = min(
        self_play.steps_per_phase,
        self_play.snapshot_steps,
        self_play.opponent_swap_steps,
    )
    if args.workers > shortest_boundary:
        raise ValueError(
            f"--workers must not exceed the shortest collection boundary "
            f"({shortest_boundary})"
        )
    configure_track(params, args.track)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.ones(1, device=device).sum().item()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    work_dir = Path(args.work_dir)
    state_path = work_dir / "trainer.pth"
    output_path = Path(args.output)
    models = [PPOActorCritic().to(device) for _ in range(self_play.num_agents)]
    initial_state = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(initial_state)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=ppo.learning_rate)
        for model in models
    ]
    pools = [deque(maxlen=self_play.opponent_window) for _ in models]
    for agent_idx, model in enumerate(models):
        pools[agent_idx].append(actor_state(model))
    phase = phase_steps = total_steps = 0

    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            saved["ppo_settings"] != asdict(ppo)
            or saved["self_play_settings"] != asdict(self_play)
        ):
            raise ValueError("Existing self-play checkpoint uses different settings")
        phase = int(saved["phase"])
        phase_steps = int(saved["phase_steps"])
        total_steps = int(saved["total_steps"])
        for model, state in zip(models, saved["models"]):
            model.load_state_dict(state)
        for optimizer, state in zip(optimizers, saved["optimizers"]):
            optimizer.load_state_dict(state)
        pools = [
            deque(
                (_copy_actor_state(item) for item in pool),
                maxlen=self_play.opponent_window,
            )
            for pool in saved["opponent_pools"]
        ]
        rng.bit_generator.state = saved["rng_state"]
        print(f"Resuming phase={phase} phase_steps={phase_steps} total_steps={total_steps}")

    setup = {
        "seed": args.seed,
        "track_csv": params.TRACK_CSV,
        "track_scale": params.TRACK_SCALE,
        "self_play": asdict(self_play),
    }
    workers = RolloutWorkers(args.workers, setup)
    updater = PPOUpdater(ppo)
    reset_workers = True
    selected_opponents = None
    started = time.monotonic()
    git = _git_metadata()
    try:
        while phase < self_play.agent_switches:
            learning_car = phase % self_play.num_agents
            model = models[learning_car]
            optimizer = optimizers[learning_car]
            next_swap = (
                (phase_steps // self_play.opponent_swap_steps) + 1
            ) * self_play.opponent_swap_steps
            next_snapshot = (
                (phase_steps // self_play.snapshot_steps) + 1
            ) * self_play.snapshot_steps
            boundary = min(
                self_play.steps_per_phase, next_swap, next_snapshot
            )
            if selected_opponents is None:
                selected_opponents = []
                for _ in range(args.workers):
                    states = []
                    for agent_idx in range(self_play.num_agents):
                        if agent_idx == learning_car:
                            states.append(_cpu_policy_state(model))
                        else:
                            states.append(_select_opponent_state(
                                models, pools, agent_idx, rng,
                                self_play.latest_opponent_probability,
                            ))
                    selected_opponents.append(states)
            for states in selected_opponents:
                states[learning_car] = _cpu_policy_state(model)
            steps = _worker_steps(
                boundary - phase_steps, args.workers, self_play.rollout_steps
            )
            rollouts = workers.collect(
                learning_car, selected_opponents, steps, reset_workers
            )
            reset_workers = False
            collected = sum(steps)
            metrics = updater.update(model, optimizer, rollouts, device, rng)
            phase_steps += collected
            total_steps += collected

            if phase_steps % self_play.snapshot_steps == 0:
                pools[learning_car].append(actor_state(model))
            if phase_steps % self_play.opponent_swap_steps == 0:
                selected_opponents = None
            if phase_steps == self_play.steps_per_phase:
                phase += 1
                phase_steps = 0
                selected_opponents = None
                reset_workers = True

            metadata = {
                "ppo_settings": asdict(ppo),
                "self_play_settings": asdict(self_play),
                "network": {"hidden_sizes": HIDDEN_SIZES, "activation": "ReLU"},
                "phase": phase,
                "phase_steps": phase_steps,
                "total_steps": total_steps,
                "track": args.track,
                "seed": args.seed,
                "sources": "alpha-RACER Appendix E; Thakkar et al. self-play",
                "git": git,
            }
            _atomic_torch_save(_deployment_checkpoint(models, metadata), output_path)
            _atomic_torch_save({
                **metadata,
                "models": [model.state_dict() for model in models],
                "optimizers": [optimizer.state_dict() for optimizer in optimizers],
                "opponent_pools": [list(pool) for pool in pools],
                "rng_state": rng.bit_generator.state,
            }, state_path)
            elapsed = time.monotonic() - started
            mean_reward = float(np.mean(np.concatenate([
                rollout["rewards"] for rollout in rollouts
            ])))
            print(
                f"phase={phase + (phase_steps > 0)}/{self_play.agent_switches} "
                f"agent={learning_car} phase_steps={phase_steps} total={total_steps} "
                f"reward={mean_reward:.5g} policy={metrics['policy_loss']:.5g} "
                f"value={metrics['value_loss']:.5g} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        workers.close()
    print(f"Training complete. Runtime checkpoint: {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Torch PPO self-play agents")
    parser.add_argument("--work-dir", default="outputs/self_play_rl/berlin-dbm")
    parser.add_argument(
        "--output", default="policy/self_play_rl/weights/self_play_rl_policy.pth"
    )
    parser.add_argument("--track", default="berlin")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
