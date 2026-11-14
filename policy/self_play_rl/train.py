"""Pure-Torch PPO self-play training with one shared policy.

Alpha-RACER supplies the PPO hyperparameters and three-agent 100K-step
viewpoint-switching schedule.  The actor/critic architecture and opponent snapshot
schedule follow the referenced Thakkar self-play implementation.
"""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import multiprocessing
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
from vehicle_model import DynamicBicycleModel, oriented_rectangle_overlap_cost

from .checkpoint import SelfPlayCheckpoint
from .network import (
    HIDDEN_SIZES,
    PPOActorCritic,
)


@dataclass(frozen=True)
class PPOSettings:
    """PPO settings from Alpha-RACER Appendix E and SB3 defaults."""

    # Appendix E: learning rate, minibatch size, epochs, and discount.
    learning_rate: float = 5e-4
    adam_epsilon: float = 1e-5  # SB3 PPO default.
    batch_size: int = 1024
    epochs: int = 5
    gamma: float = 0.99
    # SB3 PPO defaults: GAE, clipping, loss weights, and gradient limit.
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5


@dataclass(frozen=True)
class SelfPlaySettings:
    """Self-play schedule and local race collection settings."""

    # Appendix E: three agents, 99 phases, 100K steps per phase.
    num_agents: int = 3
    agent_switches: int = 99
    steps_per_phase: int = 100_000
    # Rollout timing follows old local defaults; starts are a local training choice.
    rollout_steps: int = 256
    episode_steps: int = 500
    dt: float = 0.1
    initial_speed: float = 0.0
    min_start_gap: float = 2.0
    max_start_gap: float = 6.0
    # Thakkar E2E/NonLSTM self-play configuration; extended locally to three cars.
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


def _copy_policy_state(state):
    return {
        "actor_state_dict": {
            name: np.asarray(value).copy()
            for name, value in state["actor_state_dict"].items()
        },
        "log_std": np.asarray(state["log_std"]).copy(),
    }


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
        # Resample the scene if a tight bend makes the initial bodies overlap.
        for _ in range(1000):
            base_s = float(self.rng.uniform(0.0, self.track.total_length))
            slots = self.rng.permutation(num_agents)
            gaps = self.rng.uniform(
                settings["min_start_gap"], settings["max_start_gap"], num_agents - 1
            )
            offsets = np.r_[0.0, np.cumsum(gaps)]
            states, progress = [], []
            for car_idx in range(num_agents):
                start_s = base_s - float(offsets[slots[car_idx]])
                start_e = float(self.rng.uniform(-0.05, 0.05))
                x, y = self.track.frenet_to_world(start_s, start_e)
                _, _, psi = self.track.interpolate(start_s)
                states.append(CarState(
                    x=x, y=y, psi=psi, vx=settings["initial_speed"], vy=0.0, omega=0.0
                ))
                progress.append(start_s)
            if all(
                oriented_rectangle_overlap_cost(
                    states[i], states[j], params.VEHICLE_HALF_LENGTH,
                    params.VEHICLE_HALF_WIDTH,
                ) <= 0.0
                for i in range(num_agents) for j in range(i + 1, num_agents)
            ):
                break
        else:
            raise RuntimeError("Could not sample a non-overlapping self-play start")
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

    def snapshot(self):
        """Return the complete mutable race state at a rollout boundary."""
        return {
            "rng_state": self.rng.bit_generator.state,
            "step_count": self.step_count,
            "states": np.stack([state.to_array() for state in self.environment.states]),
            "progress_s": self.environment.progress_s.copy(),
            "step_index": self.environment.step_index,
            "car_s": np.asarray(self.environment.car_s, dtype=float),
            "car_e": np.asarray(self.environment.car_e, dtype=float),
            "last_controls": np.stack([
                model.last_control.copy() for model in self.environment.vehicle_models
            ]),
        }

    def restore(self, snapshot):
        """Restore a state produced by :meth:`snapshot` without resetting."""
        self.rng.bit_generator.state = snapshot["rng_state"]
        self.step_count = int(snapshot["step_count"])
        self.environment.states = [
            CarState.from_array(values) for values in snapshot["states"]
        ]
        self.environment.progress_s = np.asarray(
            snapshot["progress_s"], dtype=float
        ).copy()
        self.environment.step_index = int(snapshot["step_index"])
        self.environment.car_s = np.asarray(snapshot["car_s"], dtype=float).tolist()
        self.environment.car_e = np.asarray(snapshot["car_e"], dtype=float).tolist()
        for model, control in zip(
            self.environment.vehicle_models, snapshot["last_controls"]
        ):
            model.last_control = np.asarray(control, dtype=float).copy()

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


def _load_worker_policies(policy_states, activation):
    models = []
    for state in policy_states:
        model = PPOActorCritic(activation).cpu().eval()
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
            if command["kind"] == "snapshot":
                connection.send({
                    "race": race.snapshot(),
                    "torch_rng_state": torch.get_rng_state().numpy().copy(),
                })
                continue
            if command["kind"] == "restore":
                race.restore(command["state"]["race"])
                torch.set_rng_state(torch.from_numpy(
                    command["state"]["torch_rng_state"]
                ))
                connection.send(True)
                continue
            if command["kind"] != "collect":
                raise ValueError(f"Unknown worker command: {command['kind']}")
            if command["reset"]:
                race.reset()
            learning_car = command["learning_car"]
            models = _load_worker_policies(command["policies"], setup["activation"])
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
    """Persistent CPU workers; policy updates remain in the main process."""

    def __init__(self, count, setup):
        self.count = int(count)
        context = multiprocessing.get_context("spawn")
        self.connections, self.processes = [], []
        for worker_id in range(self.count):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_loop, args=(child, setup, worker_id), daemon=True
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def collect_rollouts(self, learning_car, policies, steps, reset):
        """Collect one on-policy batch from all persistent CPU workers."""
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

    def snapshot(self):
        """Capture every worker's race and sampling RNG state."""
        for connection in self.connections:
            connection.send({"kind": "snapshot"})
        return [connection.recv() for connection in self.connections]

    def restore(self, states):
        """Restore all workers before collecting the next rollout."""
        if len(states) != self.count:
            raise ValueError("Worker checkpoint count does not match active workers")
        for connection, state in zip(self.connections, states):
            connection.send({"kind": "restore", "state": state})
        if not all(connection.recv() for connection in self.connections):
            raise RuntimeError("A rollout worker did not restore its checkpoint")

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

    def _prepare_batch(self, model, rollouts, device):
        """Compute old-policy values, GAE advantages and return targets."""
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
        return observations, actions, old_log_probabilities, advantages, returns

    def update(self, model, optimizer, rollouts, device, rng):
        """Compute GAE and perform all PPO epochs for one rollout batch."""
        observations, actions, old_log_probabilities, advantages, returns = (
            self._prepare_batch(model, rollouts, device)
        )
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


class SelfPlayTrainer:
    """Run one complete opponent-selection, rollout and PPO-update iteration."""

    def __init__(self, model, optimizer, workers, updater, settings, device, rng):
        self.model = model
        self.optimizer = optimizer
        self.workers = workers
        self.updater = updater
        self.settings = settings
        self.device = device
        self.rng = rng
        self.opponent_pool = deque(
            [_cpu_policy_state(model)], maxlen=settings.opponent_window
        )
        self.phase = 0
        self.phase_steps = 0
        self.total_steps = 0
        self.selected_opponents = None
        self.reset_workers = True

    @property
    def complete(self):
        return self.phase >= self.settings.agent_switches

    def state_dict(self):
        """Return algorithm scheduling state for the checkpoint manager."""
        return {
            "phase": self.phase,
            "phase_steps": self.phase_steps,
            "total_steps": self.total_steps,
            "opponent_pool": [
                _copy_policy_state(state) for state in self.opponent_pool
            ],
            "selected_opponents": (
                None if self.selected_opponents is None else [
                    [_copy_policy_state(state) for state in worker_states]
                    for worker_states in self.selected_opponents
                ]
            ),
            "reset_workers": self.reset_workers,
        }

    def load_state_dict(self, state):
        """Restore scheduling and opponent-selection state."""
        self.phase = int(state["phase"])
        self.phase_steps = int(state["phase_steps"])
        self.total_steps = int(state["total_steps"])
        self.opponent_pool = deque(
            (_copy_policy_state(item) for item in state["opponent_pool"]),
            maxlen=self.settings.opponent_window,
        )
        selected = state["selected_opponents"]
        self.selected_opponents = None if selected is None else [
            [_copy_policy_state(item) for item in worker_states]
            for worker_states in selected
        ]
        self.reset_workers = bool(state["reset_workers"])

    def _choose_opponents(self, learning_car):
        selections = []
        for _ in range(self.workers.count):
            worker_states = []
            for car_idx in range(self.settings.num_agents):
                if car_idx == learning_car:
                    worker_states.append(_cpu_policy_state(self.model))
                elif (
                    not self.opponent_pool
                    or self.rng.random() < self.settings.latest_opponent_probability
                ):
                    worker_states.append(_cpu_policy_state(self.model))
                else:
                    selected = self.opponent_pool[
                        int(self.rng.integers(0, len(self.opponent_pool)))
                    ]
                    worker_states.append(_copy_policy_state(selected))
            selections.append(worker_states)
        return selections

    def run_update(self):
        """Collect one rollout batch, update PPO, and advance self-play scheduling."""
        learning_car = self.phase % self.settings.num_agents
        next_swap = (
            (self.phase_steps // self.settings.opponent_swap_steps) + 1
        ) * self.settings.opponent_swap_steps
        next_snapshot = (
            (self.phase_steps // self.settings.snapshot_steps) + 1
        ) * self.settings.snapshot_steps
        boundary = min(self.settings.steps_per_phase, next_swap, next_snapshot)
        if self.selected_opponents is None:
            self.selected_opponents = self._choose_opponents(learning_car)
        for worker_states in self.selected_opponents:
            worker_states[learning_car] = _cpu_policy_state(self.model)

        total = min(
            boundary - self.phase_steps,
            self.workers.count * self.settings.rollout_steps,
        )
        base, remainder = divmod(total, self.workers.count)
        steps = [
            base + (worker < remainder) for worker in range(self.workers.count)
        ]
        rollouts = self.workers.collect_rollouts(
            learning_car, self.selected_opponents, steps, self.reset_workers
        )
        self.reset_workers = False
        metrics = self.updater.update(
            self.model, self.optimizer, rollouts, self.device, self.rng
        )
        collected = sum(steps)
        self.phase_steps += collected
        self.total_steps += collected

        if self.phase_steps % self.settings.snapshot_steps == 0:
            self.opponent_pool.append(_cpu_policy_state(self.model))
        if self.phase_steps % self.settings.opponent_swap_steps == 0:
            self.selected_opponents = None
        if self.phase_steps == self.settings.steps_per_phase:
            self.phase += 1
            self.phase_steps = 0
            self.selected_opponents = None
            self.reset_workers = True

        mean_reward = float(np.mean(np.concatenate([
            rollout["rewards"] for rollout in rollouts
        ])))
        return learning_car, mean_reward, metrics


def _git_metadata():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
    }


def train(args):
    """Train one policy while rotating the car that supplies its rollouts."""
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

    model = PPOActorCritic(args.activation).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo.learning_rate, eps=ppo.adam_epsilon
    )
    setup = {
        "seed": args.seed,
        "activation": args.activation,
        "track_csv": params.TRACK_CSV,
        "track_scale": params.TRACK_SCALE,
        "self_play": asdict(self_play),
    }
    workers = RolloutWorkers(args.workers, setup)
    trainer = SelfPlayTrainer(
        model, optimizer, workers, PPOUpdater(ppo), self_play, device, rng
    )
    checkpoint = SelfPlayCheckpoint(
        args.work_dir, args.output, asdict(ppo), trainer
    )
    started = time.monotonic()
    git = _git_metadata()
    try:
        resumed = checkpoint.load()
        if resumed:
            print(
                f"Resuming phase={trainer.phase} phase_steps={trainer.phase_steps} "
                f"total_steps={trainer.total_steps}"
            )

        while not trainer.complete:
            learning_car, mean_reward, metrics = trainer.run_update()

            metadata = {
                "ppo_settings": asdict(ppo),
                "self_play_settings": asdict(self_play),
                "network": {"hidden_sizes": HIDDEN_SIZES, "activation": model.activation},
                "phase": trainer.phase,
                "phase_steps": trainer.phase_steps,
                "total_steps": trainer.total_steps,
                "track": args.track,
                "seed": args.seed,
                "workers": args.workers,
                "device": str(device),
                "shared_policy": True,
                "sources": "alpha-RACER Appendix E; Thakkar et al. self-play",
                "git": git,
            }
            checkpoint.save(metadata)
            elapsed = time.monotonic() - started
            print(
                f"phase={trainer.phase + (trainer.phase_steps > 0)}/"
                f"{self_play.agent_switches} agent={learning_car} "
                f"phase_steps={trainer.phase_steps} total={trainer.total_steps} "
                f"reward={mean_reward:.5g} policy={metrics['policy_loss']:.5g} "
                f"value={metrics['value_loss']:.5g} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        workers.close()
    print(f"Training complete. Runtime checkpoint: {args.output}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train one shared Torch PPO policy")
    parser.add_argument("--work-dir", default="outputs/self_play_rl/berlin-dbm")
    parser.add_argument(
        "--output", default="policy/self_play_rl/weights/self_play_rl_policy.pth"
    )
    parser.add_argument("--track", default="berlin")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="parallel CPU rollout environments (default: 8)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="single training device: auto, cpu, cuda, cuda:0, ... (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
