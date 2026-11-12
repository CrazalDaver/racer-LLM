"""End-to-end local alpha-RACER potential training.

The pipeline has four stages:
  1. collect_rollout_data: simulate a chain of unilateral parameter changes;
  2. train_value_functions: fit own-progress initialization and relative-progress models;
  3. train_potential_function: match unilateral value differences;
  4. evaluate_test_set: report errors on held-out races.

Run this module from the repository root. Existing stage outputs are reused
unless the matching --force-* option is provided.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import os
from contextlib import nullcontext
import json
import multiprocessing
import tempfile
from multiprocessing.managers import SyncManager
import random
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import duration_to_steps, params
from policy.alpharacer.network import (
    ALPHA_INPUT_SIZE,
    ALPHA_VALUE_HIDDEN_SIZES,
    SimpleNeuralNetwork,
    build_alpha_input,
)
from policy.utils import relative_progress_gain
from simulation import run_simulation
from simulation_core.config import SimulationConfig
from track import Track, configure_track, load_centerline_csv


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz_save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def load_training_state(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def random_state():
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_random_state(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA training checkpoint requires CUDA to resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def model_directory(args, utility, player):
    return args.value_dir / ("own" if utility == "own_progress" else "relative") / f"car{player}"


def model_paths(args, utility):
    return [model_directory(args, utility, player) / "model.pth" for player in range(3)]


def collection_path(args, chain):
    return args.data.parent / "chains" / f"chain_{chain['id']:03d}" / "rollouts.npz"


def value_settings(args, utility, player):
    with np.load(args.data, allow_pickle=False) as data:
        collection = json.loads(str(data["metadata"]))["collection_settings"]
    return dict(utility=utility, player=player, seed=args.seed, epochs=args.value_epochs,
        collection=collection,
        learning_rate=args.value_learning_rate, batch_size=args.value_batch_size,
        patience=getattr(args, "early_stopping_patience", 0),
        min_delta=getattr(args, "early_stopping_min_delta", 0.0))


class ModelArtifact:
    """model.pth means completed; checkpoint.pth means training can continue."""
    def __init__(self, directory, settings, device):
        self.directory = Path(directory)
        self.model = self.directory / "model.pth"
        self.checkpoint = self.directory / "checkpoint.pth"
        self.settings = settings
        self.device = str(device)

    def complete(self):
        if not self.model.exists():
            return False
        payload = load_training_state(self.model)
        metadata = payload.get("metadata", {})
        completed = metadata.get("completed", False)
        if not completed and self.checkpoint.exists():
            state = load_training_state(self.checkpoint)
            completed = bool(state.get("completed")) and metadata.get("selection") != "shared_epoch_from_car0"
            if completed and state.get("max_epochs") != self.settings.get("epochs"):
                raise ValueError(f"Legacy model epoch limit differs: {self.model}")
        if not completed:
            raise ValueError(f"Model is not marked complete: {self.model}")
        if self.settings.get("utility") and (metadata.get("utility") != self.settings["utility"]
                or metadata.get("player") != self.settings["player"]):
            raise ValueError(f"Model belongs to a different car or utility: {self.model}")
        if self.settings.get("utility") and metadata.get("collection_settings") != self.settings["collection"]:
            raise ValueError(f"Model uses different rollout settings: {self.model}")
        if metadata.get("settings") and metadata["settings"] != self.settings:
            raise ValueError(f"Incompatible model settings: {self.model}")
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1) if self.settings.get("utility") else SimpleNeuralNetwork()
        model.load_state_dict(payload["state_dict"])
        for value in model.state_dict().values():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite model weights: {self.model}")
        return True

    def load_checkpoint(self):
        if not self.checkpoint.exists():
            return None
        state = load_training_state(self.checkpoint, self.device)
        if state.get("settings") and state["settings"] != self.settings:
            raise ValueError(f"Training checkpoint settings differ: {self.checkpoint}")
        return state

    def save_checkpoint(self, state):
        atomic_torch_save({**state, "settings": self.settings}, self.checkpoint)

    def publish(self, weights, metadata):
        atomic_torch_save(dict(format_version=1, kind="alpha_potential" if self.directory.name == "potential"
            else "alpha_value", state_dict=weights,
            metadata={**metadata, "completed": True, "settings": self.settings}), self.model)


class TrainingProgress:
    """Early stopping and checkpoint state, independent for each fitted network."""
    def __init__(self, artifact, model, optimizer, patience, min_delta, select_last=False):
        self.artifact, self.model, self.optimizer = artifact, model, optimizer
        self.patience, self.min_delta, self.select_last = patience, min_delta, select_last
        self.epoch, self.best_epoch, self.bad_epochs = 0, 0, 0
        self.best_loss = self.patience_best = float("inf")
        self.best_weights = None
        self.best_details = {}
        self.completed = False
        self.import_history = None
        state = artifact.load_checkpoint()
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            restore_random_state(state["random_state"])
            if "next_epoch" in state:
                if state.get("completed"):
                    raise ValueError("Completed legacy checkpoint needs its corresponding model.pth")
                expected = {"patience": patience, "min_delta": min_delta}
                if state.get("early_stopping") != expected:
                    raise ValueError("Legacy checkpoint early-stopping settings differ")
                if state["optimizer"]["param_groups"][0]["lr"] != artifact.settings["learning_rate"]:
                    raise ValueError("Legacy checkpoint learning rate differs")
                self.epoch = state["next_epoch"]
                self.import_history = "Legacy checkpoint: early-stopping history reset; earlier best weights unavailable"
                print(f"  {self.import_history}", flush=True)
            else:
                for key in ("epoch", "best_epoch", "bad_epochs", "best_loss", "patience_best",
                            "best_weights", "completed"):
                    setattr(self, key, state[key])
                self.import_history = state.get("import_history")
            self.best_details = state.get("best_details", {})
            print(f"  resume {artifact.directory}: epoch={self.epoch}", flush=True)

    def observe(self, epoch, loss, details=None):
        if not np.isfinite(loss):
            raise ValueError(f"Nonfinite validation MSE: {self.artifact.directory}")
        self.epoch = epoch
        if loss < self.best_loss or self.select_last:
            self.best_loss, self.best_epoch = loss, epoch
            self.best_weights = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            self.best_details = details or {}
        if loss < self.patience_best - self.min_delta:
            self.patience_best, self.bad_epochs = loss, 0
        else:
            self.bad_epochs += 1
        return bool(self.patience and self.bad_epochs >= self.patience)

    def save(self, completed=False):
        self.completed = completed
        state = {key: getattr(self, key) for key in ("epoch", "best_epoch", "bad_epochs",
                 "best_loss", "patience_best", "best_weights", "best_details", "completed")}
        self.artifact.save_checkpoint({**state, "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "random_state": random_state(),
            "import_history": self.import_history})

    def publish(self, metadata):
        if self.best_weights is None:
            raise ValueError(f"No selected model to publish: {self.artifact.directory}")
        self.artifact.publish(self.best_weights, {**metadata, "selected_epoch": self.best_epoch,
            "stopped_epoch": self.epoch, "validation_mse": self.best_loss,
            **self.best_details, "import_history": self.import_history})



PARAMETERIZATION = "paper_q_zeta_s1_s2_s3"
REWARD_DEFINITION = "relative_progress_v1"
DATA_FORMAT_VERSION = 4
STAGE_LAYOUT_VERSION = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def training_device(args):
    requested = getattr(args, "device", "auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def checkpoint_payload(model, *, kind, metadata):
    return {
        "format_version": 1,
        "kind": kind,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }


def load_checkpoint_state(path, device):
    payload = torch.load(path, map_location=device)
    return payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload


def trace_features(trace, track, joint_params, mpc_horizon):
    update_indices = np.flatnonzero(np.asarray(trace["control_update"], dtype=bool))
    rows = []
    for step in update_indices:
        observed = trace.get("car_observed_state")
        cars = [
            {
                "px": trace["car_px"][step][idx],
                "py": trace["car_py"][step][idx],
                "psi": trace["car_psi"][step][idx],
                "vx": trace["car_vx"][step][idx],
                "vy": trace["car_vy"][step][idx],
                "omega": trace["car_omega"][step][idx],
            }
            for idx in range(3)
        ]
        if observed is not None:
            cars = [dict(zip(("px", "py", "psi", "vx", "vy", "omega"), car))
                    for car in observed[step]]
        rows.append(build_alpha_input(track, params.MPC_DT, mpc_horizon, cars, joint_params))
    return np.asarray(rows, dtype=np.float32), update_indices


def trace_rewards(trace, update_indices):
    """Single-step reward: relative progress gain as delta(s_i - max_{j != i} s_j)."""
    progress = np.asarray(trace["car_px"], dtype=np.float64)
    next_progress = np.asarray(trace["car_next_px"], dtype=np.float64)
    assert progress.ndim == 2 and progress.shape[1] == 3
    assert next_progress.shape == progress.shape
    if not np.isfinite(progress).all() or not np.isfinite(next_progress).all():
        raise ValueError("Nonfinite progress in alpha-RACER reward data")
    # Sum the physical transitions belonging to each control update, including the final step.
    step_rewards = np.column_stack([
        relative_progress_gain(progress, next_progress, player)
        for player in range(3)
    ])
    ends = np.r_[update_indices[1:], len(progress)]
    return np.asarray([
        step_rewards[start:end].sum(axis=0)
        for start, end in zip(update_indices, ends)
    ], dtype=np.float32).reshape(-1, 3)


def discounted_returns(rewards, gamma, window):
    """Official fixed-S target; exclude starts without a complete future window."""
    assert 0 < window < len(rewards)
    returns = np.zeros((len(rewards)-window, rewards.shape[-1]), dtype=np.float32)
    for offset in range(window):
        returns += gamma**offset * rewards[offset:len(rewards)-window+offset]
    return returns


def sample_upstream_params(rng, count=1):
    """Official collection distribution, converted to the local theta API."""
    raw = rng.uniform([.1, .1, .12, .85, 0.], [.5, .5, .5, 1.1, 1.], size=(count, 5))
    return np.column_stack((.5 / raw[:, 2], raw[:, 3] ** 2, raw[:, :2], raw[:, 4]))


def split_races(count, validation_fraction, test_fraction):
    """--races counts training races; collect validation/test races in addition."""
    if count < 2:
        raise ValueError("Need at least two training races")
    train_fraction = 1-validation_fraction-test_fraction
    validation_count = max(2, int(round(count * validation_fraction/train_fraction)))
    test_count = max(2, int(round(count * test_fraction/train_fraction)))
    ids = np.arange(count+validation_count+test_count)
    return dict(train=ids[:count], validation=ids[count:count+validation_count],
                test=ids[count+validation_count:])


def collection_chains(splits, length, seed):
    """Independent seeded chains, kept entirely within one dataset split."""
    chains = []
    for split, ids in splits.items():
        for offset in range(0, len(ids), length):
            block = ids[offset:offset+length]
            chains.append(dict(id=len(chains), split=split, start=int(block[0]), end=int(block[-1])+1))
    seeds = np.random.SeedSequence(seed).spawn(len(chains))
    for chain, child in zip(chains, seeds):
        chain["seed"] = int(child.generate_state(1)[0])
    return chains


def potential_pairs(features, deviators, race_ids):
    """Official adjacent-race pairing: previous state, previous/current theta."""
    selected = np.zeros(len(features), dtype=bool)
    selected[race_ids] = True
    current = np.flatnonzero(selected[1:] & selected[:-1]) + 1
    current = current[np.asarray(deviators)[current] >= 0]  # Never pair across a chain restart.
    assert len(current) > 0
    left = features[current - 1].copy()
    right = left.copy()
    right[..., -15:] = features[current, :, -15:]
    players = np.repeat(deviators[current], features.shape[1])
    assert np.all((players >= 0) & (players < 3))
    changes = (right[..., -15:] - left[..., -15:]).reshape(len(current), -1, 3, 5)
    for row, race in enumerate(current):
        others = [idx for idx in range(3) if idx != deviators[race]]
        assert np.all(changes[row, :, others] == 0)
    return left.reshape(-1, ALPHA_INPUT_SIZE), right.reshape(-1, ALPHA_INPUT_SIZE), players


def predict_batches(model, rows, device, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([
            model(torch.from_numpy(rows[start:start+batch_size]).float().to(device)).cpu()
            for start in range(0, len(rows), batch_size)
        ]).numpy()


def initialize_potential(value_models, device):
    """Sum the value branches exactly after rebasing their input BatchNorms."""
    potential = SimpleNeuralNetwork().to(device)
    with torch.no_grad():
        for layer in potential.fc:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        for player, value in enumerate(value_models):
            for index in (0, 2, 4):
                source, destination = value.fc[index], potential.fc[index]
                rows = slice(player*source.out_features, (player+1)*source.out_features)
                columns = (slice(None) if index == 0 else
                           slice(player*source.in_features, (player+1)*source.in_features))
                destination.weight[rows, columns].copy_(source.weight)
                destination.bias[rows].copy_(source.bias)
                if index == 0:
                    # Express each branch's own BN in the common first-value BN coordinates.
                    common, original = value_models[0].batch_norm, value.batch_norm
                    assert not common.affine and not original.affine
                    common_std = torch.sqrt(common.running_var + common.eps)
                    original_std = torch.sqrt(original.running_var + original.eps)
                    destination.weight[rows].mul_(common_std / original_std)
                    destination.bias[rows].add_(source.weight @ (
                        (common.running_mean - original.running_mean) / original_std))
            width = value.fc[6].in_features
            potential.fc[6].weight[:, player*width:(player+1)*width].copy_(value.fc[6].weight)
            potential.fc[6].bias.add_(value.fc[6].bias)
        potential.batch_norm.load_state_dict(value_models[0].batch_norm.state_dict())
    potential.batch_norm.momentum = 0.
    potential.eval()
    return potential


# ---------------------------------------------------------------------------
# Stage 1: collect rollout data
# ---------------------------------------------------------------------------

def _collect_parallel_chain(task):
    args, chain, attempt_budget = task
    collect_rollout_data(args, _chain=chain, _attempt_budget=attempt_budget)
    return args.data


def collect_rollout_data(args, _chain=None, _attempt_budget=None):
    settings = dict(track=args.track, vehicle_model=args.vehicle_model,
                    horizon=args.horizon, duration=args.duration, gamma=args.gamma,
                    return_window=args.return_window,
                    seed=args.seed, races=args.races, initial_speed=args.initial_speed,
                    chain_length=args.batch_size,
                    validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
    if args.data.exists() and not args.force_rollouts:
        with np.load(args.data, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            count = sum(metadata["split_counts"].values())
            if data["features"].ndim != 3 or data["features"].shape[0] != count or data["features"].shape[-1] != ALPHA_INPUT_SIZE:
                raise ValueError("Rollout features have an unexpected shape")
            if data["returns"].shape != data["own_returns"].shape or data["returns"].shape != (*data["features"].shape[:2], 3):
                raise ValueError("Rollout targets have an unexpected shape")
        if (metadata.get("reward_definition") != REWARD_DEFINITION
                or metadata.get("format_version") != DATA_FORMAT_VERSION
                or metadata.get("collection_settings") != settings):
            raise ValueError("Rollout settings or format changed; use --force-rollouts to rebuild data and models")
        print(f"[stage 1/4] Reusing rollout data: {args.data}")
        return args.data

    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    splits = split_races(args.races, args.validation_fraction, args.test_fraction)
    chains = collection_chains(splits, args.batch_size, args.seed)
    total_races = sum(len(ids) for ids in splits.values())
    max_attempts = args.max_attempts or 3*total_races
    if _chain is None and (getattr(args, "n_workers", 1) > 1 or getattr(args, "resumable", False)):
        return _collect_parallel_rollouts(args, settings, splits, chains, max_attempts)
    if _chain is not None:
        length = _chain["end"] - _chain["start"]
        chains = [{**_chain, "start": 0, "end": length}]
        total_races = length
        splits = {name: np.arange(length) if name == _chain["split"] else np.array([], dtype=int)
                  for name in splits}
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    gap_diagnostics = dict(track_length_m=track.total_length, accepted_episodes=0,
        sampled_states=0, max_progress_spread_m=0.,
        episodes_at_least_half_lap=0, states_at_least_half_lap=0,
        episodes_at_least_one_lap=0, states_at_least_one_lap=0)
    all_features, all_returns, own_returns, all_params, all_deviators, starts = [], [], [], [], [], []
    chain_ids, chain_files = [], []
    # Literal initial parameters from official multi_car_blocking.py.
    initial_params = np.array([[.5/.24, 1., .2, .2, .2],
                             [.5/.15, 1., .1, .5, .2],
                             [.5/.15, 1., .1, .5, .2]])
    expected_steps = duration_to_steps(args.duration, params.SIM_DT)
    if not 0 < args.return_window < expected_steps:
        raise ValueError("--return-window must be positive and shorter than the race")
    args.data.parent.mkdir(parents=True, exist_ok=True)

    print("[stage 1/4] Collecting a chain of single-player parameter changes")
    attempts, failed_attempts = 0, []
    collection_id = uuid.uuid4().hex[:12]
    chain_index, active_chain = 0, None
    checkpoint = args.data.parent / "checkpoint.pth" if _chain is not None else args.data.with_suffix(".collection-state.pth")
    if _chain is not None and getattr(args, "resumable", False) and checkpoint.exists():
        state = load_training_state(checkpoint)
        if state["settings"] != settings or state["chain"] != _chain:
            raise ValueError("Saved collection checkpoint does not match this chain or settings")
        (all_features, all_returns, own_returns, all_params, all_deviators, starts,
         chain_ids, failed_attempts, gap_diagnostics, attempts, chain_files) = state["data"]
        collection_id, chain_attempts = state["collection_id"], state["chain_attempts"]
        active_chain = _chain["id"]
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        accepted_params = state["accepted_params"]
        config = SimulationConfig(policies=["static_alpharacer"] * 3,
            static_params_by_car=accepted_params.copy(), track_name=args.track,
            vehicle_model=args.vehicle_model, mpc_horizon=args.horizon,
            initial_speed=args.initial_speed, ego_position=starts[-1] if starts else 1,
            duration=args.duration, save_trace=False, show_plot=False, quiet=True)
        print(f"  resumed chain {_chain['id']} at {len(all_features)}/{total_races}", flush=True)
    while len(all_features) < total_races:
        if (attempts >= max_attempts
                or (_attempt_budget is not None and not _attempt_budget.acquire(False))):
            raise RuntimeError(f"Collection attempt limit reached: {len(all_features)}/{total_races} valid races; see saved failure diagnostics")
        race = len(all_features)
        if race == chains[chain_index]["end"]:
            chain_index += 1
        chain = chains[chain_index]
        if active_chain != chain["id"]:
            active_chain = chain["id"]
            rng = np.random.default_rng(chain["seed"])
            accepted_params = initial_params.copy()
            chain_attempts = 0
        local_race = race-chain["start"]
        attempts += 1
        chain_attempts += 1
        player = -1 if local_race == 0 else local_race % 3
        joint_params = accepted_params.copy()
        if local_race:
            joint_params[player] = sample_upstream_params(rng)[0]
        elif chain_attempts > 1:
            joint_params = sample_upstream_params(rng, 3)
        start_rank = int(rng.integers(1, 4))
        config = SimulationConfig(
            policies=["static_alpharacer"] * 3,
            static_params_by_car=joint_params.copy(),
            track_name=args.track,
            vehicle_model=args.vehicle_model,
            mpc_horizon=args.horizon,
            initial_speed=args.initial_speed,
            ego_position=start_rank,
            duration=args.duration,
            save_trace=False, show_plot=False, quiet=True,
        )
        trace = run_simulation(config)
        completed = len(trace["control_update"])
        reason = trace.get("failure")
        if completed != expected_steps and reason is None:
            reason = dict(error="incomplete_episode")
        for field in ("car_px", "car_py", "car_psi", "car_vx", "car_vy", "car_omega",
                      "car_observed_state", "car_next_px"):
            if reason is None and not np.isfinite(np.asarray(trace[field], dtype=float)).all():
                reason = dict(error=f"nonfinite_{field}")
        if reason is not None:
            failure_path = args.data.with_name(f"{args.data.stem}.{collection_id}.failure-{attempts:06d}.json")
            failure_path.write_text(json.dumps(dict(
                attempt=attempts, accepted_before=race + (_chain["start"] if _chain else 0),
                completed_steps=completed, expected_steps=expected_steps,
                chain=_chain or chain,
                parameters=joint_params.tolist(), ego_position=start_rank,
                settings=settings, failure=reason, trace=trace,
            ), indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + "\n")
            failed_attempts.append(str(failure_path))
            print(f"  rejected attempt {attempts}: {reason}; saved {failure_path}")
            if _chain is not None and getattr(args, "resumable", False):
                atomic_torch_save(dict(settings=settings, chain=_chain, collection_id=collection_id,
                    chain_attempts=chain_attempts, rng_state=rng.bit_generator.state,
                    accepted_params=accepted_params, data=(all_features, all_returns, own_returns,
                    all_params, all_deviators, starts, chain_ids, failed_attempts, gap_diagnostics, attempts, chain_files)), checkpoint)
            continue
        features, update_indices = trace_features(trace, track, joint_params, args.horizon)
        rewards = trace_rewards(trace, update_indices)
        if not np.isfinite(features).all() or not np.isfinite(rewards).all():
            raise ValueError(f"Nonfinite rollout data in race {race}")
        own_rewards = np.asarray(trace["car_next_px"]) - np.asarray(trace["car_px"])
        assert np.array_equal(update_indices, np.arange(expected_steps))
        relative_targets = discounted_returns(rewards, args.gamma, args.return_window)
        own_returns.append(discounted_returns(own_rewards, args.gamma, args.return_window))
        all_features.append(features[:len(relative_targets)])
        all_returns.append(relative_targets)
        all_params.append(joint_params.copy())
        all_deviators.append(player)
        progress_samples = np.vstack((trace["car_px"], np.asarray(trace["car_next_px"])[-1:]))
        spread = np.ptp(progress_samples, axis=1)
        half_lap, one_lap = spread >= .5*track.total_length, spread >= track.total_length
        gap_diagnostics["accepted_episodes"] += 1
        gap_diagnostics["sampled_states"] += len(spread)
        gap_diagnostics["max_progress_spread_m"] = max(gap_diagnostics["max_progress_spread_m"], float(spread.max()))
        gap_diagnostics["episodes_at_least_half_lap"] += int(half_lap.any())
        gap_diagnostics["states_at_least_half_lap"] += int(half_lap.sum())
        gap_diagnostics["episodes_at_least_one_lap"] += int(one_lap.any())
        gap_diagnostics["states_at_least_one_lap"] += int(one_lap.sum())
        chain_ids.append(chain["id"])
        starts.append(start_rank)
        accepted_params = joint_params.copy()
        print(f"  accepted {race + 1}/{total_races}, chain={chain['id']}, changed_car={player}, samples={len(relative_targets)}")
        if race+1 == chain["end"]:
            path = args.data if _chain is not None else args.data.with_name(f"{args.data.stem}.{collection_id}.chain-{chain['id']:03d}.npz")
            section = slice(chain["start"], chain["end"])
            if _chain is None:
                atomic_npz_save(path, features=np.stack(all_features[section]),
                    returns=np.stack(all_returns[section]), own_returns=np.stack(own_returns[section]),
                    joint_params=np.stack(all_params[section]),
                    deviating_player=np.asarray(all_deviators[section]),
                    metadata=np.asarray(json.dumps(dict(chain=chain, settings=settings))))
            chain_files.append(str(path))
        if _chain is not None and getattr(args, "resumable", False):
            atomic_torch_save(dict(settings=settings, chain=_chain, collection_id=collection_id,
                chain_attempts=chain_attempts, rng_state=rng.bit_generator.state,
                accepted_params=accepted_params, data=(all_features, all_returns, own_returns,
                all_params, all_deviators, starts, chain_ids, failed_attempts, gap_diagnostics, attempts, chain_files)), checkpoint)

    source_paths = [Path(__file__), Path("policy/alpharacer/network.py"), Path("config.py"),
                    Path(params.TRACK_CSV), Path(params.RACELINE_CSV)]
    metadata = {
        "format_version": DATA_FORMAT_VERSION,
        "parameterization": PARAMETERIZATION,
        "parameter_order": list(params.ALPHARACER_PARAMETER_NAMES),
        "reward_definition": REWARD_DEFINITION,
        "collection_settings": settings,
        "sampling": "upstream_transformed_uniform_independent_round_robin_chains",
        "chains": chains,
        "chain_files": chain_files,
        "source_asset_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_paths},
        "start_ego_positions": starts,
        "attempts": attempts,
        "collection_id": collection_id,
        "failed_attempts": failed_attempts,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "progress_gap_diagnostics": gap_diagnostics,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "effective_simulation_config": {key: str(value) if isinstance(value, Path) else value
                                        for key, value in vars(config).items()
                                        if key not in {"static_params_by_car", "ego_position"}},
        "effective_params": {key: value for key, value in vars(params).items()
                             if not key.startswith("_") and not callable(value)},
    }
    atomic_npz_save(
        args.data, features=np.stack(all_features), returns=np.stack(all_returns),
        own_returns=np.stack(own_returns),
        joint_params=np.stack(all_params), deviating_player=np.asarray(all_deviators),
        chain_id=np.asarray(chain_ids),
        **{f"{name}_ids": ids for name, ids in splits.items()},
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
    )
    print(f"  saved {args.data}")
    return args.data


def _collect_parallel_rollouts(args, settings, splits, chains, max_attempts):
    """Run existing serial collection independently for each seeded chain."""
    workers = min(args.n_workers, len(chains))
    collection_id = "resume" if getattr(args, "resumable", False) else uuid.uuid4().hex[:12]
    args.data.parent.mkdir(parents=True, exist_ok=True)
    print(f"[stage 1/4] Collecting {len(chains)} independent chains with {workers} workers", flush=True)
    context = multiprocessing.get_context("spawn")
    previous_attempts = 0
    if getattr(args, "resumable", False):
        for chain in chains:
            path = collection_path(args, chain)
            state_path = path.parent / "checkpoint.pth"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    previous_attempts += json.loads(str(data["metadata"]))["attempts"]
            elif state_path.exists():
                previous_attempts += load_training_state(state_path)["data"][-2]
    if previous_attempts > max_attempts:
        raise ValueError("Saved attempts exceed --max-attempts")
    # A short explicit socket path also works under long shared-filesystem paths.
    with tempfile.TemporaryDirectory(prefix="alpha-", dir="/tmp") as ipc_directory, SyncManager(
            address=str(Path(ipc_directory) / "manager.sock"), ctx=context) as manager:
        attempt_budget = manager.BoundedSemaphore(max_attempts - previous_attempts)
        tasks = []
        for chain in chains:
            worker_args = copy.copy(args)
            worker_args.data = collection_path(args, chain)
            worker_args.max_attempts = max_attempts
            if getattr(args, "resumable", False):
                worker_args.force_rollouts = False
            tasks.append((worker_args, chain, attempt_budget))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            paths = list(executor.map(_collect_parallel_chain, tasks))

    arrays = {name: [] for name in ("features", "returns", "own_returns", "joint_params",
                                     "deviating_player", "chain_id")}
    metadata = None
    attempts, failures, starts, chain_files = 0, [], [], []
    gap = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            current = json.loads(str(data["metadata"]))
            for name in arrays:
                arrays[name].append(data[name])
        if metadata is None:
            metadata = current
            gap = current["progress_gap_diagnostics"].copy()
        else:
            current_gap = current["progress_gap_diagnostics"]
            for key in gap:
                if key == "max_progress_spread_m":
                    gap[key] = max(gap[key], current_gap[key])
                elif key != "track_length_m":
                    gap[key] += current_gap[key]
        attempts += current["attempts"]
        failures.extend(current["failed_attempts"])
        starts.extend(current["start_ego_positions"])
        chain_files.extend(current["chain_files"])
    metadata.update(collection_settings=settings, chains=chains, chain_files=chain_files,
                    start_ego_positions=starts, attempts=attempts, collection_id=collection_id,
                    failed_attempts=failures, split_counts={name: len(ids) for name, ids in splits.items()},
                    progress_gap_diagnostics=gap, collection_workers=workers)
    atomic_npz_save(args.data, **{name: np.concatenate(parts) for name, parts in arrays.items()},
                        **{f"{name}_ids": ids for name, ids in splits.items()},
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=str)))
    if not getattr(args, "resumable", False):
        for path in paths:
            path.unlink()
    print(f"  saved {args.data}")
    return args.data


# ---------------------------------------------------------------------------
# Stage 2: train the three Monte-Carlo value functions
# ---------------------------------------------------------------------------

def train_value_functions(args, utility="relative"):
    print(f"[stage values/{utility}] Checking each car independently", flush=True)
    device = training_device(args)
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        targets = data["returns" if utility == "relative" else "own_returns"]
        dataset_metadata = json.loads(str(data["metadata"]))
        train_ids, val_ids = data["train_ids"], data["validation_ids"]
    output_paths = model_paths(args, utility)
    for player, output_path in enumerate(output_paths):
        artifact = ModelArtifact(output_path.parent, value_settings(args, utility, player), device)
        if artifact.complete():
            print(f"  reuse completed {output_path}", flush=True)
            continue
        seed_everything(args.seed + player + (0 if utility == "own_progress" else 1000))
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.value_learning_rate)
        progress = TrainingProgress(artifact, model, optimizer,
            getattr(args, "early_stopping_patience", 0), getattr(args, "early_stopping_min_delta", 0.0),
            select_last=utility == "own_progress" and not getattr(args, "early_stopping_patience", 0))
        metadata = {**dataset_metadata, "utility": utility, "player": player,
            "training_provenance": training_provenance(),
            "normalization": "official_value_running_batchnorm", "selection": "final_epoch"
            if progress.select_last else "validation_mse", "target_filter": "official_-100_le_return_lt_600"}
        x_train = torch.from_numpy(features[train_ids]).float()
        y_train = torch.from_numpy(targets[train_ids, ..., player, None]).float()
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.value_batch_size, shuffle=False)
        if len(x_train) < 2:
            raise ValueError("Need at least two value training samples for BatchNorm")
        loss_fn = nn.MSELoss()
        x_val = features[val_ids].reshape(-1, ALPHA_INPUT_SIZE)
        y_val = targets[val_ids, ..., player].reshape(-1, 1)
        val_mask = (y_val >= -100.) & (y_val < 600.)
        if progress.import_history and progress.best_weights is None:
            score = float(np.mean(((predict_batches(model, x_val, device, args.batch_size) - y_val)*val_mask) ** 2))
            progress.observe(progress.epoch, score)
        for epoch in range(progress.epoch, args.value_epochs):
            if progress.completed:
                break
            model.train()
            for x_batch, y_batch in loader:
                y_batch = y_batch.to(device)
                mask = ((y_batch >= -100.) & (y_batch < 600.)).float()
                loss = loss_fn(model(x_batch.to(device))*mask, y_batch*mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            score = float(np.mean(((predict_batches(model, x_val, device, args.batch_size) - y_val)*val_mask) ** 2))
            stopped = progress.observe(epoch + 1, score)
            completed = stopped or epoch + 1 == args.value_epochs
            if (epoch + 1) % 10 == 0 or completed:
                progress.save(completed)
            if epoch == 0 or (epoch + 1) % 10 == 0 or completed:
                print(f"  {utility} car={player} epoch={epoch + 1}/{args.value_epochs} validation_mse={score:.6g}", flush=True)
            if completed:
                print(f"  {'early stopping' if stopped else 'completed'} car={player}, best_epoch={progress.best_epoch}", flush=True)
                break
        progress.publish(metadata)
        print(f"  saved {output_path}", flush=True)
    return output_paths


# ---------------------------------------------------------------------------
# Stage 3: learn the common potential from unilateral value differences
# ---------------------------------------------------------------------------

def potential_settings(args, value_paths, initialization_paths):
    return dict(epochs=args.potential_epochs, learning_rate=args.potential_learning_rate,
        batch_size=args.batch_size, seed=args.seed, initialization_seed=args.seed + 2000,
        patience=getattr(args, "early_stopping_patience", 0),
        min_delta=getattr(args, "early_stopping_min_delta", 0.0))


def train_potential_function(args, value_paths, initialization_paths):
    device = training_device(args)
    artifact = ModelArtifact(args.potential_output.parent,
        potential_settings(args, value_paths, initialization_paths), device)
    if artifact.complete():
        print(f"[stage potential] Reusing {artifact.model}", flush=True)
        return artifact.model
    print("[stage potential] Training the common alpha-potential", flush=True)
    seed_everything(args.seed + 2000)
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        metadata = json.loads(str(data["metadata"]))
        deviators = data["deviating_player"]
        train_ids, val_ids = data["train_ids"], data["validation_ids"]
    left, right, players = potential_pairs(features, deviators, train_ids)
    val_left, val_right, val_players = potential_pairs(features, deviators, val_ids)
    if len(left)//features.shape[1] < args.batch_size-1:
        raise ValueError("Not enough training pairs for one full potential batch")
    training_pair_count = len(left)//features.shape[1]
    complete_ids = train_ids[:len(train_ids)//args.batch_size*args.batch_size].reshape(-1,args.batch_size)
    batches = [potential_pairs(features, deviators, ids) for ids in complete_ids]
    left, right, players = (np.concatenate(parts) for parts in zip(*batches))
    value_models = []
    for path in value_paths:
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        model.load_state_dict(load_checkpoint_state(path, device))
        model.eval()
        value_models.append(model)
    def value_differences(left_rows, right_rows, player_ids):
        target = np.empty((len(left_rows), 1), dtype=np.float32)
        for player, model in enumerate(value_models):
            mask = player_ids == player
            if mask.any():
                target[mask] = (predict_batches(model, right_rows[mask], device, args.batch_size)
                                - predict_batches(model, left_rows[mask], device, args.batch_size))
        return target
    target = value_differences(left, right, players)
    val_target = value_differences(val_left, val_right, val_players)
    ticks = features.shape[1]
    training_pairs_used = len(left)//ticks
    loader = DataLoader(
        TensorDataset(torch.from_numpy(left.reshape(-1,ticks,ALPHA_INPUT_SIZE)).float(),
                      torch.from_numpy(right.reshape(-1,ticks,ALPHA_INPUT_SIZE)).float(),
                      torch.from_numpy(target.reshape(-1,ticks,1)),
                      torch.from_numpy(players.reshape(-1,ticks)[:,0])),
        batch_size=args.batch_size-1, shuffle=False, drop_last=True)
    initializers = []
    for path in initialization_paths:
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        model.load_state_dict(load_checkpoint_state(path, device))
        model.eval()
        initializers.append(model)
    potential = initialize_potential(initializers, device)
    optimizer = torch.optim.Adam(potential.fc.parameters(), lr=args.potential_learning_rate)
    progress = TrainingProgress(artifact, potential, optimizer,
        getattr(args, "early_stopping_patience", 0), getattr(args, "early_stopping_min_delta", 0.0))
    loss_fn = nn.MSELoss()
    for epoch in range(progress.epoch, args.potential_epochs):
        if progress.completed:
            break
        potential.eval()
        total, count = 0.0, 0
        for x_left, x_right, y_batch, pair_players in loader:
            x_left, x_right, y_batch = x_left.to(device), x_right.to(device), y_batch.to(device)
            for player in (1,2,0):
                mask = (pair_players == player).to(device)
                if not mask.any():
                    continue
                pair_scores = potential(torch.cat((x_left[mask], x_right[mask]), dim=0))
                left_score, right_score = pair_scores.split(int(mask.sum()), dim=0)
                loss = loss_fn(right_score - left_score, y_batch[mask])
                optimizer.zero_grad()
                loss.backward()
                if epoch > 0:
                    optimizer.step()
                total += loss.item() * int(mask.sum())
                count += int(mask.sum())
        mean_loss = total / max(count, 1)
        val_prediction = (predict_batches(potential, val_right, device, args.batch_size)
                          - predict_batches(potential, val_left, device, args.batch_size))
        validation_mse = float(np.mean((val_prediction - val_target) ** 2))
        stopped = progress.observe(epoch + 1, validation_mse,
            dict(difference_mse=mean_loss, validation_difference_mse=validation_mse))
        completed = stopped or epoch + 1 == args.potential_epochs
        if (epoch + 1) % 10 == 0 or completed:
            progress.save(completed)
        if epoch == 0 or (epoch + 1) % 10 == 0 or completed:
            print(f"  epoch={epoch + 1}/{args.potential_epochs} difference_mse={mean_loss:.6g} validation_mse={validation_mse:.6g}", flush=True)
        if completed:
            print(f"  {'early stopping' if stopped else 'completed'} potential, best_epoch={progress.best_epoch}", flush=True)
            break
    progress.publish({**metadata, "initialization": "sum_own_progress_values",
        "training_provenance": training_provenance(),
        "initialization_normalization": "exact_first_layer_bn_rebase_v1",
        "normalization": "official_potential_frozen_first_value_bn",
        "tail_batches": "drop_incomplete_previous_state_pairs",
        "training_pairs_used_per_epoch": training_pairs_used,
        "training_pairs_dropped_per_epoch": training_pair_count-training_pairs_used})
    print(f"  saved {artifact.model}", flush=True)
    return artifact.model


def evaluate_test_set(args, value_paths):
    """Final errors on races unused for fitting or checkpoint selection."""
    device = training_device(args)
    with np.load(args.data, allow_pickle=False) as data:
        ids = data["test_ids"]
        features, targets = data["features"], data["returns"]
        deviators = data["deviating_player"]
        collection_metadata = json.loads(str(data["metadata"]))
    x = features[ids].reshape(-1, ALPHA_INPUT_SIZE)
    left, right, players = potential_pairs(features, deviators, ids)
    models, value_errors = [], []
    for player, path in enumerate(value_paths):
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        model.load_state_dict(load_checkpoint_state(path, device))
        error = predict_batches(model, x, device, args.batch_size) - targets[ids, ..., player].reshape(-1, 1)
        if not np.isfinite(error).all():
            raise ValueError(f"Nonfinite held-out value error for car {player}")
        value_errors.append(dict(player=player, mse=float(np.mean(error ** 2)),
                                 mae=float(np.mean(np.abs(error)))))
        models.append(model)
    potential = SimpleNeuralNetwork().to(device)
    potential.load_state_dict(load_checkpoint_state(args.potential_output, device))
    prediction = (predict_batches(potential, right, device, args.batch_size)
                  - predict_batches(potential, left, device, args.batch_size))
    pair_errors = []
    for player, model in enumerate(models):
        mask = players == player
        if mask.any():
            teacher = (predict_batches(model, right[mask], device, args.batch_size)
                       - predict_batches(model, left[mask], device, args.batch_size))
            pair_errors.append(prediction[mask] - teacher)
    error = np.concatenate(pair_errors)
    report = dict(test_races=len(ids), test_states=len(x), test_pairs=len(left),
                  split_counts=collection_metadata["split_counts"],
                  collection_chains=len(collection_metadata["chains"]),
                  progress_gap_diagnostics=collection_metadata.get("progress_gap_diagnostics"),
                  collection_attempts=collection_metadata["attempts"],
                  rejected_episodes=len(collection_metadata["failed_attempts"]),
                  future_window_steps=args.return_window, gamma=args.gamma,
                  value_vs_recorded_returns=value_errors,
                  potential_vs_value_differences=dict(mse=float(np.mean(error ** 2)),
                      mae=float(np.mean(np.abs(error))), max_abs=float(np.max(np.abs(error)))),
                  settings={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                  checkpoint_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in [*value_paths, args.potential_output,
                                         *model_paths(args, "own_progress")]})
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite held-out potential error")
    output = args.work_dir / "evaluation" / "test.json"
    report["stage_signature"] = evaluation_signature(args, value_paths)
    report["training_provenance"] = training_provenance()
    atomic_json_save(report, output)
    print("[stage 4/4] Held-out test results")
    print(json.dumps(report, indent=2))
    print(f"  saved {output}")
    return report


def training_provenance():
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        source_sha256={name: file_sha256(name) for name in (
            "policy/alpharacer/train/training.py",
            "policy/alpharacer/network.py", "config.py")})


def evaluation_signature(args, value_paths):
    return signature(dict(method="heldout_test_v1", data=file_sha256(args.data),
        values=[file_sha256(path) for path in value_paths], potential=file_sha256(args.potential_output)))


def require_models(args, value_paths, initialization_paths):
    for utility, paths in (("own_progress", initialization_paths), ("relative", value_paths)):
        for player, path in enumerate(paths):
            artifact = ModelArtifact(path.parent, value_settings(args, utility, player), "cpu")
            if not artifact.complete():
                raise ValueError(f"Missing completed prerequisite: {path}; run --stage train first")


def print_status(args):
    print(f"Work directory: {args.work_dir}")
    if not args.data.exists():
        checkpoints = list((args.data.parent / "chains").glob("*/checkpoint.pth"))
        print(f"data: RESUMABLE ({len(checkpoints)} chain checkpoints)" if checkpoints else "data: MISSING")
        return
    try:
        collect_rollout_data(args)  # Existing data only: validate collection settings.
    except (ValueError, OSError, KeyError) as error:
        print(f"data: INCOMPATIBLE ({error})")
        return
    print("data: COMPLETE")
    all_ready = True
    for utility in ("own_progress", "relative"):
        for player, path in enumerate(model_paths(args, utility)):
            artifact = ModelArtifact(path.parent, value_settings(args, utility, player), "cpu")
            try:
                if artifact.complete():
                    status = "COMPLETE"
                elif artifact.checkpoint.exists():
                    state = load_training_state(artifact.checkpoint)
                    status = f"RESUMABLE epoch={state.get('epoch', state.get('next_epoch'))}"
                else:
                    status = "MISSING"
            except Exception as error:
                status = f"INCOMPATIBLE ({error})"
            print(f"{utility}/car{player}: {status}")
            all_ready = all_ready and status == "COMPLETE"
    if not all_ready:
        print("potential: WAITING FOR VALUES")
        return
    artifact = ModelArtifact(args.potential_output.parent, potential_settings(args,
        model_paths(args, "relative"), model_paths(args, "own_progress")), "cpu")
    try:
        if artifact.complete():
            print("potential: COMPLETE")
            report_path = args.work_dir / "evaluation" / "test.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            expected = evaluation_signature(args, model_paths(args, "relative"))
            print("evaluation: COMPLETE" if report.get("stage_signature") == expected else "evaluation: MISSING OR STALE")
        elif artifact.checkpoint.exists():
            state = load_training_state(artifact.checkpoint)
            print(f"potential: RESUMABLE epoch={state.get('epoch', state.get('next_epoch'))}")
        else:
            print("potential: MISSING")
    except Exception as error:
        print(f"potential/evaluation: INCOMPATIBLE ({error})")


def run_training_pipeline(args):
    seed_everything(args.seed)
    stage = getattr(args, "stage", "all")
    if not hasattr(args, "work_dir"):
        args.work_dir = args.data.parent
    if stage == "status":
        print_status(args)
        return
    with nullcontext():
        settings = {key: value for key, value in vars(args).items()
                    if key not in {"stage", "n_workers", "device", "force_rollouts", "force_values", "force_potential"}}
        provenance = training_provenance()
        atomic_json_save(dict(settings=settings, provenance=provenance), args.work_dir / "config.json")
        if stage in {"collect", "all"}:
            collect_rollout_data(args)
        else:
            if not args.data.exists():
                raise ValueError("No completed rollouts; run --stage collect on CPU first")
            collect_rollout_data(args)
        if stage == "collect":
            return args.data
        if stage in {"train", "all", "own-values"}:
            train_value_functions(args, utility="own_progress")
        if stage in {"train", "all", "relative-values"}:
            train_value_functions(args)
        if stage in {"own-values", "relative-values"}:
            return
        value_paths, initialization_paths = model_paths(args, "relative"), model_paths(args, "own_progress")
        require_models(args, value_paths, initialization_paths)
        if stage in {"train", "all", "potential"}:
            train_potential_function(args, value_paths, initialization_paths)
        if stage == "potential":
            return args.potential_output
        potential = ModelArtifact(args.potential_output.parent,
            potential_settings(args, value_paths, initialization_paths), "cpu")
        if not potential.complete():
            raise ValueError("No compatible completed potential; run --stage train first")
        report_path = args.work_dir / "evaluation" / "test.json"
        expected = evaluation_signature(args, value_paths)
        if report_path.exists() and json.loads(report_path.read_text()).get("stage_signature") == expected:
            print(f"[stage evaluation] Reusing {report_path}")
        else:
            evaluate_test_set(args, value_paths)
        return args.potential_output


def parse_args():
    parser = argparse.ArgumentParser(description="Reusable, resumable alpha-RACER training stages.")
    parser.add_argument("--work-dir", type=Path, help="fixed experiment directory; default outputs/alpharacer/<track>-<model>")
    parser.add_argument("--stage", choices=("status", "collect", "train", "own-values", "relative-values", "potential", "evaluate", "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--track", choices=("berlin", "modena", "rounded_rectangle"), default="berlin")
    parser.add_argument("--vehicle-model", choices=("kinematic", "dbm"), default=params.VEHICLE_MODEL)
    parser.add_argument("--races", type=int, default=4000)
    parser.add_argument("--duration", type=float, default=50.)
    parser.add_argument("--horizon", type=int, default=params.MPC_HORIZON)
    parser.add_argument("--initial-speed", type=float, default=0.)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--return-window", type=int, default=350)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--value-epochs", type=int, default=50000)
    parser.add_argument("--potential-epochs", type=int, default=100000)
    parser.add_argument("--value-learning-rate", type=float, default=1e-3)
    parser.add_argument("--potential-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=241)
    parser.add_argument("--validation-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--early-stopping-patience", type=int, default=1000,
                        help="epochs without validation improvement; 0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.)
    preliminary, _ = parser.parse_known_args()
    work_dir = preliminary.work_dir or Path("outputs/alpharacer") / f"{preliminary.track}-{preliminary.vehicle_model}"
    config_path = work_dir / "config.json"
    if config_path.exists():
        stored = json.loads(config_path.read_text())["settings"]
        configurable = {action.dest for action in parser._actions} - {"stage", "device", "n_workers", "work_dir", "help"}
        parser.set_defaults(**{key: value for key, value in stored.items() if key in configurable})
    args = parser.parse_args()
    args.work_dir = work_dir.resolve()
    args.data = args.work_dir / "data" / "rollouts.npz"
    args.value_dir = args.work_dir / "values"
    args.potential_output = args.work_dir / "potential" / "model.pth"
    args.resumable = True
    args.force_rollouts = args.force_values = args.force_potential = False
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be in [0,1]")
    if not 0 < args.return_window < duration_to_steps(args.duration, params.SIM_DT):
        parser.error("--return-window must be positive and shorter than the race")
    if args.n_workers < 1 or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("workers and attempt limit must be positive")
    if args.early_stopping_patience < 0 or not np.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        parser.error("early-stopping settings must be finite and nonnegative")
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        parser.error("validation/test fractions must be positive and sum to less than 1")
    try:
        split_races(args.races, args.validation_fraction, args.test_fraction)
    except ValueError as error:
        parser.error(str(error))
    if args.value_epochs < 1 or args.potential_epochs < 1 or args.batch_size < 2 or args.value_batch_size < 1:
        parser.error("epoch/batch counts must be positive and chain length at least 2")
    if any(not np.isfinite(lr) or lr <= 0 for lr in (args.value_learning_rate, args.potential_learning_rate)):
        parser.error("learning rates must be finite and positive")
    return args


if __name__ == "__main__":
    run_training_pipeline(parse_args())
