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
import hashlib
import json
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
    ALPHA_PARAMETER_BOUNDS,
    ALPHA_VALUE_HIDDEN_SIZES,
    SimpleNeuralNetwork,
    build_alpha_input,
)
from simulation import run_simulation
from simulation_core.config import SimulationConfig
from track import Track, configure_track, load_centerline_csv


DEFAULT_DATA_PATH = Path("policy/alpharacer/train/data/rollouts.npz")
DEFAULT_VALUE_DIR = Path("policy/alpharacer/train/weights/values")
DEFAULT_POTENTIAL_PATH = Path("policy/alpharacer/train/weights/potential_local.pth")
PARAMETERIZATION = "paper_q_zeta_s1_s2_s3"
REWARD_DEFINITION = "relative_progress_v1"
DATA_FORMAT_VERSION = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    relative = np.empty_like(progress)
    next_relative = np.empty_like(next_progress)
    for player in range(3):
        opponents = [idx for idx in range(3) if idx != player]
        relative[:, player] = progress[:, player] - progress[:, opponents].max(axis=1)
        next_relative[:, player] = next_progress[:, player] - next_progress[:, opponents].max(axis=1)
    # Sum the physical transitions belonging to each control update, including the final step.
    step_rewards = next_relative - relative
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
    """Official sum-of-value branches, using the first value's BatchNorm statistics."""
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

def collect_rollout_data(args):
    settings = dict(track=args.track, vehicle_model=args.vehicle_model,
                    horizon=args.horizon, duration=args.duration, gamma=args.gamma,
                    return_window=args.return_window,
                    seed=args.seed, races=args.races, initial_speed=args.initial_speed,
                    chain_length=args.batch_size,
                    validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)
    if args.data.exists() and not args.force_rollouts:
        with np.load(args.data, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
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
    while len(all_features) < total_races:
        if attempts >= max_attempts:
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
                attempt=attempts, accepted_before=race, completed_steps=completed, expected_steps=expected_steps,
                chain=chain,
                parameters=joint_params.tolist(), ego_position=start_rank,
                settings=settings, failure=reason, trace=trace,
            ), indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + "\n")
            failed_attempts.append(str(failure_path))
            print(f"  rejected attempt {attempts}: {reason}; saved {failure_path}")
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
            path = args.data.with_name(f"{args.data.stem}.{collection_id}.chain-{chain['id']:03d}.npz")
            section = slice(chain["start"], chain["end"])
            np.savez_compressed(path, features=np.stack(all_features[section]),
                returns=np.stack(all_returns[section]), own_returns=np.stack(own_returns[section]),
                joint_params=np.stack(all_params[section]),
                deviating_player=np.asarray(all_deviators[section]),
                metadata=np.asarray(json.dumps(dict(chain=chain, settings=settings))))
            chain_files.append(str(path))

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
    np.savez_compressed(
        args.data, features=np.stack(all_features), returns=np.stack(all_returns),
        own_returns=np.stack(own_returns),
        joint_params=np.stack(all_params), deviating_player=np.asarray(all_deviators),
        chain_id=np.asarray(chain_ids),
        **{f"{name}_ids": ids for name, ids in splits.items()},
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
    )
    print(f"  saved {args.data}")
    return args.data


# ---------------------------------------------------------------------------
# Stage 2: train the three Monte-Carlo value functions
# ---------------------------------------------------------------------------

def train_value_functions(args, utility="relative"):
    directory = args.value_dir if utility == "relative" else args.value_dir / "own_progress"
    output_paths = [directory / f"value_car{player}.pth" for player in range(3)]
    if all(path.exists() for path in output_paths) and not args.force_values:
        for path in output_paths:
            payload = torch.load(path, map_location="cpu")
            if payload.get("metadata", {}).get("normalization") != "official_value_running_batchnorm":
                raise ValueError("Value checkpoint uses older normalization; use --force-values")
        print(f"[stage 2/4] Reusing value checkpoints: {args.value_dir}")
        return output_paths

    print("[stage 2/4] Training one Monte-Carlo value function per car")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        targets = data["returns" if utility == "relative" else "own_returns"]
        dataset_metadata = json.loads(str(data["metadata"]))
        train_ids, val_ids = data["train_ids"], data["validation_ids"]
    directory.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.MSELoss()

    for player, output_path in enumerate(output_paths):
        x_train = torch.from_numpy(features[train_ids]).float()
        y_train = torch.from_numpy(targets[train_ids, ..., player, None]).float()
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.value_batch_size, shuffle=False)
        if len(x_train) < 2:
            raise ValueError("Need at least two value training samples for BatchNorm")
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.value_learning_rate)
        best_loss = float("inf")

        for epoch in range(args.value_epochs):
            model.train()
            for x_batch, y_batch in loader:
                y_batch = y_batch.to(device)
                mask = ((y_batch >= -100.) & (y_batch < 600.)).float()
                loss = loss_fn(model(x_batch.to(device))*mask, y_batch*mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            x_val = features[val_ids].reshape(-1, ALPHA_INPUT_SIZE)
            y_val = targets[val_ids, ..., player].reshape(-1, 1)
            mask = (y_val >= -100.) & (y_val < 600.)
            score = float(np.mean(((predict_batches(model, x_val, device, args.batch_size) - y_val)*mask) ** 2))
            # Official own-progress models initialize Phi from their final weights.
            if ((utility == "own_progress" and epoch == args.value_epochs-1)
                    or (utility == "relative" and score < best_loss)):
                best_loss = score
                torch.save(
                    checkpoint_payload(
                        model,
                        kind="alpha_value",
                        metadata={**dataset_metadata, "utility": utility, "player": player,
                                  "normalization": "official_value_running_batchnorm",
                                  "selection": "final_epoch" if utility == "own_progress" else "validation_mse",
                                  "target_filter": "official_-100_le_return_lt_600", "validation_mse": score},
                    ),
                    output_path,
                )
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"  car={player} epoch={epoch + 1}/{args.value_epochs} validation_mse={score:.6g}")
        print(f"  saved {output_path} (selected validation MSE={best_loss:.6g})")
    return output_paths


# ---------------------------------------------------------------------------
# Stage 3: learn the common potential from unilateral value differences
# ---------------------------------------------------------------------------

def train_potential_function(args, value_paths, initialization_paths):
    if args.potential_output.exists() and not args.force_potential:
        metadata = torch.load(args.potential_output, map_location="cpu").get("metadata", {})
        if (metadata.get("normalization") != "official_potential_frozen_first_value_bn"
                or metadata.get("tail_batches") != "drop_incomplete_previous_state_pairs"):
            raise ValueError("Potential checkpoint uses older training rules; use --force-potential")
        print(f"[stage 3/4] Reusing potential checkpoint: {args.potential_output}")
        return args.potential_output

    print("[stage 3/4] Training the common alpha-potential function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        metadata = json.loads(str(data["metadata"]))
        deviators = data["deviating_player"]
        train_ids, val_ids = data["train_ids"], data["validation_ids"]
    left, right, players = potential_pairs(features, deviators, train_ids)
    val_left, val_right, val_players = potential_pairs(features, deviators, val_ids)
    if len(left)//features.shape[1] < args.batch_size-1:
        raise ValueError("Not enough training pairs for one full potential batch; reduce --batch-size or collect more races")
    training_pair_count = len(left)//features.shape[1]
    # Official batches contain B races and their B-1 internal adjacent pairs.
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
        batch_size=args.batch_size-1, shuffle=False, drop_last=True,
    )
    initializers = []
    for path in initialization_paths:
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        model.load_state_dict(load_checkpoint_state(path, device))
        model.eval()
        initializers.append(model)
    potential = initialize_potential(initializers, device)
    optimizer = torch.optim.Adam(potential.fc.parameters(), lr=args.potential_learning_rate)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    args.potential_output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.potential_epochs):
        potential.eval()  # Official training freezes BatchNorm while updating fc weights.
        total, count = 0.0, 0
        for x_left, x_right, y_batch, pair_players in loader:
            x_left, x_right, y_batch = x_left.to(device), x_right.to(device), y_batch.to(device)
            # Official order follows the chain: player 1, then 2, then 0.
            for player in (1,2,0):
                mask = (pair_players == player).to(device)
                if not mask.any():
                    continue
                pair_scores = potential(torch.cat((x_left[mask], x_right[mask]), dim=0))
                left_score, right_score = pair_scores.split(int(mask.sum()), dim=0)
                loss = loss_fn(right_score - left_score, y_batch[mask])
                optimizer.zero_grad()
                loss.backward()
                if epoch > 0:  # Official first epoch measures the initialized potential.
                    optimizer.step()
                total += loss.item() * int(mask.sum())
                count += int(mask.sum())
        mean_loss = total / max(count, 1)
        val_prediction = (predict_batches(potential, val_right, device, args.batch_size)
                          - predict_batches(potential, val_left, device, args.batch_size))
        validation_mse = float(np.mean((val_prediction - val_target) ** 2))
        if validation_mse < best_loss:
            best_loss = validation_mse
            torch.save(
                checkpoint_payload(
                    potential,
                    kind="alpha_potential",
                    metadata={**metadata, "initialization": "sum_own_progress_values",
                              "normalization": "official_potential_frozen_first_value_bn",
                              "tail_batches": "drop_incomplete_previous_state_pairs",
                              "training_pairs_used_per_epoch": training_pairs_used,
                              "training_pairs_dropped_per_epoch": training_pair_count-training_pairs_used,
                              "difference_mse": mean_loss, "validation_difference_mse": validation_mse},
                ),
                args.potential_output,
            )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch + 1}/{args.potential_epochs} difference_mse={mean_loss:.6g} validation_mse={validation_mse:.6g}")
    print(f"  saved {args.potential_output} (best validation difference MSE={best_loss:.6g})")
    return args.potential_output


def evaluate_test_set(args, value_paths):
    """Final errors on races unused for fitting or checkpoint selection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                         *[args.value_dir / "own_progress" / f"value_car{i}.pth" for i in range(3)]]})
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite held-out potential error")
    output = args.potential_output.with_suffix(".test.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("[stage 4/4] Held-out test results")
    print(json.dumps(report, indent=2))
    print(f"  saved {output}")
    return report


def run_training_pipeline(args):
    seed_everything(args.seed)
    value_paths = [args.value_dir / f"value_car{player}.pth" for player in range(3)]
    initialization_paths = [args.value_dir / "own_progress" / f"value_car{player}.pth" for player in range(3)]
    # Cache invalidation follows the dependency graph. Rebuilding an upstream
    # artifact necessarily rebuilds every model trained from that artifact.
    rebuilding_rollouts = args.force_rollouts or not args.data.exists()
    rebuilding_values = args.force_values or rebuilding_rollouts or not all(path.exists() for path in value_paths + initialization_paths)
    if rebuilding_rollouts:
        args.force_rollouts = True
    if rebuilding_values:
        args.force_values = True
        args.force_potential = True
    collect_rollout_data(args)
    initialization_paths = train_value_functions(args, utility="own_progress")
    value_paths = train_value_functions(args)
    potential_path = train_potential_function(args, value_paths, initialization_paths)
    evaluate_test_set(args, value_paths)
    return potential_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect local rollouts and train alpha-RACER value/potential models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter summary:
  paths:       --data, --value-dir, --potential-output
  rollout:     --races, --duration, --track, --vehicle-model, --horizon,
               --initial-speed, --gamma, --return-window, --max-attempts
  training:    --value-epochs, --potential-epochs, the two learning rates,
               --value-batch-size, --batch-size, --validation-fraction, --test-fraction, --seed
  cache:       --force-rollouts, --force-values, --force-potential

Choices:
  --track          berlin | modena | rounded_rectangle
  --vehicle-model  kinematic | dbm
  --gamma          0..1; --validation-fraction / --test-fraction: positive, sum < 1

Example:
  .venv/bin/python -m policy.alpharacer.train.training \\
      --races 4000 --duration 50 --track berlin --vehicle-model kinematic
""",
    )

    # Artifact paths:
    # --data PATH             rollout .npz input/output
    # --value-dir DIR         directory containing value_car0/1/2.pth
    # --potential-output PATH final locally trained potential checkpoint
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--value-dir", type=Path, default=DEFAULT_VALUE_DIR)
    parser.add_argument("--potential-output", type=Path, default=DEFAULT_POTENTIAL_PATH)

    # Rollout parameters:
    # --races N              accepted training races; validation/test are additional
    # --duration SECONDS      simulated duration of every rollout
    # --track NAME            berlin, modena, or rounded_rectangle
    # --vehicle-model MODEL   kinematic or dbm
    # --horizon N             local MPC prediction horizon
    # --initial-speed MPS     common initial longitudinal speed
    # --gamma FLOAT           Monte-Carlo return discount in [0,1]
    # reward = delta(s_i - max_{j != i} s_j)
    parser.add_argument("--races", type=int, default=4000)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--track", choices=("berlin", "modena", "rounded_rectangle"), default="berlin")
    parser.add_argument("--vehicle-model", choices=("kinematic", "dbm"), default=params.VEHICLE_MODEL)
    parser.add_argument("--horizon", type=int, default=params.MPC_HORIZON)
    parser.add_argument("--initial-speed", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--return-window", type=int, default=350, help="fixed future window in control steps (official S)")
    parser.add_argument("--max-attempts", type=int, default=None, help="collection safety limit; default 3 times the required valid races")

    # Optimization parameters:
    # --value-epochs N / --potential-epochs N
    # --value-learning-rate LR / --potential-learning-rate LR
    # --batch-size N          shared minibatch size
    # --validation-fraction F race-level validation fraction; test split defaults to 10%
    parser.add_argument("--value-epochs", type=int, default=50000)
    parser.add_argument("--potential-epochs", type=int, default=100000)
    parser.add_argument("--value-learning-rate", type=float, default=1e-3)
    parser.add_argument("--potential-learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=241, help="official independent chain length and potential batch size, in races")
    parser.add_argument("--value-batch-size", type=int, default=4096, help="official value batch size, in races")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)

    # Cache control. By default each existing stage output is reused.
    # --force-rollouts         recollect data and retrain all dependent models
    # --force-values           retrain all three value functions
    # --force-potential        retrain the final potential function
    parser.add_argument("--force-rollouts", action="store_true")
    parser.add_argument("--force-values", action="store_true")
    parser.add_argument("--force-potential", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0,1]")
    if not 0 < args.return_window < duration_to_steps(args.duration, params.SIM_DT):
        parser.error("--return-window must be positive and shorter than the race")
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if not (0.0 < args.validation_fraction < 1.0 and 0.0 < args.test_fraction < 1.0
            and args.validation_fraction + args.test_fraction < 1.0):
        parser.error("validation/test fractions must be positive and sum to less than 1")
    try:
        split_races(args.races, args.validation_fraction, args.test_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    if args.value_epochs < 1 or args.potential_epochs < 1 or args.batch_size < 2 or args.value_batch_size < 1:
        parser.error("epoch counts must be positive and --batch-size must be at least 2")
    return args


if __name__ == "__main__":
    # Typical full run:
    #   .venv/bin/python -m policy.alpharacer.train.training \
    #       --races 4000 --duration 50
    # Existing rollouts/value weights/potential weights are reused. Add the
    # corresponding --force-* flag when a particular stage must be repeated.
    run_training_pipeline(parse_args())
