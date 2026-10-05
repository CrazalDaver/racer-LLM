"""End-to-end local alpha-RACER potential training.

The pipeline has three cached stages:
  1. collect_rollout_data: simulate baseline and unilateral deviations;
  2. train_value_functions: fit one Monte-Carlo value model per car;
  3. train_potential_function: match unilateral value differences.

Run this module from the repository root. Existing stage outputs are reused
unless the matching --force-* option is provided.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import params
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
        rows.append(build_alpha_input(track, params.MPC_DT, mpc_horizon, cars, joint_params))
    return np.asarray(rows, dtype=np.float32), update_indices


def trace_rewards(trace, update_indices, progress_weight, offroad_penalty, collision_penalty):
    progress = np.asarray(trace["car_px"], dtype=np.float64)[update_indices]
    offroad = np.asarray(trace["car_offroad"], dtype=bool)[update_indices]
    collision = np.asarray(trace["collision"], dtype=bool)[update_indices]
    rewards = np.zeros_like(progress, dtype=np.float32)
    if len(progress) > 1:
        rewards[:-1] = progress_weight * np.diff(progress, axis=0)
    rewards -= offroad_penalty * offroad.astype(np.float32)
    rewards -= collision_penalty * collision[:, None].astype(np.float32)
    return rewards


def discounted_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros(rewards.shape[-1], dtype=np.float32)
    for step in range(len(rewards) - 1, -1, -1):
        running = rewards[step] + gamma * running
        returns[step] = running
    return returns


# ---------------------------------------------------------------------------
# Stage 1: collect rollout data
# ---------------------------------------------------------------------------

def collect_rollout_data(args):
    if args.data.exists() and not args.force_rollouts:
        print(f"[stage 1/3] Reusing rollout data: {args.data}")
        return args.data

    if args.groups < 1 or args.duration <= 0.0:
        raise ValueError("--groups and --duration must be positive")
    rng = np.random.default_rng(args.seed)
    configure_track(params, args.track)
    track = Track(load_centerline_csv(params.TRACK_CSV, scale=params.TRACK_SCALE))
    all_features, all_returns, all_params, all_deviators = [], [], [], []

    print("[stage 1/3] Collecting local baseline/unilateral-deviation rollouts")
    for group in range(args.groups):
        baseline = rng.uniform(
            ALPHA_PARAMETER_BOUNDS[:, 0],
            ALPHA_PARAMETER_BOUNDS[:, 1],
            size=(3, 5),
        )
        variants = [baseline]
        for player in range(3):
            changed = baseline.copy()
            changed[player] = rng.uniform(ALPHA_PARAMETER_BOUNDS[:, 0], ALPHA_PARAMETER_BOUNDS[:, 1])
            variants.append(changed)

        group_features, group_returns = [], []
        for variant_idx, joint_params in enumerate(variants):
            trace = run_simulation(SimulationConfig(
                policies=["static_alpharacer"] * 3,
                static_params_by_car=joint_params,
                vehicle_model=args.vehicle_model,
                mpc_horizon=args.horizon,
                initial_speed=args.initial_speed,
                ego_position=1 + group % 3,
                duration=args.duration,
                save_trace=False,
                show_plot=False,
                quiet=True,
            ))
            features, update_indices = trace_features(trace, track, joint_params, args.horizon)
            rewards = trace_rewards(
                trace,
                update_indices,
                args.progress_weight,
                args.offroad_penalty,
                args.collision_penalty,
            )
            group_features.append(features)
            group_returns.append(discounted_returns(rewards, args.gamma))
            print(f"  group {group + 1}/{args.groups}, rollout {variant_idx + 1}/4, samples={len(features)}")

        common_length = min(map(len, group_features))
        all_features.append(np.stack([rows[:common_length] for rows in group_features]))
        all_returns.append(np.stack([rows[:common_length] for rows in group_returns]))
        all_params.append(np.stack(variants))
        all_deviators.append(np.asarray([-1, 0, 1, 2], dtype=np.int8))

    metadata = {
        "format_version": 1,
        "parameterization": PARAMETERIZATION,
        "parameter_order": list(params.ALPHARACER_PARAMETER_NAMES),
        "track": args.track,
        "vehicle_model": args.vehicle_model,
        "horizon": args.horizon,
        "duration": args.duration,
        "gamma": args.gamma,
        "progress_weight": args.progress_weight,
        "offroad_penalty": args.offroad_penalty,
        "collision_penalty": args.collision_penalty,
        "seed": args.seed,
        "variant_order": ["baseline", "car0_deviation", "car1_deviation", "car2_deviation"],
    }
    args.data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data,
        features=np.stack(all_features),
        returns=np.stack(all_returns),
        joint_params=np.stack(all_params),
        deviating_player=np.stack(all_deviators),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"  saved {args.data}")
    return args.data


# ---------------------------------------------------------------------------
# Stage 2: train the three Monte-Carlo value functions
# ---------------------------------------------------------------------------

def train_value_functions(args):
    output_paths = [args.value_dir / f"value_car{player}.pth" for player in range(3)]
    if all(path.exists() for path in output_paths) and not args.force_values:
        print(f"[stage 2/3] Reusing value checkpoints: {args.value_dir}")
        return output_paths

    print("[stage 2/3] Training one Monte-Carlo value function per car")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        targets = data["returns"]
        dataset_metadata = json.loads(str(data["metadata"]))

    groups = features.shape[0]
    order = np.random.default_rng(args.seed).permutation(groups)
    val_count = max(1, int(round(groups * args.validation_fraction))) if groups > 1 else 0
    val_ids, train_ids = order[:val_count], order[val_count:]
    if not len(train_ids):
        train_ids = order
    args.value_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.MSELoss()

    for player, output_path in enumerate(output_paths):
        x_train = torch.from_numpy(features[train_ids].reshape(-1, ALPHA_INPUT_SIZE)).float()
        y_train = torch.from_numpy(targets[train_ids, ..., player].reshape(-1, 1)).float()
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.value_learning_rate)
        best_loss = float("inf")

        for epoch in range(args.value_epochs):
            model.train()
            for x_batch, y_batch in loader:
                loss = loss_fn(model(x_batch.to(device)), y_batch.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                if len(val_ids):
                    x_val = torch.from_numpy(features[val_ids].reshape(-1, ALPHA_INPUT_SIZE)).float().to(device)
                    y_val = torch.from_numpy(targets[val_ids, ..., player].reshape(-1, 1)).float().to(device)
                    score = loss_fn(model(x_val), y_val).item()
                else:
                    score = loss.item()
            if score < best_loss:
                best_loss = score
                torch.save(
                    checkpoint_payload(
                        model,
                        kind="alpha_value",
                        metadata={**dataset_metadata, "player": player, "validation_mse": score},
                    ),
                    output_path,
                )
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"  car={player} epoch={epoch + 1}/{args.value_epochs} validation_mse={score:.6g}")
        print(f"  saved {output_path} (best validation MSE={best_loss:.6g})")
    return output_paths


# ---------------------------------------------------------------------------
# Stage 3: learn the common potential from unilateral value differences
# ---------------------------------------------------------------------------

def train_potential_function(args, value_paths):
    if args.potential_output.exists() and not args.force_potential:
        print(f"[stage 3/3] Reusing potential checkpoint: {args.potential_output}")
        return args.potential_output

    print("[stage 3/3] Training the common alpha-potential function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        metadata = json.loads(str(data["metadata"]))

    # For each deviation, compare parameters at exactly the same baseline
    # state. Using the evolved deviation state would confound policy change
    # with the different trajectory caused by that policy.
    base = features[:, 0]
    left_parts, right_parts, player_parts = [], [], []
    for player in range(3):
        changed = base.copy()
        changed[..., -15:] = features[:, player + 1, :, -15:]
        left_parts.append(base)
        right_parts.append(changed)
        player_parts.append(np.full(base.shape[:-1], player, dtype=np.int64))
    left = np.concatenate(left_parts).reshape(-1, ALPHA_INPUT_SIZE)
    right = np.concatenate(right_parts).reshape(-1, ALPHA_INPUT_SIZE)
    players = np.concatenate(player_parts).reshape(-1)

    value_models = []
    for path in value_paths:
        model = SimpleNeuralNetwork(ALPHA_INPUT_SIZE, ALPHA_VALUE_HIDDEN_SIZES, 1).to(device)
        model.load_state_dict(load_checkpoint_state(path, device))
        model.eval()
        value_models.append(model)
    with torch.no_grad():
        left_t = torch.from_numpy(left).float().to(device)
        right_t = torch.from_numpy(right).float().to(device)
        target = torch.empty((len(left), 1), device=device)
        for player, model in enumerate(value_models):
            mask = torch.from_numpy(players == player).to(device)
            target[mask] = model(right_t[mask]) - model(left_t[mask])

    loader = DataLoader(
        TensorDataset(torch.from_numpy(left).float(), torch.from_numpy(right).float(), target.cpu()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    potential = SimpleNeuralNetwork().to(device)
    optimizer = torch.optim.Adam(potential.parameters(), lr=args.potential_learning_rate)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    args.potential_output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.potential_epochs):
        potential.train()
        total, count = 0.0, 0
        for x_left, x_right, y_batch in loader:
            x_left, x_right, y_batch = x_left.to(device), x_right.to(device), y_batch.to(device)
            pair_scores = potential(torch.cat((x_left, x_right), dim=0))
            left_score, right_score = pair_scores.split(len(x_left), dim=0)
            loss = loss_fn(right_score - left_score, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_left)
            count += len(x_left)
        mean_loss = total / max(count, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                checkpoint_payload(
                    potential,
                    kind="alpha_potential",
                    metadata={**metadata, "difference_mse": mean_loss},
                ),
                args.potential_output,
            )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch={epoch + 1}/{args.potential_epochs} difference_mse={mean_loss:.6g}")
    print(f"  saved {args.potential_output} (best difference MSE={best_loss:.6g})")
    return args.potential_output


def run_training_pipeline(args):
    seed_everything(args.seed)
    value_paths = [args.value_dir / f"value_car{player}.pth" for player in range(3)]
    # Cache invalidation follows the dependency graph. Rebuilding an upstream
    # artifact necessarily rebuilds every model trained from that artifact.
    rebuilding_rollouts = args.force_rollouts or not args.data.exists()
    rebuilding_values = args.force_values or rebuilding_rollouts or not all(path.exists() for path in value_paths)
    if rebuilding_rollouts:
        args.force_rollouts = True
    if rebuilding_values:
        args.force_values = True
        args.force_potential = True
    collect_rollout_data(args)
    value_paths = train_value_functions(args)
    return train_potential_function(args, value_paths)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect local rollouts and train alpha-RACER value/potential models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter summary:
  paths:       --data, --value-dir, --potential-output
  rollout:     --groups, --duration, --track, --vehicle-model, --horizon,
               --initial-speed, --gamma, and the three reward weights
  training:    --value-epochs, --potential-epochs, the two learning rates,
               --batch-size, --validation-fraction, --seed
  cache:       --force-rollouts, --force-values, --force-potential

Choices:
  --track          berlin | modena | rounded_rectangle
  --vehicle-model  kinematic | dbm
  --gamma          0..1; --validation-fraction 0..<1

Example:
  .venv/bin/python -m policy.alpharacer.train.training \\
      --groups 100 --duration 12 --track berlin --vehicle-model kinematic
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
    # --groups N              number of groups; each group runs baseline + 3 deviations
    # --duration SECONDS      simulated duration of every rollout
    # --track NAME            berlin, modena, or rounded_rectangle
    # --vehicle-model MODEL   kinematic or dbm
    # --horizon N             local MPC prediction horizon
    # --initial-speed MPS     common initial longitudinal speed
    # --gamma FLOAT           Monte-Carlo return discount in [0,1]
    # reward = progress_weight*delta_s - offroad_penalty*I_offroad
    #          - collision_penalty*I_collision
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--track", choices=("berlin", "modena", "rounded_rectangle"), default="berlin")
    parser.add_argument("--vehicle-model", choices=("kinematic", "dbm"), default=params.VEHICLE_MODEL)
    parser.add_argument("--horizon", type=int, default=params.MPC_HORIZON)
    parser.add_argument("--initial-speed", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--offroad-penalty", type=float, default=2.0)
    parser.add_argument("--collision-penalty", type=float, default=1.0)

    # Optimization parameters:
    # --value-epochs N / --potential-epochs N
    # --value-learning-rate LR / --potential-learning-rate LR
    # --batch-size N          shared minibatch size
    # --validation-fraction F group-level value validation fraction [0,1)
    parser.add_argument("--value-epochs", type=int, default=100)
    parser.add_argument("--potential-epochs", type=int, default=200)
    parser.add_argument("--value-learning-rate", type=float, default=1e-3)
    parser.add_argument("--potential-learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)

    # Cache control. By default each existing stage output is reused.
    # --force-rollouts         recollect data, then train only missing later stages
    # --force-values           retrain all three value functions
    # --force-potential        retrain the final potential function
    parser.add_argument("--force-rollouts", action="store_true")
    parser.add_argument("--force-values", action="store_true")
    parser.add_argument("--force-potential", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0,1]")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0,1)")
    if args.value_epochs < 1 or args.potential_epochs < 1 or args.batch_size < 1:
        parser.error("epoch counts and --batch-size must be positive")
    return args


if __name__ == "__main__":
    # Typical full run:
    #   .venv/bin/python -m policy.alpharacer.train.training \
    #       --groups 100 --duration 12
    # Existing rollouts/value weights/potential weights are reused. Add the
    # corresponding --force-* flag when a particular stage must be repeated.
    run_training_pipeline(parse_args())
