"""Shared alpha-RACER network and feature definitions.

Keeping these definitions outside the online policy makes locally trained
checkpoints structurally identical to the models used at inference time.
"""

import math
from pathlib import Path

import numpy as np

from config import params

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from ..utils import progress_delta, track_curvature_at, track_heading_error


ALPHA_INPUT_SIZE = 39
ALPHA_VALUE_HIDDEN_SIZES = (128, 128, 64)
ALPHA_POTENTIAL_HIDDEN_SIZES = (384, 384, 192)
ALPHA_PARAMETER_BOUNDS = params.ALPHARACER_PARAMETER_BOUNDS


class SimpleNeuralNetwork(nn.Module if nn is not None else object):
    """Batch-normalized MLP shared by value and potential approximators."""

    def __init__(self, input_size=ALPHA_INPUT_SIZE, hidden_sizes=ALPHA_POTENTIAL_HIDDEN_SIZES, output_size=1):
        if nn is None:
            raise ImportError("PyTorch is required for alpha-RACER models.")
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size, affine=False)
        layers = []
        current = input_size
        for width in hidden_sizes:
            layers.extend((nn.Linear(current, width), nn.ReLU()))
            current = width
        layers.append(nn.Linear(current, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 2:
            x = self.batch_norm(x)
        elif x.ndim == 3:
            shape = x.shape
            x = self.batch_norm(x.reshape(-1, shape[-1])).reshape(shape)
        return self.fc(x)


def load_potential_checkpoint(weight_path, device):
    """Load one raw-upstream or metadata-bearing local potential checkpoint."""
    if torch is None:
        raise ImportError("PyTorch is required for alpha-RACER models.")
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"alpha-RACER weight file not found: {path}")
    model = SimpleNeuralNetwork().to(device)
    checkpoint = torch.load(str(path), map_location=device)
    wrapped = isinstance(checkpoint, dict) and "state_dict" in checkpoint
    metadata = checkpoint.get("metadata", {}) if wrapped else {}
    if wrapped and "parameterization" not in metadata:
        raise ValueError(
            "Local alpha-RACER checkpoint metadata must declare its "
            "parameterization; retrain or add "
            "parameterization='paper_q_zeta_s1_s2_s3'."
        )
    parameterization = metadata.get("parameterization", "upstream_transformed")
    if parameterization not in {
        "paper_q_zeta_s1_s2_s3",
        "upstream_transformed",
    }:
        raise ValueError(
            f"Unsupported alpha-RACER checkpoint parameterization: {parameterization}"
        )
    model.load_state_dict(checkpoint["state_dict"] if wrapped else checkpoint)
    model.eval()
    return path, model, metadata, parameterization


def alpha_network_input(paper_input, parameterization):
    """Convert paper theta only for legacy upstream checkpoint inputs."""
    if parameterization == "paper_q_zeta_s1_s2_s3":
        return paper_input
    state_features = paper_input[..., :-15]
    theta = paper_input[..., -15:].reshape(*paper_input.shape[:-1], 3, 5)
    q, zeta, s1, s2, s3 = theta.unbind(dim=-1)
    upstream = torch.stack(
        (
            s1,
            s2,
            0.5 / torch.clamp(q, min=1e-6),
            torch.sqrt(torch.clamp(zeta, min=0.0)),
            s3,
        ),
        dim=-1,
    )
    return torch.cat(
        (state_features, upstream.reshape(*paper_input.shape[:-1], 15)),
        dim=-1,
    )


def paper_to_upstream_alpha_params(paper_params):
    """Adapt paper theta to the transformed coordinates seen by official weights."""
    q, zeta, s1, s2, s3 = np.asarray(paper_params, dtype=float)
    return np.asarray(
        [
            s1,
            s2,
            1.0 / max(2.0 * q, 1e-6),
            np.sqrt(max(zeta, 0.0)),
            s3,
        ],
        dtype=float,
    )


def build_alpha_input(track, mpc_dt, mpc_horizon, cars, joint_params):
    """Build the canonical 39-D feature vector used online and in training."""
    cars = [dict(car) for car in list(cars)[:3]]
    if not cars:
        raise ValueError("At least one car is required to build alpha-RACER features.")
    while len(cars) < 3:
        cars.append({"px": cars[0]["px"], "py": 0.0, "vx": 0.0, "vy": 0.0, "omega": 0.0, "psi": 0.0})

    s0, s1, s2 = (float(car["px"]) for car in cars)
    features = [
        progress_delta(s2, s1, track.total_length),
        progress_delta(s1, s0, track.total_length),
        progress_delta(s2, s0, track.total_length),
    ]

    blocks = []
    for car in cars:
        state = car.get("state")
        psi = float(car.get("psi", getattr(state, "psi", 0.0)))
        blocks.append(
            [
                float(car["py"]),
                (float(car["heading_error"]) if "heading_error" in car else
                 track_heading_error(track, psi, float(car["px"]))),
                float(car.get("vx", getattr(state, "vx", 0.0))),
                float(car.get("vy", getattr(state, "vy", 0.0))),
                float(car.get("omega", getattr(state, "omega", 0.0))),
            ]
        )
    features.extend(block[0] for block in blocks)
    for block in blocks:
        features.extend(block[1:])

    features.extend(track_curvature_at(track, float(car["px"])) for car in cars)
    for car in cars:
        lookahead = max(float(car.get("vx", 0.0)), 0.0) * float(mpc_dt) * int(mpc_horizon)
        features.append(track_curvature_at(track, float(car["px"]) + lookahead))

    features.extend(np.asarray(joint_params, dtype=float).reshape(3, 5).ravel().tolist())
    result = np.asarray(features, dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    if result.shape != (ALPHA_INPUT_SIZE,):
        raise ValueError(f"alpha-RACER input must have {ALPHA_INPUT_SIZE} values, got {result.shape}")
    return result
