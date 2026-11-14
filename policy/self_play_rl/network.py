"""Torch networks and checkpoints for the direct-action self-play policy."""

from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

# The input size is the same as the Alpha-Racer released code.
OBSERVATION_SIZE = 15
ACTION_SIZE = 2
# Thakkar's E2E/NonLSTM configuration supplies these hidden-layer sizes.
HIDDEN_SIZES = (128, 128, 128)
CHECKPOINT_FORMAT = "self_play_rl_torch_v2_shared"


def _mlp(output_size, activation="tanh"):
    """Build a three-layer MLP with the selected activation."""
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
    if activation not in activations:
        raise ValueError(f"Unsupported self-play activation: {activation}")
    layers = []
    input_size = OBSERVATION_SIZE
    for hidden_size in HIDDEN_SIZES:
        layers.extend((nn.Linear(input_size, hidden_size), activations[activation]()))
        input_size = hidden_size
    layers.append(nn.Linear(input_size, output_size))
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    """
    Separate actor and critic networks for the shared racing policy.

    Attributes:
        - actor: MLP mapping 15-value observation to 2-value predicted action.
        - critic: MLP mapping 15-value observation to 1-value estimated state value.
    """

    def __init__(self, activation="tanh"):
        super().__init__()
        self.activation = activation.lower()
        self.actor = _mlp(ACTION_SIZE, self.activation)
        self.critic = _mlp(1, self.activation)

        # SB3 PPO: orthogonal hidden layers and small policy output weights.
        for network, output_gain in ((self.actor, 0.01), (self.critic, 1.0)):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    gain = output_gain if layer is network[-1] else math.sqrt(2.0)
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

        # The log standard deviation is a learnable parameter, initialized to zero.
        self.log_std = nn.Parameter(torch.zeros(ACTION_SIZE))

    def distribution(self, observation):
        mean = self.actor(observation)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, observation):
        distribution = self.distribution(observation)
        action = distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return action, log_probability, value

    def evaluate(self, observation, action):
        distribution = self.distribution(observation)
        return (
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
            self.critic(observation).squeeze(-1),
        )


class TorchActorPolicy:
    """Inference-only wrapper around the shared self-play actor."""

    def __init__(self, actor, device):
        self.actor = actor.to(device).eval()
        self.device = torch.device(device)

    def predict(self, observation, deterministic=True):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (OBSERVATION_SIZE,) or not np.all(np.isfinite(values)):
            raise ValueError("Self-play actor requires one finite 15-value observation")
        with torch.no_grad():
            tensor = torch.from_numpy(values).to(self.device).unsqueeze(0)
            action = self.actor(tensor).squeeze(0).cpu().numpy()
        return action, None


def actor_state(model):
    """Return one CPU copy suitable for worker transfer or snapshot storage."""
    return {name: value.detach().cpu().clone() for name, value in model.actor.state_dict().items()}


def load_actor_checkpoint(weight_path, device="cpu"):
    """Load the single actor shared by every car at runtime."""
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported self-play checkpoint format in {path}")
    state = checkpoint.get("actor_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Self-play checkpoint {path} has no shared actor")
    metadata = checkpoint.get("metadata", {})
    # Earlier v2 files always used ReLU; their metadata may lack this field.
    activation = metadata.get("network", {}).get("activation", "relu").lower()
    actor = _mlp(ACTION_SIZE, activation)
    actor.load_state_dict(state)
    return path, TorchActorPolicy(actor, device), metadata
