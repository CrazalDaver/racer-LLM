"""Atomic training and runtime checkpoints for the self-play baseline."""

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from .network import CHECKPOINT_FORMAT, actor_state


TRAINING_CHECKPOINT_FORMAT = "self_play_rl_trainer_v3_complete"


class SelfPlayCheckpoint:
    """Save and restore one complete PPO-update boundary."""

    def __init__(self, work_dir, output_path, ppo_settings, trainer):
        self.state_path = Path(work_dir) / "trainer.pth"
        self.output_path = Path(output_path)
        self.ppo_settings = dict(ppo_settings)
        self.self_play_settings = asdict(trainer.settings)
        self.trainer = trainer

    @staticmethod
    def _atomic_save(value, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    @staticmethod
    def _runtime_state(model, metadata):
        metadata = dict(metadata)
        metadata["network"] = {
            **metadata.get("network", {}), "activation": model.activation,
        }
        return {
            "format": CHECKPOINT_FORMAT,
            "actor_state_dict": actor_state(model),
            "metadata": metadata,
        }

    def save(self, metadata):
        """Atomically save training state, then publish its matching actor."""
        trainer = self.trainer
        state = {
            **metadata,
            "format": TRAINING_CHECKPOINT_FORMAT,
            "ppo_settings": self.ppo_settings,
            "self_play_settings": self.self_play_settings,
            "model": trainer.model.state_dict(),
            "activation": trainer.model.activation,
            "optimizer": trainer.optimizer.state_dict(),
            "training_state": trainer.state_dict(),
            "rng_state": trainer.rng.bit_generator.state,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "worker_count": trainer.workers.count,
            "worker_states": trainer.workers.snapshot(),
        }
        self._atomic_save(state, self.state_path)
        self._atomic_save(
            self._runtime_state(trainer.model, metadata), self.output_path
        )

    def load(self):
        """Restore all mutable training state, or return False for a new run."""
        if not self.state_path.exists():
            return False
        trainer = self.trainer
        device = trainer.device
        saved = torch.load(self.state_path, map_location="cpu", weights_only=False)
        if saved.get("format") != TRAINING_CHECKPOINT_FORMAT:
            raise ValueError(
                "Existing self-play checkpoint is not the complete shared-policy "
                "format; use a new --work-dir"
            )
        if (
            saved.get("ppo_settings") != self.ppo_settings
            or saved.get("self_play_settings") != self.self_play_settings
        ):
            raise ValueError("Existing self-play checkpoint uses different settings")
        # Older complete shared-policy checkpoints were always ReLU.
        saved_activation = saved.get(
            "activation", saved.get("network", {}).get("activation", "relu")
        ).lower()
        if saved_activation != trainer.model.activation:
            raise ValueError(
                f"Checkpoint activation is {saved_activation}; use "
                f"--activation {saved_activation} or a new --work-dir"
            )
        if int(saved.get("worker_count", -1)) != trainer.workers.count:
            raise ValueError(
                "Exact self-play resume requires the same --workers value "
                f"({saved.get('worker_count')} saved, {trainer.workers.count} requested)"
            )

        trainer.model.load_state_dict(saved["model"])
        trainer.optimizer.load_state_dict(saved["optimizer"])
        for optimizer_state in trainer.optimizer.state.values():
            for name, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[name] = value.to(device)
        trainer.load_state_dict(saved["training_state"])
        trainer.rng.bit_generator.state = saved["rng_state"]
        random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            cuda_states = saved.get("cuda_rng_states", [])
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    "Exact CUDA resume requires the same number of visible GPUs"
                )
            torch.cuda.set_rng_state_all(cuda_states)
        trainer.workers.restore(saved["worker_states"])

        metadata = {
            key: saved[key]
            for key in (
                "ppo_settings", "self_play_settings", "network", "phase",
                "phase_steps", "total_steps", "track", "seed", "workers",
                "device", "shared_policy", "sources", "git",
            )
            if key in saved
        }
        self._atomic_save(
            self._runtime_state(trainer.model, metadata), self.output_path
        )
        return True
