"""Reusable learned-potential scoring and unilateral maximization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

try:
    import torch
except ImportError:
    torch = None

from policy.alpharacer.network import (
    ALPHA_PARAMETER_BOUNDS,
    alpha_network_input,
    build_alpha_input,
    load_potential_checkpoint,
)

@dataclass(frozen=True)
class PotentialOptimizationResult:
    theta: tuple[float, ...]
    potential: float
    llm_potential: float
    predicted_difference: float
    method: str
    seed: int
    starts: int
    converged_starts: int
    successful: bool
    boundary_saturated: tuple[bool, ...]
    messages: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


class LearnedPotential:
    """Checkpoint-aware alpha-RACER potential with the canonical 39-D input."""

    def __init__(self, track, mpc_dt, mpc_horizon, weight_path, device=None):
        if torch is None:
            raise ImportError("PyTorch is required for learned-potential evaluation")
        self.track = track
        self.mpc_dt = float(mpc_dt)
        self.mpc_horizon = int(mpc_horizon)
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        (
            self.weight_path,
            self.model,
            self.metadata,
            self.parameterization,
        ) = load_potential_checkpoint(weight_path, self.device)

    def _network_input(self, paper_input):
        return alpha_network_input(paper_input, self.parameterization)

    def state_features(self, cars, joint_params):
        row = build_alpha_input(
            self.track, self.mpc_dt, self.mpc_horizon, cars, joint_params
        )
        return np.asarray(row[:-15], dtype=np.float32)

    def score(self, state_features, joint_params_batch):
        joint = np.asarray(joint_params_batch, dtype=np.float32)
        single = joint.ndim == 2
        joint = joint.reshape(-1, 3, 5)
        state = np.broadcast_to(
            np.asarray(state_features, dtype=np.float32), (len(joint), 24)
        )
        rows = np.concatenate((state, joint.reshape(len(joint), 15)), axis=1)
        with torch.no_grad():
            values = self.model(
                self._network_input(
                    torch.tensor(rows, dtype=torch.float32, device=self.device)
                )
            ).detach().cpu().numpy().reshape(-1)
        return float(values[0]) if single else values

    def checkpoint_record(self):
        return {
            "path": str(self.weight_path),
            "parameterization": self.parameterization,
            "metadata": self.metadata,
        }


class PotentialOracleOptimizer:
    """Maximize one ego block while holding evaluator-known opponents fixed."""

    def __init__(
        self,
        potential: LearnedPotential,
        *,
        sobol_points=8,
        max_iterations=40,
        seed=0,
        boundary_tolerance=1e-5,
    ):
        self.potential = potential
        self.bounds = np.asarray(ALPHA_PARAMETER_BOUNDS, dtype=float)
        self.sobol_points = max(1, int(sobol_points))
        self.max_iterations = max(1, int(max_iterations))
        self.seed = int(seed)
        self.boundary_tolerance = float(boundary_tolerance)

    def settings_dict(self):
        return {
            "method": "fixed_sobol_multistart_then_scipy_l_bfgs_b",
            "sobol_points": self.sobol_points,
            "max_iterations": self.max_iterations,
            "seed": self.seed,
            "bounds": self.bounds.tolist(),
            "checkpoint": self.potential.checkpoint_record(),
        }

    def maximize(self, cars, joint_params, ego_idx, llm_theta):
        joint = np.asarray(joint_params, dtype=float).reshape(3, 5).copy()
        llm_theta = np.asarray(llm_theta, dtype=float).reshape(5)
        state_features = self.potential.state_features(cars, joint)
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        span = upper - lower

        exponent = int(np.ceil(np.log2(self.sobol_points)))
        unit = qmc.Sobol(d=5, scramble=True, seed=self.seed).random_base2(exponent)
        starts = lower + unit[: self.sobol_points] * span
        starts = np.vstack((llm_theta, joint[ego_idx], starts))
        starts = np.unique(np.clip(starts, lower, upper), axis=0)

        def score(theta):
            candidate = joint.copy()
            candidate[ego_idx] = theta
            return self.potential.score(state_features, candidate)

        llm_potential = score(llm_theta)
        best_theta = llm_theta.copy()
        best_value = llm_potential
        converged = 0
        messages = []
        for start in starts:
            result = minimize(
                lambda theta: -score(theta),
                start,
                method="L-BFGS-B",
                bounds=[tuple(row) for row in self.bounds],
                # The network evaluates float32 inputs; a finite-difference
                # step near machine-double epsilon would look identically zero.
                options={"maxiter": self.max_iterations, "ftol": 1e-10, "eps": 1e-4},
            )
            value = score(result.x)
            converged += int(result.success)
            messages.append(str(result.message))
            if np.isfinite(value) and value > best_value:
                best_theta = np.asarray(result.x, dtype=float).copy()
                best_value = float(value)

        tolerance = self.boundary_tolerance * np.maximum(span, 1.0)
        saturated = (best_theta - lower <= tolerance) | (
            upper - best_theta <= tolerance
        )
        return PotentialOptimizationResult(
            theta=tuple(float(value) for value in best_theta),
            potential=float(best_value),
            llm_potential=float(llm_potential),
            predicted_difference=float(best_value - llm_potential),
            method="fixed_sobol_multistart_then_scipy_l_bfgs_b",
            seed=self.seed,
            starts=len(starts),
            converged_starts=converged,
            successful=bool(np.isfinite(best_value) and converged > 0),
            boundary_saturated=tuple(bool(value) for value in saturated),
            messages=tuple(messages),
        )


def observation_cars(observation):
    """Convert a three-car RaceObservation without changing global role order."""
    if len(observation.cars) < 3:
        raise ValueError("The learned potential requires exactly three race cars")
    return [
        {
            "px": car.s,
            "py": car.e,
            "vx": car.state.vx,
            "vy": car.state.vy,
            "omega": car.state.omega,
            "psi": car.state.psi,
            "state": car.state,
        }
        for car in observation.cars[:3]
    ]
