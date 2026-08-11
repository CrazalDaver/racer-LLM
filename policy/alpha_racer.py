import math
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from state import CarState

from .alpha_model import (
    ALPHA_PARAMETER_BOUNDS,
    SimpleNeuralNetwork,
    build_alpha_input,
)
from .mpc import MPCPolicy
from .utils import progress_delta

class AlphaRacerPolicy(MPCPolicy):
    """
    alpha-RACER online policy that searches for parameters maximizing the
    learned alpha-potential function.

    Implementation of the alpha-potential based policy from the paper:
    https://arxiv.org/pdf/2412.08855v3

    The repository-wide parameterization is the paper's
    theta=(q, zeta, s1, s2, s3). Released weights are detected as legacy
    checkpoints and receive an internal, differentiable input conversion;
    this never leaks their transformed coordinates into the policy API.
    """

    PARAM_BOUNDS = ALPHA_PARAMETER_BOUNDS

    def __init__(
        self,
        *args,
        weight_path="train/weights/alpha_racer/upstream/potential_myopic_mpc.pth",
        search_points=7,
        search_iters=6,
        opt_style="grad",
        gradient_rate=1e-4,
        initial_params=None,
        opponent_default_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if torch is None:
            raise ImportError("PyTorch is required to use AlphaRacerPolicy.")

        self.weight_path = Path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"alpha-RACER weight file not found: {self.weight_path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.potential_function = SimpleNeuralNetwork().to(self.device)
        checkpoint = torch.load(str(self.weight_path), map_location=self.device)
        # Local checkpoints include experiment metadata; released upstream
        # files are raw state dicts. Supporting both keeps the baseline intact.
        wrapped_checkpoint = isinstance(checkpoint, dict) and "state_dict" in checkpoint
        state_dict = checkpoint["state_dict"] if wrapped_checkpoint else checkpoint
        metadata = checkpoint.get("metadata", {}) if wrapped_checkpoint else {}
        if wrapped_checkpoint and "parameterization" not in metadata:
            raise ValueError(
                "Local alpha-RACER checkpoint metadata must declare its parameterization; "
                "retrain or add parameterization='paper_q_zeta_s1_s2_s3'."
            )
        # Released files are raw state dicts and are the only implicitly
        # legacy format. Locally produced wrapped checkpoints are explicit.
        self.weight_parameterization = metadata.get("parameterization", "upstream_transformed")
        if self.weight_parameterization not in {"paper_q_zeta_s1_s2_s3", "upstream_transformed"}:
            raise ValueError(f"Unsupported alpha-RACER checkpoint parameterization: {self.weight_parameterization}")
        self.potential_function.load_state_dict(state_dict)
        self.potential_function.eval()

        self.search_points = max(2, int(search_points))
        self.search_iters = max(1, int(search_iters))
        self.opt_style = str(opt_style).strip().lower()
        if self.opt_style not in {"grad", "linear"}:
            raise ValueError(f"Unknown AlphaRacer optimization style: {opt_style}")
        self.gradient_rate = float(gradient_rate)
        if initial_params is None:
            initial_params = [2.0, 0.95**2, 0.3, 0.2, 0.0]
        self.params = np.asarray(initial_params, dtype=float).copy()
        if self.params.shape != (5,):
            raise ValueError("initial_params must contain q, zeta, s1, s2, s3")
        self.opponent_default_params = np.asarray(
            self.params if opponent_default_params is None else opponent_default_params,
            dtype=float,
        ).copy()
        self.current_mode = "potential"
        self.joint_params = np.repeat(self.opponent_default_params[None, :], 3, axis=0)
        if self.car_idx is not None and 0 <= self.car_idx < 3:
            self.joint_params[self.car_idx] = self.params
        self._warned_missing_opponents = False
        # This is this agent's own estimate of the joint policy parameters.
        # It is deliberately not shared with other AlphaRacerPolicy instances:
        # each agent solves the paper's online potential maximization from its
        # own model/view and executes only its own five-parameter block.
        self._joint_plan_ready = False
    
    def _ordered_cars(self, state: CarState, current_s: float, opponents):
        cars = [
            {
                "px": current_s,
                "py": self.track.project(
                    state.x,
                    state.y,
                    s_hint=current_s,
                    search_radius=8.0,
                    psi_hint=state.psi,
                    max_heading_error=math.pi / 2.0,
                )[1],
                "vx": state.vx,
                "vy": state.vy,
                "omega": state.omega,
                "psi": state.psi,
                "state": state,
            }
        ]
        for opponent in opponents or []:
            cars.append(dict(opponent))

        if len(cars) < 3:
            if not self._warned_missing_opponents:
                self._log(f"[potential-function policy][{self.warning_label}] expected 3 cars; padding missing opponents.")
                self._warned_missing_opponents = True
            while len(cars) < 3:
                cars.append({"px": current_s, "py": 0.0, "vx": 0.0, "vy": 0.0, "omega": 0.0, "psi": state.psi})
        return cars[:3]

    def _build_joint_input(self, cars, joint_params):
        """Build the canonical 39-D input without reordering vehicle roles."""
        return build_alpha_input(self.track, self.mpc.dt, self.mpc.horizon, cars, joint_params)

    def _network_input(self, paper_input):
        """Convert only legacy checkpoint inputs, preserving autograd."""
        if self.weight_parameterization == "paper_q_zeta_s1_s2_s3":
            return paper_input
        state_features = paper_input[..., :-15]
        theta = paper_input[..., -15:].reshape(*paper_input.shape[:-1], 3, 5)
        q, zeta, s1, s2, s3 = theta.unbind(dim=-1)
        upstream = torch.stack(
            (s1, s2, 0.5 / torch.clamp(q, min=1e-6), torch.sqrt(torch.clamp(zeta, min=0.0)), s3),
            dim=-1,
        )
        return torch.cat((state_features, upstream.reshape(*paper_input.shape[:-1], 15)), dim=-1)

    def _potential(self, paper_input):
        return self.potential_function(self._network_input(paper_input))

    def _score_joint_params(self, cars, joint_params_batch):
        # use the learned alpha-potential function to score a batch of joint parameters, returning a numpy array of scores
        rows = [self._build_joint_input(cars, row) for row in joint_params_batch]
        with torch.no_grad():
            x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=self.device)
            return self._potential(x).detach().cpu().numpy().reshape(-1)

    def _maximize_joint_params(self, cars, initial_joint):
        """Warm-started optimizer matching the paper's default online phase."""
        joint = np.asarray(initial_joint, dtype=np.float32).reshape(3, 5).copy()
        bounds = self.PARAM_BOUNDS.astype(np.float32)
        if self.opt_style == "linear":
            for _ in range(self.search_iters):
                for flat_idx in range(15):
                    car_idx, dim = divmod(flat_idx, 5)
                    candidates = np.repeat(joint[None, :, :], self.search_points, axis=0)
                    candidates[:, car_idx, dim] = np.linspace(
                        bounds[dim, 0], bounds[dim, 1], self.search_points
                    )
                    scores = self._score_joint_params(cars, candidates)
                    joint = candidates[int(np.argmax(scores))].copy()
            return joint
        # Canonical projected gradient ascent. All checkpoints optimize the
        # same normalized paper coordinates xi in [0,1]^15. Official weights
        # differ only at _network_input(), after theta has been reconstructed.
        base = self._build_joint_input(cars, joint)
        state_features = torch.tensor(base[None, :-15], dtype=torch.float32, device=self.device)
        lower = torch.tensor(bounds[:, 0], dtype=torch.float32, device=self.device).repeat(3)
        span = torch.tensor(bounds[:, 1] - bounds[:, 0], dtype=torch.float32, device=self.device).repeat(3)
        theta0 = torch.tensor(joint.reshape(-1), dtype=torch.float32, device=self.device)
        xi = ((theta0 - lower) / span).clamp(0.0, 1.0)[None, :]
        for _ in range(self.search_iters):
            xi = xi.detach().requires_grad_(True)
            theta = lower + xi[0] * span
            x = torch.cat((state_features, theta[None, :]), dim=-1)
            score = self._potential(x).sum()
            gradient = torch.autograd.grad(score, xi)[0]
            with torch.no_grad():
                xi = (xi + self.gradient_rate * gradient).clamp(0.0, 1.0)
        theta = lower + xi.detach()[0] * span
        return theta.cpu().numpy().reshape(3, 5)

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        if self._joint_plan_ready:  # cache
            self._joint_plan_ready = False
        else:
            # Standalone use (outside simulation.py): still optimize the full
            # joint theta.  Here the policy is role 0 and opponent blocks are
            # private latent estimates; only role 0 is executed.
            cars = self._ordered_cars(state, current_s, opponents or [])
            local_joint = np.repeat(
                self.opponent_default_params[None, :], 3, axis=0
            )
            local_joint[0] = self.params
            local_joint = self._maximize_joint_params(cars, local_joint)
            self.params = local_joint[0].copy()
        return super().get_reference_trajectory(state, current_s, opponents=opponents)


def plan_alpharacer_policy_params(policies, states, car_s, car_e):
    """Run the paper's online alpha-RACER update independently for each agent.

    Every AlphaRacerPolicy maximizes its own learned potential over the full
    joint theta, retains that private joint estimate, and executes only the
    five-parameter block belonging to its vehicle.  The low-level MPC remains
    a separate per-vehicle optimization, as in equations (1a)-(1h).
    """
    alpha_indices = [idx for idx, policy in enumerate(policies) if isinstance(policy, AlphaRacerPolicy)]
    if not alpha_indices:
        return

    cars = [
        {
            "px": float(car_s[idx]),
            "py": float(car_e[idx]),
            "vx": float(states[idx].vx),
            "vy": float(states[idx].vy),
            "omega": float(states[idx].omega),
            "psi": float(states[idx].psi),
            "state": states[idx],
        }
        for idx in range(min(len(states), 3))
    ]
    if not cars:
        return

    for idx in alpha_indices:
        if idx >= 3:
            continue
        policy = policies[idx]
        joint = policy._maximize_joint_params(cars, policy.joint_params)
        policy.joint_params = joint.copy()
        policy.params = joint[idx].copy()
        policy._joint_plan_ready = True
