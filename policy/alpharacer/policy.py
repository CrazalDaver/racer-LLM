import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .network import (
    ALPHA_PARAMETER_BOUNDS,
    alpha_network_input,
    build_alpha_input,
    load_potential_checkpoint,
)
from ..mpc import MPCPolicy

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
        weight_path="policy/alpharacer/train/weights/potential.pth",
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        (
            self.weight_path,
            self.potential_function,
            self.weight_metadata,
            self.weight_parameterization,
        ) = load_potential_checkpoint(weight_path, self.device)

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
        self._oracle_rollout_frozen = False
        if self.car_idx is not None and 0 <= self.car_idx < 3:
            self.joint_params[self.car_idx] = self.params
        # This is this agent's own estimate of the joint policy parameters.
        # It is deliberately not shared with other AlphaRacerPolicy instances:
        # each agent solves the paper's online potential maximization from its
        # own model/view and executes only its own five-parameter block.

    def _build_joint_input(self, cars, joint_params):
        """Build the canonical 39-D input without reordering vehicle roles."""
        return build_alpha_input(self.track, self.mpc.dt, self.mpc.horizon, cars, joint_params)

    def _network_input(self, paper_input):
        """Convert only legacy checkpoint inputs, preserving autograd."""
        return alpha_network_input(
            paper_input,
            self.weight_parameterization,
        )

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

    def get_action(self, observation):
        """Optimize private joint theta from the frozen global car ordering, then MPC."""
        if self.car_idx is None or self.car_idx >= 3:
            raise ValueError("AlphaRacerPolicy supports the three-car RaceObservation ordering")
        cars = []
        for car in observation.cars[:3]:
            cars.append(
                {
                    "px": car.s,
                    "py": car.e,
                    "vx": car.vx_body,
                    "vy": car.vy_body,
                    "omega": car.state.omega,
                    "psi": car.state.psi,
                    "heading_error": -car.heading_error,  # network uses tangent minus body
                    "state": car.state,
                }
            )
        if len(cars) < 3:
            raise ValueError("AlphaRacerPolicy requires a three-car RaceObservation")
        if not self._oracle_rollout_frozen:
            joint = self._maximize_joint_params(cars, self.joint_params)
            self.joint_params = joint.copy()
            self.params = joint[self.car_idx].copy()
        return super().get_action(observation)

    def freeze_for_oracle_rollout(self):
        """Hold the current high-level parameters in a diagnostic branch."""
        self._oracle_rollout_frozen = True
