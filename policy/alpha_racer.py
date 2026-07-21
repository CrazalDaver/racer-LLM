import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from state import CarState

from .mpc import MPCPolicy
from .utils import progress_delta, track_curvature_at, track_heading_error

class _LearnedAlphaPotentialFunction(nn.Module if nn is not None else object):
    """Neural network approximation of the alpha-potential function."""

    def __init__(self, input_size, hidden_size, output_size):
        if nn is None:
            raise ImportError("PyTorch is required for the learned alpha-potential policy.")
        super().__init__()
        layers = []
        self.batch_norm = nn.BatchNorm1d(input_size, affine=False)
        current = input_size
        for width in hidden_size:
            layers.append(nn.Linear(current, width))
            layers.append(nn.ReLU())
            current = width
        layers.append(nn.Linear(current, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.batch_norm(x)
        elif len(x.shape) == 3:
            flat = self.batch_norm(x.reshape((-1, x.shape[-1])))
            x = flat.reshape(x.shape)
        return self.fc(x)


class AlphaRacerPolicy(MPCPolicy):
    """
    alpha-RACER online policy that searches for parameters maximizing the
    learned alpha-potential function.

    Implementation of the alpha-potential based policy from the paper:
    https://arxiv.org/pdf/2412.08855v3

    The original model scores 5 policy parameters per car:
    sf1, sf2, lookahead_factor, speed_factor, blocking. This project uses
    q, zeta, s1, s2, s3 in MPC, so the selected parameters are mapped as:
    The upstream implementation stores q/2 as ``lookahead_factor`` and
    sqrt(zeta) as ``speed_factor``.  This project's MPC directly multiplies
    its tracking cost by q_weight, matching the paper's Q=qI definition, so
    the executable map is q=2*lookahead_factor, zeta=speed_factor^2,
    s1=sf1, s2=sf2, s3=blocking.
    """

    ORIGINAL_PARAM_BOUNDS = np.asarray(
        [
            [0.1, 0.5],   # s1
            [0.1, 0.5],   # s2
            [0.15, 0.5],  # lookahead_factor = q/2
            [0.85, 1.1],  # speed_factor = zeta^0.5
            [0.0, 1.0],   # s3
        ],
        dtype=float,
    )

    def __init__(
        self,
        *args,
        weight_path="alpha_racer_weights/p_models_rel/model_multi_myopic_mpc.pth",
        search_points=7,
        search_iters=6,
        opt_style="grad",
        gradient_rate=1e-4,
        initial_original_params=None,
        opponent_default_original_params=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if torch is None:
            raise ImportError("PyTorch is required to use AlphaRacerPolicy.")

        self.weight_path = Path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"alpha-RACER weight file not found: {self.weight_path}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.potential_function = _LearnedAlphaPotentialFunction(39, [384, 384, 192], 1).to(self.device)
        state_dict = torch.load(str(self.weight_path), map_location=self.device)
        self.potential_function.load_state_dict(state_dict)
        self.potential_function.eval()

        self.search_points = max(2, int(search_points))
        self.search_iters = max(1, int(search_iters))
        self.opt_style = str(opt_style).strip().lower()
        if self.opt_style not in {"grad", "linear"}:
            raise ValueError(f"Unknown AlphaRacer optimization style: {opt_style}")
        self.gradient_rate = float(gradient_rate)
        if initial_original_params is None:
            initial_original_params = [0.3, 0.2, 0.25, 0.95, 0.0]
        self.original_params = np.asarray(initial_original_params, dtype=float).copy()
        if self.original_params.shape != (5,):
            raise ValueError("initial_original_params must contain exactly five values")
        self.opponent_default_original_params = np.asarray(
            self.original_params
            if opponent_default_original_params is None
            else opponent_default_original_params,
            dtype=float,
        ).copy()
        self.current_mode = "potential"
        self.params = self._map_original_params(self.original_params)
        self.joint_original_params = np.repeat(
            self.opponent_default_original_params[None, :], 3, axis=0
        )
        if self.car_idx is not None and 0 <= self.car_idx < 3:
            self.joint_original_params[self.car_idx] = self.original_params
        self._warned_missing_opponents = False
        # This is this agent's own estimate of the joint policy parameters.
        # It is deliberately not shared with other AlphaRacerPolicy instances:
        # each agent solves the paper's online potential maximization from its
        # own model/view and executes only its own five-parameter block.
        self._joint_plan_ready = False
    
    def _map_original_params(self, original_params):
        sf1, sf2, lookahead_factor, speed_factor, blocking = np.asarray(original_params, dtype=float)
        return np.asarray(
            [
                float(np.clip(2.0 * lookahead_factor, 1e-6, None)),
                float(np.clip(speed_factor * speed_factor, 0.1, 3.0)),
                float(max(sf1, 0.0)),
                float(max(sf2, 1e-3)),
                float(np.clip(blocking, 0.0, 5.0)),
            ],
            dtype=float,
        )

    def _wrap_delta(self, s_a, s_b):
        return progress_delta(float(s_a), float(s_b), self.track.total_length)

    def _car_feature_block(self, car):
        s = float(car["px"])
        state = car.get("state")
        psi = float(car.get("psi", getattr(state, "psi", 0.0)))
        vx = float(car.get("vx", getattr(state, "vx", 0.0)))
        vy = float(car.get("vy", getattr(state, "vy", 0.0)))
        omega = float(car.get("omega", getattr(state, "omega", 0.0)))
        return [
            float(car["py"]),
            track_heading_error(self.track, psi, s),
            vx,
            vy,
            omega,
        ]

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
        cars = list(cars[:3])
        # Pad missing cars with default values if fewer than 3 are provided.
        while len(cars) < 3:
            cars.append({"px": cars[0]["px"], "py": 0.0, "vx": 0.0, "vy": 0.0, "omega": 0.0, "psi": 0.0})
        s0, s1, s2 = [float(car["px"]) for car in cars]

        # longitudal distance deltas between the three cars, in order: (s2-s1), (s1-s0), (s2-s0)
        # 3 features
        features = [
            self._wrap_delta(s2, s1),
            self._wrap_delta(s1, s0),
            self._wrap_delta(s2, s0),
        ]

        # Frenet states features for each car: 3 * 5 = 15 features
        for car in cars:
            features.extend(self._car_feature_block(car))

        # track curvature features for each car: 3 features
        for car in cars:
            s = float(car["px"])
            features.append(track_curvature_at(self.track, s))
        
        # lookahead curvature features for each car: 3 features
        for car in cars:
            s = float(car["px"])
            features.append(
                track_curvature_at(
                    self.track,
                    s + max(float(car.get("vx", 0.0)), 0.0) * self.mpc.dt * self.mpc.horizon,
                )
            )

        # policy parameters for each car: 3 * 5 = 15 features
        param_blocks = np.asarray(joint_params, dtype=float).reshape(3, 5)
        for block in param_blocks:
            features.extend(block.tolist())

        x = np.asarray(features, dtype=np.float32)
        x[~np.isfinite(x)] = 0.0
        if x.shape[0] != 39:
            raise ValueError(f"Potential-function policy input must have 39 values, got {x.shape[0]}")
        return x

    def _score_joint_params(self, cars, joint_params_batch):
        # use the learned alpha-potential function to score a batch of joint parameters, returning a numpy array of scores
        rows = [self._build_joint_input(cars, row) for row in joint_params_batch]
        with torch.no_grad():
            x = torch.tensor(np.asarray(rows), dtype=torch.float32, device=self.device)
            return self.potential_function(x).detach().cpu().numpy().reshape(-1)

    def _maximize_joint_params(self, cars, initial_joint):
        """Warm-started optimizer matching the paper's default online phase."""
        joint = np.asarray(initial_joint, dtype=np.float32).reshape(3, 5).copy()
        bounds = self.ORIGINAL_PARAM_BOUNDS.astype(np.float32)
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
        # gradient-based optimization, using the learned alpha-potential function as a differentiable surrogate for the true objective
        base = self._build_joint_input(cars, joint)  # 39-D input for the current joint parameters
        x = torch.tensor(base[None, :], dtype=torch.float32, device=self.device)
        lower = torch.tensor(bounds[:, 0], dtype=torch.float32, device=self.device).repeat(3)  # lower bounds for the 15 policy parameters
        upper = torch.tensor(bounds[:, 1], dtype=torch.float32, device=self.device).repeat(3)  # repeat: 3 cars
        for _ in range(self.search_iters):
            x = x.detach().requires_grad_(True)
            score = self.potential_function(x).sum()
            gradient = torch.autograd.grad(score, x)[0]
            with torch.no_grad():
                policy_values = x[0, -15:] + self.gradient_rate * gradient[0, -15:]
                x[0, -15:] = torch.maximum(torch.minimum(policy_values, upper), lower)
        return x.detach().cpu().numpy()[0, -15:].reshape(3, 5)

    def get_reference_trajectory(self, state: CarState, current_s: float, opponents=None) -> np.ndarray:
        if self._joint_plan_ready:  # cache
            self._joint_plan_ready = False
        else:
            # Standalone use (outside simulation.py): still optimize the full
            # joint theta.  Here the policy is role 0 and opponent blocks are
            # private latent estimates; only role 0 is executed.
            cars = self._ordered_cars(state, current_s, opponents or [])
            local_joint = np.repeat(
                self.opponent_default_original_params[None, :], 3, axis=0
            )
            local_joint[0] = self.original_params
            local_joint = self._maximize_joint_params(cars, local_joint)
            self.original_params = local_joint[0].copy()
        self.params = self._map_original_params(self.original_params)
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
        joint = policy._maximize_joint_params(cars, policy.joint_original_params)
        policy.joint_original_params = joint.copy()
        policy.original_params = joint[idx].copy()
        policy.params = policy._map_original_params(policy.original_params)
        policy._joint_plan_ready = True


