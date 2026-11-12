"""Low-rate LLM mode selection with per-step learned-potential refinement."""

import numpy as np
import torch

from ..alpharacer.policy import AlphaRacerPolicy
from .base import LLMPolicyBase
from .mode import ModeDecisionMixin


class LLMModeAlphaRacerPolicy(
    ModeDecisionMixin,
    LLMPolicyBase,
    AlphaRacerPolicy,
):
    """Choose a mode with the LLM, then optimize near that mode every step.

    ``LLMPolicyBase.get_action`` first reads the unified ``RaceObservation`` and
    updates the mode when due. It then calls ``AlphaRacerPolicy.get_action``,
    whose learned potential searches inside the selected mode's local bounds
    before MPC computes the physical action.
    """

    def __init__(self, *args, search_radius, **kwargs):
        """Validate the local search radius and initialize both decision layers."""
        radius = np.asarray(search_radius, dtype=float)
        if radius.shape != (5,) or not np.isfinite(radius).all() or np.any(radius < 0):
            raise ValueError("search_radius must contain five finite nonnegative values")
        kwargs["include_target_memory"] = False
        super().__init__(*args, **kwargs)
        if self.car_idx is None or not 0 <= self.car_idx < 3:
            raise ValueError("LLMModeAlphaRacerPolicy requires car_idx in 0..2")
        self.search_radius = radius.copy()
        bounds = np.asarray(self.PARAM_BOUNDS, dtype=float)
        for name, preset in self.mode_presets.items():
            if (
                preset.shape != (5,)
                or not np.isfinite(preset).all()
                or np.any(preset < bounds[:, 0])
                or np.any(preset > bounds[:, 1])
            ):
                raise ValueError(f"Mode preset {name!r} is outside AlphaRacer bounds")
        self.params = self.mode_presets[self.current_mode].copy()
        self.last_llm_params = self.params.copy()
        self.joint_params[self.car_idx] = self.params

    def apply_response(self, response, observation, observed):
        """Apply the selected preset and use it as the next local-search center."""
        super().apply_response(response, observation, observed)
        self.joint_params[self.car_idx] = self.params.copy()

    def _raceline_frenet_point(
        self,
        raceline_s,
        projection_s_hint=None,
        *,
        centerline_s_hint=None,
    ):
        """Accept both historical projection-hint names at the shared boundary."""
        if centerline_s_hint is not None:
            if projection_s_hint is not None:
                raise ValueError("Only one centerline projection hint may be supplied")
            projection_s_hint = centerline_s_hint
        return super()._raceline_frenet_point(raceline_s, projection_s_hint)

    def _maximize_joint_params(self, cars, initial_joint):
        """Optimize all cars while constraining this car near the selected preset."""
        joint = np.asarray(initial_joint, dtype=np.float32).reshape(3, 5).copy()
        global_bounds = np.asarray(self.PARAM_BOUNDS, dtype=np.float32)
        lower = np.tile(global_bounds[:, 0], (3, 1))
        upper = np.tile(global_bounds[:, 1], (3, 1))
        center = np.asarray(self.mode_presets[self.current_mode], dtype=np.float32)
        lower[self.car_idx] = np.maximum(lower[self.car_idx], center - self.search_radius)
        upper[self.car_idx] = np.minimum(upper[self.car_idx], center + self.search_radius)
        joint = np.clip(joint, lower, upper)

        if self.opt_style == "linear":
            for _ in range(self.search_iters):
                for flat_idx in range(15):
                    car, dim = divmod(flat_idx, 5)
                    candidates = np.repeat(joint[None, :, :], self.search_points, axis=0)
                    candidates[:, car, dim] = np.linspace(
                        lower[car, dim], upper[car, dim], self.search_points
                    )
                    scores = self._score_joint_params(cars, candidates)
                    if not np.isfinite(scores).all():
                        raise RuntimeError("Nonfinite learned potential during local search")
                    joint = candidates[int(np.argmax(scores))].copy()
            return joint

        base = self._build_joint_input(cars, joint)
        state = torch.tensor(base[None, :-15], dtype=torch.float32, device=self.device)
        global_lower = torch.tensor(
            np.tile(global_bounds[:, 0], 3), device=self.device
        )
        span = torch.tensor(
            np.tile(global_bounds[:, 1] - global_bounds[:, 0], 3),
            device=self.device,
        )
        low = (torch.tensor(lower.reshape(-1), device=self.device) - global_lower) / span
        high = (torch.tensor(upper.reshape(-1), device=self.device) - global_lower) / span
        xi = (
            (torch.tensor(joint.reshape(-1), device=self.device) - global_lower) / span
        )[None, :]
        for _ in range(self.search_iters):
            xi = xi.detach().requires_grad_(True)
            values = global_lower + xi[0] * span
            score = self._potential(torch.cat((state, values[None, :]), dim=-1)).sum()
            gradient = torch.autograd.grad(score, xi)[0]
            if not torch.isfinite(score).all() or not torch.isfinite(gradient).all():
                raise RuntimeError("Nonfinite learned potential or gradient during local search")
            with torch.no_grad():
                xi = torch.maximum(low, torch.minimum(high, xi + self.gradient_rate * gradient))
        answer = (global_lower + xi.detach()[0] * span).cpu().numpy().reshape(3, 5)
        if not np.isfinite(answer).all():
            raise RuntimeError("Nonfinite parameters from local potential search")
        return answer
