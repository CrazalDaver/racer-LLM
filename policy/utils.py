import math

import numpy as np

from track import Track


def relative_progress(progress, player_idx):
    """Return one player's progress minus the furthest-ahead opponent.

    ``progress`` may be one joint progress vector or a batch whose last axis
    lists cars.  This is the single numerical definition of the one-step
    utility used by alpha-RACER, IBR, self-play RL, and rollout diagnostics.
    """
    values = np.asarray(progress, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("relative progress requires at least two cars")
    player_idx = int(player_idx)
    if not 0 <= player_idx < values.shape[-1]:
        raise IndexError("player_idx is outside the progress vector")
    if not np.isfinite(values).all():
        raise ValueError("relative progress requires finite values")
    opponents = np.delete(values, player_idx, axis=-1)
    return values[..., player_idx] - np.max(opponents, axis=-1)


def relative_progress_gain(start_progress, final_progress, player_idx):
    """Paper utility over an interval: final relative progress minus initial."""
    return relative_progress(final_progress, player_idx) - relative_progress(
        start_progress, player_idx
    )


def progress_delta(s_a: float, s_b: float, total_length: float) -> float:
    """Signed shortest progress from ``s_b`` to ``s_a`` on a closed track."""
    return math.atan2(
        math.sin((s_a - s_b) * 2.0 * math.pi / total_length),
        math.cos((s_a - s_b) * 2.0 * math.pi / total_length),
    ) * total_length / (2.0 * math.pi)


def angle_delta(angle_a: float, angle_b: float) -> float:
    """Return the wrapped signed angle from ``angle_b`` to ``angle_a``."""
    return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))


def track_heading_error(track: Track, psi: float, s: float, reference_minus_heading: bool = True) -> float:
    """Wrapped heading error relative to the centerline at progress ``s``."""
    _, _, ref_psi = track.interpolate(s)
    if reference_minus_heading:
        return angle_delta(ref_psi, psi)
    return angle_delta(psi, ref_psi)


def track_curvature_at(track: Track, s: float, ds: float = 0.25) -> float:
    """Estimate centerline curvature with a centered heading difference."""
    _, _, psi_prev = track.interpolate(s - ds)
    _, _, psi_next = track.interpolate(s + ds)
    return angle_delta(psi_next, psi_prev) / (2.0 * ds)

