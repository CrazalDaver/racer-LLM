"""Track geometry and its offline minimum-curvature candidate generator."""

from .track import RaceLine, Track, load_centerline_csv, load_raceline_csv

__all__ = ["RaceLine", "Track", "load_centerline_csv", "load_raceline_csv"]

from dataclasses import dataclass
from pathlib import Path
from config import TRACKS as _TRACK_CONFIGS, TRACK_DIR


@dataclass(frozen=True)
class TrackSpec:
    centerline: str
    raceline: str
    scale: float = 1.0
    half_width: float = 0.5


TRACKS = {
    name: TrackSpec(
        str(c.path), str(TRACK_DIR / (name + "_raceline.csv")), c.scale, c.half_width_m
    )
    for name, c in _TRACK_CONFIGS.items()
}

TRACK_ALIASES = {
    "berlin_2018": "berlin",
    "modena_2019": "modena",
    "rounded-rectangle": "rounded_rectangle",
    "rectangle": "rounded_rectangle",
}


def get_track_spec(name: str = "berlin") -> TrackSpec:
    """Return a registered alpha-RACER track configuration."""
    key = str(name).strip().lower()
    key = TRACK_ALIASES.get(key, key)
    try:
        return TRACKS[key]
    except KeyError as exc:
        choices = ", ".join(TRACKS)
        raise ValueError(f"Unknown track {name!r}; choose one of: {choices}") from exc


def configure_track(
    sim_params, name: str = "berlin", *, scale: float = None
) -> TrackSpec:
    """Apply a registered track to the mutable simulation parameters."""
    spec = get_track_spec(name)
    for filename in (spec.centerline, spec.raceline):
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Track file is missing: {filename}")
    sim_params.TRACK_CSV = spec.centerline
    sim_params.RACELINE_CSV = spec.raceline
    applied_scale = spec.scale if scale is None else float(scale)
    applied_half_width = spec.half_width * applied_scale / spec.scale
    sim_params.TRACK_SCALE = applied_scale
    sim_params.TRACK_HALF_WIDTH = applied_half_width
    sim_params.TRACK_WIDTH = 2.0 * applied_half_width
    # Track-boundary handling uses the vehicle centre as a point. Vehicle
    # width is reserved for rectangle contact detection and visualization.
    sim_params.MPC_CONSTRAINT_HALF_WIDTH = applied_half_width
    ref_boundary_margin = float(getattr(sim_params, "REF_BOUNDARY_MARGIN", 0.0))
    sim_params.REF_LATERAL_CLIP = max(applied_half_width - ref_boundary_margin, 0.0)
    return spec
