import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackSpec:
    centerline: str
    raceline: str
    scale: float = 1.0
    half_width: float = 0.5


# alpha-RACER's numerical simulator applies track_info.scale=2.0 to Berlin.
# Modena uses the same 1:31 source-to-ref_trajs conversion, so it follows the
# same numerical scaling. Rounded rectangle uses a separate 1:10 conversion
# and already has a 0.5 m half-width at scale 1.0.
TRACKS: Mapping[str, TrackSpec] = {
    "berlin": TrackSpec(
        "track_csv/berlin_2018.csv",
        "track_csv/berlin_2018_raceline_with_speeds.csv",
        2.0,
        0.44,
    ),
    "modena": TrackSpec(
        "track_csv/modena_2019.csv",
        "track_csv/modena_2019_with_speeds.csv",
        2.0,
        0.37,
    ),
    "rounded_rectangle": TrackSpec(
        "track_csv/rounded_rectangle.csv",
        "track_csv/rounded_rectangle_with_speeds.csv",
        1.0,
        0.50,
    ),
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


def configure_track(sim_params, name: str = "berlin", *, scale: float = None) -> TrackSpec:
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
    vehicle_half_width = float(getattr(sim_params, "VEHICLE_HALF_WIDTH", 0.0))
    center_half_width = applied_half_width - vehicle_half_width
    if center_half_width <= 0.0:
        raise ValueError(
            "Track half-width must be greater than VEHICLE_HALF_WIDTH "
            f"({applied_half_width:g} <= {vehicle_half_width:g})"
        )
    # Both the MPC boundary and reference clipping operate on the vehicle
    # center; subtracting half the car width keeps the full body on track.
    sim_params.MPC_CONSTRAINT_HALF_WIDTH = center_half_width
    ref_boundary_margin = float(getattr(sim_params, "REF_BOUNDARY_MARGIN", 0.0))
    sim_params.REF_LATERAL_CLIP = max(center_half_width - ref_boundary_margin, 0.0)
    return spec


class ProjectablePath:
    """
    Base class for paths that support projection of points onto the path.
    """
    def _precompute_projection(self, projection_waypoints: np.ndarray, set_total_length: bool = False):
        self._projection_waypoints = np.asarray(projection_waypoints, dtype=float)
        self._seg_vectors = np.diff(self._projection_waypoints, axis=0) # The racetrack is composed of connected vector segments.
        self._seg_lengths = np.linalg.norm(self._seg_vectors, axis=1)
        self._cum_len = np.cumsum(np.insert(self._seg_lengths, 0, 0))
        self._tangents = self._seg_vectors / (self._seg_lengths[:, None] + 1e-8)
        self._normals = np.array([[-t[1], t[0]] for t in self._tangents])  # perpendicular 
        self._normals = self._normals / (np.linalg.norm(self._normals, axis=1, keepdims=True) + 1e-8)

        if set_total_length:
            self.total_length = self._cum_len[-1]

        self.seg_vectors = self._seg_vectors
        self.seg_lengths = self._seg_lengths
        self.cum_len = self._cum_len
        self.tangents = self._tangents
        self.normals = self._normals

    def project(
        self,
        x: float,
        y: float,
        s_hint: float = None,
        search_radius: float = None,
        psi_hint: float = None,
        max_heading_error: float = None,
        allow_global_fallback: bool = True,
    ) -> Tuple[float, float]:
        result = self._project_impl(
            x,
            y,
            s_hint=s_hint,
            search_radius=search_radius,
            psi_hint=psi_hint,
            max_heading_error=max_heading_error,
        )
        if result is not None:
            return result
        if s_hint is not None and search_radius is not None:
            result = self._project_impl(
                x,
                y,
                s_hint=s_hint,
                search_radius=search_radius * 2.0,
            )
            if result is not None:
                return result
        if allow_global_fallback:
            return self._project_impl(x, y)
        raise ValueError("Projection failed within the local progress window.")

    def _project_impl(  # 
        self,
        x: float,
        y: float,
        s_hint: float = None,
        search_radius: float = None,
        psi_hint: float = None,
        max_heading_error: float = None,
    ) -> Tuple[float, float]:
        pos = np.array([x, y])
        best_dist = np.inf
        best_idx = -1
        best_t = 0.0
        best_closest = None

        for i, (p0, p1) in enumerate(zip(self._projection_waypoints[:-1], self._projection_waypoints[1:])):
            if not _is_near_progress(self.cum_len[i], s_hint, search_radius, self.total_length):
                continue
            if not _is_heading_compatible(self.tangents[i], psi_hint, max_heading_error):
                continue
            seg_vec = p1 - p0
            seg_len = self.seg_lengths[i]
            if seg_len < 1e-6:
                continue
            t = np.dot(pos - p0, seg_vec) / (seg_len * seg_len)
            t = np.clip(t, 0.0, 1.0)
            closest = p0 + t * seg_vec
            dist = np.linalg.norm(pos - closest)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_t = t
                best_closest = closest

        if best_idx == -1:
            return None

        normal = self.normals[best_idx]
        signed_e = np.dot(pos - best_closest, normal)
        s = self.cum_len[best_idx] + best_t * self.seg_lengths[best_idx]
        return _unwrap_progress(s % self.total_length, s_hint, self.total_length), signed_e


class Track(ProjectablePath):
    def __init__(self, waypoints: np.ndarray, closed: bool = True):
        self.waypoints = np.asarray(waypoints, dtype=float)
        if closed and np.linalg.norm(self.waypoints[0] - self.waypoints[-1]) > 1e-6:
            self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        self._precompute_projection(self.waypoints, set_total_length=True)

    def interpolate(self, s: float) -> Tuple[float, float, float]:
        if self.total_length > 0:
            s = s % self.total_length
        if s <= 0:
            return self.waypoints[0][0], self.waypoints[0][1], self._get_psi_at_seg(0)
        if s >= self.total_length:
            idx = len(self.waypoints) - 2
            return self.waypoints[-1][0], self.waypoints[-1][1], self._get_psi_at_seg(idx)

        idx = np.searchsorted(self.cum_len[1:], s)
        local_s = s - self.cum_len[idx]
        t = local_s / (self.seg_lengths[idx] + 1e-8)
        p0, p1 = self.waypoints[idx], self.waypoints[idx + 1]
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        psi = self._get_psi_at_seg(idx)
        return x, y, psi

    def frenet_to_world(self, s: float, e: float) -> Tuple[float, float]:
        x, y, psi = self.interpolate(s)
        return x - e * math.sin(psi), y + e * math.cos(psi)

    def _get_psi_at_seg(self, idx: int) -> float:
        if idx >= len(self.tangents):
            idx = len(self.tangents) - 1
        dx, dy = self.tangents[idx]
        return math.atan2(dy, dx)


def create_test_track(num_points=300, radius=14.0):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.column_stack([x, y])


class RaceLine(ProjectablePath):
    """
    alpha-RACER style raceline.

    Supported CSV layouts:
    - s_m,x_m,y_m,vx_mps
    - s_m,x_m,y_m,vx_mps,dev_m
    Extra columns are preserved in `extra`.
    """

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 4:
            raise ValueError("Raceline must have at least columns: s_m, x_m, y_m, vx_mps")

        self.data = data
        self.s = data[:, 0]
        self.x = data[:, 1]
        self.y = data[:, 2]
        self.speed = data[:, 3]
        self.dev = data[:, 4] if data.shape[1] >= 5 else np.zeros(len(data))
        self.extra = data[:, 5:] if data.shape[1] > 5 else np.empty((len(data), 0))
        self.psi = self._compute_heading()
        self.waypoints = np.column_stack([self.x, self.y])
        self.total_length = float(self.s[-1] + np.linalg.norm(self.waypoints[0] - self.waypoints[-1]))
        self._precompute_segments()

    def interpolate(self, s: float) -> dict:
        s_wrapped = s % self.total_length
        s_ext = np.r_[self.s, self.total_length]
        x_ext = np.r_[self.x, self.x[0]]
        y_ext = np.r_[self.y, self.y[0]]
        speed_ext = np.r_[self.speed, self.speed[0]]
        dev_ext = np.r_[self.dev, self.dev[0]]
        psi_unwrapped = np.unwrap(self.psi)
        psi0_closed = psi_unwrapped[-1] + math.atan2(
            math.sin(psi_unwrapped[0] - psi_unwrapped[-1]),
            math.cos(psi_unwrapped[0] - psi_unwrapped[-1]),
        )
        psi_ext = np.r_[psi_unwrapped, psi0_closed]

        return {
            "s_m": s_wrapped,
            "x_m": np.interp(s_wrapped, s_ext, x_ext),
            "y_m": np.interp(s_wrapped, s_ext, y_ext),
            "vx_mps": np.interp(s_wrapped, s_ext, speed_ext),
            "psi_rad": np.interp(s_wrapped, s_ext, psi_ext),
            "dev_m": np.interp(s_wrapped, s_ext, dev_ext),
        }

    def nearest_index(self, x: float, y: float) -> int:
        dx = self.x - x
        dy = self.y - y
        return int(np.argmin(dx * dx + dy * dy))

    def _compute_heading(self):
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        return np.arctan2(dy, dx)

    def _precompute_segments(self):
        self._precompute_projection(np.vstack([self.waypoints, self.waypoints[0]]))


def load_centerline_csv(path, scale: float = 1.0) -> np.ndarray:
    data = np.genfromtxt(Path(path), delimiter=",", comments="#")
    data = np.asarray(data, dtype=float)
    scaled = data.copy()
    scaled[:, :2] *= float(scale)
    return scaled


def load_raceline_csv(path, scale: float = 1.0) -> RaceLine:
    data = np.genfromtxt(Path(path), delimiter=",", comments="#")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Some upstream alpha-RACER files have an uncommented text header.
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[1] == 3:
        # Native alpha-RACER layout: x_m, y_m, vx_mps.  RaceLine uses an
        # explicit progress column, so derive it from consecutive points.
        xy = data[:, :2]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        data = np.column_stack([s, xy, data[:, 2]])
    scaled = data.copy()
    scaled[:, 0] *= float(scale)
    scaled[:, 1:3] *= float(scale)
    if scaled.shape[1] >= 5:
        scaled[:, 4] *= float(scale)
    return RaceLine(scaled)


def _is_near_progress(seg_s: float, s_hint: float, search_radius: float, total_length: float) -> bool:
    if s_hint is None or search_radius is None:
        return True
    delta = abs((seg_s - s_hint) % total_length)
    return min(delta, total_length - delta) <= search_radius


def _is_heading_compatible(tangent: np.ndarray, psi_hint: float, max_heading_error: float) -> bool:
    if psi_hint is None or max_heading_error is None:
        return True
    seg_psi = math.atan2(tangent[1], tangent[0])
    err = math.atan2(math.sin(seg_psi - psi_hint), math.cos(seg_psi - psi_hint))
    return abs(err) <= max_heading_error


def _unwrap_progress(s: float, s_hint: float, total_length: float) -> float:
    if s_hint is None:
        return s
    return s_hint + math.atan2(
        math.sin((s - s_hint) * 2.0 * math.pi / total_length),
        math.cos((s - s_hint) * 2.0 * math.pi / total_length),
    ) * total_length / (2.0 * math.pi)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run simulation.py on a selectable alpha-RACER track.")
    parser.add_argument("--track", default="berlin", help="berlin (default), modena, or rounded_rectangle")
    parser.add_argument("--list", action="store_true", help="list registered tracks and exit")
    parser.add_argument("--scale", type=float, default=None, help="override the selected track's configured scale")
    parser.add_argument("--policy", default="centerline")
    parser.add_argument("--policies", nargs="+", default=None, help="optional per-car policy names")
    parser.add_argument("--num-cars", type=int, default=None)
    parser.add_argument("--vehicle-model", choices=["kinematic", "dbm"], default=None)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--ego-position", type=int, default=1)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    """Command-line entry point for selecting a track and running simulation."""
    args = _parse_args()
    if args.list:
        for name, spec in TRACKS.items():
            print(
                f"{name}: scale={spec.scale:g}, half_width={spec.half_width:g}, "
                f"centerline={spec.centerline}, raceline={spec.raceline}"
            )
        return

    from simulation import run_simulation

    run_simulation(
        policy_type=args.policy,
        track_name=args.track,
        track_scale=args.scale,
        policy_names=args.policies,
        num_cars=args.num_cars,
        vehicle_model_type=args.vehicle_model,
        v0=args.v0,
        ego_position=args.ego_position,
        save_trace=not args.no_trace,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
