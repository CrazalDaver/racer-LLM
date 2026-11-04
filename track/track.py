"""
Unified Class definition for track centerlines, optimal racelines, and related utilities,
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np


def frenet_denominator(raw, minimum=0.05, smoothing=1e-4, *, sqrt=np.sqrt):
    """Shared smooth guard for 1-kappa*e; sqrt may be numeric or symbolic."""
    return 0.5 * (raw + minimum + sqrt((raw - minimum)**2 + smoothing**2))


class ProjectablePath:
    """
    Base class for track paths that support projection of points onto the path.
    
    Methods:
        - project: Projects a point (x, y) onto the path and returns the closest
          point on the path.        
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

    def _project_impl(
        self,
        x: float,
        y: float,
        s_hint: float = None,
        search_radius: float = None,
        psi_hint: float = None,
        max_heading_error: float = None,
    ) -> Tuple[float, float]:
        """
        Projects a point (x, y) onto the path and returns the closest point on the path.
        """
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
    """
    Track centerline and related utilities.

    Methods:
        - interpolate: Interpolates the track at a given progress s and returns
          the corresponding (x, y) coordinates and heading psi.
        - frenet_to_world: Converts Frenet coordinates (s, e) to world coordinates (x, y).
        - curvature_profile: Returns a periodic, numerically smoothed centerline curvature table.
    """
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

    def local_frame(self, centerline_s: float) -> Tuple[float, float, float, float, float]:
        """Get the centerline position, heading and left unit normal at centerline_s.

        Input:
            centerline_s: distance along the centerline (m), wrapped internally.

        Output:
            (x, y, psi, normal_x, normal_y): centerline position (m),
            segment heading (rad), and the unit normal pointing left.
            A point at lateral offset e is (x + e*normal_x, y + e*normal_y).

        Uses the existing polyline interpolation;
        it does not fit or smooth the track.
        """
        x, y, psi = self.interpolate(centerline_s)  # World XY (m), centerline heading (rad).
        return x, y, psi, -math.sin(psi), math.cos(psi)

    def frenet_to_world(self, s: float, e: float) -> Tuple[float, float]:
        x, y, _, normal_x, normal_y = self.local_frame(s)  # s/e are centerline Frenet coordinates.
        return x + e * normal_x, y + e * normal_y

    def _get_psi_at_seg(self, idx: int) -> float:
        if idx >= len(self.tangents):
            idx = len(self.tangents) - 1
        dx, dy = self.tangents[idx]
        return math.atan2(dy, dx)

    def state_kinematics(self, state, s, e, curvature):
        """Compute world-frame and track-relative velocities from a vehicle state.

        Input:
            - state: vehicle state; psi is the body heading in radians,
              vx/vy are forward/leftward velocities in the body frame (m/s).
            - s, e: vehicle position already projected onto the centerline (m).
              Positive e is to the left of the track. No projection is done here.
            - curvature: estimated centerline curvature at s (1/m).

        Output:
            A dictionary containing the track heading and body-minus-track
            heading error (rad), the supplied curvature (1/m), world-frame
            vx/vy, speed magnitude, and track-relative rates s_dot/e_dot (m/s).

        Rotate the body velocity into the world frame and the track frame.
        In the track frame, e_dot is the leftward velocity; s_dot is the
        tangential velocity divided by 1 - curvature*e. The denominator uses
        the same lower-bound protection as the MPC. This is an approximation:
        the track uses straight segments, while its curvature is smoothed.
        """
        heading = self.interpolate(s)[2]
        chi = math.atan2(math.sin(state.psi-heading), math.cos(state.psi-heading))
        c, sn = math.cos(state.psi), math.sin(state.psi)
        tangent_speed = state.vx*math.cos(chi) - state.vy*math.sin(chi)
        normal_speed = state.vx*math.sin(chi) + state.vy*math.cos(chi)
        return dict(track_heading=heading, heading_error=chi, curvature=float(curvature),
                    vx_world=state.vx*c-state.vy*sn,
                    vy_world=state.vx*sn+state.vy*c,
                    speed_mps=math.hypot(state.vx,state.vy),
                    s_dot=tangent_speed/frenet_denominator(1-curvature*e),
                    e_dot=normal_speed)

    def curvature_profile(self, sample_step: float = 0.05, heading_window: float = 0.25):
        """Estimate centerline curvature from changes in the track heading.

        Input:
            - sample_step: target spacing between curvature samples (m).
            - heading_window: distance before and after each sample at which
              to read the track heading (m).

        Output:
            - progress: uniformly spaced s values from 0 to total_length (m).
            - curvature: estimated curvature at each s (1/m), positive for
              left turns. The last value repeats the first to close the lap.

        The centerline is a polyline: its heading is constant along each
        segment and jumps at vertices. Estimate curvature over a finite
        distance instead of differentiating these jumps:
            curvature(s) = wrapped_heading_change / (2 * heading_window)
        where the headings are read at s - heading_window and
        s + heading_window. Wrap their difference to [-pi, pi] so crossing
        the angle boundary does not create a false sharp turn.
        The resulting table is used for curvature interpolation in the MPC;
        it does not change the polyline used for position and projection.
        """
        sample_step = float(sample_step)
        heading_window = float(heading_window)
        if sample_step <= 0.0 or heading_window <= 0.0:
            raise ValueError("curvature sampling distances must be positive")
        sample_count = max(2, int(math.ceil(self.total_length / sample_step)))
        progress = np.linspace(0.0, self.total_length, sample_count + 1)
        curvature = np.empty_like(progress)
        for idx, s in enumerate(progress[:-1]):
            _, _, psi_before = self.interpolate(s - heading_window)
            _, _, psi_after = self.interpolate(s + heading_window)
            heading_delta = math.atan2(
                math.sin(psi_after - psi_before),
                math.cos(psi_after - psi_before),
            )
            curvature[idx] = heading_delta / (2.0 * heading_window)
        curvature[-1] = curvature[0]
        return progress, curvature


# def create_test_track(num_points=300, radius=14.0):
#     # Create a simple circular test track
#     t = np.linspace(0, 2 * np.pi, num_points)
#     x = radius * np.cos(t)
#     y = radius * np.sin(t)
#     return np.column_stack([x, y])


class RaceLine(ProjectablePath):
    """
    alpha-RACER style raceline.

    Supported CSV layouts:
    - s_m,x_m,y_m,vx_mps
    - s_m,x_m,y_m,vx_mps,dev_m
    Extra columns are preserved in `extra`.

    Methods:
        - interpolate: Interpolates the raceline at a given progress s and returns
          the corresponding (x, y) coordinates, speed, heading psi, and deviation e.
        - nearest_index: Returns the index of the closest point on the raceline to a given (x, y) position.
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


# Tool functions

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
