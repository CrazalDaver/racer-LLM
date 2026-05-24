import math
from pathlib import Path
from typing import Tuple

import numpy as np


class Track:
    def __init__(self, waypoints: np.ndarray, closed: bool = True):
        self.waypoints = np.asarray(waypoints, dtype=float)
        if closed and np.linalg.norm(self.waypoints[0] - self.waypoints[-1]) > 1e-6:
            self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        self._precompute()

    def _precompute(self):
        self.seg_vectors = np.diff(self.waypoints, axis=0)
        self.seg_lengths = np.linalg.norm(self.seg_vectors, axis=1)
        self.cum_len = np.cumsum(np.insert(self.seg_lengths, 0, 0))
        self.total_length = self.cum_len[-1]
        self.tangents = self.seg_vectors / (self.seg_lengths[:, None] + 1e-8)
        self.normals = np.array([[-t[1], t[0]] for t in self.tangents])
        self.normals = self.normals / (np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-8)

    def project(self, x: float, y: float, s_hint: float = None, search_radius: float = None) -> Tuple[float, float]:
        pos = np.array([x, y])
        best_dist = np.inf
        best_idx = -1
        best_t = 0.0
        best_closest = None

        for i, (p0, p1) in enumerate(zip(self.waypoints[:-1], self.waypoints[1:])):
            if not self._segment_is_near_hint(self.cum_len[i], s_hint, search_radius):
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
            return self.project(x, y)

        normal = self.normals[best_idx]
        signed_e = np.dot(pos - best_closest, normal)
        s = self.cum_len[best_idx] + best_t * self.seg_lengths[best_idx]
        return s % self.total_length, signed_e

    def _segment_is_near_hint(self, seg_s: float, s_hint: float, search_radius: float) -> bool:
        if s_hint is None or search_radius is None:
            return True
        return _circular_distance(seg_s, s_hint, self.total_length) <= search_radius

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


class RaceLine:
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
        psi_ext = np.r_[np.unwrap(self.psi), np.unwrap(self.psi)[0]]

        return {
            "s_m": s_wrapped,
            "x_m": np.interp(s_wrapped, s_ext, x_ext),
            "y_m": np.interp(s_wrapped, s_ext, y_ext),
            "vx_mps": np.interp(s_wrapped, s_ext, speed_ext),
            "psi_rad": np.interp(s_wrapped, s_ext, psi_ext),
            "dev_m": np.interp(s_wrapped, s_ext, dev_ext),
        }

    def project(self, x: float, y: float, s_hint: float = None, search_radius: float = None) -> Tuple[float, float]:
        pos = np.array([x, y])
        best_dist = np.inf
        best_idx = -1
        best_t = 0.0
        best_closest = None

        for i, (p0, p1) in enumerate(zip(self._closed_waypoints[:-1], self._closed_waypoints[1:])):
            if not self._segment_is_near_hint(self._cum_len[i], s_hint, search_radius):
                continue
            seg_vec = p1 - p0
            seg_len = self._seg_lengths[i]
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
            return self.project(x, y)

        normal = self._normals[best_idx]
        signed_e = np.dot(pos - best_closest, normal)
        s = self._cum_len[best_idx] + best_t * self._seg_lengths[best_idx]
        return s % self.total_length, signed_e

    def _segment_is_near_hint(self, seg_s: float, s_hint: float, search_radius: float) -> bool:
        if s_hint is None or search_radius is None:
            return True
        return _circular_distance(seg_s, s_hint, self.total_length) <= search_radius

    def _compute_heading(self):
        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        return np.arctan2(dy, dx)

    def _precompute_segments(self):
        self._closed_waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        seg_vectors = np.diff(self._closed_waypoints, axis=0)
        self._seg_lengths = np.linalg.norm(seg_vectors, axis=1)
        self._cum_len = np.cumsum(np.insert(self._seg_lengths, 0, 0))
        tangents = seg_vectors / (self._seg_lengths[:, None] + 1e-8)
        self._normals = np.array([[-t[1], t[0]] for t in tangents])
        self._normals = self._normals / (np.linalg.norm(self._normals, axis=1, keepdims=True) + 1e-8)


def load_centerline_csv(path) -> np.ndarray:
    return np.genfromtxt(Path(path), delimiter=",", comments="#")


def load_raceline_csv(path) -> RaceLine:
    return RaceLine(np.genfromtxt(Path(path), delimiter=",", comments="#"))


def _circular_distance(a: float, b: float, length: float) -> float:
    delta = abs((a - b) % length)
    return min(delta, length - delta)
