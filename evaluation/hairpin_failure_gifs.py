"""Render the saved P3 LLM/rule hairpin failures as three local-view GIFs.

This is a result-visualization utility; it does not run or modify simulation.

Usage:
    .venv/bin/python evaluation/hairpin_failure_gifs.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from animate_trace import _track_boundaries, _vehicle_corners
from track import Track, load_centerline_csv, load_raceline_csv


CASES = {
    "raw_llm": REPO_ROOT / "result/frenet_llm_alpha_screen_raw_llm/summary_traces/trial_003.npz",
    "explicit_rule_llm": REPO_ROOT / "result/frenet_llm_alpha_screen_explicit_rule_llm/summary_traces/trial_003.npz",
    "rule_only": REPO_ROOT / "result/frenet_llm_alpha_screen_rule_only/summary_traces/trial_003.npz",
}
OUTPUT_DIR = REPO_ROOT / "result/frenet_llm_alpha_screen/hairpin"
START_TIME = 14.0
END_TIME = 21.0
FRAME_STEP_SECONDS = 0.20
FRAME_INTERVAL_MS = 120


def _scalar_string(value):
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array)


def render_case(label: str, trace_path: Path) -> Path:
    data = np.load(trace_path, allow_pickle=True)
    track_scale = float(np.asarray(data["track_scale"]).item())
    track_csv = _scalar_string(data["track_csv"])
    raceline_csv = _scalar_string(data["raceline_csv"])
    track = Track(load_centerline_csv(track_csv, scale=track_scale))
    raceline = load_raceline_csv(raceline_csv, scale=track_scale)
    half_width = float(np.asarray(data["track_half_width"]).item())
    half_length = float(np.asarray(data["vehicle_half_length"]).item())
    vehicle_half_width = float(np.asarray(data["vehicle_half_width"]).item())
    left, right = _track_boundaries(track.waypoints, half_width)

    local_s = np.linspace(54.0, 73.0, 500)
    local_xy = np.asarray([track.frenet_to_world(s, 0.0) for s in local_s])
    margin = 1.2
    xlim = (local_xy[:, 0].min() - margin, local_xy[:, 0].max() + margin)
    ylim = (local_xy[:, 1].min() - margin, local_xy[:, 1].max() + margin)

    time = data["time"]
    raw_indices = np.flatnonzero((time >= START_TIME) & (time <= END_TIME))
    simulation_dt = float(np.asarray(data["simulation_dt"]).item())
    stride = max(1, int(round(FRAME_STEP_SECONDS / simulation_dt)))
    frames = raw_indices[::stride]

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.fill(
        np.r_[left[:, 0], right[::-1, 0]],
        np.r_[left[:, 1], right[::-1, 1]],
        color="#e8e8e8",
        zorder=0,
    )
    ax.plot(left[:, 0], left[:, 1], color="#555555", linewidth=1.2)
    ax.plot(right[:, 0], right[:, 1], color="#555555", linewidth=1.2)
    center_xy = np.asarray([track.frenet_to_world(s, 0.0) for s in np.linspace(0, track.total_length, 1500)])
    ax.plot(center_xy[:, 0], center_xy[:, 1], "--", color="#999999", linewidth=0.8, label="centerline")
    ax.plot(raceline.x, raceline.y, color="#b23aee", linewidth=1.2, alpha=0.85, label="raceline")

    colors = ["#e66101", "#1f77b4", "#2ca02c"]
    labels = [label.replace("_", " "), "Alpha opponent 1", "Alpha opponent 2"]
    boxes = []
    trails = []
    for car_idx in range(3):
        trail, = ax.plot([], [], color=colors[car_idx], linewidth=1.4, alpha=0.75)
        box = Polygon(np.zeros((4, 2)), closed=True, facecolor=colors[car_idx], edgecolor="white", linewidth=0.8, zorder=5)
        ax.add_patch(box)
        trails.append(trail)
        boxes.append(box)

    reference, = ax.plot([], [], "o-", color="#e66101", markersize=2.4, linewidth=0.8, alpha=0.8, label="ego MPC reference")
    status = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"P3 hairpin — {label.replace('_', ' ')}")

    car_x = data["car_x"]
    car_y = data["car_y"]
    car_psi = data["car_psi"]
    car_s = data["car_px"]
    car_e = data["car_py"]
    car_v = data["car_vx"]
    modes = data["car_policy_mode"].astype(str)
    ref_x = data["ref_horizon_x"]
    ref_y = data["ref_horizon_y"]

    def update(frame_idx):
        trail_start = max(0, frame_idx - int(round(3.0 / simulation_dt)))
        for car_idx in range(3):
            boxes[car_idx].set_xy(_vehicle_corners(
                car_x[frame_idx, car_idx], car_y[frame_idx, car_idx],
                car_psi[frame_idx, car_idx], half_length, vehicle_half_width,
            ))
            trails[car_idx].set_data(
                car_x[trail_start:frame_idx + 1, car_idx],
                car_y[trail_start:frame_idx + 1, car_idx],
            )
        reference.set_data(ref_x[frame_idx], ref_y[frame_idx])
        status.set_text(
            f"t={time[frame_idx]:4.1f} s   mode={modes[frame_idx, 0]}\n"
            f"ego: s={car_s[frame_idx, 0]:5.1f}  e={car_e[frame_idx, 0]:+5.2f} m  "
            f"v={car_v[frame_idx, 0]:4.2f} m/s"
        )
        return [*boxes, *trails, reference, status]

    animation = FuncAnimation(
        fig, update, frames=frames, interval=FRAME_INTERVAL_MS, blit=False,
    )
    output = OUTPUT_DIR / f"{label}.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=1000 // FRAME_INTERVAL_MS), dpi=90)
    plt.close(fig)
    print(f"Saved {output.relative_to(REPO_ROOT)}")
    return output


def main():
    missing = [str(path) for path in CASES.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing trace(s):\n" + "\n".join(missing))
    for label, path in CASES.items():
        render_case(label, path)


if __name__ == "__main__":
    main()
