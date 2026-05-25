import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from config import params
from track import load_centerline_csv, load_raceline_csv


def _scalar_string(value, default):
    if value is None:
        return default
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def load_trace(path):
    data = np.load(path, allow_pickle=False)
    track_csv = _scalar_string(data["track_csv"], params.TRACK_CSV) if "track_csv" in data.files else params.TRACK_CSV
    raceline_csv = (
        _scalar_string(data["raceline_csv"], params.RACELINE_CSV)
        if "raceline_csv" in data.files
        else params.RACELINE_CSV
    )
    return data, track_csv, raceline_csv


def make_animation(trace_path, stride=1, interval=30, save_path=None):
    data, track_csv, raceline_csv = load_trace(trace_path)
    waypoints = load_centerline_csv(track_csv)
    raceline = load_raceline_csv(raceline_csv)

    time = data["time"]
    x = data["x"]
    y = data["y"]
    ref_x = data["ref_x"]
    ref_y = data["ref_y"]
    ref_horizon_x = data["ref_horizon_x"]
    ref_horizon_y = data["ref_horizon_y"]
    px = data["px"]
    py = data["py"]
    vx = data["vx"]
    throttle = data["throttle"]
    steer = data["steer"]

    frames = np.arange(0, len(time), max(1, stride))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(waypoints[:, 0], waypoints[:, 1], "k--", alpha=0.5, label="Track")
    ax.plot(raceline.x, raceline.y, "m--", alpha=0.65, label="Raceline")
    full_traj, = ax.plot(x, y, color="0.75", linewidth=0.8, label="Full trajectory")
    trail, = ax.plot([], [], "b-", linewidth=1.6, label="Trajectory so far")
    horizon_line, = ax.plot([], [], color="tab:orange", linewidth=1.2, marker=".", markersize=3, label="ref_traj horizon")
    car_ref_line, = ax.plot([], [], color="tab:cyan", linewidth=1.0, alpha=0.75, label="Car to ref")
    car_point, = ax.plot([], [], "bo", markersize=6, label="Car")
    ref_point, = ax.plot([], [], "o", color="tab:orange", markersize=6, label="ref_traj[0]")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Racer trace animation")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    margin = 1.0
    ax.set_xlim(np.nanmin(waypoints[:, 0]) - margin, np.nanmax(waypoints[:, 0]) + margin)
    ax.set_ylim(np.nanmin(waypoints[:, 1]) - margin, np.nanmax(waypoints[:, 1]) + margin)

    def update(frame_idx):
        i = int(frames[frame_idx])
        trail.set_data(x[: i + 1], y[: i + 1])
        car_point.set_data([x[i]], [y[i]])
        ref_point.set_data([ref_x[i]], [ref_y[i]])
        car_ref_line.set_data([x[i], ref_x[i]], [y[i], ref_y[i]])
        horizon_line.set_data(ref_horizon_x[i], ref_horizon_y[i])
        info.set_text(
            f"t={time[i]:.2f}s\n"
            f"px={px[i]:.2f} m, py={py[i]:.2f} m\n"
            f"vx={vx[i]:.2f} m/s\n"
            f"throttle={throttle[i]:.2f}, steer={steer[i]:.2f}"
        )
        return trail, car_point, ref_point, car_ref_line, horizon_line, info

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    if save_path:
        animation.save(save_path)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    return animation


def main():
    parser = argparse.ArgumentParser(description="Animate a saved racer simulation trace.")
    parser.add_argument("--trace", default=params.TRACE_OUTPUT, help="Path to simulation_trace.npz")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth recorded frame")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval in milliseconds")
    parser.add_argument("--save", default=None, help="Optional output path, e.g. trace.gif or trace.mp4")
    args = parser.parse_args()
    make_animation(args.trace, stride=args.stride, interval=args.interval, save_path=args.save)


if __name__ == "__main__":
    main()
