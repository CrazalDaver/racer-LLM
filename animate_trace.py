import argparse
import sys
from pathlib import Path

import numpy as np

from config import params
from plotting import compute_lap_stats
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


def _save_animation(animation, save_path, interval, dpi):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1000.0 / interval)))
    suffix = output.suffix.lower()

    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        writer = FFMpegWriter(fps=fps)
    else:
        writer = None

    try:
        animation.save(str(output), writer=writer, dpi=dpi)
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            "Failed to save animation with the current Python interpreter: "
            f"{sys.executable}. Try running with "
            "F:\\Anacode\\envs\\racer\\python.exe animate_trace.py --save trace.gif"
        ) from exc
    print(f"Saved animation to {output}")


def make_animation(trace_path, stride=1, interval=30, save_path=None, dpi=100):
    import matplotlib

    if save_path:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data, track_csv, raceline_csv = load_trace(trace_path)
    waypoints = load_centerline_csv(track_csv)
    raceline = load_raceline_csv(raceline_csv)

    time = data["time"]
    x = data["x"]
    y = data["y"]
    car_x = data["car_x"] if "car_x" in data.files else None
    car_y = data["car_y"] if "car_y" in data.files else None
    ref_x = data["ref_x"]
    ref_y = data["ref_y"]
    ref_horizon_x = data["ref_horizon_x"]
    ref_horizon_y = data["ref_horizon_y"]
    raceline_horizon_x = data["raceline_horizon_x"] if "raceline_horizon_x" in data.files else None
    raceline_horizon_y = data["raceline_horizon_y"] if "raceline_horizon_y" in data.files else None
    px = data["px"]
    py = data["py"]
    track_length = float(np.asarray(data["track_length"]).item()) if "track_length" in data.files else None
    lap_stats = compute_lap_stats(time, px, track_length)
    ref_speed = data["ref_speed"] if "ref_speed" in data.files else None
    vx = data["vx"]
    throttle = data["throttle"]
    steer = data["steer"]
    offroad = data["offroad"] if "offroad" in data.files else np.abs(py) > (params.TRACK_WIDTH / 2.0)
    collision = data["collision"] if "collision" in data.files else np.zeros_like(offroad, dtype=bool)

    frames = np.arange(0, len(time), max(1, stride))
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(waypoints[:, 0], waypoints[:, 1], "k--", alpha=0.5, label="Track")
    ax.plot(raceline.x, raceline.y, "m--", alpha=0.65, label="Raceline")
    full_traj, = ax.plot(x, y, color="0.75", linewidth=0.8, label="Full trajectory")
    trail, = ax.plot([], [], "b-", linewidth=1.6, label="Trajectory so far")
    opponent_trails = []
    opponent_points = []
    if car_x is not None and car_y is not None and car_x.ndim == 2 and car_x.shape[1] > 1:
        colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink"]
        for car_idx in range(1, car_x.shape[1]):
            color = colors[(car_idx - 1) % len(colors)]
            opponent_trail, = ax.plot([], [], color=color, linewidth=1.2, label=f"Opponent {car_idx}")
            opponent_point, = ax.plot([], [], "o", color=color, markersize=5)
            opponent_trails.append(opponent_trail)
            opponent_points.append(opponent_point)
    horizon_line, = ax.plot([], [], linestyle="None", color="tab:orange", marker=".", markersize=4, label="ref_traj points")
    raceline_points, = ax.plot([], [], linestyle="None", color="m", marker=".", markersize=4, alpha=0.8, label="raceline source points")
    car_ref_line, = ax.plot([], [], color="tab:cyan", linewidth=1.0, alpha=0.75, label="Car to ref_traj[0]")
    car_point, = ax.plot([], [], "bo", markersize=6, label="Car")
    ref_point, = ax.plot([], [], "o", color="tab:orange", markersize=6, label="ref_traj[0]")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    offroad_text = ax.text(
        0.02,
        0.78,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="red",
        fontsize=22,
        fontweight="bold",
    )
    collision_text = ax.text(
        0.02,
        0.70,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="darkred",
        fontsize=18,
        fontweight="bold",
    )

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
        if car_x is not None and car_y is not None and car_x.ndim == 2:
            for local_idx, car_idx in enumerate(range(1, car_x.shape[1])):
                opponent_trails[local_idx].set_data(car_x[: i + 1, car_idx], car_y[: i + 1, car_idx])
                opponent_points[local_idx].set_data([car_x[i, car_idx]], [car_y[i, car_idx]])
        car_point.set_data([x[i]], [y[i]])
        ref_point.set_data([ref_x[i]], [ref_y[i]])
        car_ref_line.set_data([x[i], ref_x[i]], [y[i], ref_y[i]])
        horizon_line.set_data(ref_horizon_x[i], ref_horizon_y[i])
        if raceline_horizon_x is not None and raceline_horizon_y is not None:
            raceline_points.set_data(raceline_horizon_x[i], raceline_horizon_y[i])
        else:
            raceline_points.set_data([], [])
        offroad_text.set_text("OFF TRACK" if offroad[i] else "")
        collision_text.set_text("NEAR COLLISION" if collision[i] else "")
        ref_speed_text = "" if ref_speed is None else f"\nref_v={ref_speed[i]:.2f} m/s"
        completed_laps = int(np.searchsorted(lap_stats["crossing_times"], time[i], side="right"))
        current_lap = completed_laps + 1
        previous_crossing = time[0] if completed_laps == 0 else lap_stats["crossing_times"][completed_laps - 1]
        current_lap_text = f"lap={current_lap}, lap_t={time[i] - previous_crossing:.2f}s"
        if completed_laps > 0:
            current_lap_text += f", last={lap_stats['lap_times'][completed_laps - 1]:.2f}s"
        info.set_text(
            f"t={time[i]:.2f}s\n"
            f"{current_lap_text}\n"
            f"px={px[i]:.2f} m, py={py[i]:.2f} m\n"
            f"vx={vx[i]:.2f} m/s{ref_speed_text}\n"
            f"throttle={throttle[i]:.2f}, steer={steer[i]:.2f}"
        )
        return (
            trail,
            *opponent_trails,
            *opponent_points,
            car_point,
            ref_point,
            car_ref_line,
            horizon_line,
            raceline_points,
            info,
            offroad_text,
            collision_text,
        )

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    if save_path:
        _save_animation(animation, save_path, interval, dpi)
        plt.close(fig)
    else:
        plt.show()
    return animation


def main():
    parser = argparse.ArgumentParser(description="Animate a saved racer simulation trace.")
    parser.add_argument("--trace", default=params.TRACE_OUTPUT, help="Path to simulation_trace.npz")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth recorded frame")
    parser.add_argument("--interval", type=int, default=30, help="Frame interval in milliseconds")
    parser.add_argument(
        "--save",
        nargs="?",
        const="simulation_trace.gif",
        default=None,
        help="Optional output path. Use --save for simulation_trace.gif, or pass trace.gif/trace.mp4.",
    )
    parser.add_argument("--dpi", type=int, default=100, help="Saved animation DPI")
    args = parser.parse_args()
    make_animation(args.trace, stride=args.stride, interval=args.interval, save_path=args.save, dpi=args.dpi)


if __name__ == "__main__":
    print("Starting animation...")
    main()
    # make_animation("F:/desktop/racer-LLM/result/grid_search/019_vw2_q1_h8.npz", stride=1, interval=30, dpi=100)
    # make_animation("F:/desktop/racer-LLM/result/grid_search/013_vw1_q2_h8.npz", stride=1, interval=30, dpi=100)