import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_results(
    waypoints,
    raceline,
    traj_x,
    traj_y,
    velocities,
    lateral_errors,
    mpc_losses,
    mpc_track_losses,
    mpc_action_losses,
    longitudinal_positions,
    frenet_ref_lines,
    params,
):
    half_width = params.TRACK_WIDTH / 2.0

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 2, 1)
    plt.plot(waypoints[:, 0], waypoints[:, 1], "k--", alpha=0.5, label="Track")
    if raceline is not None:
        plt.plot(raceline.x, raceline.y, "m--", alpha=0.8, label="Raceline")
    plt.plot(traj_x, traj_y, "b-", label="Trajectory")
    off_idx = [i for i, e in enumerate(lateral_errors) if abs(e) > half_width]
    if off_idx:
        off_x = [traj_x[i // 2] for i in off_idx if i // 2 < len(traj_x)]
        off_y = [traj_y[i // 2] for i in off_idx if i // 2 < len(traj_y)]
        plt.scatter(off_x, off_y, c="red", s=20, alpha=0.6, label="Offroad")
    for idx, (x, y, rx, ry) in enumerate(frenet_ref_lines):
        ref_label = "Car to ref_traj" if idx == 0 else None
        plt.plot([x, rx], [y, ry], color="tab:cyan", alpha=0.55, linewidth=0.9, label=ref_label)
        plt.scatter([rx], [ry], c="tab:cyan", s=10, alpha=0.7)
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectory")

    plt.subplot(3, 2, 2)
    plt.plot(np.arange(len(velocities)) * params.DT * 2, velocities, "g-")
    plt.grid()
    plt.title("Velocity")
    plt.xlabel("Time (s)")

    plt.subplot(3, 2, 3)
    plt.plot(np.arange(len(lateral_errors)) * params.DT, lateral_errors, "r-")
    plt.axhline(y=half_width, ls="--", c="k")
    plt.axhline(y=-half_width, ls="--", c="k")
    plt.grid()
    plt.title("Lateral error")
    plt.xlabel("Time (s)")

    plt.subplot(3, 2, 4)
    off_flag = [1 if abs(e) > half_width else 0 for e in lateral_errors]
    plt.plot(np.arange(len(off_flag)) * params.DT, off_flag, drawstyle="steps-post", c="orange")
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.title("Offroad penalty")

    plt.subplot(3, 2, 5)
    loss_time = np.arange(len(mpc_losses)) * params.DT
    plt.plot(loss_time, mpc_losses, "m-", label="Total loss")
    plt.plot(loss_time, mpc_track_losses, "b-", alpha=0.8, label="Track loss")
    plt.plot(loss_time, mpc_action_losses, "g-", alpha=0.8, label="Action change loss")
    plt.grid()
    plt.title("MPC loss")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.subplot(3, 2, 6)
    px_time = np.arange(len(longitudinal_positions)) * params.DT
    plt.plot(px_time, longitudinal_positions, "c-")
    plt.grid()
    plt.title("Longitudinal px")
    plt.xlabel("Time (s)")
    plt.ylabel("Arc length (m)")

    plt.tight_layout()
    plt.show()
