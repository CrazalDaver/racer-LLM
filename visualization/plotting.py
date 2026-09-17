"""Plot saved traces and in-memory simulation results."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from config import params as sim_params
from track import load_centerline_csv, load_raceline_csv


def _scalar_string(value):
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def _string_list(value):
    if value is None:
        return []
    array = np.asarray(value)
    if array.shape == ():
        return [str(array.item())]
    return [str(item) for item in array.tolist()]


def _time_step(time, fallback=0.1):
    if len(time) < 2:
        return fallback
    return float(np.median(np.diff(time)))


def _as_optional_array(values):
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return array if array.size else None


def _time_for(values, dt, multiplier=1):
    return np.arange(len(values)) * dt * multiplier


def _same_length(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _loss_term_enabled(weight, values):
    """Use the configured objective weight, with data fallback for old traces."""
    if weight is not None:
        return np.isfinite(weight) and abs(float(weight)) > 1e-12
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return bool(finite.size and np.any(np.abs(finite) > 1e-12))


def _mode_runs(modes):
    runs = []
    if not modes:
        return runs
    start = 0
    current = modes[0]
    for idx, mode in enumerate(modes[1:], start=1):
        if mode != current:
            runs.append((current, start, idx))
            start = idx
            current = mode
    runs.append((current, start, len(modes)))
    return runs


def _plot_direct_parameter_histograms(ax, decisions, bounds):
    """Show compact discrete frequencies, with one sample per LLM call."""
    names = ("q", "zeta", "s1", "s2", "s3")
    rounding_digits = (2, 3, 3, 3, 3)
    decisions = np.asarray(decisions, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    ax.set_axis_off()
    ax.set_title(
        f"Direct LLM parameter choices ({len(decisions)} decisions)",
        pad=10,
    )
    positions = (
        (0.03, 0.55, 0.28, 0.34),
        (0.36, 0.55, 0.28, 0.34),
        (0.69, 0.55, 0.28, 0.34),
        (0.19, 0.08, 0.28, 0.34),
        (0.53, 0.08, 0.28, 0.34),
    )
    for idx, (name, position, digits) in enumerate(
        zip(names, positions, rounding_digits)
    ):
        inset = ax.inset_axes(position)
        values = decisions[:, idx]
        values = values[np.isfinite(values)]
        lo, hi = bounds[idx]
        if values.size:
            # Preserve the LLM's discrete choices instead of smoothing them
            # into values it never emitted. Rounding only removes irrelevant
            # floating-point precision before exact frequency counting.
            span = max(hi - lo, 1.0e-9)
            rounded = np.round(values, digits)
            choices, counts = np.unique(rounded, return_counts=True)
            separations = np.diff(choices)
            positive_separations = separations[separations > 0.0]
            width = 0.018 * span
            if positive_separations.size:
                width = min(width, 0.65 * float(np.min(positive_separations)))
            width = max(width, 0.006 * span)
            color = f"C{idx}"
            inset.bar(
                choices,
                counts,
                width=width,
                color=color,
                alpha=0.72,
                edgecolor="black",
                linewidth=0.55,
            )
            mean = float(np.mean(values))
            inset.axvline(mean, color="black", linestyle="--", linewidth=1.0)
            inset.text(
                0.97,
                0.93,
                f"mean={mean:.3g}",
                transform=inset.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
        inset.set_xlim(lo, hi)
        inset.set_ylim(bottom=0.0)
        inset.set_title(name, fontsize=9, pad=1)
        inset.tick_params(axis="both", labelsize=7)
        inset.locator_params(axis="y", nbins=3, integer=True)
        inset.grid(axis="y", alpha=0.2)


def _direct_parameter_decisions(data, ego_index=0):
    """Recover one parameter vector per direct-parameter LLM call."""
    if "car_llm_prompt_variants" not in data.files:
        return None
    variants = _string_list(data["car_llm_prompt_variants"])
    if ego_index >= len(variants) or not variants[ego_index].startswith("direct_params"):
        return None
    if "policy_params" not in data.files:
        return None

    params_history = np.asarray(data["policy_params"], dtype=float)
    if params_history.ndim != 2 or params_history.shape[1] != 5:
        return None
    control_update = (
        np.asarray(data["control_update"], dtype=bool)
        if "control_update" in data.files
        else np.ones(len(params_history), dtype=bool)
    )
    control_params = params_history[control_update]
    if "llm_raw_call_indices" in data.files:
        call_indices = np.asarray(data["llm_raw_call_indices"], dtype=int)
        if call_indices.ndim == 2 and ego_index < call_indices.shape[0]:
            # Call indices are one-based policy-control calls in the trace.
            selected = call_indices[ego_index]
            selected = selected[selected > 0] - 1
            selected = selected[selected < len(control_params)]
            if selected.size:
                return control_params[selected]
    return control_params


def compute_lap_stats(time, progress, track_length):
    time = np.asarray(time, dtype=float)
    progress = np.asarray(progress, dtype=float)
    if track_length is None or track_length <= 0 or len(time) == 0 or len(progress) == 0:
        return {"crossing_times": [], "lap_times": [], "current_lap": 1, "current_elapsed": np.nan}

    n = min(len(time), len(progress))
    time = time[:n]
    progress = progress[:n]
    valid = np.isfinite(time) & np.isfinite(progress)
    if not np.any(valid):
        return {"crossing_times": [], "lap_times": [], "current_lap": 1, "current_elapsed": np.nan}

    time = time[valid]
    progress = progress[valid]
    progress = np.maximum.accumulate(progress)
    start_progress = progress[0]
    start_time = time[0]
    max_progress = progress[-1]

    crossing_times = []
    lap_times = []
    previous_crossing = start_time
    lap_number = 1
    while start_progress + lap_number * track_length <= max_progress:
        threshold = start_progress + lap_number * track_length
        idx = int(np.searchsorted(progress, threshold, side="left"))
        if idx <= 0 or idx >= len(progress):
            break
        p0, p1 = progress[idx - 1], progress[idx]
        t0, t1 = time[idx - 1], time[idx]
        ratio = 0.0 if abs(p1 - p0) < 1e-9 else (threshold - p0) / (p1 - p0)
        crossing_time = float(t0 + np.clip(ratio, 0.0, 1.0) * (t1 - t0))
        crossing_times.append(crossing_time)
        lap_times.append(crossing_time - previous_crossing)
        previous_crossing = crossing_time
        lap_number += 1

    current_lap = len(crossing_times) + 1
    current_elapsed = float(time[-1] - previous_crossing)
    current_progress = float((max_progress - start_progress) % track_length)
    return {
        "crossing_times": crossing_times,
        "lap_times": lap_times,
        "current_lap": current_lap,
        "current_elapsed": current_elapsed,
        "current_progress": current_progress,
    }


def format_lap_summary(lap_stats, max_laps=4):
    lap_times = lap_stats.get("lap_times", [])
    if lap_times:
        shown = lap_times[-max_laps:]
        first_lap = len(lap_times) - len(shown) + 1
        lines = [f"Lap {first_lap + idx}: {lap_time:.2f}s" for idx, lap_time in enumerate(shown)]
    else:
        lines = ["No completed lap"]
    current_elapsed = lap_stats.get("current_elapsed", np.nan)
    current_lap = lap_stats.get("current_lap", 1)
    if np.isfinite(current_elapsed):
        lines.append(f"Current lap {current_lap}: {current_elapsed:.2f}s")
    return "\n".join(lines)


def _align_reference_sample(reference, actual, multiplier):
    reference = np.asarray(reference, dtype=float)
    if multiplier > 1 and len(reference) >= len(actual) * multiplier:
        reference = reference[::multiplier]
    return _same_length(actual, reference)


def _add_trajectory_time_marks(ax, traj_x, traj_y, dt, stride=1, interval=5.0):
    traj_x = np.asarray(traj_x, dtype=float)
    traj_y = np.asarray(traj_y, dtype=float)
    if interval is None or interval <= 0 or len(traj_x) == 0 or len(traj_y) == 0:
        return

    n = min(len(traj_x), len(traj_y))
    max_time = (n - 1) * dt * stride
    mark_times = np.arange(interval, max_time + 0.5 * interval, interval)
    first_label = True
    for mark_time in mark_times:
        idx = int(round(mark_time / (dt * stride)))
        if idx < 0 or idx >= n:
            continue
        x = traj_x[idx]
        y = traj_y[idx]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        label = "5s marks" if first_label else None
        ax.scatter([x], [y], marker="o", s=32, c="yellow", edgecolors="black", linewidths=0.7, zorder=5, label=label)
        ax.annotate(
            f"{mark_time:.0f}s",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "0.4", "alpha": 0.8},
            zorder=6,
        )
        first_label = False


def _unwrap_reference_s(ref_s, actual_s, total_length):
    ref_s = np.asarray(ref_s, dtype=float)
    actual_s = np.asarray(actual_s, dtype=float)
    if total_length is None or total_length <= 0:
        return ref_s
    delta = np.arctan2(
        np.sin((ref_s - actual_s) * 2.0 * np.pi / total_length),
        np.cos((ref_s - actual_s) * 2.0 * np.pi / total_length),
    ) * total_length / (2.0 * np.pi)
    return actual_s + delta


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
    mpc_velocity_losses,
    mpc_heading_losses,
    longitudinal_positions,
    frenet_ref_lines,
    params,
    throttle_history=None,
    steering_history=None,
    other_trajectories=None,
    ref_speed_history=None,
    ref_lateral_errors=None,
    ref_longitudinal_positions=None,
    policy_params_history=None,
    policy_names=None,
    policy_mode_history=None,
    track_length=None,
    trajectory_stride=2,
    velocity_dt_multiplier=2,
    trajectory_time_mark_interval=5.0,
    velocity_track_weight=None,
    heading_track_weight=None,
    direct_param_decisions=None,
    parameter_bounds=None,
):
    half_width = getattr(params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0)
    offroad_half_width = params.OFFROAD_RECOVERY_THRESHOLD
    dt = params.SIM_DT

    fig, axes = plt.subplots(4, 2, figsize=(14, 17), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.12, wspace=0.12, hspace=0.18)
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(waypoints[:, 0], waypoints[:, 1], "k--", alpha=0.5, label="Track")
    if raceline is not None:
        ax.plot(raceline.x, raceline.y, "m--", alpha=0.8, label="Raceline")
    ax.plot(traj_x, traj_y, "b-", label="Trajectory")
    _add_trajectory_time_marks(
        ax,
        traj_x,
        traj_y,
        dt,
        stride=trajectory_stride,
        interval=trajectory_time_mark_interval,
    )
    if other_trajectories:
        colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink"]
        for idx, (other_x, other_y) in enumerate(other_trajectories):
            ax.plot(
                other_x,
                other_y,
                color=colors[idx % len(colors)],
                linewidth=1.0,
                alpha=0.8,
                label=f"Opponent {idx + 1}",
            )
    off_idx = [i for i, e in enumerate(lateral_errors) if abs(e) > offroad_half_width]
    if off_idx:
        off_x = [traj_x[i // trajectory_stride] for i in off_idx if i // trajectory_stride < len(traj_x)]
        off_y = [traj_y[i // trajectory_stride] for i in off_idx if i // trajectory_stride < len(traj_y)]
        ax.scatter(off_x, off_y, c="red", s=20, alpha=0.6, label="Offroad")
    for idx, (x, y, rx, ry) in enumerate(frenet_ref_lines):
        ref_label = "Car to ref_traj" if idx == 0 else None
        ax.plot([x, rx], [y, ry], color="tab:cyan", alpha=0.55, linewidth=0.9, label=ref_label)
        ax.scatter([rx], [ry], c="tab:cyan", s=10, alpha=0.7)
    ax.axis("equal")
    lap_time = _time_for(longitudinal_positions, dt)
    lap_stats = compute_lap_stats(lap_time, longitudinal_positions, track_length)
    ax.text(
        0.02,
        0.02,
        format_lap_summary(lap_stats),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.4", "alpha": 0.85},
    )
    ax.legend()
    ax.set_title("Trajectory")

    ax = axes[1]
    velocities = np.asarray(velocities, dtype=float)
    ax.plot(_time_for(velocities, dt, velocity_dt_multiplier), velocities, "g-", label="Actual vx")
    ref_speed_history = _as_optional_array(ref_speed_history)
    if ref_speed_history is not None:
        actual_v, ref_v = _align_reference_sample(ref_speed_history, velocities, velocity_dt_multiplier)
        ax.plot(_time_for(ref_v, dt, velocity_dt_multiplier), ref_v, "k--", alpha=0.8, label="Raceline vx")
    ax.grid(alpha=0.3)
    ax.set_title("Velocity vs raceline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s")
    ax.legend()

    ax = axes[2]
    lateral_errors = np.asarray(lateral_errors, dtype=float)
    lateral_time = np.arange(len(lateral_errors)) * dt
    ax.plot(lateral_time, lateral_errors, "r-", label="Actual py")
    ref_lateral_errors = _as_optional_array(ref_lateral_errors)
    if ref_lateral_errors is not None:
        actual_e, ref_e = _same_length(lateral_errors, ref_lateral_errors)
        ax.plot(_time_for(ref_e, dt), ref_e, "k--", alpha=0.8, label="Raceline/ref py")
    ax.axhline(y=half_width, ls="--", c="k")
    ax.axhline(y=-half_width, ls="--", c="k")
    if abs(offroad_half_width - half_width) > 1e-9:
        ax.axhline(y=offroad_half_width, ls=":", c="orange", label="Offroad threshold")
        ax.axhline(y=-offroad_half_width, ls=":", c="orange")
    ax.grid(alpha=0.3)
    ax.set_title("Lateral error vs raceline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m")
    ax.legend()

    ax = axes[3]
    policy_names = [name.lower() for name in _string_list(policy_names)]
    policy_mode_history = _string_list(policy_mode_history)
    policy_params_history = _as_optional_array(policy_params_history)
    uses_llm = any("llm" in name for name in policy_names) or any(mode in {"stable", "push", "overtake", "block", "custom"} for mode in policy_mode_history)
    if direct_param_decisions is not None and len(direct_param_decisions):
        if parameter_bounds is None:
            parameter_bounds = sim_params.ALPHARACER_PARAMETER_BOUNDS
        _plot_direct_parameter_histograms(
            ax,
            direct_param_decisions,
            parameter_bounds,
        )
    elif uses_llm and policy_mode_history:
        mode_colors = {
            "stable": "tab:blue",
            "push": "tab:green",
            "overtake": "tab:orange",
            "block": "tab:red",
            "custom": "tab:gray",
            "fixed": "0.65",
        }
        n_modes = len(policy_mode_history)
        mode_time = np.arange(n_modes + 1) * dt
        seen_modes = []
        for mode, start, end in _mode_runs(policy_mode_history):
            color = mode_colors.get(mode, "tab:brown")
            ax.axvspan(mode_time[start], mode_time[end], ymin=0.08, ymax=0.92, color=color, alpha=0.75)
            center = 0.5 * (mode_time[start] + mode_time[end])
            width = mode_time[end] - mode_time[start]
            if width >= 0.25:
                ax.text(
                    center,
                    0.5,
                    mode,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                    clip_on=True,
                )
            if mode not in seen_modes:
                seen_modes.append(mode)
        for mode in seen_modes:
            ax.barh(
                [-1],
                [0],
                color=mode_colors.get(mode, "tab:brown"),
                alpha=0.75,
                label=mode,
            )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(mode_time[0], mode_time[-1] if len(mode_time) else 1.0)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title("LLM policy mode")
        ax.set_xlabel("Time (s)")
        ax.legend(ncol=min(3, max(1, len(seen_modes))), fontsize=8, loc="upper right")
    else:
        off_flag = [1 if abs(e) > offroad_half_width else 0 for e in lateral_errors]
        ax.plot(lateral_time, off_flag, drawstyle="steps-post", c="orange")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3)
        ax.set_title("Offroad penalty")

    ax = axes[4]
    loss_time = np.arange(len(mpc_losses)) * dt
    ax.plot(loss_time, mpc_losses, "m-", label="Total loss")
    ax.plot(loss_time, mpc_track_losses, "b-", alpha=0.8, label="Track loss")
    ax.plot(loss_time, mpc_action_losses, "g-", alpha=0.8, label="Action change loss")
    if _loss_term_enabled(velocity_track_weight, mpc_velocity_losses):
        ax.plot(
            loss_time,
            mpc_velocity_losses,
            color="tab:orange",
            alpha=0.8,
            label="Velocity loss",
        )
    if _loss_term_enabled(heading_track_weight, mpc_heading_losses):
        ax.plot(
            loss_time,
            mpc_heading_losses,
            color="tab:red",
            alpha=0.8,
            label="Heading loss",
        )
    ax.grid(alpha=0.3)
    ax.set_title("MPC loss")
    ax.set_xlabel("Time (s)")
    ax.legend()

    ax = axes[5]
    longitudinal_positions = np.asarray(longitudinal_positions, dtype=float)
    px_time = np.arange(len(longitudinal_positions)) * dt
    ax.plot(px_time, longitudinal_positions, "c-", label="Actual px")
    ref_longitudinal_positions = _as_optional_array(ref_longitudinal_positions)
    if ref_longitudinal_positions is not None:
        actual_px, ref_px = _same_length(longitudinal_positions, ref_longitudinal_positions)
        ref_px = _unwrap_reference_s(ref_px, actual_px, track_length)
        ax.plot(_time_for(ref_px, dt), ref_px, "k--", alpha=0.8, label="Raceline/ref px")
    for idx, crossing_time in enumerate(lap_stats["crossing_times"]):
        label = "Lap crossing" if idx == 0 else None
        ax.axvline(crossing_time, color="tab:red", linestyle=":", linewidth=1.0, alpha=0.8, label=label)
        ax.text(
            crossing_time,
            ax.get_ylim()[1],
            f" lap {idx + 1}\n{lap_stats['lap_times'][idx]:.2f}s",
            va="top",
            ha="right",
            fontsize=8,
            color="tab:red",
        )
    ax.grid(alpha=0.3)
    ax.set_title("Longitudinal px vs raceline s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arc length (m)")
    ax.legend()

    ax = axes[6]
    steering_history = [] if steering_history is None else steering_history
    steering_time = np.arange(len(steering_history)) * dt
    ax.plot(steering_time, steering_history, color="tab:blue", label="Steering")
    ax.axhline(1.0, ls="--", c="0.7", linewidth=0.8)
    ax.axhline(-1.0, ls="--", c="0.7", linewidth=0.8)
    ax.axhline(0.0, ls=":", c="0.5", linewidth=0.8)
    ax.set_ylim(-1.08, 1.08)
    ax.grid(alpha=0.3)
    ax.set_title("Steering command")
    ax.set_xlabel("Time (s)")
    ax.legend()

    ax = axes[7]
    throttle_history = [] if throttle_history is None else throttle_history
    throttle_time = np.arange(len(throttle_history)) * dt
    ax.plot(throttle_time, throttle_history, color="tab:orange", label="Throttle")
    ax.axhline(1.0, ls="--", c="0.7", linewidth=0.8)
    ax.axhline(-1.0, ls="--", c="0.7", linewidth=0.8)
    ax.axhline(0.0, ls="--", c="0.7", linewidth=0.8)
    ax.set_ylim(-1.08, 1.08)
    ax.grid(alpha=0.3)
    ax.set_title("Throttle command")
    ax.set_xlabel("Time (s)")
    ax.legend()

    plt.show()
    return fig


def plot_trace_npz(trace_path="simulation_trace.npz"):
    trace_path = Path(trace_path)
    data = np.load(trace_path, allow_pickle=True)
    time = data["time"]
    dt = _time_step(time)
    track_width = float(np.asarray(data["track_width"]).item()) if "track_width" in data.files else 2.0
    track_half_width = (
        float(np.asarray(data["track_half_width"]).item())
        if "track_half_width" in data.files
        else track_width / 2.0
    )
    if "offroad_recovery_threshold" in data.files:
        offroad_recovery_threshold = float(
            np.asarray(data["offroad_recovery_threshold"]).item()
        )
    elif "offroad_half_width" in data.files:  # compatibility with old traces
        offroad_recovery_threshold = float(np.asarray(data["offroad_half_width"]).item())
    else:
        offroad_recovery_threshold = 0.55
    track_length = float(np.asarray(data["track_length"]).item()) if "track_length" in data.files else None
    params = SimpleNamespace(
        DT=dt,
        SIM_DT=dt,
        TRACK_WIDTH=track_width,
        TRACK_HALF_WIDTH=track_half_width,
        OFFROAD_RECOVERY_THRESHOLD=offroad_recovery_threshold,
    )
    velocity_track_weight = (
        float(np.asarray(data["velocity_track_weight"]).item())
        if "velocity_track_weight" in data.files
        else None
    )
    heading_track_weight = (
        float(np.asarray(data["heading_track_weight"]).item())
        if "heading_track_weight" in data.files
        else None
    )
    direct_param_decisions = _direct_parameter_decisions(data)
    parameter_bounds = (
        np.asarray(data["llm_parameter_bounds"], dtype=float)
        if "llm_parameter_bounds" in data.files
        else np.asarray(list(sim_params.LLM_PARAMETER_BOUNDS.values()), dtype=float)
    )

    track_csv = _scalar_string(data["track_csv"]) if "track_csv" in data.files else "track_csv/berlin_2018.csv"
    raceline_csv = (
        _scalar_string(data["raceline_csv"])
        if "raceline_csv" in data.files
        else "track_csv/berlin_2018_raceline_with_speeds.csv"
    )
    track_scale = (
        float(np.asarray(data["track_scale"]).item())
        if "track_scale" in data.files
        else 1.0
    )
    waypoints = load_centerline_csv(track_csv, scale=track_scale)
    raceline = load_raceline_csv(raceline_csv, scale=track_scale)

    x = np.asarray(data["x"])
    y = np.asarray(data["y"])
    ref_x = np.asarray(data["ref_x"]) if "ref_x" in data.files else np.full_like(x, np.nan)
    ref_y = np.asarray(data["ref_y"]) if "ref_y" in data.files else np.full_like(y, np.nan)
    ref_lines = [
        (x[i], y[i], ref_x[i], ref_y[i])
        for i in range(99, len(x), 100)
        if np.isfinite(ref_x[i]) and np.isfinite(ref_y[i])
    ]

    other_trajectories = None
    if "car_x" in data.files and np.asarray(data["car_x"]).ndim == 2 and np.asarray(data["car_x"]).shape[1] > 1:
        car_x = np.asarray(data["car_x"])
        car_y = np.asarray(data["car_y"])
        other_trajectories = [(car_x[:, idx], car_y[:, idx]) for idx in range(1, car_x.shape[1])]

    return plot_simulation_results(
        waypoints,
        raceline,
        x,
        y,
        np.asarray(data["vx"]),
        np.asarray(data["py"]),
        np.asarray(data["mpc_loss"]),
        np.asarray(data["mpc_track_loss"]),
        np.asarray(data["mpc_action_loss"]),
        np.asarray(data["mpc_velocity_loss"]) if "mpc_velocity_loss" in data.files else np.zeros_like(data["mpc_loss"]),
        np.asarray(data["mpc_heading_loss"]) if "mpc_heading_loss" in data.files else np.zeros_like(data["mpc_loss"]),
        np.asarray(data["px"]),
        ref_lines,
        params,
        throttle_history=np.asarray(data["throttle"]) if "throttle" in data.files else None,
        steering_history=np.asarray(data["steer"]) if "steer" in data.files else None,
        other_trajectories=other_trajectories,
        ref_speed_history=np.asarray(data["ref_speed"]) if "ref_speed" in data.files else None,
        ref_lateral_errors=np.asarray(data["ref_py"]) if "ref_py" in data.files else None,
        ref_longitudinal_positions=np.asarray(data["ref_px"]) if "ref_px" in data.files else None,
        policy_params_history=np.asarray(data["policy_params"]) if "policy_params" in data.files else None,
        policy_names=np.asarray(data["policy_names"]) if "policy_names" in data.files else None,
        policy_mode_history=np.asarray(data["policy_mode"]) if "policy_mode" in data.files else None,
        track_length=track_length,
        trajectory_stride=1,
        velocity_dt_multiplier=1,
        velocity_track_weight=velocity_track_weight,
        heading_track_weight=heading_track_weight,
        direct_param_decisions=direct_param_decisions,
        parameter_bounds=parameter_bounds,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot simulation results from a saved .npz trace.")
    parser.add_argument("trace", nargs="?", default="simulation_trace.npz", help="Path to simulation trace .npz.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_trace_npz(args.trace)


if __name__ == "__main__":
    main()
    # plot_trace_npz("LLM.npz")
    # plot_trace_npz("F:/desktop/racer-LLM/result/grid_search/019_vw2_q1_h8.npz")
    # plot_trace_npz("F:/desktop/racer-LLM/result/grid_search/013_vw1_q2_h8.npz")
