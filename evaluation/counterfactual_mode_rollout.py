#!/usr/bin/env python3
"""Offline counterfactual evaluation of one recorded LLM mode decision.

This is a detachable evaluator: it does not modify the simulator or policies.
For sampled LLM decisions, the ego is restored from the trace and executes
each fixed mode preset. Opponents replay their recorded trajectories exactly.
All branches run for the short horizon; ambiguous samples are rerun to the long
horizon. A safe mode is acceptable when its progress reaches a configurable
fraction of the best safe progress.

Typical usage is documented in run_counterfactual_mode_rollouts.sh.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collision import oriented_rectangle_overlap_cost
from config import make_vehicle_params, params
from mpc_controller import MPCController
from policy import StaticAlphaRacerPolicy, progress_delta
from simulation_core.observation import RaceObservation
from simulation import create_vehicle_model
from state import CarState
from track import Track, load_centerline_csv, load_raceline_csv


MODES = ("stable", "push", "overtake", "block")


def scalar(data, name, default):
    return np.asarray(data[name]).item() if name in data.files else default


def recorded_state(data, frame, car):
    return CarState(
        x=float(data["car_x"][frame, car]),
        y=float(data["car_y"][frame, car]),
        psi=float(data["car_psi"][frame, car]),
        vx=float(data["car_vx"][frame, car]),
        vy=float(data["car_vy"][frame, car]),
        omega=float(data["car_omega"][frame, car]),
    )


def build_policy(data, track, raceline, mode, car_idx):
    model_type = str(scalar(data, "vehicle_model", params.VEHICLE_MODEL))
    vehicle_params = make_vehicle_params(model_type)
    mpc = MPCController(
        dt=float(scalar(data, "mpc_dt", params.MPC_DT)),
        horizon=int(scalar(data, "mpc_horizon", params.MPC_HORIZON)),
        Q=params.Q,
        R=params.R,
        max_vel_rate=params.MAX_VEL_RATE,
        max_steer_delta=params.MPC_MAX_STEER_RATE * float(scalar(data, "mpc_dt", params.MPC_DT)),
        db_params=vehicle_params,
        track_width=2.0 * float(scalar(data, "track_half_width", params.TRACK_HALF_WIDTH)),
        track_constraint_half_width=float(scalar(data, "mpc_constraint_half_width", params.MPC_CONSTRAINT_HALF_WIDTH)),
        max_state_speed=params.MAX_STATE_SPEED,
        velocity_track_weight=float(scalar(data, "velocity_track_weight", params.VELOCITY_TRACK_WEIGHT)),
        heading_track_weight=float(scalar(data, "heading_track_weight", params.HEADING_TRACK_WEIGHT)),
        track_length=track.total_length,
        min_longitudinal_separation=params.MPC_MIN_LONGITUDINAL_SEPARATION,
        min_lateral_separation=params.MPC_MIN_LATERAL_SEPARATION,
        max_constrained_opponents=params.MPC_MAX_CONSTRAINED_OPPONENTS,
        collision_superellipse_exponent=params.MPC_COLLISION_SUPERELLIPSE_EXPONENT,
        enforce_track_boundary=bool(scalar(data, "mpc_enforce_track_boundary", True)),
        enforce_collision_avoidance=bool(scalar(data, "mpc_enforce_collision_avoidance", True)),
        track_slack_weight=float(scalar(data, "track_slack_weight", params.MPC_TRACK_SLACK_WEIGHT)),
        collision_slack_weight=float(scalar(data, "mpc_collision_slack_weight", params.MPC_COLLISION_SLACK_WEIGHT)),
        enforce_friction_speed_limit=bool(scalar(data, "mpc_enforce_friction_speed_limit", True)),
        friction_coefficient=float(scalar(data, "mpc_friction_coefficient", params.MPC_FRICTION_COEFFICIENT)),
        track=track,
    )
    policy = StaticAlphaRacerPolicy(
        track,
        mpc,
        params=np.asarray(params.LLM_MODE_PRESETS[mode], dtype=float),
        raceline=raceline,
        max_speed=params.MAX_RACELINE_SPEED,
        ref_lateral_clip=float(scalar(data, "ref_lateral_clip", params.REF_LATERAL_CLIP)),
        car_idx=car_idx,
        policy_name=f"counterfactual_{mode}",
    )
    return policy, create_vehicle_model(
        model_type,
        dt=float(scalar(data, "simulation_dt", params.SIM_DT)),
        db_params=vehicle_params,
        max_speed=params.MAX_STATE_SPEED,
    )


def simulate_branch(data, frame, ego_idx, mode, duration, track, raceline, record_trajectory=False):
    policy, plant = build_policy(data, track, raceline, mode, ego_idx)
    sim_dt = float(scalar(data, "simulation_dt", params.SIM_DT))
    control_dt = float(scalar(data, "control_dt", params.CONTROL_DT))
    control_steps = int(round(control_dt / sim_dt))
    steps = min(int(round(duration / sim_dt)), len(data["time"]) - frame - 1)
    state = recorded_state(data, frame, ego_idx)
    current_s = float(data["car_px"][frame, ego_idx])
    initial_s = current_s
    opponent_indices = [car for car in range(data["car_px"].shape[1]) if car != ego_idx]
    initial_leading_opponent_s = max(
        float(data["car_px"][frame, car]) for car in opponent_indices
    )
    initial_relative_lead = initial_s - initial_leading_opponent_s
    previous = max(0, frame - 1)
    warm = np.asarray(
        [data["car_throttle"][previous, ego_idx], data["car_steer"][previous, ego_idx]],
        dtype=float,
    )
    policy.last_u = warm.copy()
    policy.last_feasible_u = warm.copy()
    action = None
    collision = False
    offroad = False
    boundary = False
    solver_failures = 0
    stalled_steps = 0
    minimum_gap = math.inf
    max_abs_e = 0.0
    progress_history = [current_s]
    trajectory = {
        "time": [0.0], "ego_x": [float(state.x)], "ego_y": [float(state.y)],
        "opponent_x": [[float(data["car_x"][frame, car]) for car in opponent_indices]],
        "opponent_y": [[float(data["car_y"][frame, car]) for car in opponent_indices]],
    }

    for step in range(steps):
        source_frame = frame + step
        current_s, e = track.project(
            state.x,
            state.y,
            s_hint=current_s,
            search_radius=params.PROJECTION_SEARCH_RADIUS,
            psi_hint=state.psi,
            max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
        )
        max_abs_e = max(max_abs_e, abs(float(e)))
        boundary = boundary or abs(e) > float(scalar(data, "track_half_width", params.TRACK_HALF_WIDTH))
        offroad = offroad or abs(e) > float(scalar(data, "offroad_recovery_threshold", params.OFFROAD_RECOVERY_THRESHOLD))

        opponent_states = []
        scene_states = []
        scene_s = []
        scene_e = []
        for car in range(data["car_x"].shape[1]):
            other = state if car == ego_idx else recorded_state(data, source_frame, car)
            scene_states.append(other)
            scene_s.append(current_s if car == ego_idx else float(data["car_px"][source_frame, car]))
            scene_e.append(e if car == ego_idx else float(data["car_py"][source_frame, car]))
            if car != ego_idx:
                opponent_states.append(other)
                minimum_gap = min(minimum_gap, abs(progress_delta(scene_s[car], current_s, track.total_length)))

        if step % control_steps == 0:
            obs = RaceObservation.from_runtime(
                [policy] * len(scene_states), scene_states, scene_s, scene_e,
                float(data["time"][frame]) + step * sim_dt,
            )
            action = policy.get_action(obs)
            solver_failures += int(not policy.mpc.last_solver_success)
        if state.vx < 0.05:
            stalled_steps += 1
        collision = collision or any(
            oriented_rectangle_overlap_cost(
                state, other, params.VEHICLE_HALF_LENGTH, params.VEHICLE_HALF_WIDTH
            ) > 0.0
            for other in opponent_states
        )
        if collision or offroad:
            break
        state = CarState.from_array(plant.step(state.to_array(), action.throttle, action.steer))
        progress_history.append(current_s)
        if record_trajectory:
            next_frame = min(source_frame + 1, len(data["time"]) - 1)
            trajectory["time"].append((step + 1) * sim_dt)
            trajectory["ego_x"].append(float(state.x))
            trajectory["ego_y"].append(float(state.y))
            trajectory["opponent_x"].append(
                [float(data["car_x"][next_frame, car]) for car in opponent_indices]
            )
            trajectory["opponent_y"].append(
                [float(data["car_y"][next_frame, car]) for car in opponent_indices]
            )

    final_s, final_e = track.project(
        state.x,
        state.y,
        s_hint=current_s,
        search_radius=params.PROJECTION_SEARCH_RADIUS,
        psi_hint=state.psi,
        max_heading_error=params.PROJECTION_MAX_HEADING_ERROR,
    )
    progress = float(final_s - initial_s)
    final_source_frame = min(frame + max(len(progress_history) - 1, 0), len(data["time"]) - 1)
    final_leading_opponent_s = max(
        float(data["car_px"][final_source_frame, car]) for car in opponent_indices
    )
    final_relative_lead = float(final_s - final_leading_opponent_s)
    result = {
        "mode": mode,
        "requested_duration_s": float(duration),
        "actual_duration_s": float((len(progress_history) - 1) * sim_dt),
        "progress_m": progress,
        "initial_relative_lead_m": float(initial_relative_lead),
        "final_relative_lead_m": final_relative_lead,
        "relative_progress_gain_m": float(final_relative_lead - initial_relative_lead),
        "final_speed_mps": float(state.vx),
        "max_abs_lateral_m": max(max_abs_e, abs(float(final_e))),
        "minimum_longitudinal_gap_m": None if not np.isfinite(minimum_gap) else float(minimum_gap),
        "boundary_violation": bool(boundary),
        "offroad": bool(offroad),
        "collision": bool(collision),
        "solver_failure_count": int(solver_failures),
        "stalled_fraction": float(stalled_steps / max(steps, 1)),
        "safe": bool(not collision and not offroad and solver_failures == 0),
    }
    if record_trajectory:
        result["_trajectory"] = {name: np.asarray(values) for name, values in trajectory.items()}
    return result


def decision_candidates(data, ego_idx, long_seconds, track):
    sim_dt = float(scalar(data, "simulation_dt", params.SIM_DT))
    control_dt = float(scalar(data, "control_dt", params.CONTROL_DT))
    calls = np.asarray(data["llm_raw_call_indices"])[ego_idx]
    frames = ((calls[calls >= 1] - 1) * round(control_dt / sim_dt)).astype(int)
    future = int(math.ceil(long_seconds / sim_dt))
    reset = np.asarray(
        data["car_deadlock_reset_event"]
        if "car_deadlock_reset_event" in data.files
        else np.zeros((len(data["collision"]), data["car_x"].shape[1]), dtype=bool),
        dtype=bool,
    )
    collision = np.asarray(data["collision"], dtype=bool)
    curvature_s, curvature = track.curvature_profile(sample_step=0.1, heading_window=0.5)
    usable = []
    for frame in frames:
        end = frame + future
        if end >= len(data["time"]):
            continue
        if collision[frame:end + 1].any() or reset[frame:end + 1].any():
            continue
        if bool(data["car_offroad"][frame, ego_idx]) or float(data["car_vx"][frame, ego_idx]) < 0.05:
            continue
        ego_s = float(data["car_px"][frame, ego_idx])
        ego_v = float(data["car_vx"][frame, ego_idx])
        current_kappa = float(np.interp(ego_s % track.total_length, curvature_s, curvature))
        lookahead_s = (ego_s + np.linspace(0.0, 8.0, 17)) % track.total_length
        upcoming_kappa = float(np.max(np.abs(np.interp(lookahead_s, curvature_s, curvature))))
        front_gap = math.inf
        front_closing = 0.0
        rear_gap = math.inf
        rear_closing = 0.0
        for car in range(data["car_px"].shape[1]):
            if car == ego_idx:
                continue
            ds = progress_delta(float(data["car_px"][frame, car]), ego_s, track.total_length)
            dv = ego_v - float(data["car_vx"][frame, car])
            delta_s = float(ds)
            if delta_s > 0.0 and delta_s < front_gap:
                front_gap, front_closing = delta_s, max(dv, 0.0)
            elif delta_s < 0.0 and abs(delta_s) < rear_gap:
                rear_gap, rear_closing = abs(delta_s), max(-dv, 0.0)
        mode = str(data["car_policy_mode"][frame, ego_idx])
        usable.append({
            "frame": int(frame), "mode": mode,
            "current_abs_curvature": abs(current_kappa),
            "upcoming_abs_curvature": upcoming_kappa,
            "front_gap_m": None if not np.isfinite(front_gap) else float(front_gap),
            "front_closing_mps": float(front_closing),
            "rear_gap_m": None if not np.isfinite(rear_gap) else float(rear_gap),
            "rear_closing_mps": float(rear_closing),
        })
    return usable


def select_frames(data, ego_idx, sample_count, long_seconds, track, strategy):
    candidates = decision_candidates(data, ego_idx, long_seconds, track)
    if len(candidates) <= sample_count:
        for item in candidates:
            item["sampling_reasons"] = ["all_usable"]
        return candidates
    if strategy == "uniform":
        indices = np.linspace(0, len(candidates) - 1, sample_count).round().astype(int)
        selected = [dict(candidates[i]) for i in np.unique(indices)]
        for item in selected:
            item["sampling_reasons"] = ["uniform_time"]
        return selected

    selected = {}
    quota = max(1, sample_count // 5)

    def add_ranked(label, key, reverse=True, count=quota, eligible=lambda row: True):
        ranked = sorted((row for row in candidates if eligible(row)), key=key, reverse=reverse)
        for item in ranked[:count]:
            if item["frame"] not in selected and len(selected) >= sample_count:
                break
            reasons = selected.setdefault(item["frame"], dict(item, sampling_reasons=[]))["sampling_reasons"]
            if label not in reasons:
                reasons.append(label)

    add_ranked("upcoming_curvature", lambda row: row["upcoming_abs_curvature"])
    add_ranked(
        "front_interaction",
        lambda row: row["front_closing_mps"] / max(row["front_gap_m"] or math.inf, 0.2),
        eligible=lambda row: row["front_gap_m"] is not None and row["front_closing_mps"] > 0.0,
    )
    add_ranked(
        "rear_interaction",
        lambda row: row["rear_closing_mps"] / max(row["rear_gap_m"] or math.inf, 0.2),
        eligible=lambda row: row["rear_gap_m"] is not None and row["rear_closing_mps"] > 0.0,
    )
    mode_counts = {}
    for item in candidates:
        mode_counts[item["mode"]] = mode_counts.get(item["mode"], 0) + 1
    if len(mode_counts) > 1:
        add_ranked("rare_mode", lambda row: 1.0 / mode_counts[row["mode"]])

    changes = []
    for idx, item in enumerate(candidates):
        if idx and item["mode"] != candidates[idx - 1]["mode"]:
            changes.extend([candidates[idx - 1], item])
    for item in changes:
        if item["frame"] not in selected and len(selected) >= sample_count:
            break
        reasons = selected.setdefault(item["frame"], dict(item, sampling_reasons=[]))["sampling_reasons"]
        if "mode_change" not in reasons:
            reasons.append("mode_change")

    uniform_indices = np.linspace(0, len(candidates) - 1, sample_count).round().astype(int)
    for idx in uniform_indices:
        if len(selected) >= sample_count:
            break
        item = candidates[idx]
        selected.setdefault(item["frame"], dict(item, sampling_reasons=[]))["sampling_reasons"].append("uniform_fill")
    if len(selected) < sample_count:
        for item in candidates:
            if len(selected) >= sample_count:
                break
            selected.setdefault(item["frame"], dict(item, sampling_reasons=[]))["sampling_reasons"].append("time_fill")
    return sorted(selected.values(), key=lambda row: row["frame"])


def label_acceptability(branches, ratio):
    safe = [row for row in branches if row["safe"]]
    if not safe:
        for row in branches:
            row["acceptable"] = False
            row["regret_m"] = None
            row["best_safe"] = False
        return [], None
    # Primary score: change in signed lead over the most advanced recorded
    # opponent. Positive means ego closes the gap or extends its lead; negative
    # means ego falls farther behind or loses part of its lead. Direct argmax
    # is sign-independent. A literal ratio*best is ill-posed for a negative
    # best score, so use a symmetric 20%-of-|best| regret allowance.
    best = max(row["relative_progress_gain_m"] for row in safe)
    threshold = best - (1.0 - ratio) * max(abs(best), 1.0)
    acceptable = []
    for row in branches:
        # Constrained regret is defined only for safe branches. Unsafe choices
        # are counted separately rather than hidden behind an arbitrary crash
        # penalty expressed in metres.
        row["regret_m"] = (
            float(best - row["relative_progress_gain_m"])
            if row["safe"] else None
        )
        row["best_safe"] = bool(row["safe"] and row["regret_m"] <= 1e-9)
        row["acceptable"] = bool(
            row["safe"] and row["relative_progress_gain_m"] >= threshold
        )
        if row["acceptable"]:
            acceptable.append(row["mode"])
    return acceptable, float(best)


def save_overview_plot(output_dir, samples):
    """Write a compact mode-by-scene score map for this temporary audit."""
    if not samples:
        return None
    scores = np.full((len(samples), len(MODES)), np.nan)
    labels = []
    for row_idx, sample in enumerate(samples):
        branches = sample["long_branches"] or sample["short_branches"]
        by_mode = {row["mode"]: row for row in branches}
        for col_idx, mode in enumerate(MODES):
            if by_mode[mode]["safe"]:
                scores[row_idx, col_idx] = by_mode[mode]["relative_progress_gain_m"]
        reasons = ",".join(sample["sampling_reasons"][:2])
        labels.append(f"{sample['sample']:02d}  t={sample['time_s']:.0f}s  {reasons}")

    finite = scores[np.isfinite(scores)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1.0)
    fig_height = max(4.0, 0.47 * len(samples) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, fig_height))
    image_handle = ax.imshow(scores, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for row_idx, sample in enumerate(samples):
        branches = sample["long_branches"] or sample["short_branches"]
        by_mode = {row["mode"]: row for row in branches}
        for col_idx, mode in enumerate(MODES):
            branch = by_mode[mode]
            mark = "×" if not branch["safe"] else ("✓" if branch["acceptable"] else "")
            value = "unsafe" if not branch["safe"] else f"{branch['relative_progress_gain_m']:+.1f} {mark}"
            ax.text(col_idx, row_idx, value, ha="center", va="center", fontsize=8)
        recorded_col = MODES.index(sample["recorded_mode"])
        ax.add_patch(plt.Rectangle((recorded_col - 0.48, row_idx - 0.48), 0.96, 0.96,
                                   fill=False, edgecolor="black", linewidth=2.0))
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_yticks(range(len(samples)), labels)
    ax.set_xlabel("Fixed counterfactual mode (black outline = recorded choice)")
    ax.set_title("Relative progress gain over the leading recorded opponent [m]")
    fig.colorbar(image_handle, ax=ax, shrink=0.8, label="relative progress gain [m]")
    fig.tight_layout()
    path = output_dir / "temporary_counterfactual_overview.svg"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def save_case_gif(output_dir, sample, branches, track, half_width, stride):
    """Animate all fixed-mode branches around one informative local scene."""
    colors = {"stable": "#377eb8", "push": "#e41a1c", "overtake": "#4daf4a", "block": "#984ea3"}
    source_s = sample["source_progress_m"]
    s_values = np.linspace(source_s - 5.0, source_s + 32.0, 420)
    center = np.asarray([track.frenet_to_world(float(s), 0.0) for s in s_values])
    left = np.asarray([track.frenet_to_world(float(s), half_width) for s in s_values])
    right = np.asarray([track.frenet_to_world(float(s), -half_width) for s in s_values])
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(center[:, 0], center[:, 1], "--", color="0.65", linewidth=1)
    ax.plot(left[:, 0], left[:, 1], color="0.25", linewidth=1)
    ax.plot(right[:, 0], right[:, 1], color="0.25", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(left[:, 0].min(), right[:, 0].min()) - 1, max(left[:, 0].max(), right[:, 0].max()) + 1)
    ax.set_ylim(min(left[:, 1].min(), right[:, 1].min()) - 1, max(left[:, 1].max(), right[:, 1].max()) + 1)
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    trails, points = {}, {}
    for branch in branches:
        mode = branch["mode"]
        trails[mode], = ax.plot([], [], color=colors[mode], linewidth=2, label=mode)
        points[mode], = ax.plot([], [], "o", color=colors[mode], markersize=5)
    first_traj = branches[0]["_trajectory"]
    opponent_points = [ax.plot([], [], "s", color="0.2", markersize=6)[0]
                       for _ in range(first_traj["opponent_x"].shape[1])]
    ax.legend(loc="best", fontsize=8)
    max_frames = max(len(branch["_trajectory"]["time"]) for branch in branches)
    animation_frames = list(range(0, max_frames, max(1, stride)))
    if animation_frames[-1] != max_frames - 1:
        animation_frames.append(max_frames - 1)

    def update(frame_idx):
        artists = []
        for branch in branches:
            mode, trajectory = branch["mode"], branch["_trajectory"]
            idx = min(frame_idx, len(trajectory["time"]) - 1)
            trails[mode].set_data(trajectory["ego_x"][:idx + 1], trajectory["ego_y"][:idx + 1])
            points[mode].set_data([trajectory["ego_x"][idx]], [trajectory["ego_y"][idx]])
            artists.extend((trails[mode], points[mode]))
        opponent_idx = min(frame_idx, len(first_traj["time"]) - 1)
        for car_idx, point in enumerate(opponent_points):
            point.set_data([first_traj["opponent_x"][opponent_idx, car_idx]],
                           [first_traj["opponent_y"][opponent_idx, car_idx]])
            artists.append(point)
        ax.set_title(f"TEMPORARY audit: sample {sample['sample']} at t={sample['time_s']:.1f}s; +{first_traj['time'][opponent_idx]:.1f}s")
        return artists

    animation = FuncAnimation(fig, update, frames=animation_frames, interval=100, blit=False)
    path = output_dir / f"temporary_case_{sample['sample']:02d}.gif"
    animation.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return path.name


def write_temporary_report(output_dir, payload, overview_name):
    samples = payload["samples"]
    informative = [sample for sample in samples if sample["informative"]]
    interesting = [sample for sample in samples if sample.get("temporary_gif")]
    rows = []
    for sample in interesting:
        branches = sample["long_branches"] or sample["short_branches"]
        gains = ", ".join(f"{row['mode']} {row['relative_progress_gain_m']:+.1f}" for row in branches)
        gif = html.escape(sample["temporary_gif"])
        rows.append(
            f"<tr><td>{sample['sample']}</td><td>{sample['time_s']:.1f}</td>"
            f"<td>{html.escape(', '.join(sample['sampling_reasons']))}</td>"
            f"<td>{html.escape(sample['recorded_mode'])}</td>"
            f"<td>{html.escape(', '.join(sample['final_acceptable_modes']) or 'none')}</td>"
            f"<td>{html.escape(gains)}</td><td><a href='{gif}'><img src='{gif}'></a></td></tr>"
        )
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Temporary counterfactual report</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:1150px}} img{{max-width:100%}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:.45rem;vertical-align:top}} td img{{width:250px}} .note{{color:#555}}</style></head>
<body><h1>TEMPORARY counterfactual mode report</h1>
<p><b>{len(samples)} sampled decisions; {len(informative)} informative scenes; recorded choice accepted in {sum(s['recorded_mode_acceptable'] for s in samples)}/{len(samples) or 1} scenes.</b></p>
<p>Sampling: {html.escape(payload['sampling'])}</p>
{f'<img src="{html.escape(overview_name)}">' if overview_name else ''}
<p class="note">A check mark means acceptable and a cross means unsafe. The black outline is the recorded LLM choice. High overall acceptance is not automatically good: if all four modes are accepted, that scene provides little evidence about mode selection.</p>
<h2>Selected informative or rejected cases</h2>
<table><tr><th>#</th><th>time [s]</th><th>sampling reason</th><th>recorded</th><th>acceptable</th><th>relative gains [m]</th><th>local replay</th></tr>
{''.join(rows) if rows else '<tr><td colspan="7">No discriminative case was found among these samples.</td></tr>'}</table>
<p class="note">Opponents replay their recorded trajectories and do not react to the counterfactual ego. This is a screening audit, not a replacement for closed-loop races.</p>
</body></html>"""
    path = output_dir / "temporary_counterfactual_report.html"
    path.write_text(report)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Offline fixed-mode counterfactual rollouts.")
    parser.add_argument("--trace", type=Path, required=True, help="Recorded 50 s LLM NPZ trace.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12, help="Number of usable LLM decisions.")
    parser.add_argument("--sampling-strategy", choices=("stratified", "uniform"), default="stratified",
                        help="Prefer curves/interactions/rare modes, or sample uniformly in time.")
    parser.add_argument("--short-seconds", type=float, default=2.0)
    parser.add_argument("--long-seconds", type=float, default=5.0)
    parser.add_argument("--progress-ratio", type=float, default=0.80,
                        help="Safe modes at least this fraction of best progress are acceptable.")
    parser.add_argument("--extend-if-acceptable-count", type=int, default=2,
                        help="Run long branches when at least this many short modes remain acceptable.")
    parser.add_argument("--max-gifs", type=int, default=4,
                        help="Maximum temporary GIFs for informative/rejected cases (0 disables).")
    parser.add_argument("--gif-stride", type=int, default=5,
                        help="Use every Nth simulator frame in temporary GIFs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.progress_ratio <= 1.0:
        raise ValueError("--progress-ratio must be in (0,1]")
    if not 0.0 < args.short_seconds <= args.long_seconds:
        raise ValueError("Require 0 < short-seconds <= long-seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.trace, allow_pickle=True)
    ego_idx = int(scalar(data, "ego_index", 0))
    track = Track(load_centerline_csv(str(scalar(data, "track_csv", params.TRACK_CSV)), scale=float(scalar(data, "track_scale", 1.0))))
    raceline = load_raceline_csv(str(scalar(data, "raceline_csv", params.RACELINE_CSV)), scale=float(scalar(data, "track_scale", 1.0)))
    selected = select_frames(
        data, ego_idx, args.samples, args.long_seconds, track, args.sampling_strategy
    )
    samples = []
    flat_rows = []
    gif_count = 0
    for number, selected_row in enumerate(selected, start=1):
        frame = selected_row["frame"]
        chosen = str(data["car_policy_mode"][frame, ego_idx])
        print(f"[{number}/{len(selected)}] t={data['time'][frame]:.2f}s recorded={chosen} sample={','.join(selected_row['sampling_reasons'])}", flush=True)
        short = [
            simulate_branch(data, frame, ego_idx, mode, args.short_seconds, track, raceline, True)
            for mode in MODES
        ]
        short_ok, short_best = label_acceptability(short, args.progress_ratio)
        extend = len(short_ok) >= args.extend_if_acceptable_count
        long_rows = []
        if extend and args.long_seconds > args.short_seconds:
            long_rows = [
                simulate_branch(data, frame, ego_idx, mode, args.long_seconds, track, raceline, True)
                for mode in MODES
            ]
            final_ok, final_best = label_acceptability(long_rows, args.progress_ratio)
        else:
            final_ok, final_best = short_ok, short_best
        sample = {
            "sample": number,
            "frame": int(frame),
            "time_s": float(data["time"][frame]),
            "recorded_mode": chosen,
            "recorded_basis": str(data["car_policy_basis"][frame, ego_idx]),
            "recorded_reason": str(data["car_policy_reason"][frame, ego_idx]),
            "source_progress_m": float(data["car_px"][frame, ego_idx]),
            "sampling_reasons": selected_row["sampling_reasons"],
            "scene_features": {key: value for key, value in selected_row.items()
                               if key not in ("frame", "sampling_reasons")},
            "extended": extend,
            "short_acceptable_modes": short_ok,
            "final_acceptable_modes": final_ok,
            "recorded_mode_acceptable": chosen in final_ok,
            "informative": len(final_ok) < len(MODES),
            "short_branches": short,
            "long_branches": long_rows,
        }
        final_rows = long_rows or short
        chosen_row = next(row for row in final_rows if row["mode"] == chosen)
        sample["recorded_mode_safe"] = bool(chosen_row["safe"])
        sample["recorded_mode_regret_m"] = chosen_row["regret_m"]
        sample["best_safe_modes"] = [
            row["mode"] for row in final_rows if row["best_safe"]
        ]
        interesting = sample["informative"] or not sample["recorded_mode_acceptable"]
        if interesting and gif_count < args.max_gifs:
            sample["temporary_gif"] = save_case_gif(
                args.output_dir, sample, final_rows, track,
                float(scalar(data, "track_half_width", params.TRACK_HALF_WIDTH)), args.gif_stride,
            )
            gif_count += 1
        samples.append(sample)
        for stage, rows in (("short", short), ("long", long_rows)):
            for row in rows:
                scalar_row = {key: value for key, value in row.items() if key != "_trajectory"}
                flat_rows.append({
                    "sample": number, "time_s": sample["time_s"], "recorded_mode": chosen,
                    "basis": sample["recorded_basis"], "stage": stage, **scalar_row,
                })
        for row in short + long_rows:
            row.pop("_trajectory", None)

    acceptance = [row["recorded_mode_acceptable"] for row in samples]
    safe_regrets = [
        row["recorded_mode_regret_m"]
        for row in samples
        if row["recorded_mode_regret_m"] is not None
    ]
    informative_regrets = [
        row["recorded_mode_regret_m"]
        for row in samples
        if row["informative"] and row["recorded_mode_regret_m"] is not None
    ]
    payload = {
        "trace": str(args.trace),
        "settings": vars(args) | {"trace": str(args.trace), "output_dir": str(args.output_dir)},
        "sampling": (
            f"{args.sampling_strategy} sampling of actual LLM decisions; exclude low-speed/offroad starts "
            "and source windows containing collision/reset. Stratified sampling prioritizes upcoming "
            "curvature, closing front/rear traffic, rare recorded modes, and mode changes, then fills in time."
        ),
        "opponents": "Frozen replay of recorded six-state trajectories; no counterfactual reaction.",
        "acceptability": (
            "Safe branch whose change in signed lead over the most advanced frozen opponent is within "
            "(1-progress_ratio)*max(abs(best_relative_gain),1 m) of the best safe branch. Positive "
            "means closing/extending a lead and negative means falling behind/losing a lead. Final "
            "signed lead and absolute ego progress are retained as diagnostics."
        ),
        "sample_count": len(samples),
        "recorded_mode_acceptance_rate": float(np.mean(acceptance)) if acceptance else None,
        "recorded_mode_safe_rate": (
            float(np.mean([row["recorded_mode_safe"] for row in samples]))
            if samples else None
        ),
        "recorded_mode_mean_regret_m_safe_choices": (
            float(np.mean(safe_regrets)) if safe_regrets else None
        ),
        "recorded_mode_median_regret_m_safe_choices": (
            float(np.median(safe_regrets)) if safe_regrets else None
        ),
        "informative_sample_count": int(sum(row["informative"] for row in samples)),
        "recorded_mode_mean_regret_m_informative_safe_choices": (
            float(np.mean(informative_regrets)) if informative_regrets else None
        ),
        "recorded_mode_acceptance_rate_informative": (
            float(np.mean([row["recorded_mode_acceptable"] for row in samples if row["informative"]]))
            if any(row["informative"] for row in samples) else None
        ),
        "samples": samples,
    }
    overview_name = save_overview_plot(args.output_dir, samples)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    if flat_rows:
        fields = list(flat_rows[0])
        with (args.output_dir / "branches.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
    report_path = write_temporary_report(args.output_dir, payload, overview_name)
    print(f"Saved {len(samples)} samples to {args.output_dir}")
    print(f"Recorded-mode acceptance rate: {payload['recorded_mode_acceptance_rate']}")
    print(
        "Recorded-mode mean regret (safe choices): "
        f"{payload['recorded_mode_mean_regret_m_safe_choices']} m"
    )
    print(f"Temporary report: {report_path}")


if __name__ == "__main__":
    main()
