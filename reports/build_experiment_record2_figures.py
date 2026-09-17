"""Build the compact figures used by ``LLM-racer-experiment record2``.

The script reads retained JSON/CSV/NPZ experiment outputs and writes PDF figures
directly into the Overleaf mirror. It never reruns a policy or changes results.

Usage:
    MPLCONFIGDIR=/tmp/racer_llm_matplotlib \
      .venv/bin/python reports/build_experiment_record2_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.vq import kmeans2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "LLM-racer-experiment record 2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "stable": "#4c78a8",
    "push": "#f58518",
    "overtake": "#54a24b",
    "block": "#b279a2",
    "supported": "#4c78a8",
    "ambiguous": "#bab0ac",
    "contradicted": "#e45756",
}
MODES = ["stable", "push", "overtake", "block"]


def load_json(path: str):
    with (ROOT / path).open() as handle:
        return json.load(handle)


def save(fig, name: str):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=220)
    plt.close(fig)


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)


def build_shared_mpc_hairpin():
    weights = np.array([0, 1, 3, 10])
    progress = np.array([64.2, 89.1, 95.2, 98.0])
    boundary = np.array([6.06, 0.34, 0.20, 0.0])
    final_speed = np.array([0.0, 5.0, 5.0, 5.0])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    bars = axes[0].bar(range(4), progress, color="#4c78a8")
    axes[0].set_xticks(range(4), [str(v) for v in weights])
    axes[0].set_xlabel(r"Reference-speed weight $w_v$")
    axes[0].set_ylabel("Progress after 25 s (m)")
    axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set_ylim(0, 108)
    style_axis(axes[0])

    x = np.arange(4)
    width = 0.36
    b1 = axes[1].bar(x - width / 2, boundary, width, label="Boundary time", color="#e45756")
    b2 = axes[1].bar(x + width / 2, final_speed, width, label="Final speed", color="#54a24b")
    axes[1].set_xticks(x, [str(v) for v in weights])
    axes[1].set_xlabel(r"Reference-speed weight $w_v$")
    axes[1].set_ylabel("Seconds / m s$^{-1}$")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].bar_label(b1, fmt="%.2g", fontsize=7)
    axes[1].bar_label(b2, fmt="%.1f", fontsize=7)
    style_axis(axes[1])
    fig.suptitle("Case-specific hairpin diagnostic (not a general stall guarantee)", fontsize=11)
    fig.tight_layout()
    save(fig, "record2_shared_mpc_hairpin.pdf")


def build_basis_and_action():
    audit = load_json("result/llm_basis_ab_berlin_rho3e4_reset_50s/audit/summary.json")
    variants = audit["variants"]
    grounding_names = ["Action-first", "Evidence-first"]
    grounding_keys = ["structured_action_first_minimal_v2", "evidence_first_minimal_v2"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    left = np.zeros(2)
    for status in ["supported", "ambiguous", "contradicted"]:
        values = np.array([
            variants[key]["grounding_counts"].get(status, 0) for key in grounding_keys
        ])
        axes[0].barh(grounding_names, values, left=left, color=COLORS[status], label=status.title())
        for y, (start, value) in enumerate(zip(left, values)):
            if value:
                axes[0].text(start + value / 2, y, str(int(value)), ha="center", va="center", fontsize=8)
        left += values
    axes[0].set_xlim(0, 150)
    axes[0].set_xlabel("Decisions")
    axes[0].set_title("Traffic-basis grounding")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axes[0].spines[["top", "right"]].set_visible(False)

    policy_names = ["Raw", "Action-first", "Evidence-first"]
    policy_keys = ["raw", "structured_action_first_minimal_v2", "evidence_first_minimal_v2"]
    bottom = np.zeros(3)
    for mode in MODES:
        values = np.array([variants[key]["mode_counts"].get(mode, 0) for key in policy_keys])
        axes[1].bar(policy_names, values, bottom=bottom, color=COLORS[mode], label=mode.title())
        bottom += values
    axes[1].set_ylim(0, 150)
    axes[1].set_ylabel("Decisions")
    axes[1].set_title("Mode output on the same 50 s screen")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axes[1])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    save(fig, "record2_basis_grounding_and_modes.pdf")


def build_prompt_and_two_call():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))

    prompt_modes = {
        "Raw": [0, 100, 0, 0],
        "Balanced": [1, 77, 22, 0],
    }
    bottom = np.zeros(2)
    for idx, mode in enumerate(MODES):
        values = np.array([prompt_modes[name][idx] for name in prompt_modes])
        axes[0].bar(list(prompt_modes), values, bottom=bottom, color=COLORS[mode], label=mode.title())
        bottom += values
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Decisions")
    axes[0].set_title("One neutral sentence")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    style_axis(axes[0])

    progress = [191.26, 202.61]
    bars = axes[1].bar(["Raw", "Balanced"], progress, color=["#4c78a8", "#f58518"])
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[1].set_ylim(175, 208)
    axes[1].set_ylabel("Mean progress (m)")
    axes[1].set_title("P2/P3, 50 s")
    style_axis(axes[1])

    names = ["One call", "Two calls"]
    latency = [3.17, 6.80]
    support = [37 / 45 * 100, 34 / 45 * 100]
    x = np.arange(2)
    axes[2].bar(x - 0.18, latency, 0.36, color="#e45756", label="Latency (s)")
    ax2 = axes[2].twinx()
    ax2.bar(x + 0.18, support, 0.36, color="#4c78a8", label="Supported (%)")
    axes[2].set_xticks(x, names)
    axes[2].set_ylim(0, 8)
    ax2.set_ylim(0, 100)
    axes[2].set_ylabel("Latency (s)")
    ax2.set_ylabel("Supported basis (%)")
    axes[2].set_title("Controlled split-call test")
    axes[2].spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    handles = [axes[2].patches[0], ax2.patches[0]]
    axes[2].legend(handles, ["Latency", "Supported"], frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "record2_prompt_and_two_call.pdf")


def build_counterfactual():
    data = load_json("result/common_scene_mode_dev_v2_60/summary.json")
    variants = ["raw", "balanced", "evidence_first"]
    labels = ["Raw", "Balanced", "Evidence-first"]
    summary = data["summary"]["primary_1s"]

    regrets = {variant: [] for variant in variants}
    for scene in data["scenes"]:
        for variant in variants:
            regrets[variant].append(scene["decisions"][variant]["primary_1s"]["regret_m"])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    bottom = np.zeros(3)
    for mode in MODES:
        values = np.array([summary[v]["mode_counts"].get(mode, 0) for v in variants])
        axes[0].bar(labels, values, bottom=bottom, color=COLORS[mode], label=mode.title())
        bottom += values
    axes[0].set_ylim(0, 60)
    axes[0].set_ylabel("Decisions")
    axes[0].set_title("Modes on 60 paired real states")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axes[0])

    positions = np.arange(3)
    rng = np.random.default_rng(12)
    for position, variant in zip(positions, variants):
        values = np.maximum(np.asarray(regrets[variant]), 1e-5)
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        axes[1].scatter(position + jitter, values, s=13, color="#4c78a8", alpha=0.42, edgecolors="none")
    means = [np.mean(regrets[v]) for v in variants]
    axes[1].scatter(positions, means, marker="D", color="#e45756", s=25, label="Mean")
    axes[1].axhline(0.05, color="#555555", linestyle="--", linewidth=0.9, label="0.05 m")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("One-second safe regret (m)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(8e-6, 0.6)
    axes[1].set_title("Most states weakly separate presets")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])
    fig.tight_layout()
    save(fig, "record2_counterfactual_utility.pdf")


def _one_second_samples(path: Path, car_indices):
    data = np.load(path, allow_pickle=True)
    time = np.asarray(data["time"], dtype=float)
    params = np.asarray(data["car_policy_params"], dtype=float)
    indices = np.unique(np.minimum((np.arange(0, 50) / np.median(np.diff(time))).astype(int), len(time) - 1))
    return time[indices], [params[indices, car, :] for car in car_indices], np.asarray(data["llm_parameter_bounds"], dtype=float)


def _cluster_ids(values, bounds, seed=7):
    scaled = (values - bounds[:, 0]) / np.maximum(bounds[:, 1] - bounds[:, 0], 1e-9)
    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(scaled, 5, minit="++", iter=50, seed=rng)
    order = np.argsort(centroids[:, 0])
    remap = np.empty_like(order)
    remap[order] = np.arange(5)
    return remap[labels]


def build_alpha_dynamics():
    self_path = ROOT / "result/alpha_local_selfplay_parameter_trace_p3_50s/summary_traces/trial_001.npz"
    ibr_path = ROOT / "result/alpha_local_p3_vs_traj_ibr_parameter_trace_50s/summary_traces/trial_001.npz"
    t_self, self_values, bounds = _one_second_samples(self_path, [0, 1, 2])
    t_ibr, ibr_values, _ = _one_second_samples(ibr_path, [0])
    names = [r"$q$", r"$\zeta$", r"$s_1$", r"$s_2$", r"$s_3$"]

    fig = plt.figure(figsize=(9.3, 7.0))
    grid = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1.25], hspace=0.22)
    for dim, name in enumerate(names):
        ax = fig.add_subplot(grid[dim, 0])
        for car, values in enumerate(self_values):
            ax.step(t_self, values[:, dim], where="post", color="#4c78a8", alpha=0.38 + 0.2 * car, linewidth=1.0)
        ax.step(t_ibr, ibr_values[0][:, dim], where="post", color="#e45756", linewidth=1.35)
        ax.set_ylabel(name, rotation=0, labelpad=13)
        ax.set_xlim(0, 49)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        if dim < 4:
            ax.set_xticklabels([])

    ax = fig.add_subplot(grid[5, 0])
    sequences = self_values + ibr_values
    ylabels = ["Self car 0", "Self car 1", "Self car 2", r"$\alpha$ vs IBR"]
    cluster_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for row, values in enumerate(sequences):
        ids = _cluster_ids(values, bounds, seed=7 + row)
        for index, cluster in enumerate(ids):
            ax.broken_barh([(t_self[index], 1.0)], (row - 0.34, 0.68), facecolors=cluster_colors[int(cluster)])
    ax.set_yticks(range(4), ylabels)
    ax.set_xlim(0, 50)
    ax.set_xlabel("Simulation time (s)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in cluster_colors]
    ax.legend(handles, [f"C{i}" for i in range(5)], ncol=5, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.34))
    fig.suptitle("Local Alpha-Racer parameters move through continuous, race-dependent phases", fontsize=11, y=0.995)
    fig.subplots_adjust(top=0.96, bottom=0.10)
    save(fig, "record2_alpha_parameter_dynamics.pdf")


def _read_csv(path: str):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_preset_cross_track():
    sources = {
        "Berlin": "result/mode_preset_audit_gym_controlled_sep03/berlin_3s/rollouts.csv",
        "Modena (local)": "result/local_raceline_and_interactive_audits/modena_3s/rollouts.csv",
        "Rounded (local)": "result/local_raceline_and_interactive_audits/rounded_rectangle_3s/rollouts.csv",
    }
    scenarios = ["clear_straight", "clear_curve_approach", "closing_on_front", "rear_vehicle_closing", "front_side_vehicle"]
    short_names = ["Straight", "Curve", "Front closing", "Rear closing", "Front-side"]
    rows = []
    unsafe = []
    labels = []
    for track, path in sources.items():
        entries = _read_csv(path)
        lookup = {(r["scenario"], r["mode"]): r for r in entries}
        for scenario, short in zip(scenarios, short_names):
            raw = []
            row_unsafe = []
            for mode in MODES:
                entry = lookup[(scenario, mode)]
                metric_key = "ego_progress_m" if scenario.startswith("clear_") else "target_relative_gain_m"
                raw.append(float(entry[metric_key]))
                row_unsafe.append(entry["unsafe"].lower() == "true")
            best = max(raw)
            rows.append([value - best for value in raw])
            unsafe.append(row_unsafe)
            labels.append(f"{track}: {short}")

    matrix = np.asarray(rows)
    unsafe = np.asarray(unsafe)
    cmap = LinearSegmentedColormap.from_list("regret", ["#e45756", "#f3d5d2", "#ffffff", "#4c78a8"])
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-2.5, vmax=0.15)
    ax.set_xticks(range(4), [m.title() for m in MODES])
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            text = f"{matrix[row, col]:.2f}"
            if unsafe[row, col]:
                text += "\nunsafe"
            ax.text(col, row, text, ha="center", va="center", fontsize=7,
                    color="white" if matrix[row, col] < -1.4 else "black",
                    fontweight="bold" if unsafe[row, col] else "normal")
    divider_positions = [4.5, 9.5]
    for pos in divider_positions:
        ax.axhline(pos, color="#333333", linewidth=1.2)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Outcome relative to the best preset in that scene (m)")
    ax.set_title("Preset rankings change with track and raceline")
    fig.tight_layout()
    save(fig, "record2_mode_preset_cross_track.pdf")


def build_block_audit():
    entries = _read_csv("result/local_raceline_and_interactive_audits/interactive_block/rollouts.csv")
    controllers = ["stable", "push", "no_block_s3_zero", "block"]
    labels = ["Stable", "Push", r"No block ($s_3=0$)", "Block"]
    safe_counts = []
    collision_counts = []
    offroad_counts = []
    max_block_reference = []
    for controller in controllers:
        selected = [row for row in entries if row["ego_variant"] == controller]
        safe_counts.append(sum(row["unsafe"].lower() == "false" for row in selected))
        collision_counts.append(sum(row["collision"].lower() == "true" for row in selected))
        offroad_counts.append(sum(row["ego_offroad"].lower() == "true" for row in selected))
        max_block_reference.append(max(float(row["max_abs_ego_block_reference_m"]) for row in selected))

    effects = _read_csv("result/local_raceline_and_interactive_audits/interactive_block/block_effects.csv")
    grouped = {}
    for row in effects:
        grouped.setdefault(row["challenge"], []).append(float(row["challenger_relative_gain_reduction_m"]))
    challenges = ["near_moderate", "nominal", "far_aggressive"]
    reductions = [np.mean(grouped[name]) for name in challenges]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
    x = np.arange(4)
    axes[0].bar(x, safe_counts, color="#4c78a8")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0, 6.5)
    axes[0].set_ylabel("Safe branches / 6")
    axes[0].set_title("Control scene is mostly unsafe")
    style_axis(axes[0])

    width = 0.36
    axes[1].bar(x - width / 2, collision_counts, width, label="Collision", color="#e45756")
    axes[1].bar(x + width / 2, offroad_counts, width, label="Off-road", color="#b279a2")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 6.5)
    axes[1].set_ylabel("Branches / 6")
    axes[1].set_title("Failure type")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])

    bars = axes[2].bar(["Near", "Nominal", "Far"], reductions, color="#54a24b")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[2].set_ylabel("Challenger gain reduction (m)")
    axes[2].set_title("Block effect vs $s_3=0$")
    style_axis(axes[2])
    fig.tight_layout()
    save(fig, "record2_block_interaction_audit.pdf")


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    })
    build_shared_mpc_hairpin()
    build_basis_and_action()
    build_prompt_and_two_call()
    build_counterfactual()
    build_alpha_dynamics()
    build_preset_cross_track()
    build_block_audit()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
