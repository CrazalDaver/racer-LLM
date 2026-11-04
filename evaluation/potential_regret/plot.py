#!/usr/bin/env python3
"""Plot spatial and per-race views of potential-oracle rollout regret."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track import Track, load_centerline_csv


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
AUTO_BIN_FINE_RESOLUTION_M = 0.25
AUTO_BIN_MIN_WIDTH_M = 0.5
AUTO_BIN_MAX_WIDTH_M = 5.0
REGRET_CMAP = LinearSegmentedColormap.from_list(
    "regret_green_zero_yellow_red",
    ("#18823b", "#f4e7a1", "#b2182b"),
)


def _scalar(data, key):
    return np.asarray(data[key]).item()


def load_race(path):
    with np.load(path, allow_pickle=False) as data:
        if "potential_regret_decisions_json" not in data.files:
            raise ValueError(f"{path}: trace has no potential-regret decisions")
        decisions = json.loads(str(_scalar(data, "potential_regret_decisions_json")))
        settings = json.loads(str(_scalar(data, "potential_regret_settings_json")))
        validation = json.loads(
            str(_scalar(data, "potential_regret_snapshot_validation_json"))
        )
        track_csv = str(_scalar(data, "track_csv"))
        track_scale = float(_scalar(data, "track_scale"))
        ego_position = int(_scalar(data, "ego_position"))
    usable = [row for row in decisions if row.get("status") == "ok"]
    return {
        "path": str(path),
        "decisions": usable,
        "all_decisions": decisions,
        "settings": settings,
        "snapshot_validation": validation,
        "track_csv": track_csv,
        "track_scale": track_scale,
        "ego_position": ego_position,
    }


def _race_bin_rows(race, edges):
    grouped = {idx: [] for idx in range(len(edges) - 1)}
    for row in race["decisions"]:
        idx = min(
            int(np.searchsorted(edges, row["lap_relative_s_m"], side="right") - 1),
            len(edges) - 2,
        )
        if idx >= 0:
            grouped[idx].append(row)
    result = {}
    for idx, rows in grouped.items():
        if not rows:
            continue
        regret = np.asarray([row["rollout_regret_m"] for row in rows], dtype=float)
        result[idx] = {
            "median_regret_m": float(np.median(regret)),
            "p90_regret_m": float(np.percentile(regret, 90)),
            "decision_count": len(rows),
            "interaction_rate": float(
                np.mean([row["context"]["interaction"] for row in rows])
            ),
            "negative_regret_rate": float(np.mean(regret < 0.0)),
            "collision_events": int(
                np.sum(
                    [
                        row["llm_branch"]["physical_collision_seconds"] > 0.0
                        for row in rows
                    ]
                )
            ),
            "rank_change_events": int(
                np.sum([row["llm_branch"]["rank_change"] != 0 for row in rows])
            ),
            "overtake_events": int(
                np.sum([row["llm_branch"]["rank_change"] > 0 for row in rows])
            ),
        }
    return result


def aggregate(races, track_length, bin_width):
    bin_count = max(1, int(np.ceil(track_length / bin_width)))
    edges = np.linspace(0.0, track_length, bin_count + 1)
    per_race = [_race_bin_rows(race, edges) for race in races]
    rows = []
    for idx in range(bin_count):
        contributions = [bins[idx] for bins in per_race if idx in bins]
        row = {
            "bin": idx,
            "s_start_m": float(edges[idx]),
            "s_end_m": float(edges[idx + 1]),
            "s_center_m": float(0.5 * (edges[idx] + edges[idx + 1])),
            "race_count": len(contributions),
            "coverage_percent": 100.0 * len(contributions) / len(races),
            "decision_count": int(
                sum(item["decision_count"] for item in contributions)
            ),
        }
        for name in (
            "median_regret_m",
            "p90_regret_m",
            "interaction_rate",
            "negative_regret_rate",
        ):
            row[name] = (
                float(np.median([item[name] for item in contributions]))
                if contributions
                else np.nan
            )
        for name in ("collision_events", "rank_change_events", "overtake_events"):
            row[name] = int(sum(item[name] for item in contributions))
        rows.append(row)
    return edges, rows


def automatic_bin_width(races, track_length):
    """Choose spatial resolution from distinct observed position support.

    Positions falling in the same 0.25 m cell count once, so deterministic
    repeats do not manufacture finer spatial evidence. The resulting target is
    approximately one distinct occupied fine cell per displayed bin, bounded
    to avoid either false precision or an unreadably coarse map.
    """
    positions = np.asarray(
        [
            float(row["lap_relative_s_m"]) % track_length
            for race in races
            for row in race["decisions"]
            if np.isfinite(row.get("lap_relative_s_m", np.nan))
        ],
        dtype=float,
    )
    if not positions.size:
        return min(AUTO_BIN_MAX_WIDTH_M, max(AUTO_BIN_MIN_WIDTH_M, track_length))
    fine_cell_count = max(1, int(np.ceil(track_length / AUTO_BIN_FINE_RESOLUTION_M)))
    occupied = np.unique(
        np.clip(
            np.floor(positions / track_length * fine_cell_count).astype(int),
            0,
            fine_cell_count - 1,
        )
    ).size
    raw_width = track_length / max(1, occupied)
    return float(np.clip(raw_width, AUTO_BIN_MIN_WIDTH_M, AUTO_BIN_MAX_WIDTH_M))


def _track_segments(track, sample_step=0.2):
    samples = np.arange(0.0, track.total_length + sample_step, sample_step)
    samples[-1] = track.total_length
    points = np.asarray([track.interpolate(float(s))[:2] for s in samples])
    segments = np.stack((points[:-1], points[1:]), axis=1)
    centers = 0.5 * (samples[:-1] + samples[1:])
    return segments, centers


def _draw_map(
    ax, track, edges, rows, field, title, min_races, cmap, label,
    fixed_limits=None, zero_centered=False,
):
    segments, centers = _track_segments(track)
    bin_ids = np.clip(np.searchsorted(edges, centers, side="right") - 1, 0, len(rows) - 1)
    values = np.asarray([rows[idx][field] for idx in bin_ids], dtype=float)
    enough = np.asarray([rows[idx]["race_count"] >= min_races for idx in bin_ids])
    finite = values[np.isfinite(values) & enough]
    ax.add_collection(
        LineCollection(segments, linewidths=6.0, colors="0.75", zorder=1)
    )
    shown_segments = segments[enough & np.isfinite(values)]
    shown_values = values[enough & np.isfinite(values)]
    collection = LineCollection(shown_segments, linewidths=6.0, cmap=cmap, zorder=2)
    if fixed_limits is not None:
        collection.set_clim(*fixed_limits)
    elif zero_centered:
        limit = (
            float(np.percentile(np.abs(finite), 95))
            if finite.size
            else 1.0
        )
        limit = max(limit, 1e-3)
        collection.set_norm(TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    elif finite.size:
        lo, hi = np.percentile(finite, [5, 95])
        if np.isclose(lo, hi):
            delta = max(abs(float(lo)) * 0.1, 1e-3)
            lo, hi = lo - delta, hi + delta
        collection.set_clim(float(lo), float(hi))
    collection.set_array(shown_values)
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar = ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label(label)


def plot_maps(path, track, edges, rows, min_races):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    _draw_map(
        axes[0, 0], track, edges, rows, "median_regret_m",
        "Median one-second rollout regret", min_races, REGRET_CMAP, "metres",
        zero_centered=True,
    )
    _draw_map(
        axes[0, 1], track, edges, rows, "p90_regret_m",
        "P90 one-second rollout regret", min_races, REGRET_CMAP, "metres",
        zero_centered=True,
    )
    _draw_map(
        axes[1, 0], track, edges, rows, "coverage_percent",
        "Independent race coverage", 1, "viridis", "% of input races",
        fixed_limits=(0.0, 100.0),
    )
    _draw_map(
        axes[1, 1], track, edges, rows, "interaction_rate",
        "Interaction rate", min_races, "magma", "fraction",
        fixed_limits=(0.0, 1.0),
    )
    fig.suptitle(
        f"Potential-oracle regret by track position (grey: < {min_races} races)",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_traces(path, races):
    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, constrained_layout=True
    )
    for index, race in enumerate(races):
        rows = race["decisions"]
        if not rows:
            continue
        time = np.asarray([row["race_time_s"] for row in rows])
        regret = np.asarray([row["rollout_regret_m"] for row in rows])
        label = f"race {index + 1}, start P{race['ego_position']}"
        axes[0].plot(time, regret, marker="o", markersize=3, label=label)
        for row in rows:
            if row["llm_branch"]["physical_collision_seconds"] > 0.0:
                axes[0].axvline(row["race_time_s"], color="black", alpha=0.15)
        delta = np.asarray([row["delta_theta"] for row in rows], dtype=float)
        for dim, name in enumerate(PARAMETER_NAMES):
            axes[1].plot(
                time,
                delta[:, dim],
                alpha=0.65,
                label=name if index == 0 else None,
            )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("rollout regret (m)")
    axes[0].set_title("Per-decision regret (vertical markers: LLM-branch collision)")
    axes[0].legend(loc="best")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("oracle - LLM parameter")
    axes[1].set_xlabel("race time (s)")
    axes[1].legend(ncol=5, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_outputs(output_dir, races, edges, rows, binning):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "potential_regret_spatial_bins.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_ok = [row for race in races for row in race["decisions"]]
    regret = np.asarray([row["rollout_regret_m"] for row in all_ok], dtype=float)
    predicted = np.asarray(
        [row["optimizer"]["predicted_difference"] for row in all_ok], dtype=float
    )
    if len(all_ok) >= 2:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(predicted, regret).statistic)
    else:
        spearman = None
    delta = (
        np.asarray([row["delta_theta"] for row in all_ok], dtype=float)
        if all_ok else np.empty((0, 5), dtype=float)
    )

    def branch_exposure(name):
        if not all_ok:
            return None
        branches = [row[name] for row in all_ok]
        return {
            "summed_physical_collision_seconds": float(
                sum(branch["physical_collision_seconds"] for branch in branches)
            ),
            "summed_ego_track_boundary_violation_seconds": float(
                sum(
                    branch["track_boundary_violation_seconds_by_car"][0]
                    for branch in branches
                )
            ),
            "summed_ego_recovery_active_seconds": float(
                sum(
                    branch["recovery_active_seconds_by_car"][0]
                    for branch in branches
                )
            ),
            "summed_ego_mpc_solver_failures": int(
                sum(branch["mpc_solver_failures_by_car"][0] for branch in branches)
            ),
            "note": "Summed one-second branch exposures overlap in race time.",
        }
    summary = {
        "result_class": "exploratory",
        "trace_paths": [race["path"] for race in races],
        "successful_decisions": len(all_ok),
        "failed_or_missing_decisions": int(
            sum(len(race["all_decisions"]) - len(race["decisions"]) for race in races)
        ),
        "snapshot_validation": [race["snapshot_validation"] for race in races],
        "median_rollout_regret_m": (
            float(np.median([row["rollout_regret_m"] for row in all_ok]))
            if all_ok else None
        ),
        "oracle_win_rate": (
            float(np.mean([row["rollout_regret_m"] > 0.0 for row in all_ok]))
            if all_ok else None
        ),
        "negative_regret_rate": (
            float(np.mean([row["rollout_regret_m"] < 0.0 for row in all_ok]))
            if all_ok else None
        ),
        "potential_rollout_sign_agreement": (
            float(np.mean(np.sign(predicted) == np.sign(regret)))
            if all_ok else None
        ),
        "potential_pairwise_ranking_accuracy": (
            float(np.mean(np.sign(predicted) == np.sign(regret)))
            if all_ok else None
        ),
        "potential_rollout_spearman": spearman,
        "optimizer_convergence_rate": (
            float(np.mean([row["optimizer"]["successful"] for row in all_ok]))
            if all_ok else None
        ),
        "optimizer_any_boundary_saturation_rate": (
            float(
                np.mean(
                    [any(row["optimizer"]["boundary_saturated"]) for row in all_ok]
                )
            )
            if all_ok else None
        ),
        "optimizer_boundary_saturation_rate_by_parameter": (
            dict(
                zip(
                    PARAMETER_NAMES,
                    np.mean(
                        np.asarray(
                            [
                                row["optimizer"]["boundary_saturated"]
                                for row in all_ok
                            ],
                            dtype=float,
                        ),
                        axis=0,
                    ).tolist(),
                )
            )
            if all_ok else None
        ),
        "delta_theta_mean": (
            dict(zip(PARAMETER_NAMES, np.mean(delta, axis=0).tolist()))
            if all_ok else None
        ),
        "delta_theta_mean_absolute": (
            dict(zip(PARAMETER_NAMES, np.mean(np.abs(delta), axis=0).tolist()))
            if all_ok else None
        ),
        "llm_branch_safety_exposure": branch_exposure("llm_branch"),
        "oracle_branch_safety_exposure": branch_exposure("oracle_branch"),
        "spatial_binning": binning,
        "bin_edges_m": edges.tolist(),
    }
    summary_path = output_dir / "potential_regret_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bin-width",
        default="auto",
        metavar="METRES|auto",
        help="spatial bin width; default auto uses distinct observed positions",
    )
    parser.add_argument("--min-races", type=int, default=1)
    args = parser.parse_args()
    if args.min_races < 1:
        parser.error("--min-races must be at least 1")

    races = [load_race(path) for path in args.traces]
    pairs = {(race["track_csv"], race["track_scale"]) for race in races}
    if len(pairs) != 1:
        raise ValueError("All traces in one track map must use the same track and scale")
    track_csv, track_scale = next(iter(pairs))
    track_path = Path(track_csv)
    if not track_path.is_absolute():
        track_path = ROOT / track_path
    track = Track(load_centerline_csv(track_path, scale=track_scale))
    if str(args.bin_width).strip().lower() == "auto":
        requested_bin_width = automatic_bin_width(races, track.total_length)
        binning_mode = "auto-distinct-position-support"
    else:
        try:
            requested_bin_width = float(args.bin_width)
        except ValueError:
            parser.error("--bin-width must be a positive number or 'auto'")
        if requested_bin_width <= 0.0:
            parser.error("--bin-width must be positive")
        binning_mode = "fixed"
    edges, rows = aggregate(races, track.total_length, requested_bin_width)
    actual_bin_width = float(edges[1] - edges[0])
    binning = {
        "mode": binning_mode,
        "requested_width_m": requested_bin_width,
        "actual_equal_width_m": actual_bin_width,
        "bin_count": len(rows),
        "auto_fine_resolution_m": (
            AUTO_BIN_FINE_RESOLUTION_M if binning_mode.startswith("auto") else None
        ),
        "auto_width_bounds_m": (
            [AUTO_BIN_MIN_WIDTH_M, AUTO_BIN_MAX_WIDTH_M]
            if binning_mode.startswith("auto") else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_path = args.output_dir / "potential_regret_track_maps.png"
    traces_path = args.output_dir / "potential_regret_time_traces.png"
    plot_maps(maps_path, track, edges, rows, args.min_races)
    plot_traces(traces_path, races)
    csv_path, summary_path = write_outputs(
        args.output_dir, races, edges, rows, binning
    )
    print(f"Saved {maps_path}")
    print(f"Saved {traces_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")
    print(
        f"Spatial bins: {len(rows)} x {actual_bin_width:.3f} m "
        f"({binning_mode})"
    )


if __name__ == "__main__":
    main()
