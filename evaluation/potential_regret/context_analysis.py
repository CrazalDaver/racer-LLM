#!/usr/bin/env python3
"""Relate exact online potential-regret gaps to action and race context.

The analysis is read-only and consumes traces produced by the opt-in online
potential-regret evaluator.  Because decisions within a race are dependent,
the primary grouped statistic is race-first: compute a median within each race
and context, then take the median across contributing races.  Raw decision
statistics are retained for transparency, but are not treated as independent
experimental replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
GEOMETRY_ORDER = ("straight", "corner_entry", "apex", "corner_exit")
DISTANCE_ORDER = ("close<=2.5m", "mid2.5-6m", "far>6m")
CORNER_CLOSE_ORDER = (
    "straight+not_close",
    "straight+close",
    "corner+not_close",
    "corner+close",
)


def _scalar(trace, key):
    return np.asarray(trace[key]).item()


def _model_label(trace, path: Path) -> str:
    ego = int(_scalar(trace, "ego_index"))
    names = np.asarray(trace.get("llm_model_names", []), dtype=str).reshape(-1)
    if names.size > ego and names[ego]:
        return str(names[ego])
    return path.parent.parent.name


def _distance_band(distance_m: float) -> str:
    if distance_m <= 2.5:
        return "close<=2.5m"
    if distance_m <= 6.0:
        return "mid2.5-6m"
    return "far>6m"


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            decisions = json.loads(
                str(np.asarray(trace["potential_regret_decisions_json"]).item())
            )
            model = _model_label(trace, path)
            ego_position = int(_scalar(trace, "ego_position"))
        race_id = str(path)
        for decision in decisions:
            if decision.get("status") != "ok":
                continue
            context = decision["context"]
            relative = float(context["nearest_opponent_relative_progress_m"])
            distance = abs(relative)
            geometry = str(context["track_geometry"])
            close = distance <= 2.5
            corner = geometry != "straight"
            theta = np.asarray(decision["llm_theta"], dtype=float)
            row = {
                "model": model,
                "race_id": race_id,
                "ego_position": ego_position,
                "decision_index": int(decision["decision_index"]),
                "race_time_s": float(decision["race_time_s"]),
                "lap_relative_s_m": float(decision["lap_relative_s_m"]),
                "rollout_regret_m": float(decision["rollout_regret_m"]),
                "track_geometry": geometry,
                "interaction": bool(context["interaction"]),
                "attack_defend": str(context["attack_defend"]),
                "nearest_opponent_relative_progress_m": relative,
                "nearest_opponent_distance_m": distance,
                "opponent_relation": "ahead" if relative > 0 else "behind",
                "distance_band": _distance_band(distance),
                "corner_close": (
                    ("corner" if corner else "straight")
                    + "+"
                    + ("close" if close else "not_close")
                ),
            }
            row.update(dict(zip(PARAMETER_NAMES, theta.tolist())))
            rows.append(row)
    return rows


def grouped_summary(rows: list[dict], dimension: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], str(row[dimension]))].append(row)
    output = []
    for (model, level), selected in sorted(groups.items()):
        gaps = np.asarray([row["rollout_regret_m"] for row in selected])
        by_race = defaultdict(list)
        for row in selected:
            by_race[row["race_id"]].append(row["rollout_regret_m"])
        race_medians = np.asarray(
            [np.median(values) for values in by_race.values()], dtype=float
        )
        output.append(
            {
                "model": model,
                "dimension": dimension,
                "level": level,
                "decision_count": len(selected),
                "race_count": len(by_race),
                "raw_mean_gap_m": float(np.mean(gaps)),
                "raw_median_gap_m": float(np.median(gaps)),
                "raw_p90_gap_m": float(np.percentile(gaps, 90)),
                "positive_gap_percent": float(100.0 * np.mean(gaps > 0.0)),
                "race_first_median_gap_m": float(np.median(race_medians)),
                "race_median_min_m": float(np.min(race_medians)),
                "race_median_max_m": float(np.max(race_medians)),
            }
        )
    return output


def action_summary(rows: list[dict]) -> tuple[list[dict], dict]:
    table = []
    summary = {}
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        keys = [
            tuple(round(float(row[name]), 6) for name in PARAMETER_NAMES)
            for row in selected
        ]
        counts = Counter(keys)
        dominant, dominant_count = counts.most_common(1)[0]
        dominant_gaps = np.asarray(
            [
                row["rollout_regret_m"]
                for row, key in zip(selected, keys)
                if key == dominant
            ],
            dtype=float,
        )
        other_gaps = np.asarray(
            [
                row["rollout_regret_m"]
                for row, key in zip(selected, keys)
                if key != dominant
            ],
            dtype=float,
        )
        summary[model] = {
            "decisions": len(selected),
            "unique_theta_6dp": len(counts),
            "dominant_theta": list(dominant),
            "dominant_share_percent": 100.0 * dominant_count / len(selected),
            "dominant_mean_gap_m": float(np.mean(dominant_gaps)),
            "dominant_median_gap_m": float(np.median(dominant_gaps)),
            "non_dominant_decisions": int(other_gaps.size),
            "non_dominant_mean_gap_m": (
                float(np.mean(other_gaps)) if other_gaps.size else None
            ),
            "non_dominant_median_gap_m": (
                float(np.median(other_gaps)) if other_gaps.size else None
            ),
        }
        for theta, count in counts.most_common():
            mask = np.asarray([key == theta for key in keys], dtype=bool)
            gaps = np.asarray(
                [row["rollout_regret_m"] for row in selected], dtype=float
            )[mask]
            table.append(
                {
                    "model": model,
                    **dict(zip(PARAMETER_NAMES, theta)),
                    "decision_count": int(count),
                    "share_percent": float(100.0 * count / len(selected)),
                    "mean_gap_m": float(np.mean(gaps)),
                    "median_gap_m": float(np.median(gaps)),
                }
            )
    return table, summary


def _lookup(summary_rows, model, dimension, level):
    for row in summary_rows:
        if (
            row["model"] == model
            and row["dimension"] == dimension
            and row["level"] == level
        ):
            return row["race_first_median_gap_m"]
    return np.nan


def plot_context(path: Path, rows: list[dict], summaries: list[dict], action_info: dict):
    models = sorted({row["model"] for row in rows})
    colors = dict(zip(models, plt.cm.Set2(np.linspace(0.1, 0.9, len(models)))))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    shares = [action_info[model]["dominant_share_percent"] for model in models]
    axes[0, 0].bar(models, shares, color=[colors[model] for model in models])
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].set_ylabel("dominant action share (%)")
    axes[0, 0].set_title("LLM action concentration")
    for idx, value in enumerate(shares):
        axes[0, 0].text(idx, value + 2, f"{value:.0f}%", ha="center")

    x = np.arange(len(GEOMETRY_ORDER))
    width = 0.8 / len(models)
    for model_idx, model in enumerate(models):
        values = [
            _lookup(summaries, model, "track_geometry", level)
            for level in GEOMETRY_ORDER
        ]
        axes[0, 1].bar(
            x + (model_idx - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model,
            color=colors[model],
        )
    axes[0, 1].set_xticks(x, [item.replace("corner_", "") for item in GEOMETRY_ORDER])
    axes[0, 1].set_ylabel("race-first median gap (m / 1 s)")
    axes[0, 1].set_title("Gap by track geometry")
    axes[0, 1].legend()

    x = np.arange(len(DISTANCE_ORDER))
    for model_idx, model in enumerate(models):
        values = [
            _lookup(summaries, model, "distance_band", level)
            for level in DISTANCE_ORDER
        ]
        axes[1, 0].bar(
            x + (model_idx - (len(models) - 1) / 2) * width,
            values,
            width,
            label=model,
            color=colors[model],
        )
    axes[1, 0].set_xticks(x, ["close", "mid", "far"])
    axes[1, 0].set_ylabel("race-first median gap (m / 1 s)")
    axes[1, 0].set_title("Gap by nearest-opponent distance")

    focus = next((model for model in models if "14b" in model.lower()), models[-1])
    values = [
        _lookup(summaries, focus, "corner_close", level)
        for level in CORNER_CLOSE_ORDER
    ]
    labels = ["straight\nnot close", "straight\nclose", "corner\nnot close", "corner\nclose"]
    axes[1, 1].bar(labels, values, color=colors[focus])
    axes[1, 1].set_ylabel("race-first median gap (m / 1 s)")
    axes[1, 1].set_title(f"{focus}: geometry × proximity")

    for ax in axes.flat:
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.grid(axis="y", color="0.9", linewidth=0.7)
    fig.suptitle("Potential-maximizer proposal gap context audit")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    missing = [str(path) for path in args.traces if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trace(s): " + ", ".join(missing))

    rows = load_rows(args.traces)
    if not rows:
        raise SystemExit("No successful potential-regret decisions found")
    dimensions = (
        "track_geometry",
        "interaction",
        "attack_defend",
        "distance_band",
        "opponent_relation",
        "corner_close",
    )
    context = [item for dim in dimensions for item in grouped_summary(rows, dim)]
    actions, action_info = action_summary(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "decision_context.csv", rows)
    _write_csv(args.output_dir / "context_summary.csv", context)
    _write_csv(args.output_dir / "parameter_concentration.csv", actions)
    plot_context(
        args.output_dir / "context_gap_analysis.png", rows, context, action_info
    )
    payload = {
        "result_class": "exploratory",
        "claim_limit": (
            "Three deterministic races per model; decisions within a race are "
            "dependent. Grouped primary values are race-first medians and do not "
            "establish causality. The proposal maximizes a learned potential, not "
            "true rollout return."
        ),
        "trace_paths": [str(path) for path in args.traces],
        "action_concentration": action_info,
        "context_summary": context,
    }
    (args.output_dir / "context_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved context audit to {args.output_dir}")


if __name__ == "__main__":
    main()
