"""Plot the factual-grounding result from an LLM scene-understanding audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/racer_llm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot supported/ambiguous/contradicted traffic-basis counts."
    )
    parser.add_argument("summary", type=Path, help="Scene-audit summary.json")
    parser.add_argument("output", type=Path, help="Output PDF or PNG")
    parser.add_argument(
        "--variant", default="evidence_first_minimal_v2", help="Variant key"
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    counts = summary["variants"][args.variant]["grounding_counts"]
    supported = counts.get("supported", 0)
    ambiguous = counts.get("ambiguous", 0)
    contradicted = counts.get("contradicted", 0)

    rows = [
        ("All decisions", [supported, ambiguous, contradicted]),
        ("Outside deadband", [supported, 0, contradicted]),
    ]
    colors = ["#2E7D6E", "#D6A33A", "#B84A4A"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    for y, (label, values) in enumerate(reversed(rows)):
        total = sum(values)
        left = 0.0
        for value, color in zip(values, colors):
            if not value:
                continue
            percentage = 100.0 * value / total
            ax.barh(
                y,
                percentage,
                left=left,
                height=0.48,
                color=color,
                edgecolor="white",
                linewidth=1,
            )
            if percentage >= 7:
                ax.text(
                    left + percentage / 2,
                    y,
                    f"{value} ({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
            else:
                ax.text(
                    left + percentage + 1,
                    y,
                    f"{value} ({percentage:.1f}%)",
                    ha="left",
                    va="center",
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                )
            left += percentage

    decisive = supported + contradicted
    total = decisive + ambiguous
    ax.set_yticks(
        [0, 1],
        [f"Outside deadband\n(n={decisive})", f"All decisions\n(n={total})"],
    )
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of audited traffic-basis labels")
    ax.set_title("Traffic-basis factual grounding", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(
        handles,
        ["Supported", "Ambiguous", "Contradicted"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.28),
        frameon=False,
    )
    fig.text(
        0.12,
        0.005,
        "Supported = exact code-derived match; ambiguous = |closing speed| <= "
        "0.1 m/s; contradicted = decisive mismatch.",
        fontsize=7.8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
