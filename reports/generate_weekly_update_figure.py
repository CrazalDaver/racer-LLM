"""Generate the compact August 25 progress/safety figure from saved sweeps."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("Alpha\nself-play", "velocity_w10_alpha_selfplay"),
    ("Pure\nrules", "velocity_w10_rule_vs_alpha"),
    ("Raw\nLLM", "velocity_w10_raw_llm_vs_alpha"),
    ("Rule-prompt\nLLM", "velocity_w10_rule_prompt_llm_vs_alpha"),
    ("Parameter-\nIBR", "velocity_w10_parameter_ibr_vs_alpha"),
]


def load_case(directory):
    path = ROOT / "result" / directory
    summary = json.loads((path / "summary.json").read_text())
    progress = [trial["final_progress"][0] for trial in summary["trials"]]
    maximum_lateral_error = []
    for trial_index in range(1, len(summary["trials"]) + 1):
        trace = np.load(
            path / "summary_traces" / f"trial_{trial_index:03d}.npz",
            allow_pickle=True,
        )
        maximum_lateral_error.append(float(np.max(np.abs(trace["car_py"][:, 0]))))
    return float(np.mean(progress)), float(np.max(maximum_lateral_error))


labels = [label for label, _ in CASES]
values = [load_case(directory) for _, directory in CASES]
progress = [value[0] for value in values]
maximum_lateral_error = [value[1] for value in values]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
bars = axes[0].bar(x, progress, color="#4878a8")
axes[0].set_ylabel("Mean ego progress (m)")
axes[0].set_ylim(0, 120)
axes[0].set_xticks(x, labels)
axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

bars = axes[1].bar(x, maximum_lateral_error, color="#6b9f78")
axes[1].axhline(0.44, color="#cc7a29", linestyle="--", linewidth=1, label="nominal boundary")
axes[1].axhline(0.55, color="#b04444", linestyle=":", linewidth=1.2, label="recovery threshold")
axes[1].set_ylabel(r"Maximum ego $|e|$ (m)")
axes[1].set_ylim(0, 0.65)
axes[1].set_xticks(x, labels)
axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
axes[1].legend(frameon=False, fontsize=7, loc="upper left")

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="y", alpha=0.2)

fig.tight_layout()
output = ROOT / "figures" / "august25_shared_mpc_comparison.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, bbox_inches="tight")
print(output)
