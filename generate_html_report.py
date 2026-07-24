"""Generate a self-contained HTML report from simulation NPZ/JSON results.

NPZ rows contain the diagnostic figure from ``plotting.py`` on the left and
the animation from ``animate_trace.py`` on the right. Position-sweep JSON rows
contain aggregate plots on the left and a trial table on the right. Images are
embedded into the HTML, so the report can be moved or shared as one file.

Either edit ``NPZ_FILES`` and run this file without arguments, or pass paths:

    .venv/bin/python generate_html_report.py LLM_alpha_1.npz LLM_alpha_2.npz
    .venv/bin/python generate_html_report.py result/no_safeguard_LLM_alpha.json
    .venv/bin/python generate_html_report.py trace.npz sweep.json -o report.html
    .venv/bin/python generate_html_report.py result/*.npz -o report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import tempfile
from pathlib import Path
from unittest import mock

import matplotlib

# Must be selected before importing pyplot through plotting/animation modules.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from animate_trace import make_animation
from plotting import plot_trace_npz


# Option 1: put any mixture of .npz and position-sweep .json files here.
NPZ_FILES = [
    # "LLM_alpha_1.npz",
    # "LLM_alpha_2.npz",
    # "LLM_alpha_3.npz",
]


def _data_uri(path: Path, media_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _scalar(data, key, default=""):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.tolist()


def _trace_description(trace_path: Path) -> str:
    with np.load(trace_path, allow_pickle=False) as data:
        ego_position = _scalar(data, "ego_position", "?")
        policies = _scalar(data, "policy_names", [])
        duration = float(data["time"][-1] - data["time"][0]) if len(data["time"]) else 0.0
    policies_text = ", ".join(map(str, policies)) if policies else "unknown"
    return (
        f"ego start position: {ego_position} &nbsp;·&nbsp; "
        f"duration: {duration:.2f} s &nbsp;·&nbsp; policies: {html.escape(policies_text)}"
    )


def _load_sweep_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    required = {"policy_names", "trials", "winner_counts"}
    missing = sorted(required.difference(summary))
    if missing:
        raise ValueError(
            f"Unsupported JSON structure in {json_path}; missing: {', '.join(missing)}"
        )
    if not isinstance(summary["trials"], list):
        raise ValueError(f"Invalid trials field in {json_path}; expected a list.")
    return summary


def _plot_sweep_summary(summary):
    policies = [str(value) for value in summary["policy_names"]]
    roles = list(summary["winner_counts"])
    positions = [int(value) for value in summary.get("ego_positions", [])]
    if not positions:
        positions = sorted({int(trial["ego_position"]) for trial in summary["trials"]})

    fig, (ax_wins, ax_progress) = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    counts_by_position = summary.get("winner_counts_by_position", {})
    bottom = np.zeros(len(positions), dtype=float)
    colors = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown"]
    for role_idx, role in enumerate(roles):
        counts = np.asarray(
            [counts_by_position.get(str(position), {}).get(role, 0) for position in positions],
            dtype=float,
        )
        policy = policies[role_idx] if role_idx < len(policies) else "?"
        ax_wins.bar(
            positions,
            counts,
            bottom=bottom,
            color=colors[role_idx % len(colors)],
            label=f"{role} ({policy})",
        )
        bottom += counts
    ax_wins.set_title("Winner counts by ego starting position")
    ax_wins.set_xlabel("Ego starting position")
    ax_wins.set_ylabel("Wins")
    ax_wins.set_xticks(positions)
    ax_wins.grid(axis="y", alpha=0.25)
    ax_wins.legend()

    car_count = len(policies)
    offsets = np.linspace(-0.24, 0.24, car_count) if car_count > 1 else np.zeros(1)
    width = min(0.7 / max(car_count, 1), 0.24)
    for car_idx in range(car_count):
        means = []
        errors = []
        for position in positions:
            values = [
                float(trial["final_progress"][car_idx])
                for trial in summary["trials"]
                if int(trial["ego_position"]) == position
                and len(trial.get("final_progress", [])) > car_idx
            ]
            means.append(float(np.mean(values)) if values else np.nan)
            errors.append(float(np.std(values)) if len(values) > 1 else 0.0)
        role = roles[car_idx] if car_idx < len(roles) else f"car{car_idx}"
        ax_progress.bar(
            np.asarray(positions) + offsets[car_idx],
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=colors[car_idx % len(colors)],
            label=f"{role} ({policies[car_idx]})",
        )
    ax_progress.set_title("Mean final progress by ego starting position")
    ax_progress.set_xlabel("Ego starting position")
    ax_progress.set_ylabel("Final progress (m)")
    ax_progress.set_xticks(positions)
    ax_progress.grid(axis="y", alpha=0.25)
    ax_progress.legend()
    return fig


def _sweep_description(summary) -> str:
    policies = ", ".join(html.escape(str(value)) for value in summary["policy_names"])
    trials = len(summary["trials"])
    repeats = summary.get("trials_per_position", "?")
    return f"{trials} trials &nbsp;·&nbsp; {repeats} per position &nbsp;·&nbsp; policies: {policies}"


def _sweep_table(summary) -> str:
    policies = [str(value) for value in summary["policy_names"]]
    winner_counts = summary["winner_counts"]
    count_text = " &nbsp;·&nbsp; ".join(
        f"<strong>{html.escape(str(role))}</strong>: {int(count)}"
        for role, count in winner_counts.items()
    )
    body = []
    for trial in summary["trials"]:
        progress = ", ".join(f"{float(value):.2f}" for value in trial.get("final_progress", []))
        body.append(
            "<tr>"
            f"<td>{int(trial.get('trial', len(body) + 1))}</td>"
            f"<td>P{int(trial['ego_position'])}</td>"
            f"<td>{int(trial.get('repeat', 1))}</td>"
            f"<td>{html.escape(str(trial.get('winner', '?')))}</td>"
            f"<td>{html.escape(str(trial.get('winner_policy', '?')))}</td>"
            f"<td>{html.escape(progress)}</td>"
            "</tr>"
        )
    policies_text = ", ".join(html.escape(value) for value in policies)
    return f"""
      <div class="summary-box">
        <h3>Winner totals</h3>
        <p>{count_text}</p>
        <p class="policies">Progress order: {policies_text}</p>
      </div>
      <div class="table-scroll">
        <table>
          <thead><tr><th>Trial</th><th>Ego start</th><th>Repeat</th><th>Winner</th><th>Policy</th><th>Final progress (m)</th></tr></thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>"""


def _replace_none_with_nan(value):
    """Restore non-finite numeric NPZ values serialized as JSON null."""
    if value is None:
        return np.nan
    if isinstance(value, list):
        return [_replace_none_with_nan(item) for item in value]
    return value


def _write_json_trace(npz_contents, output_path):
    """Rebuild a temporary NPZ from a trial's embedded JSON trace."""
    arrays = {
        str(name): np.asarray(_replace_none_with_nan(value))
        for name, value in npz_contents.items()
    }
    np.savez(output_path, **arrays)


def _trial_from_npz(trace_path, trial_number, repeat=1):
    with np.load(trace_path, allow_pickle=False) as data:
        policies = [str(value) for value in np.asarray(data["policy_names"]).tolist()]
        final_progress = np.asarray(data["car_px"], dtype=float)[-1]
        winner_index = int(np.nanargmax(final_progress))
        ego_position = int(_scalar(data, "ego_position", 1))
        start_ranks = _scalar(data, "start_ranks", list(range(len(final_progress))))
    roles = ["ego", *[f"oppo{idx}" for idx in range(1, len(final_progress))]]
    return {
        "trial": int(trial_number),
        "ego_position": ego_position,
        "repeat": int(repeat),
        "winner": roles[winner_index],
        "winner_index": winner_index,
        "winner_policy": policies[winner_index],
        "final_progress": final_progress.tolist(),
        "start_ranks": start_ranks,
    }, policies


def _combined_summary(trials, policies):
    roles = ["ego", *[f"oppo{idx}" for idx in range(1, len(policies))]]
    positions = sorted({int(trial["ego_position"]) for trial in trials})
    winner_counts = {role: 0 for role in roles}
    counts_by_position = {
        str(position): {role: 0 for role in roles} for position in positions
    }
    trials_by_position = {position: 0 for position in positions}
    for trial in trials:
        role = str(trial["winner"])
        position = int(trial["ego_position"])
        if role not in winner_counts:
            winner_counts[role] = 0
            for counts in counts_by_position.values():
                counts[role] = 0
        winner_counts[role] += 1
        counts_by_position[str(position)][role] += 1
        trials_by_position[position] += 1
    repeat_counts = set(trials_by_position.values())
    trials_per_position = repeat_counts.pop() if len(repeat_counts) == 1 else "varies"
    return {
        "policy_names": list(policies),
        "trials_per_position": trials_per_position,
        "ego_positions": positions,
        "winner_counts": winner_counts,
        "winner_counts_by_position": counts_by_position,
        "trials": trials,
    }


def generate_report(
    result_paths,
    output="simulation_report.html",
    animation_stride=2,
    animation_interval=30,
    animation_dpi=80,
    plot_dpi=110,
):
    paths = [Path(path).expanduser().resolve() for path in result_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Result file(s) not found: " + ", ".join(missing))
    if not paths:
        raise ValueError("No NPZ or JSON files were provided.")

    rows = []
    with tempfile.TemporaryDirectory(prefix="racer_html_report_") as temp_dir:
        temp_dir = Path(temp_dir)
        trace_jobs = []
        aggregate_trials = []
        aggregate_policies = None
        report_trial_number = 0

        # Expand embedded JSON traces first. Older summary-only JSON files still
        # contribute to the final statistics but cannot produce animations.
        for source_index, source_path in enumerate(paths, start=1):
            suffix = source_path.suffix.lower()
            if suffix == ".npz":
                report_trial_number += 1
                trial, policies = _trial_from_npz(source_path, report_trial_number)
                aggregate_trials.append(trial)
                aggregate_policies = aggregate_policies or policies
                trace_jobs.append((source_path, source_path.name, str(source_path)))
            elif suffix == ".json":
                summary = _load_sweep_json(source_path)
                policies = [str(value) for value in summary["policy_names"]]
                aggregate_policies = aggregate_policies or policies
                for source_trial in summary["trials"]:
                    report_trial_number += 1
                    trial = {key: value for key, value in source_trial.items() if key != "npz"}
                    trial["trial"] = report_trial_number
                    aggregate_trials.append(trial)
                    embedded_trace = source_trial.get("npz")
                    if embedded_trace:
                        temp_trace = temp_dir / (
                            f"source_{source_index}_trial_{report_trial_number}.npz"
                        )
                        _write_json_trace(embedded_trace, temp_trace)
                        label = (
                            f"{source_path.name} — trial "
                            f"{source_trial.get('trial', report_trial_number)} "
                            f"(ego P{source_trial.get('ego_position', '?')}, "
                            f"repeat {source_trial.get('repeat', 1)})"
                        )
                        trace_jobs.append((temp_trace, label, str(source_path)))
            else:
                raise ValueError(
                    f"Unsupported result type: {source_path} (use .npz or .json)"
                )

        for index, (trace_path, display_name, source_label) in enumerate(
            trace_jobs, start=1
        ):
            print(f"[{index}/{len(trace_jobs)}] Plotting {display_name} ...")
            plot_path = temp_dir / f"plot_{index}.png"
            with mock.patch.object(plt, "show"):
                figure = plot_trace_npz(trace_path)
            figure.savefig(plot_path, dpi=plot_dpi, bbox_inches="tight")
            plt.close(figure)

            print(f"[{index}/{len(trace_jobs)}] Animating {display_name} ...")
            gif_path = temp_dir / f"animation_{index}.gif"
            make_animation(
                trace_path,
                stride=animation_stride,
                interval=animation_interval,
                save_path=gif_path,
                dpi=animation_dpi,
            )
            escaped_name = html.escape(display_name)
            rows.append(
                {
                    "name": escaped_name,
                    "path": html.escape(source_label),
                    "description": _trace_description(trace_path),
                    "plot": _data_uri(plot_path, "image/png"),
                    "plot_caption": "Simulation diagnostics",
                    "right_content": (
                        f'<img src="{_data_uri(gif_path, "image/gif")}" '
                        f'alt="Animation for {escaped_name}">'
                        "<figcaption>Simulation animation</figcaption>"
                    ),
                }
            )

        if aggregate_trials:
            combined = _combined_summary(aggregate_trials, aggregate_policies)
            summary_path = temp_dir / "overall_summary.png"
            summary_figure = _plot_sweep_summary(combined)
            summary_figure.savefig(summary_path, dpi=plot_dpi, bbox_inches="tight")
            plt.close(summary_figure)
            rows.append(
                {
                    "name": "Overall statistics",
                    "path": html.escape(", ".join(str(path) for path in paths)),
                    "description": _sweep_description(combined),
                    "plot": _data_uri(summary_path, "image/png"),
                    "plot_caption": "Aggregate results across all selected runs",
                    "right_content": _sweep_table(combined),
                }
            )

    row_html = "\n".join(
        f"""
        <section class="experiment">
          <header>
            <h2>{row['name']}</h2>
            <div class="description">{row['description']}</div>
            <div class="path">{row['path']}</div>
          </header>
          <div class="media">
            <figure class="plot">
              <img src="{row['plot']}" alt="Diagnostic plots for {row['name']}">
              <figcaption>{row['plot_caption']}</figcaption>
            </figure>
            <figure class="right-panel">
              {row['right_content']}
            </figure>
          </div>
        </section>"""
        for row in rows
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Racer simulation report</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
    body {{ margin: 0; padding: 28px; color: #202124; background: #f3f5f7; }}
    main {{ max-width: 1800px; margin: auto; }}
    h1 {{ margin: 0 0 24px; font-size: 28px; }}
    .experiment {{ margin: 0 0 30px; padding: 20px; background: white;
                   border: 1px solid #dfe3e8; border-radius: 12px;
                   box-shadow: 0 2px 8px rgba(0,0,0,.05); }}
    header {{ margin-bottom: 14px; }}
    h2 {{ margin: 0 0 7px; font-size: 21px; }}
    .description {{ color: #3c4043; font-size: 14px; }}
    .path {{ margin-top: 5px; color: #777; font: 12px ui-monospace, monospace;
             overflow-wrap: anywhere; }}
    .media {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(300px, 2fr);
              gap: 18px; align-items: start; }}
    figure {{ margin: 0; min-width: 0; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #e5e7eb;
           border-radius: 6px; background: white; }}
    figcaption {{ margin-top: 7px; text-align: center; color: #666; font-size: 13px; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    .summary-box {{ padding: 14px; margin-bottom: 12px; background: #f7f8fa;
                    border: 1px solid #e1e4e8; border-radius: 6px; }}
    .summary-box p {{ margin: 6px 0; }}
    .summary-box .policies {{ color: #666; font-size: 13px; }}
    .table-scroll {{ overflow-x: auto; max-height: 720px; overflow-y: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 7px; border-bottom: 1px solid #e5e7eb; text-align: left;
              white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #f2f4f6; }}
    @media (max-width: 900px) {{
      body {{ padding: 12px; }}
      .media {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      body {{ background: white; padding: 0; }}
      .experiment {{ box-shadow: none; break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Racer simulation report</h1>
    {row_html}
  </main>
</body>
</html>
"""

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    print(f"Saved self-contained report to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a self-contained HTML report from NPZ traces and sweep JSON files."
    )
    parser.add_argument(
        "results",
        nargs="*",
        help="NPZ/JSON paths supplied as positional arguments.",
    )
    parser.add_argument(
        "--results",
        dest="result_options",
        nargs="+",
        metavar="PATH",
        help="NPZ/JSON path list (alternative to positional paths).",
    )
    
    parser.add_argument("-o", "--output", default="simulation_report.html")
    parser.add_argument("--stride", type=int, default=2, help="Animation frame stride.")
    parser.add_argument("--interval", type=int, default=30, help="GIF frame interval in ms.")
    parser.add_argument("--animation-dpi", type=int, default=80)
    parser.add_argument("--plot-dpi", type=int, default=110)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_paths = args.result_options or args.results or NPZ_FILES
    if not selected_paths:
        raise SystemExit("Provide NPZ/JSON paths on the command line or add them to NPZ_FILES.")
    generate_report(
        selected_paths,
        output=args.output,
        animation_stride=args.stride,
        animation_interval=args.interval,
        animation_dpi=args.animation_dpi,
        plot_dpi=args.plot_dpi,
    )
