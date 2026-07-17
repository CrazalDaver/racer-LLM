"""Generate a self-contained HTML report from simulation NPZ traces.

Each trace occupies one row: the diagnostic figure from ``plotting.py`` is on
the left and the animation from ``animate_trace.py`` is on the right. Images
are embedded into the HTML, so the report can be moved or shared as one file.

Either edit ``NPZ_FILES`` and run this file without arguments, or pass paths:

    .venv/bin/python generate_html_report.py LLM_alpha_1.npz LLM_alpha_2.npz
    .venv/bin/python generate_html_report.py result/*.npz -o report.html
"""

from __future__ import annotations

import argparse
import base64
import html
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


# Option 1: put the desired traces here, then run the script without arguments.
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


def generate_report(
    npz_paths,
    output="simulation_report.html",
    animation_stride=2,
    animation_interval=30,
    animation_dpi=80,
    plot_dpi=110,
):
    paths = [Path(path).expanduser().resolve() for path in npz_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("NPZ file(s) not found: " + ", ".join(missing))
    if not paths:
        raise ValueError("No NPZ files were provided.")

    rows = []
    with tempfile.TemporaryDirectory(prefix="racer_html_report_") as temp_dir:
        temp_dir = Path(temp_dir)
        for index, trace_path in enumerate(paths, start=1):
            print(f"[{index}/{len(paths)}] Plotting {trace_path.name} ...")
            plot_path = temp_dir / f"plot_{index}.png"
            # The existing plotting helper is interactive; suppress only its show call.
            with mock.patch.object(plt, "show"):
                figure = plot_trace_npz(trace_path)
            figure.savefig(plot_path, dpi=plot_dpi, bbox_inches="tight")
            plt.close(figure)

            print(f"[{index}/{len(paths)}] Animating {trace_path.name} ...")
            gif_path = temp_dir / f"animation_{index}.gif"
            make_animation(
                trace_path,
                stride=animation_stride,
                interval=animation_interval,
                save_path=gif_path,
                dpi=animation_dpi,
            )

            rows.append(
                {
                    "name": html.escape(trace_path.name),
                    "path": html.escape(str(trace_path)),
                    "description": _trace_description(trace_path),
                    "plot": _data_uri(plot_path, "image/png"),
                    "gif": _data_uri(gif_path, "image/gif"),
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
              <figcaption>Simulation diagnostics</figcaption>
            </figure>
            <figure class="animation">
              <img src="{row['gif']}" alt="Animation for {row['name']}">
              <figcaption>Simulation animation</figcaption>
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
        description="Create a self-contained HTML report with a plot and GIF for each NPZ."
    )
    parser.add_argument(
        "npz",
        nargs="*",
        help="NPZ paths. If omitted, the NPZ_FILES list in this script is used.",
    )
    parser.add_argument("-o", "--output", default="simulation_report.html")
    parser.add_argument("--stride", type=int, default=2, help="Animation frame stride.")
    parser.add_argument("--interval", type=int, default=30, help="GIF frame interval in ms.")
    parser.add_argument("--animation-dpi", type=int, default=80)
    parser.add_argument("--plot-dpi", type=int, default=110)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_paths = args.npz or NPZ_FILES
    if not selected_paths:
        raise SystemExit("Provide NPZ paths on the command line or add them to NPZ_FILES.")
    generate_report(
        selected_paths,
        output=args.output,
        animation_stride=args.stride,
        animation_interval=args.interval,
        animation_dpi=args.animation_dpi,
        plot_dpi=args.plot_dpi,
    )
