#!/usr/bin/env python3
"""Audit raw versus evidence-first LLM decisions in saved NPZ traces.

This checks factual grounding of the structured basis, not mode accuracy.
Small deadbands mark borderline measurements ambiguous and never prescribe an
action.

Usage:
  .venv/bin/python evaluation/llm_scene_understanding_audit.py \
    result/llm_basis_ab_berlin_raw result/llm_basis_ab_berlin_evidence_first \
    --output-dir result/llm_basis_ab_berlin/audit
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/racer_llm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = ("stable", "push", "overtake", "block")
STRUCTURED_VARIANTS = {
    "structured_action_first_v1",
    "evidence_first_v1",
    "structured_action_first_minimal_v2",
    "evidence_first_minimal_v2",
}


def discover_npz(inputs):
    found = set()
    for item in inputs:
        if item.is_file() and item.suffix == ".npz":
            found.add(item.resolve())
        elif item.is_dir():
            found.update(path.resolve() for path in item.rglob("*.npz"))
    return sorted(found)


def parse_object(text):
    try:
        value = json.loads(str(text))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", str(text), flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def sign_state(value, deadband):
    if value > deadband:
        return "positive"
    if value < -deadband:
        return "negative"
    return "ambiguous"


def verify_basis(label, observation):
    """Return status and the observation used to check one basis label."""
    traffic = observation.get("traffic", {})
    front, rear = traffic.get("front"), traffic.get("rear")
    if label == "no_front_vehicle":
        return ("supported" if front is None else "contradicted", f"traffic.front={'null' if front is None else 'present'}")
    if label == "no_rear_vehicle":
        return ("supported" if rear is None else "contradicted", f"traffic.rear={'null' if rear is None else 'present'}")

    gap_claims = {
        "closing_on_front_vehicle": (front, "positive", "front"),
        "front_vehicle_pulling_away": (front, "negative", "front"),
        "rear_vehicle_closing": (rear, "positive", "rear"),
        "rear_vehicle_falling_behind": (rear, "negative", "rear"),
    }
    if label in gap_claims:
        vehicle, expected, subject = gap_claims[label]
        if vehicle is None:
            return "contradicted", f"traffic.{subject}=null"
        value = float(vehicle.get("closing_speed_meters_per_second", 0.0))
        observed = sign_state(value, 0.1)
        status = "ambiguous" if observed == "ambiguous" else "supported" if observed == expected else "contradicted"
        return status, f"{subject}.closing_speed={value:.2f} m/s"

    constant_gap_claims = {
        "front_gap_approximately_constant": (front, "front"),
        "rear_gap_approximately_constant": (rear, "rear"),
    }
    if label in constant_gap_claims:
        vehicle, subject = constant_gap_claims[label]
        if vehicle is None:
            return "contradicted", f"traffic.{subject}=null"
        value = float(vehicle.get("closing_speed_meters_per_second", 0.0))
        observed = sign_state(value, 0.1)
        return ("supported" if observed == "ambiguous" else "contradicted",
                f"{subject}.closing_speed={value:.2f} m/s")

    lateral = float(observation.get("ego", {}).get("left_position_percent_of_half_width", 0.0))
    lateral_state = sign_state(lateral, 5.0)
    lateral_claims = {
        "ego_left_of_center": "positive",
        "ego_right_of_center": "negative",
        "ego_near_center": "ambiguous",
    }
    if label in lateral_claims:
        expected = lateral_claims[label]
        status = "supported" if lateral_state == expected else "ambiguous" if lateral_state == "ambiguous" else "contradicted"
        return status, f"ego.lateral={lateral:.1f}% half-width"

    if label.startswith("road_"):
        samples = observation.get("road_preview", {}).get("samples_near_to_far", [])
        turns = [float(sample["centerline_left_turn_degrees_from_current"])
                 for sample in samples
                 if sample.get("centerline_left_turn_degrees_from_current") is not None]
        left = any(value > 1.0 for value in turns)
        right = any(value < -1.0 for value in turns)
        observed = ("road_changes_direction" if left and right else
                    "road_turns_left" if left else
                    "road_turns_right" if right else "road_direction_unclear")
        return ("supported" if label == observed else "contradicted",
                f"preview_turns={','.join(f'{value:.1f}' for value in turns)} deg")
    return "unknown", "unrecognized basis"


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit_trace(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"policy_names", "car_llm_prompt_variants", "llm_raw_call_indices",
                    "llm_raw_prompts", "llm_raw_responses", "car_px"}
        missing = sorted(required - set(data.files))
        if missing:
            return [], None, f"missing {', '.join(missing)}"
        names = np.asarray(data["policy_names"], dtype=str)
        variants = np.asarray(data["car_llm_prompt_variants"], dtype=str)
        candidates = [i for i, name in enumerate(names)
                      if name in {
                          "llm_alpharacer",
                          "llm_structured_action_first",
                          "llm_evidence_first",
                      }]
        if not candidates:
            return [], None, "no raw/evidence-first LLM car"
        car = candidates[0]
        variant = str(variants[car])
        calls = np.asarray(data["llm_raw_call_indices"], dtype=int)[car]
        prompts = np.asarray(data["llm_raw_prompts"], dtype=str)[car]
        responses = np.asarray(data["llm_raw_responses"], dtype=str)[car]
        errors = np.asarray(data.get("llm_raw_errors", np.full_like(responses, "")), dtype=str)[car]
        all_latency = np.asarray(data.get("llm_raw_latency_seconds", np.full((len(names), len(calls)), np.nan)), dtype=float)
        latencies = all_latency[car]
        decisions = []
        for slot, call in enumerate(calls):
            if call <= 0 or errors[slot]:
                continue
            observation, response = parse_object(prompts[slot]), parse_object(responses[slot])
            if observation is None or response is None:
                continue
            mode, basis = str(response.get("mode", "")).lower(), str(response.get("basis", ""))
            if mode not in MODES:
                continue
            grounding, evidence = (verify_basis(basis, observation) if variant in STRUCTURED_VARIANTS
                                   else ("not_applicable", "raw free-text basis"))
            decisions.append({
                "trace": relative(path), "variant": variant, "call": int(call),
                "basis": basis, "mode": mode, "reason": str(response.get("reason", "")),
                "grounding": grounding, "grounding_evidence": evidence,
                "latency_seconds": float(latencies[slot]) if np.isfinite(latencies[slot]) else None,
            })

        progress = np.asarray(data["car_px"], dtype=float)
        speeds = np.asarray(data.get("car_vx", np.zeros_like(progress)), dtype=float)
        boundary = np.asarray(data.get("car_track_boundary_violation", np.zeros_like(progress, dtype=bool)), dtype=bool)
        offroad = np.asarray(data.get("car_offroad", np.zeros_like(progress, dtype=bool)), dtype=bool)
        final = progress[-1]
        trace = {
            "trace": relative(path), "variant": variant,
            "final_progress": float(final[car]), "won": int(np.nanargmax(final)) == car,
            "mean_speed": float(np.mean(speeds[:, car])),
            "boundary_fraction": float(np.mean(boundary[:, car])),
            "offroad_fraction": float(np.mean(offroad[:, car])), "decision_count": len(decisions),
        }
        return decisions, trace, None


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize(decisions, traces, excluded):
    summary = {"scope_note": "Basis grounding is factual consistency, not mode accuracy or optimality.",
               "measurement_tolerances": {"closing_speed_mps": 0.1, "lateral_percent": 5.0, "road_turn_degrees": 1.0},
               "excluded": excluded, "variants": {}}
    for variant in sorted({row["variant"] for row in traces}):
        ds = [row for row in decisions if row["variant"] == variant]
        ts = [row for row in traces if row["variant"] == variant]
        decisive = [row for row in ds if row["grounding"] in {"supported", "contradicted"}]
        summary["variants"][variant] = {
            "traces": len(ts), "decisions": len(ds),
            "mode_counts": dict(Counter(row["mode"] for row in ds)),
            "basis_counts": dict(Counter(row["basis"] for row in ds)),
            "grounding_counts": dict(Counter(row["grounding"] for row in ds)),
            "supported_fraction_excluding_ambiguous": mean([row["grounding"] == "supported" for row in decisive]),
            "mean_latency_seconds": mean([row["latency_seconds"] for row in ds]),
            "wins": int(sum(row["won"] for row in ts)),
            "mean_final_progress": mean([row["final_progress"] for row in ts]),
            "mean_speed": mean([row["mean_speed"] for row in ts]),
            "mean_boundary_fraction": mean([row["boundary_fraction"] for row in ts]),
            "mean_offroad_fraction": mean([row["offroad_fraction"] for row in ts]),
        }
    return summary


def plot_summary(summary, output):
    variants = list(summary["variants"])
    display_names = {
        "raw": "raw",
        "structured_action_first_v1": "action-first",
        "evidence_first_v1": "evidence-first",
        "structured_action_first_minimal_v2": "action-first minimal",
        "evidence_first_minimal_v2": "evidence-first minimal",
    }
    display = [display_names.get(name, name) for name in variants]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x, bottom = np.arange(len(variants)), np.zeros(len(variants))
    for mode in MODES:
        values = [summary["variants"][name]["mode_counts"].get(mode, 0) /
                  max(summary["variants"][name]["decisions"], 1) for name in variants]
        axes[0].bar(x, values, bottom=bottom, label=mode)
        bottom += values
    axes[0].set_xticks(x, display, rotation=20, ha="right")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("decision fraction")
    axes[0].set_title("Mode distribution"); axes[0].legend(fontsize=8)

    structured = [name for name in variants if name in STRUCTURED_VARIANTS]
    structured_x = np.arange(len(structured))
    grounding_bottom = np.zeros(len(structured))
    grounding_colors = {"supported": "#54a24b", "ambiguous": "#eeca3b", "contradicted": "#e45756"}
    for status in ("supported", "ambiguous", "contradicted"):
        values = []
        for name in structured:
            stats = summary["variants"][name]
            values.append(stats["grounding_counts"].get(status, 0) / max(stats["decisions"], 1))
        axes[1].bar(structured_x, values, bottom=grounding_bottom, label=status,
                    color=grounding_colors[status])
        grounding_bottom += values
    axes[1].set_xticks(structured_x, [display_names[name] for name in structured], rotation=20, ha="right")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("decision fraction")
    axes[1].set_title("Structured-basis grounding"); axes[1].legend(fontsize=8)

    width = 0.36
    axes[2].bar(x - width / 2, [summary["variants"][v]["mean_final_progress"] for v in variants],
                width, label="final progress (m)")
    twin = axes[2].twinx()
    twin.bar(x + width / 2, [summary["variants"][v]["mean_speed"] for v in variants],
             width, color="#f58518", label="mean speed (m/s)")
    axes[2].set_xticks(x, display, rotation=20, ha="right"); axes[2].set_title("Race outcome")
    h1, l1 = axes[2].get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=8)
    fig.suptitle("Raw vs evidence-first LLM A/B audit")
    fig.tight_layout(); fig.savefig(output, dpi=180, bbox_inches="tight"); plt.close(fig)


def write_csv(rows, path):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_html(summary, decisions, path):
    def number(value, digits=2):
        return "N/A" if value is None else f"{value:.{digits}f}"

    rows = []
    for variant, stats in summary["variants"].items():
        grounded = stats["supported_fraction_excluding_ambiguous"]
        rows.append(f"<tr><td>{html.escape(variant)}</td><td>{stats['traces']}</td><td>{stats['decisions']}</td>"
                    f"<td>{html.escape(str(stats['mode_counts']))}</td><td>{'N/A' if grounded is None else f'{100*grounded:.1f}%'}</td>"
                    f"<td>{stats['wins']}</td><td>{number(stats['mean_final_progress'], 1)}</td><td>{number(stats['mean_speed'])}</td>"
                    f"<td>{number(stats['mean_latency_seconds'])}</td></tr>")
    bad = [row for row in decisions if row["grounding"] == "contradicted"]
    examples = "".join(f"<tr><td>{html.escape(r['trace'])}</td><td>{r['call']}</td><td>{html.escape(r['basis'])}</td>"
                       f"<td>{html.escape(r['grounding_evidence'])}</td><td>{html.escape(r['mode'])}</td></tr>" for r in bad[:30])
    examples = examples or "<tr><td colspan='5'>No contradicted structured basis labels.</td></tr>"
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LLM basis A/B</title>
<style>body{{font:15px system-ui;max-width:1250px;margin:30px auto;padding:0 18px}}table{{border-collapse:collapse;width:100%;margin:14px 0}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}th{{background:#f2f4f7}}img{{max-width:100%}}.note{{background:#fff8dc;padding:12px;border-left:4px solid #d9a400}}</style></head><body>
<h1>Raw vs evidence-first LLM</h1><p class='note'>{html.escape(summary['scope_note'])} Deadbands only mark borderline observations ambiguous.</p>
<img src='audit.png'><h2>Summary</h2><table><tr><th>Variant</th><th>Traces</th><th>Decisions</th><th>Mode counts</th><th>Supported basis*</th><th>Wins</th><th>Mean progress</th><th>Mean speed</th><th>Mean latency</th></tr>{''.join(rows)}</table>
<p>*Among non-ambiguous structured labels. Raw free text is intentionally not assigned this score.</p>
<h2>Contradicted structured basis</h2><table><tr><th>Trace</th><th>Call</th><th>Basis</th><th>Observation</th><th>Mode</th></tr>{examples}</table></body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/llm_basis_ab_berlin/audit")
    args = parser.parse_args()
    decisions, traces, excluded = [], [], {}
    for path in discover_npz(args.inputs):
        ds, trace, error = audit_trace(path)
        if error:
            excluded[relative(path)] = error
        else:
            decisions.extend(ds); traces.append(trace)
    if not traces:
        print("No compatible traces found.", file=sys.stderr); return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(decisions, traces, excluded)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(decisions, args.output_dir / "decisions.csv"); write_csv(traces, args.output_dir / "traces.csv")
    plot_summary(summary, args.output_dir / "audit.png"); write_html(summary, decisions, args.output_dir / "report.html")
    print(f"Audited {len(decisions)} decisions from {len(traces)} traces.")
    print(f"Report: {args.output_dir / 'report.html'}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
