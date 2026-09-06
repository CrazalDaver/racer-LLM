#!/usr/bin/env python3
"""Probe whether a factual traffic basis affects subsequent LLM mode choice.

The script reuses observations and one-call outputs from saved NPZ traces.  It
does not run the simulator.  For each sampled observation, a separate mode-only
call receives the recorded generated basis, a code-derived oracle basis, and a
direction-reversed counterfactual basis.  Identical basis queries are cached.

Usage:
  .venv/bin/python evaluation/llm_basis_intervention_probe.py \
    result/llm_directional_minimal_ab_berlin_evidence_first \
    --output-dir result/llm_basis_intervention_berlin
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import os
from pathlib import Path
import re
import time
import urllib.request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/racer_llm_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traffic_basis import (
    COUNTERFACTUAL_BASIS,
    DYNAMIC_TRAFFIC_BASIS_LABELS,
    oracle_traffic_bases,
    verify_traffic_basis,
)


ROOT = Path(__file__).resolve().parents[1]
MODES = ("stable", "push", "overtake", "block")
SYSTEM_PROMPT = """You are the tactical decision maker for an alpha-RACER reference-trajectory MPC.
Select exactly one mode from stable, push, overtake, and block.

The user message contains a frozen numerical racing observation and a basis
label supplied by an upstream traffic-relation stage. Use the basis to focus on
that observable relationship, while using the complete observation to choose
the tactical mode. There is no fixed basis-to-mode mapping.

Modes:
- stable: neutral raceline following for controlled progress.
- push: faster reference progress.
- overtake: create a passing offset around a vehicle ahead.
- block: create a defensive offset against a vehicle behind.

Return only the required JSON with the selected mode and a factual reason under
20 words."""


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
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else None


def load_candidates(paths):
    candidates = []
    for path in discover_npz(paths):
        with np.load(path, allow_pickle=False) as data:
            required = {"policy_names", "llm_raw_call_indices", "llm_raw_prompts", "llm_raw_responses"}
            if not required.issubset(data.files):
                continue
            names = np.asarray(data["policy_names"], dtype=str)
            cars = [i for i, name in enumerate(names) if name == "llm_evidence_first"]
            if not cars:
                continue
            car = cars[0]
            calls = np.asarray(data["llm_raw_call_indices"], dtype=int)[car]
            prompts = np.asarray(data["llm_raw_prompts"], dtype=str)[car]
            responses = np.asarray(data["llm_raw_responses"], dtype=str)[car]
            errors = np.asarray(
                data.get("llm_raw_errors", np.full_like(responses, "")), dtype=str
            )[car]
            for slot, call in enumerate(calls):
                if call <= 0 or errors[slot]:
                    continue
                observation = parse_object(prompts[slot])
                response = parse_object(responses[slot])
                if observation is None or response is None:
                    continue
                generated_basis = str(response.get("basis", response.get("traffic_basis", "")))
                generated_mode = str(response.get("mode", ""))
                grounding, _ = verify_traffic_basis(generated_basis, observation)
                for subject, fact in oracle_traffic_bases(observation).items():
                    if fact["label"] not in DYNAMIC_TRAFFIC_BASIS_LABELS:
                        continue
                    candidates.append({
                        "trace": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        "call": int(call),
                        "subject": subject,
                        "observation": observation,
                        "generated_basis": generated_basis,
                        "generated_grounding": grounding,
                        "one_call_mode": generated_mode,
                        "oracle_basis": fact["label"],
                        "oracle_evidence": fact["evidence"],
                        "counterfactual_basis": COUNTERFACTUAL_BASIS[fact["label"]],
                    })
    return candidates


def stratified_sample(candidates, samples_per_relation):
    grouped = defaultdict(list)
    for item in candidates:
        grouped[item["oracle_basis"]].append(item)
    selected = []
    for label in DYNAMIC_TRAFFIC_BASIS_LABELS:
        rows = grouped[label]
        if not rows:
            continue
        # Evenly cover early and late observations without random sampling.
        indices = np.linspace(0, len(rows) - 1, min(samples_per_relation, len(rows)), dtype=int)
        selected.extend(rows[int(index)] for index in indices)
    return selected


def query_mode(base_url, model, timeout, temperature, observation, basis):
    prompt = json.dumps(
        {"observation": observation, "provided_basis": basis},
        indent=2,
        ensure_ascii=True,
    )
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": list(MODES)},
            "reason": {"type": "string", "maxLength": 120},
        },
        "required": ["mode", "reason"],
        "additionalProperties": False,
    }
    request_body = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 96},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    result = parse_object(body.get("response", "")) or {}
    mode = str(result.get("mode", ""))
    if mode not in MODES:
        raise ValueError(f"Invalid mode response: {result}")
    return {
        "mode": mode,
        "reason": str(result.get("reason", "")),
        "latency_seconds": time.perf_counter() - started,
    }


def run_probe(cases, args):
    rows = []
    for index, case in enumerate(cases, start=1):
        labels = {
            "generated": case["generated_basis"],
            "oracle": case["oracle_basis"],
            "counterfactual": case["counterfactual_basis"],
        }
        results_by_label = {}
        for label in dict.fromkeys(labels.values()):
            results_by_label[label] = query_mode(
                args.base_url,
                args.model,
                args.timeout,
                args.temperature,
                case["observation"],
                label,
            )
        row = {key: value for key, value in case.items() if key != "observation"}
        for condition, label in labels.items():
            result = results_by_label[label]
            row[f"{condition}_conditioned_mode"] = result["mode"]
            row[f"{condition}_conditioned_reason"] = result["reason"]
            row[f"{condition}_latency_seconds"] = result["latency_seconds"]
        row["oracle_counterfactual_mode_changed"] = (
            row["oracle_conditioned_mode"] != row["counterfactual_conditioned_mode"]
        )
        row["generated_oracle_mode_agree"] = (
            row["generated_conditioned_mode"] == row["oracle_conditioned_mode"]
        )
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {case['oracle_basis']}: "
            f"one-call={case['one_call_mode']}, oracle={row['oracle_conditioned_mode']}, "
            f"counterfactual={row['counterfactual_conditioned_mode']}"
        )
    return rows


def fraction(values):
    return float(np.mean(values)) if values else None


def summarize(rows):
    conditions = {
        "one_call": "one_call_mode",
        "generated_conditioned": "generated_conditioned_mode",
        "oracle_conditioned": "oracle_conditioned_mode",
        "counterfactual_conditioned": "counterfactual_conditioned_mode",
    }
    changed_count = sum(row["oracle_counterfactual_mode_changed"] for row in rows)
    change_by_relation = {}
    for label in DYNAMIC_TRAFFIC_BASIS_LABELS:
        relation_rows = [row for row in rows if row["oracle_basis"] == label]
        relation_changed = sum(
            row["oracle_counterfactual_mode_changed"] for row in relation_rows
        )
        change_by_relation[label] = {
            "cases": len(relation_rows),
            "changed": relation_changed,
            "fraction": fraction(
                [row["oracle_counterfactual_mode_changed"] for row in relation_rows]
            ),
        }
    return {
        "scope_note": "Oracle basis is a factual traffic relation, not a correct-mode label.",
        "cases": len(rows),
        "oracle_basis_counts": dict(Counter(row["oracle_basis"] for row in rows)),
        "generated_basis_grounding": dict(Counter(row["generated_grounding"] for row in rows)),
        "mode_counts": {
            condition: dict(Counter(row[field] for row in rows))
            for condition, field in conditions.items()
        },
        "oracle_counterfactual_mode_change_fraction": fraction(
            [row["oracle_counterfactual_mode_changed"] for row in rows]
        ),
        "oracle_counterfactual_mode_change_count": changed_count,
        "oracle_counterfactual_mode_unchanged_count": len(rows) - changed_count,
        "oracle_counterfactual_change_by_relation": change_by_relation,
        "generated_oracle_conditioned_mode_agreement": fraction(
            [row["generated_oracle_mode_agree"] for row in rows]
        ),
        "mode_by_oracle_basis": {
            label: dict(Counter(
                row["oracle_conditioned_mode"] for row in rows if row["oracle_basis"] == label
            ))
            for label in DYNAMIC_TRAFFIC_BASIS_LABELS
        },
    }


def write_outputs(rows, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (output_dir / "decisions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    changed = summary["oracle_counterfactual_mode_change_count"]
    unchanged = summary["oracle_counterfactual_mode_unchanged_count"]
    total = max(summary["cases"], 1)
    changed_fraction = changed / total
    axes[0].barh([0], [changed_fraction], color="#2e7d6e", label="mode changed")
    axes[0].barh(
        [0], [1.0 - changed_fraction], left=[changed_fraction],
        color="#b8bec6", label="unchanged"
    )
    axes[0].text(
        changed_fraction / 2, 0, f"{changed}/{total}\n({changed_fraction:.1%})",
        ha="center", va="center", color="white", fontweight="bold"
    )
    axes[0].text(
        changed_fraction + (1.0 - changed_fraction) / 2, 0,
        f"{unchanged}/{total}\n({1.0 - changed_fraction:.1%})",
        ha="center", va="center", color="#333333", fontweight="bold"
    )
    axes[0].set_xlim(0, 1); axes[0].set_yticks([])
    axes[0].set_xlabel("fraction of paired observations")
    axes[0].set_title("Mode response to reversed basis")
    axes[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)

    labels = list(DYNAMIC_TRAFFIC_BASIS_LABELS)
    relation_stats = summary["oracle_counterfactual_change_by_relation"]
    values = [relation_stats[label]["fraction"] or 0.0 for label in labels]
    x = np.arange(len(labels))
    axes[1].bar(x, values, color="#4c78a8")
    for index, label in enumerate(labels):
        stats = relation_stats[label]
        axes[1].text(
            index, values[index] + 0.035,
            f"{stats['changed']}/{stats['cases']}", ha="center", va="bottom"
        )
    display_labels = [
        "front:\nclosing", "front:\nopening", "rear:\nclosing", "rear:\nopening"
    ]
    axes[1].set_xticks(x, display_labels)
    axes[1].set_ylim(0, 1.12); axes[1].set_ylabel("paired mode-change fraction")
    axes[1].set_title("Change rate by intervened relation")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18); axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "probe.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "probe.pdf", bbox_inches="tight")
    plt.close(fig)

    detail_rows = "".join(
        f"<tr><td>{html.escape(row['oracle_basis'])}</td><td>{html.escape(row['generated_basis'])}</td>"
        f"<td>{html.escape(row['one_call_mode'])}</td><td>{html.escape(row['oracle_conditioned_mode'])}</td>"
        f"<td>{html.escape(row['counterfactual_conditioned_mode'])}</td>"
        f"<td>{'yes' if row['oracle_counterfactual_mode_changed'] else 'no'}</td></tr>"
        for row in rows
    )
    output_dir.joinpath("report.html").write_text(f"""<!doctype html><html><head><meta charset='utf-8'>
<title>LLM basis intervention</title><style>body{{font:15px system-ui;max-width:1200px;margin:30px auto;padding:0 18px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:7px}}th{{background:#f2f4f7}}img{{max-width:100%}}.note{{background:#fff8dc;padding:12px;border-left:4px solid #d9a400}}</style></head><body>
<h1>Traffic-basis intervention probe</h1><p class='note'>{html.escape(summary['scope_note'])}</p>
<img src='probe.png'><p>Cases: {summary['cases']}. Oracle/counterfactual mode-change fraction:
{summary['oracle_counterfactual_mode_change_fraction']:.1%}. Generated/oracle conditioned-mode agreement:
{summary['generated_oracle_conditioned_mode_agreement']:.1%}.</p>
<h2>Decisions</h2><table><tr><th>Oracle basis</th><th>Generated basis</th><th>Original one-call mode</th><th>Oracle-conditioned mode</th><th>Counterfactual-conditioned mode</th><th>Changed?</th></tr>{detail_rows}</table>
</body></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="NPZ file(s) or result directories")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result/llm_basis_intervention_berlin")
    parser.add_argument("--samples-per-relation", type=int, default=3)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    candidates = load_candidates(args.inputs)
    cases = stratified_sample(candidates, max(args.samples_per_relation, 1))
    if not cases:
        raise SystemExit("No dynamic traffic-relation observations found.")
    rows = run_probe(cases, args)
    summary = summarize(rows)
    write_outputs(rows, summary, args.output_dir)
    print(f"Summary: {args.output_dir / 'summary.json'}")
    print(f"Report: {args.output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
