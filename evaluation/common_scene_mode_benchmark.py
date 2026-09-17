#!/usr/bin/env python3
"""Paired LLM-mode evaluation on a frozen bank of reachable race scenes.

The bank is sampled from real 50 s traces using only observable geometry and
traffic relationships, never the recorded LLM mode. Every prompt variant sees
the exact same JSON observation. The four mode presets are then replayed from
the same simulator state, producing a local constrained regret for the mode
chosen by each prompt.

This is a development benchmark. Frozen-opponent rollouts and short horizons
make it useful for paired screening, not a replacement for closed-loop races.

Example:
  .venv/bin/python evaluation/common_scene_mode_benchmark.py \
    --source raw_p2=result/.../trial_001.npz \
    --source balanced_p3=result/.../trial_001.npz \
    --output-dir result/common_scene_mode_dev_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.counterfactual_mode_rollout import (  # noqa: E402
    MODES,
    build_policy,
    label_acceptability,
    recorded_state,
    scalar,
    simulate_branch,
)
from evaluation.traffic_basis import TRAFFIC_BASIS_LABELS, verify_traffic_basis  # noqa: E402
from policy.experimental_balanced_raw_llm import BalancedRawLLMPolicy  # noqa: E402
from policy.llm import LLMAlphaRacerPolicy  # noqa: E402
from config import params  # noqa: E402
from track import Track, load_centerline_csv, load_raceline_csv  # noqa: E402


CATEGORIES = ("ordinary", "high_curvature", "front_closing", "rear_closing")


def parse_json_object(text):
    value = json.loads(str(text))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def signed_motion(vehicle):
    if vehicle is None:
        return 0.0
    motion = vehicle.get("gap_motion") or {}
    speed = float(motion.get("speed_meters_per_second", 0.0))
    if motion.get("trend") == "closing":
        return speed
    if motion.get("trend") == "opening":
        return -speed
    return 0.0


def scene_signature(observation):
    """Coarse observation signature used only to avoid near-duplicate scenes."""
    ego = observation["ego"]
    traffic = observation["traffic"]
    preview = observation["road_preview"]["samples_near_to_far"]

    def vehicle_signature(vehicle):
        if vehicle is None:
            return None
        return (
            round(float(vehicle["longitudinal_gap_meters"]) * 2.0) / 2.0,
            (vehicle.get("gap_motion") or {}).get("trend"),
            round(abs(signed_motion(vehicle)) * 2.0) / 2.0,
        )

    lateral = ego.get("lateral_position") or {}
    signature = (
        round(float(ego["lap_progress_percent"]) / 2.0) * 2.0,
        round(float(ego["speed_percent_of_maximum"]) / 10.0) * 10.0,
        lateral.get("side"),
        round(float(lateral.get("percent_of_half_width", 0.0)) / 10.0) * 10.0,
        vehicle_signature(traffic.get("front")),
        vehicle_signature(traffic.get("rear")),
        round(max(float(row["centerline_curvature"]["magnitude_per_meter"])
                  for row in preview) / 0.05) * 0.05,
    )
    return hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:12]


def candidate_scenes(label, path, long_seconds, all_cars_as_ego=True):
    data = np.load(path, allow_pickle=True)
    sim_dt = float(scalar(data, "simulation_dt", params.SIM_DT))
    control_dt = float(scalar(data, "control_dt", params.CONTROL_DT))
    control_steps = int(round(control_dt / sim_dt))
    future_steps = int(math.ceil(long_seconds / sim_dt))
    resets = np.asarray(
        data["car_deadlock_reset_event"]
        if "car_deadlock_reset_event" in data.files
        else np.zeros((len(data["collision"]), data["car_x"].shape[1]), dtype=bool),
        dtype=bool,
    )
    collisions = np.asarray(data["collision"], dtype=bool)
    track_csv = str(scalar(data, "track_csv", params.TRACK_CSV))
    track_key = (
        "modena" if "modena" in track_csv.lower()
        else "rounded_rectangle" if "rounded" in track_csv.lower()
        else "berlin" if "berlin" in track_csv.lower()
        else Path(track_csv).stem
    )
    track = Track(load_centerline_csv(
        track_csv,
        scale=float(scalar(data, "track_scale", 1.0)),
    ))
    raceline = load_raceline_csv(
        str(scalar(data, "raceline_csv", params.RACELINE_CSV)),
        scale=float(scalar(data, "track_scale", 1.0)),
    )
    source_ego = int(scalar(data, "ego_index", 0))
    ego_indices = range(data["car_x"].shape[1]) if all_cars_as_ego else (source_ego,)
    rows = []
    for ego_idx in ego_indices:
        template, _ = build_policy(data, track, raceline, "stable", ego_idx)
        observer = LLMAlphaRacerPolicy(
            track,
            template.mpc,
            params=np.asarray(params.LLM_MODE_PRESETS["stable"], dtype=float),
            raceline=raceline,
            max_speed=params.MAX_RACELINE_SPEED,
            ref_lateral_clip=float(scalar(data, "ref_lateral_clip", params.REF_LATERAL_CLIP)),
            car_idx=ego_idx,
            policy_name="common_scene_observer",
            output_mode="mode",
        )
        # Candidate times follow the actual one-second high-level cadence even
        # when the source policy was not an LLM.
        frames = range(int(round(1.0 / sim_dt)), len(data["time"]), int(round(1.0 / sim_dt)))
        for slot, frame in enumerate(frames, start=1):
            end = frame + future_steps
            if end >= len(data["time"]):
                continue
            # The initial state must be valid. Future resets/collisions are excluded
            # because a frozen recorded trajectory would otherwise jump or terminate.
            if bool(data["car_offroad"][frame, ego_idx]) or float(data["car_vx"][frame, ego_idx]) < 0.2:
                continue
            if collisions[frame:end + 1].any() or resets[frame:end + 1].any():
                continue
            state = recorded_state(data, frame, ego_idx)
            current_s = float(data["car_px"][frame, ego_idx])
            opponents = []
            for car_idx in range(data["car_x"].shape[1]):
                if car_idx == ego_idx:
                    continue
                other = recorded_state(data, frame, car_idx)
                opponents.append({
                    "car_idx": car_idx,
                    "px": float(data["car_px"][frame, car_idx]),
                    "py": float(data["car_py"][frame, car_idx]),
                    "vx": max(float(other.vx), 0.0),
                    "vy": float(other.vy),
                    "omega": float(other.omega),
                    "psi": float(other.psi),
                    "state": other,
                })
            observation = observer._scene_features(state, current_s, opponents=opponents)
            # Remove source-policy history from the paired physical-scene test.
            # This makes every source policy and every evaluated prompt start
            # from the same stateless tactical question.
            observation["memory"] = {
                "previous_mode": None,
                "recent_modes_oldest_to_newest": [],
            }
            front, rear = observation["traffic"].get("front"), observation["traffic"].get("rear")
            curvature = max(
                float(row["centerline_curvature"]["magnitude_per_meter"])
                for row in observation["road_preview"]["samples_near_to_far"]
            )
            front_pressure = (
                max(signed_motion(front), 0.0) / max(float(front["longitudinal_gap_meters"]), 0.2)
                if front is not None else 0.0
            )
            rear_pressure = (
                max(signed_motion(rear), 0.0) / max(float(rear["longitudinal_gap_meters"]), 0.2)
                if rear is not None else 0.0
            )
            rows.append({
                "source": label,
                "track": track_key,
                "trace": str(path),
                "ego_idx": int(ego_idx),
                "frame": int(frame),
                "slot": int(slot),
                "time_s": float(data["time"][frame]),
                "observation": observation,
                "signature": scene_signature(observation),
                "upcoming_abs_curvature": float(curvature),
                "front_pressure": float(front_pressure),
                "rear_pressure": float(rear_pressure),
            })
    data.close()
    return rows


def select_scene_bank(candidates, per_category):
    selected = []
    used_signatures = set()
    tracks = sorted({row["track"] for row in candidates})

    def try_add(category, row):
        if row["signature"] in used_signatures:
            return False
        # Suppress adjacent global states from one source trajectory even when
        # viewed from different cars.
        if any(old["trace"] == row["trace"] and abs(old["time_s"] - row["time_s"]) < 1.0
               for old in selected):
            return False
        item = dict(row)
        item["category"] = category
        selected.append(item)
        used_signatures.add(row["signature"])
        return True

    def add(category, ranked, eligible=lambda row: True):
        count = 0
        # Allocate the first quota evenly across tracks. This prevents Berlin's
        # sharper absolute curvature or rounded rectangle's long straight from
        # monopolising a capability slice.
        base, remainder = divmod(per_category, max(len(tracks), 1))
        for track_idx, track_name in enumerate(tracks):
            track_target = base + int(track_idx < remainder)
            track_count = 0
            for row in ranked:
                if track_count >= track_target:
                    break
                if row["track"] != track_name or not eligible(row):
                    continue
                if try_add(category, row):
                    track_count += 1
                    count += 1
        # Fill any unavailable track quota from the best remaining candidates.
        for row in ranked:
            if count >= per_category:
                break
            if eligible(row) and try_add(category, row):
                count += 1

    add(
        "rear_closing",
        sorted(candidates, key=lambda row: row["rear_pressure"], reverse=True),
        lambda row: row["rear_pressure"] > 0.0,
    )
    add(
        "front_closing",
        sorted(candidates, key=lambda row: row["front_pressure"], reverse=True),
        lambda row: row["front_pressure"] > 0.0,
    )
    add(
        "high_curvature",
        sorted(candidates, key=lambda row: row["upcoming_abs_curvature"], reverse=True),
    )
    ordinary = sorted(
        candidates,
        key=lambda row: (
            row["front_pressure"] + row["rear_pressure"],
            row["upcoming_abs_curvature"],
            row["time_s"],
        ),
    )
    add("ordinary", ordinary)
    return sorted(selected, key=lambda row: (CATEGORIES.index(row["category"]), row["source"], row["time_s"]))


def response_schema(variant):
    mode = {"type": "string", "enum": list(MODES)}
    basis = {"type": "string", "maxLength": 40}
    reason = {"type": "string", "maxLength": 120}
    if variant == "evidence_first":
        basis = {"type": "string", "enum": list(TRAFFIC_BASIS_LABELS)}
        properties = {"basis": basis, "mode": mode, "reason": reason}
        required = ["basis", "mode", "reason"]
    else:
        properties = {"mode": mode, "basis": basis, "reason": reason}
        required = ["mode", "basis", "reason"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def system_prompts():
    prompt_dir = ROOT / "policy" / "prompts"
    raw = (prompt_dir / "mode_system.txt").read_text().strip()
    return {
        "raw": raw,
        "balanced": f"{raw}\n\n{BalancedRawLLMPolicy.PROMPT_ADDITION}",
        "evidence_first": (prompt_dir / "mode_evidence_first_system.txt").read_text().strip(),
    }


def query_ollama(base_url, model, temperature, timeout, system, observation, variant):
    request_body = {
        "model": model,
        "system": system,
        "prompt": json.dumps(observation, indent=2, ensure_ascii=True),
        "format": response_schema(variant),
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 192},
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
    latency = time.perf_counter() - started
    parsed = parse_json_object(body.get("response", ""))
    mode = str(parsed.get("mode", "")).strip().lower()
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode!r}")
    return {
        "mode": mode,
        "basis": str(parsed.get("basis", ""))[:80],
        "reason": " ".join(str(parsed.get("reason", "")).split())[:160],
        "latency_seconds": float(latency),
    }


def branch_summary(branches, ratio=0.8):
    acceptable, best = label_acceptability(branches, ratio)
    return {
        "best_safe_gain_m": best,
        "best_safe_modes": [row["mode"] for row in branches if row["best_safe"]],
        "acceptable_modes_at_80_percent_rule": acceptable,
        "branches": branches,
    }


def evaluate_decision(decision, horizon_result, observation):
    chosen = next(row for row in horizon_result["branches"] if row["mode"] == decision["mode"])
    decision["safe"] = bool(chosen["safe"])
    decision["regret_m"] = chosen["regret_m"]
    decision["chosen_relative_gain_m"] = float(chosen["relative_progress_gain_m"])
    decision["best_safe_modes"] = horizon_result["best_safe_modes"]
    decision["accepted_at_80_percent_rule"] = decision["mode"] in horizon_result["acceptable_modes_at_80_percent_rule"]
    if decision["basis"] in TRAFFIC_BASIS_LABELS:
        status, evidence = verify_traffic_basis(decision["basis"], observation)
    else:
        status, evidence = "not_structured", "free-text basis"
    decision["basis_grounding"] = status
    decision["basis_grounding_evidence"] = evidence


def summarize(scenes, variants, horizon_key):
    output = {}
    for variant in variants:
        decisions = [scene["decisions"][variant][horizon_key] for scene in scenes]
        safe = [row for row in decisions if row["safe"]]
        regrets = [float(row["regret_m"]) for row in safe if row["regret_m"] is not None]
        structured = [row for row in decisions if row["basis_grounding"] != "not_structured"]
        output[variant] = {
            "scene_count": len(decisions),
            "mode_counts": dict(Counter(row["mode"] for row in decisions)),
            "safe_choice_rate": float(np.mean([row["safe"] for row in decisions])),
            "mean_regret_m_safe_choices": float(np.mean(regrets)) if regrets else None,
            "median_regret_m_safe_choices": float(np.median(regrets)) if regrets else None,
            "near_best_rate_within_0_05_m": float(np.mean([
                row["safe"] and row["regret_m"] is not None and row["regret_m"] <= 0.05
                for row in decisions
            ])),
            "accepted_rate_at_80_percent_rule": float(np.mean([
                row["accepted_at_80_percent_rule"] for row in decisions
            ])),
            "mean_latency_seconds": float(np.mean([
                scene["decisions"][variant]["query"]["latency_seconds"] for scene in scenes
            ])),
            "basis_grounding_counts": dict(Counter(
                row["basis_grounding"] for row in structured
            )),
        }
    return output


def save_plot(output_dir, scenes, variants, horizon_key):
    values = np.full((len(scenes), len(variants)), np.nan)
    unsafe = np.zeros_like(values, dtype=bool)
    for i, scene in enumerate(scenes):
        for j, variant in enumerate(variants):
            row = scene["decisions"][variant][horizon_key]
            unsafe[i, j] = not row["safe"]
            if row["regret_m"] is not None:
                values[i, j] = row["regret_m"]
    finite = values[np.isfinite(values)]
    vmax = max(float(np.max(finite)) if finite.size else 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.42 * len(scenes) + 1.7)))
    image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(len(scenes)):
        for j in range(len(variants)):
            label = "unsafe" if unsafe[i, j] else f"{values[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(variants)), variants)
    ax.set_yticks(range(len(scenes)), [f"{s['scene_id']} ({s['category']})" for s in scenes])
    ax.set_title(f"Paired local mode regret ({horizon_key.replace('_', ' ')})")
    fig.colorbar(image, ax=ax, label="regret [m]; lower is better")
    fig.tight_layout()
    path = output_dir / "paired_regret.svg"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def write_report(output_dir, payload, plot_name):
    primary = payload["summary"][payload["primary_horizon"]]
    summary_rows = []
    for variant, row in primary.items():
        summary_rows.append(
            f"<tr><td>{html.escape(variant)}</td><td>{html.escape(str(row['mode_counts']))}</td>"
            f"<td>{row['safe_choice_rate']:.2f}</td>"
            f"<td>{row['mean_regret_m_safe_choices']:.3f}</td>"
            f"<td>{row['near_best_rate_within_0_05_m']:.2f}</td>"
            f"<td>{html.escape(str(row['basis_grounding_counts']))}</td></tr>"
        )
    scene_rows = []
    for scene in payload["scenes"]:
        choices = ", ".join(
            f"{variant}: {scene['decisions'][variant][payload['primary_horizon']]['mode']} "
            f"(r={scene['decisions'][variant][payload['primary_horizon']]['regret_m']})"
            for variant in payload["variants"]
        )
        scene_rows.append(
            f"<tr><td>{scene['scene_id']}</td><td>{scene['category']}</td>"
            f"<td>{scene['source']} @ {scene['time_s']:.1f}s</td><td>{html.escape(choices)}</td></tr>"
        )
    report = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Common-scene mode benchmark</title><style>
body{{font:15px system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.45rem;vertical-align:top}}img{{max-width:100%}}.note{{color:#555}}
</style></head><body><h1>Common-scene mode benchmark</h1>
<p><b>Primary horizon: one high-level decision interval ({payload['primary_horizon'].replace('_', ' ')}).</b></p>
<p>{html.escape(payload['scope'])}</p><img src='{html.escape(plot_name)}'>
<h2>Paired summary</h2><table><tr><th>prompt</th><th>mode counts</th><th>safe rate</th>
<th>mean regret [m], safe</th><th>regret &le; 0.05 m</th><th>basis grounding</th></tr>
{''.join(summary_rows)}</table>
<h2>Frozen scene bank</h2><table><tr><th>ID</th><th>slice</th><th>source</th><th>choice and regret</th></tr>
{''.join(scene_rows)}</table>
<p class='note'>Opponents replay recorded trajectories and do not react. Unsafe choices are counted separately and are excluded from mean regret rather than assigned an arbitrary crash penalty. This is a development screening benchmark, not a final held-out race result.</p>
</body></html>"""
    path = output_dir / "report.html"
    path.write_text(report)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Paired LLM-mode evaluation on common real-trace scenes.")
    parser.add_argument("--source", action="append", required=True, metavar="LABEL=TRACE",
                        help="Real 50 s source trace; repeat for multiple state distributions.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=("raw", "balanced", "evidence_first"),
                        help="Prompt variant to evaluate; default evaluates all three.")
    parser.add_argument("--scenes-per-category", type=int, default=3,
                        help="Scenes from each of ordinary/curve/front/rear slices.")
    parser.add_argument("--primary-seconds", type=float, default=1.0,
                        help="One intervention interval; matches the online LLM update period.")
    parser.add_argument("--diagnostic-seconds", type=float, default=3.0,
                        help="Optional longer fixed-mode persistence diagnostic; 0 disables.")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Select and save the scene bank without MPC rollouts or LLM calls.")
    parser.add_argument("--source-ego-only", action="store_true",
                        help="Use only each trace's recorded ego; default treats every car as an ego view.")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = args.variant or ["raw", "balanced", "evidence_first"]
    sources = []
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"--source must be LABEL=TRACE, got {item!r}")
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        sources.append((label, path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        row for label, path in sources
        for row in candidate_scenes(
            label, path, max(args.primary_seconds, args.diagnostic_seconds),
            all_cars_as_ego=not args.source_ego_only,
        )
    ]
    selected = select_scene_bank(candidates, args.scenes_per_category)
    if not selected:
        raise RuntimeError("No usable scenes")
    manifest = {
        "selection_rule": (
            "Reachable real-trace states selected without recorded modes. Rank front/rear closing pressure "
            "and preview curvature, plus low-pressure/low-curvature ordinary scenes; suppress near duplicates "
            "and adjacent frames. Source windows containing collision or reset are excluded."
        ),
        "requested_per_category": args.scenes_per_category,
        "scenes": [{key: value for key, value in row.items() if key != "observation"} | {
            "observation": row["observation"]
        } for row in selected],
    }
    (args.output_dir / "scene_manifest.local.json").write_text(json.dumps(manifest, indent=2))
    if args.manifest_only:
        print(f"Saved {len(selected)} scenes to {args.output_dir / 'scene_manifest.local.json'}")
        return

    loaded = {str(path): np.load(path, allow_pickle=True) for _, path in sources}
    tracks = {}
    racelines = {}
    prompts = system_prompts()
    scenes = []
    horizons = [("primary_1s", args.primary_seconds)]
    if args.diagnostic_seconds > 0 and not math.isclose(args.diagnostic_seconds, args.primary_seconds):
        horizons.append(("diagnostic_3s", args.diagnostic_seconds))

    for scene_number, selected_row in enumerate(selected, start=1):
        scene_id = f"S{scene_number:02d}"
        print(f"[{scene_number}/{len(selected)}] {scene_id} {selected_row['category']} "
              f"{selected_row['source']} t={selected_row['time_s']:.1f}s", flush=True)
        data = loaded[selected_row["trace"]]
        if selected_row["trace"] not in tracks:
            tracks[selected_row["trace"]] = Track(load_centerline_csv(
                str(scalar(data, "track_csv", params.TRACK_CSV)),
                scale=float(scalar(data, "track_scale", 1.0)),
            ))
            racelines[selected_row["trace"]] = load_raceline_csv(
                str(scalar(data, "raceline_csv", params.RACELINE_CSV)),
                scale=float(scalar(data, "track_scale", 1.0)),
            )
        track, raceline = tracks[selected_row["trace"]], racelines[selected_row["trace"]]
        horizon_results = {}
        for horizon_key, seconds in horizons:
            branches = [
                simulate_branch(
                    data, selected_row["frame"], selected_row["ego_idx"],
                    mode, seconds, track, raceline, False,
                )
                for mode in MODES
            ]
            horizon_results[horizon_key] = branch_summary(branches)

        decisions = {}
        for variant in variants:
            try:
                query = query_ollama(
                    args.base_url, args.model, args.temperature, args.timeout,
                    prompts[variant], selected_row["observation"], variant,
                )
                error = ""
            except Exception as exc:  # Preserve the scene benchmark if one local call fails.
                query = {"mode": "stable", "basis": "error", "reason": str(exc), "latency_seconds": math.nan}
                error = str(exc)
            decisions[variant] = {"query": query, "error": error}
            for horizon_key, _ in horizons:
                evaluated = dict(query)
                evaluate_decision(evaluated, horizon_results[horizon_key], selected_row["observation"])
                decisions[variant][horizon_key] = evaluated

        scenes.append({
            "scene_id": scene_id,
            "category": selected_row["category"],
            "source": selected_row["source"],
            "trace": selected_row["trace"],
            "frame": selected_row["frame"],
            "ego_idx": selected_row["ego_idx"],
            "time_s": selected_row["time_s"],
            "scene_features": {
                key: selected_row[key] for key in
                ("upcoming_abs_curvature", "front_pressure", "rear_pressure")
            },
            "observation": selected_row["observation"],
            "horizons": horizon_results,
            "decisions": decisions,
        })

    for data in loaded.values():
        data.close()
    summary = {key: summarize(scenes, variants, key) for key, _ in horizons}
    payload = {
        "scope": (
            "Paired development benchmark on reachable Berlin P2/P3 states. Every prompt sees the same "
            "observation; every mode starts from the same state. Synthetic target scenes are intentionally "
            "excluded from the aggregate score."
        ),
        "model": args.model,
        "temperature": args.temperature,
        "variants": variants,
        "primary_horizon": horizons[0][0],
        "horizons_seconds": {key: seconds for key, seconds in horizons},
        "regret_definition": (
            "best safe change in signed lead minus the chosen safe mode's change in signed lead. "
            "Unsafe choices are reported separately and have regret=null."
        ),
        "opponent_model": "Frozen replay of the source trace; no reaction to counterfactual ego.",
        "scene_selection": manifest["selection_rule"],
        "summary": summary,
        "scenes": scenes,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))

    flat = []
    for scene in scenes:
        for variant in variants:
            for horizon_key, _ in horizons:
                row = scene["decisions"][variant][horizon_key]
                flat.append({
                    "scene_id": scene["scene_id"], "category": scene["category"],
                    "source": scene["source"], "time_s": scene["time_s"],
                    "variant": variant, "horizon": horizon_key,
                    "mode": row["mode"], "safe": row["safe"], "regret_m": row["regret_m"],
                    "chosen_relative_gain_m": row["chosen_relative_gain_m"],
                    "basis": row["basis"], "basis_grounding": row["basis_grounding"],
                })
    with (args.output_dir / "decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot_name = save_plot(args.output_dir, scenes, variants, horizons[0][0])
    report = write_report(args.output_dir, payload, plot_name)
    print(json.dumps(summary[horizons[0][0]], indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
