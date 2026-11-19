"""CLI: run an official-logic session and write config.json, races.jsonl and per-race tick arrays.
Usage (repo root, official environment):
  /Users/lee/alpha-RACER-run/venv/bin/python -m official_port.main --opp1 mpc --opp2 mpc --races 101 --out DIR
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from .config import EGOS, OPPONENTS, RaceConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def source_hashes():
    """md5 of every file in official_port except outputs and caches."""
    out = {}
    for root, dirs, files in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            out[os.path.relpath(p, HERE)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return out


def git_identity():
    def run(*a):
        try:
            return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "."))}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp1", default="mpc", choices=OPPONENTS)
    ap.add_argument("--opp2", default="mpc", choices=OPPONENTS)
    ap.add_argument("--ego", default="alpharacer", choices=EGOS)
    ap.add_argument("--fixed-params", default=None, help="ego 'fixed': sf1,sf2,lookahead,speed,blocking")
    ap.add_argument("--rl-weights", default=None, help="actor checkpoint for rl cars")
    g = ap.add_argument_group("LLM ego (--ego llm)")
    g.add_argument("--llm-model")
    g.add_argument("--llm-base-url", default="http://localhost:8000")
    g.add_argument("--llm-backend", default="vllm", choices=("vllm", "ollama"))
    g.add_argument("--llm-seed", type=int, default=0)
    g.add_argument("--llm-timeout", type=float, default=20.0)
    g.add_argument("--llm-temperature", type=float, default=0.0)
    g.add_argument("--llm-thinking", action="store_true")
    g.add_argument("--llm-max-tokens", type=int, default=192)
    g.add_argument("--llm-interval", type=float, default=1.0, help="decision interval, s")
    g.add_argument("--llm-prompt-module", default="base", choices=("base", "reference", "reference_mpc"))
    g.add_argument("--llm-current-params", action="store_true")
    g.add_argument("--llm-memory-length", type=int, default=0, help="recent_decisions entries (0: off)")
    ap.add_argument("--seed", default="0", help="integer, or 'none' for an unseeded draw stream")
    ap.add_argument("--races", type=int, default=101)
    ap.add_argument("--start-ep", type=int, default=0)
    ap.add_argument("--no-ticks", action="store_true", help="do not save per-tick arrays")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fixed = None if a.fixed_params is None else [float(v) for v in a.fixed_params.split(",")]
    if (a.ego == "fixed") != (fixed is not None):
        ap.error("--fixed-params is required with, and only with, --ego fixed")
    seed = None if a.seed == "none" else int(a.seed)
    llm = None
    if a.ego == "llm":
        if not a.llm_model:
            ap.error("--ego llm needs --llm-model")
        llm = {"model": a.llm_model, "base_url": a.llm_base_url, "backend": a.llm_backend, "seed": a.llm_seed,
               "timeout": a.llm_timeout, "temperature": a.llm_temperature, "thinking": a.llm_thinking,
               "max_tokens": a.llm_max_tokens, "interval_s": a.llm_interval, "prompt_module": a.llm_prompt_module,
               "include_current_parameters": a.llm_current_params, "memory_length": a.llm_memory_length}
    return RaceConfig(opp1=a.opp1, opp2=a.opp2, ego=a.ego, fixed_params=fixed, rl_weights=a.rl_weights, llm=llm, seed=seed, n_races=a.races,
                      start_ep=a.start_ep, save_ticks=not a.no_ticks), a.out


def main(argv=None):
    cfg, out = parse(argv)
    from .simulation import Session  # heavy imports (jax, casadi) after argument parsing
    os.makedirs(os.path.join(out, "ticks"), exist_ok=True)
    meta = {"config": cfg.to_dict(), "argv": sys.argv, "git": git_identity(), "sources": source_hashes(),
            "python": sys.version, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=1)
    t0 = time.time()

    def on_race(summary, ticks):
        summary["wall_s"] = round(time.time() - t0, 1)
        for k, rec in enumerate(summary["end"]):  # LLM prompts/responses go to their own file
            io = rec.pop("llm_io", None)
            if io is not None:
                os.makedirs(os.path.join(out, "llm_io"), exist_ok=True)
                with open(os.path.join(out, "llm_io", f"race_{summary['ep_no']:03d}_car{k}.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in io)
        with open(os.path.join(out, "races.jsonl"), "a") as f:
            f.write(json.dumps(summary) + "\n")
        if ticks is not None:
            np.savez_compressed(os.path.join(out, "ticks", f"race_{summary['ep_no']:03d}.npz"), **ticks)
        print(f"race {summary['ep_no']} G{summary['group']} progress {np.round(summary['progress'], 1).tolist()} "
              f"winner {summary['winner']} wins {summary['n_wins']} (official rule {summary['n_wins_official_rule']}) "
              f"{summary['wall_s']:.0f}s", flush=True)

    Session(cfg).run(on_race)


if __name__ == "__main__":
    main()
