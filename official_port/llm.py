"""LLM ego policy in the official world: every decision interval (default 1 s = 10 ticks, decided at
ticks 1, 11, 21, ...) the model returns the five local-convention parameters theta = (q, zeta, s1, s2, s3)
for the official MPC reference. They map to the official coordinates exactly as the C01/F02
correspondence: lookahead = .5 / q (the MPC position weight 1 / (2 lookahead) is q), speed = sqrt(zeta)
(points advance at speed**2 * raceline speed = zeta * raceline speed), (sf1, sf2, blocking) = (s1, s2, s3).
Bounds are the official clip ranges in these coordinates.

The request path (OpenAI-compatible chat, per-call seed, JSON schema, thinking switch, raw I/O record)
reuses policy/llm/base.py LLMPolicyBase methods unchanged. The observation has the local v3 fields
(policy/llm/parameter.py format_observation), computed from official quantities: generator s/e
(centerline, left positive), heading error = -theta_diff, body vx, progress/lateral rates as finite
differences over the last tick, opponents' delta s wrapped by the centerline length, and a 12 m / 7
sample road preview on the centerline (official custom_fn curvature, geometric raceline offset and
raceline speed at the nearest raceline point). A failed call keeps the previous parameters; before the
first successful answer the local LLM initializer (2, .95**2, .3, .2, 0) is used.
"""
import hashlib
import json
import math
import os

import numpy as np

from car_dynamics.controllers_jax.waypoint import custom_fn
from policy.llm.base import LLMPolicyBase

from .policies import Policy
from .world import DT

PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
# Official clip ranges mapped to (q, zeta, s1, s2, s3): lookahead [.15, .5], speed [.85, 1.1].
PARAMETER_BOUNDS = {"q": (0.5 / 0.5, 0.5 / 0.15), "zeta": (0.85 ** 2, 1.1 ** 2), "s1": (0.1, 0.5),
                    "s2": (0.1, 0.5), "s3": (0.0, 1.0)}
INITIAL_THETA = (2.0, 0.95 ** 2, 0.3, 0.2, 0.0)  # local LLM_POLICY_PARAMS
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
PROMPT_MODULE_FILES = {"base": (), "reference": ("params_reference_module.txt",),
                       "reference_mpc": ("params_reference_module.txt", "params_mpc_module.txt")}
ROAD_PREVIEW_M, ROAD_SAMPLES = 12.0, 7  # local LLM_ROAD_PREVIEW_LENGTH_METERS / SAMPLE_COUNT
OFF_TRACK_E = 0.55  # official recovery threshold


def theta_to_official(theta):
    q, zeta, s1, s2, s3 = (float(v) for v in theta)
    return [s1, s2, 0.5 / q, math.sqrt(zeta), s3]


def _r(value, digits=3):
    return LLMPolicyBase._rounded(value, digits)


class _Client(LLMPolicyBase):
    """LLMPolicyBase used only for its request/record methods (its MPC-coupled __init__ is not run)."""
    response_format = "params"

    def __init__(self, cfg, system_prompt, car_idx):
        self.model_name = str(cfg["model"])
        self.base_url = str(cfg.get("base_url", "http://localhost:11434")).rstrip("/").removesuffix("/v1")
        self.backend = str(cfg.get("backend", "vllm")).lower()
        assert self.backend in ("ollama", "vllm"), self.backend
        self.seed = int(cfg.get("seed", 0))
        self.timeout = float(cfg.get("timeout", 20.0))
        self.temperature = float(np.clip(cfg.get("temperature", 0.0), 0.0, 2.0))
        self.thinking = bool(cfg.get("thinking", False))
        self.num_predict = int(cfg.get("max_tokens", 192))
        self.system_prompt = system_prompt
        self.car_idx = car_idx
        self._llm_call_count = 0
        self.llm_raw_io_history = []
        self._reset_response_metadata()

    def response_schema(self):
        props = {n: {"type": "number", "minimum": float(f"{lo:.4g}"), "maximum": float(f"{hi:.4g}")}
                 for n, (lo, hi) in PARAMETER_BOUNDS.items()}
        return {"type": "object", "properties": props, "required": list(PARAMETER_NAMES),
                "additionalProperties": False}


class LLMPolicy(Policy):
    name = "llm"

    def __init__(self, car, cfg, world):
        assert car == 0, "the LLM drives the ego"
        super().__init__(car, theta_to_official(INITIAL_THETA))
        self.cfg = dict(cfg)
        self.interval_ticks = max(1, int(round(float(cfg.get("interval_s", 1.0)) / DT)))
        self.race_duration_s = float(cfg.get("race_duration_s", 50.0))
        self.include_current_parameters = bool(cfg.get("include_current_parameters", False))
        self.memory_length = int(cfg.get("memory_length", 0))  # 0: no recent_decisions block
        self.include_raceline_speed = bool(cfg.get("include_raceline_speed", True))
        module = cfg.get("prompt_module", "base")
        paths = [os.path.join(PROMPT_DIR, "params_system.txt")] + [
            os.path.join(PROMPT_DIR, f) for f in PROMPT_MODULE_FILES[module]]
        text = "\n\n".join(open(p, encoding="utf-8").read().strip() for p in paths)
        system_prompt = text.replace("{decision_interval_seconds}", f"{self.interval_ticks * DT:g}")
        self.prompt_info = {"module": module, "files": [os.path.relpath(p, PROMPT_DIR) for p in paths],
                            "sha256": hashlib.sha256(system_prompt.encode()).hexdigest()}
        self.client = _Client(self.cfg, system_prompt, car)
        self.world = world
        self.gen = world.generators[0]
        self.path = np.asarray(self.gen.path)  # centerline rows: arc length, x, y, speed
        self.lap = float(self.path[-1, 0])
        self.start_race(0, None, None)  # race 0 has no official reset; later races call this from Session

    def start_race(self, ep, draws, ego_beliefs):  # the official draws are consumed but not used
        self.ep = ep
        self.theta = list(INITIAL_THETA)
        self.own = theta_to_official(self.theta)
        self.prev = None
        self.decisions, self.recent, self.open_entry = [], [], None
        self.events = {"contact": False, "off_track": False}
        self.client._llm_call_count = 0
        self.client.seed = int(np.random.SeedSequence([int(self.cfg.get("seed", 0)), ep]).generate_state(1)[0])
        self.client.llm_raw_io_history = []

    # --- per-tick hooks (Session) ---
    def before_tick(self, session, ms):
        s = [float(m.s) for m in ms]
        e = [float(m.e) for m in ms]
        rates = self._rates(s, e)
        self.prev = (s, e)
        if (session.i - 1) % self.interval_ticks == 0:
            self._decide(session, ms, s, e, rates)

    def after_tick(self, session, ms, contact):
        self.events["contact"] |= contact[0] > 0. or contact[1] > 0.  # pairs (0,1), (0,2)
        self.events["off_track"] |= abs(float(ms[0].e)) > OFF_TRACK_E

    # --- observation ---
    def _wrap(self, d):
        return (d + self.lap / 2) % self.lap - self.lap / 2

    def _rates(self, s, e):
        if self.prev is None:
            return [(0., 0.)] * 3  # first tick of the race: cars start at rest
        ps, pe = self.prev
        return [(self._wrap(s[k] - ps[k]) / DT, (e[k] - pe[k]) / DT) for k in range(3)]

    def _road(self, s0):
        P, RL = self.path, self.world.raceline
        out = []
        for d in np.linspace(0., ROAD_PREVIEW_M, ROAD_SAMPLES):
            t = (s0 + d) % self.lap
            pos, _, info = custom_fn(t, P)
            nxt, _, _ = custom_fn((t + 0.1) % self.lap, P)
            x, y = float(pos[0]), float(pos[1])
            hx, hy = float(nxt[0]) - x, float(nxt[1]) - y
            n = math.hypot(hx, hy)
            j = int(np.argmin((RL[:, 0] - x) ** 2 + (RL[:, 1] - y) ** 2))
            sample = {"distance_ahead_m": _r(d, 1), "centerline_curvature_per_m": _r(info["curv"], 3),
                      "raceline_offset_m": _r(((RL[j, 0] - x) * -hy + (RL[j, 1] - y) * hx) / n, 3)}
            if self.include_raceline_speed:
                sample["raceline_speed_mps"] = _r(RL[j, 2], 2)
            out.append(sample)
        return out

    def observation(self, session, ms, s, e, rates):
        m = ms[0]
        traffic = [{"id": k, "relative_longitudinal_position_m": _r(self._wrap(s[k] - s[0]), 2),
                    "relative_lateral_position_m": _r(e[k] - e[0], 3),
                    "progress_rate_mps": _r(rates[k][0], 2), "lateral_rate_mps": _r(rates[k][1], 2)}
                   for k in (1, 2)]
        scene = {
            "race_time": {"elapsed_s": _r((session.i - 1) * DT, 1), "total_s": _r(self.race_duration_s, 1)},
            "ego_state": {"speed_mps": _r(max(m.vx, 0.), 2), "progress_rate_mps": _r(rates[0][0], 2),
                          "lateral_rate_mps": _r(rates[0][1], 2), "lateral_position_m": _r(e[0], 3),
                          "heading_error_rad": _r(-float(m.theta_diff), 3)},
            "traffic_state": {"all_opponents": traffic},
            "road_preview": {"samples_near_to_far": self._road(s[0])},
        }
        if self.include_current_parameters:
            scene["current_parameters"] = {n: _r(v) for n, v in zip(PARAMETER_NAMES, self.theta)}
        return scene

    # --- decision ---
    def _close_entry(self, session):
        entry = self.open_entry
        if entry is None or self.memory_length <= 0:
            return
        cum = session._cum
        self.recent.append({
            "parameters": entry["parameters"],
            "interval_seconds": _r((session.i - entry["tick"]) * DT, 2),
            "distance_travelled_m": {("ego" if k == 0 else f"car_{k}"): _r(cum[k] - entry["cum"][k], 2)
                                     for k in range(3)},
            "ego_touched_another_car": bool(self.events["contact"]),
            "ego_center_more_than_0.55_m_from_centerline": bool(self.events["off_track"]),
        })
        self.recent = self.recent[-self.memory_length:]

    def _decide(self, session, ms, s, e, rates):
        self._close_entry(session)
        observed = self.observation(session, ms, s, e, rates)
        payload = observed if self.memory_length <= 0 else {
            "recent_decisions": json.loads(json.dumps(self.recent)), "current_observation": observed}
        prompt = json.dumps(payload, indent=2, ensure_ascii=True)
        c = self.client
        c._llm_call_count = session.i  # request seed from (race seed, car, tick)
        c._reset_response_metadata()
        error = None
        try:
            response = c.call_llm(prompt)
            theta = [float(np.clip(float(response[n]), *PARAMETER_BOUNDS[n])) for n in PARAMETER_NAMES]
            if not all(np.isfinite(theta)):
                raise ValueError(f"nonfinite parameters {theta}")
            self.theta = theta
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        c._record_raw_io(prompt, error=None if error is None else error)
        self.own = theta_to_official(self.theta)
        io = c.llm_raw_io_history[-1]
        self.decisions.append({"tick": session.i, "theta": list(self.theta), "official": list(self.own),
                               "error": error, "latency_s": io["latency_seconds"], "seed": io["seed"],
                               "completion_tokens": io["eval_count"], "prompt_tokens": io["prompt_eval_count"]})
        self.open_entry = {"tick": session.i, "cum": list(session._cum),
                           "parameters": {n: _r(v) for n, v in zip(PARAMETER_NAMES, self.theta)}}
        self.events = {"contact": False, "off_track": False}

    def record(self):
        n_err = sum(d["error"] is not None for d in self.decisions)
        return {"params": list(map(float, self.own)), "theta": list(map(float, self.theta)),
                "decisions": len(self.decisions), "errors": n_err, "prompt": self.prompt_info,
                "llm_io": [dict(io, **d) for io, d in zip(self.client.llm_raw_io_history, self.decisions)]}
