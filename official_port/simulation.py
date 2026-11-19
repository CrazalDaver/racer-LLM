"""Session: the official CarNode.timer_callback order as an explicit loop over races and ticks.
Per tick: measure all cars; for car 0, 1, 2 run the official MPC reference with the car's parameters,
take the policy override (IBR), apply recovery/boost and step; apply contact decay; update policies
(ego, opp, opp1) from the 24 measured features. Race start: official draws (5 per car, then the ego's
2 belief rows) from the session RandomState, which consumes them in the official np.random order."""
import random

import numpy as np
import torch

from .config import RaceConfig
from .controller import OfficialController
from .policies import make_policy
from .world import EP_LEN, TRACK_LENGTH_WRAP, World, features, start_group


def _draw_params(rng):
    return [rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.5), rng.uniform(0.12, 0.5), rng.uniform(0.85, 1.1),
            rng.uniform(0., 1.0)]


class Session:
    def __init__(self, config: RaceConfig):
        self.config = config
        seed = config.seed
        self.rng = np.random.RandomState(seed)
        if seed is not None:  # run.py also seeds these; the race logic does not draw from them
            random.seed(seed)
            torch.manual_seed(seed)
        self.world = World()
        self.ctrl = OfficialController(self.world.raceline, self.world.raceline_dev)
        specs = (config.ego, config.opp1, config.opp2)
        self.policies = [make_policy(k, s, config.fixed_params if k == 0 else None) for k, s in enumerate(specs)]
        self.ep_no = 0
        self.i = 0
        self.last_i = [-1, -1, -1]  # official per-car raceline search start for CarNode.mpc
        self.n_wins = [0, 0, 0]
        if config.start_ep > 0:
            self.ep_no = config.start_ep - 1
            self._next_race()

    def _next_race(self):
        """Official reset block (without the win count, done in _finish_race)."""
        self.last_i = [-1, -1, -1]
        self.ep_no += 1
        self.world.reset(self.ep_no)
        draws = [_draw_params(self.rng) for _ in range(3)]
        ego_beliefs = [_draw_params(self.rng) for _ in range(2)]
        for p in self.policies:
            p.start_race(self.ep_no, draws, ego_beliefs)

    def _tick(self, rec):
        ms = self.world.measure()
        s_now = [float(m.s) for m in ms]
        if self._last_s is not None:  # unwrap the lap for cumulative progress (noros record rule)
            for k in range(3):
                d = s_now[k] - self._last_s[k]
                d = d - TRACK_LENGTH_WRAP if d > 75. else (d + TRACK_LENGTH_WRAP if d < -75. else d)
                self._cum[k] += d
        self._last_s = s_now
        used, actions = [], []
        for k in range(3):
            o1, o2 = [c for c in range(3) if c != k]
            m, a, b = ms[k], ms[o1], ms[o2]
            p = list(self.policies[k].params())
            steer, throttle, _, _, self.last_i[k] = self.ctrl.mpc(
                (m.px, m.py, m.psi), (m.s, m.e, m.vx), (a.s, a.e, a.vx), (b.s, b.e, b.vx),
                p[0], p[1], p[2]*2, p[3]**2, p[4], last_i=self.last_i[k])
            override = self.policies[k].action_override(ms, self.world.raceline)
            actions.append(self.world.recover_and_step(k, m, steer, throttle, override))
            used.append(p)
        contact = self.world.apply_contacts(ms, self.ctrl)
        feats = features(ms, self.world.obs)
        for p in self.policies:
            p.update(self.i, feats)
        if rec is not None:
            rec["state"].append([[m.px, m.py, m.psi, m.vx, m.vy, m.omega] for m in ms])
            rec["s"].append(s_now)
            rec["e"].append([float(m.e) for m in ms])
            rec["cum"].append(list(self._cum))
            rec["params"].append(used)
            rec["action"].append([[float(v) for v in act] for act in actions])
            rec["contact"].append(contact)
        return ms

    def run_race(self):
        """One race of EP_LEN ticks (the current ep_no); returns the summary and per-tick arrays."""
        self._last_s, self._cum = None, [0., 0., 0.]
        rec = {k: [] for k in ("state", "s", "e", "cum", "params", "action", "contact")} if self.config.save_ticks else None
        start_params = [list(map(float, p.params())) for p in self.policies]
        for i in range(1, EP_LEN + 1):
            self.i = i
            self._tick(rec)
        summary = self._finish_race(start_params)
        return summary, ({k: np.asarray(v) for k, v in rec.items()} if rec is not None else None)

    def _finish_race(self, start_params):
        s = list(self._last_s)  # official win rule uses s measured at the last tick (before its step)
        s_official = [v + 150.1 if v < 50. else v for v in s]
        winner_official = None
        for k in range(3):
            if all(s_official[k] > s_official[j] for j in range(3) if j != k):
                winner_official = k
        if winner_official is not None:
            self.n_wins[winner_official] += 1
        return {"ep_no": self.ep_no, "group": start_group(self.ep_no), "s": s, "progress": list(self._cum),
                "winner_official_rule": winner_official, "winner_progress": int(np.argmax(self._cum)),
                "n_wins_official": list(self.n_wins), "start_params": start_params,
                "end": [p.record() for p in self.policies], "policies": [p.name for p in self.policies]}

    def run(self, on_race=None):
        out = []
        for r in range(self.config.n_races):
            if r > 0:
                self._next_race()
            summary, ticks = self.run_race()
            out.append(summary)
            if on_race is not None:
                on_race(summary, ticks)
        return out
