"""Direct-command RL policy in the official world, and its self-play training race.

Input: the official evaluation code's 15-value rl_obs (multi_car_comp_blocking.py, OPP_Stretegy == 'rl')
  [s_k - s_r, e_k, e_r, theta_diff_k, vx_k, vy_k, omega_k, theta_diff_r, vx_r, vy_r, omega_r,
   curv_k, curv_r, curv_lookahead_k, curv_lookahead_r]
for car k and rival r (body-frame velocities, generator s/e/curvature). The official PPO weights were
not released, so this input is defined for training and evaluation alike with three port choices
(notes R1-R3): all values come from the tick-start measurement (the official code reads the
post-step velocities of cars that already moved this tick); car 2 against the ego uses its own
curvatures (the official line copies car 1's); s_k - s_r is lap-wrapped at +-75 m with 150.087 m
like the alpha features (the official difference is raw). The rival follows the official rule:
car 1 takes the ego if |s0 - s1| < |s1 - s2|, else car 2 (symmetrically for car 2); car 0 takes
the nearer of cars 1 and 2. The command is executed as is (the environment clips it); the official
recovery still decays velocity and turns the heading when |e| > .55, and the throttle boost and
recovery steering apply only to MPC commands, so they do not touch it.
"""
import copy

import numpy as np

from .controller import OfficialController
from .policies import Policy
from .world import TRACK_LENGTH_WRAP, World, start_poses


def rival_of(ms, k):
    if k == 0:
        return 1 if abs(ms[0].s - ms[1].s) < abs(ms[0].s - ms[2].s) else 2
    j = 3 - k
    return 0 if abs(ms[0].s - ms[k].s) < abs(ms[k].s - ms[j].s) else j


def rl_observation(ms, k):
    r = rival_of(ms, k)
    a, b = ms[k], ms[r]
    ds = float(a.s) - float(b.s)
    ds = ds - TRACK_LENGTH_WRAP if ds > 75. else (ds + TRACK_LENGTH_WRAP if ds < -75. else ds)
    return np.asarray([ds, a.e, b.e, a.theta_diff, a.vx, a.vy, a.omega, b.theta_diff, b.vx, b.vy, b.omega,
                       a.curv, b.curv, a.curv_lookahead, b.curv_lookahead], dtype=np.float32)


class RLPolicy(Policy):
    """Trained shared actor (deterministic mean). Its parameter row is unused (no MPC call)."""
    name = "rl"
    uses_mpc = False

    def __init__(self, car, init, weight_path):
        super().__init__(car, init)
        from policy.self_play_rl.network import load_actor_checkpoint
        self.weight_path = weight_path
        _, self.actor, self.metadata = load_actor_checkpoint(weight_path)
        assert self.metadata.get("race_class") == "official_port.rl:PortTrainingRace", \
            f"{weight_path} was not trained in the official world"

    def action_override(self, ms, raceline):
        action, _ = self.actor.predict(rl_observation(ms, self.car), deterministic=True)
        return [float(action[0]), float(action[1])]


class PortTrainingRace:
    """Self-play race for policy/self_play_rl/train.py (--race-class official_port.rl:PortTrainingRace).
    Starts: one official start group drawn uniformly, its three poses assigned to the cars in a random
    order. Reward: change of the learning car's progress minus the furthest-ahead opponent's
    (relative_progress, the local definition), on lap-unwrapped generator progress."""

    def __init__(self, setup, seed):
        self.setup = setup
        self.rng = np.random.default_rng(seed)
        self.world = World()
        self.ctrl = OfficialController(self.world.raceline, self.world.raceline_dev)
        self.episode_steps = int(setup["self_play"]["episode_steps"])
        assert setup["self_play"]["num_agents"] == 3
        self.reset()

    def reset(self):
        ep = int(self.rng.choice([0, 34, 67]))
        poses, _ = start_poses(ep)
        perm = self.rng.permutation(3)
        for g in self.world.generators:
            g.last_i = -1
        for k in range(3):
            self.world.obs[k] = self.world.envs[k].reset(pose=poses[perm[k]])
        self.step_count = 0
        self.cum = np.zeros(3)
        self.ms = self.world.measure()
        self.last_s = [float(m.s) for m in self.ms]

    def observations(self):
        return [rl_observation(self.ms, k) for k in range(3)]

    def step(self, actions, learning_car):
        from policy.utils import relative_progress
        prev = relative_progress(self.cum, learning_car)
        ms = self.ms
        for k, act in enumerate(actions):
            self.world.recover_and_step(k, ms[k], 0., 0., override=[act.throttle, act.steer])
        contact = self.world.apply_contacts(ms, self.ctrl)
        self.step_count += 1
        self.ms = self.world.measure()
        s_now = [float(m.s) for m in self.ms]
        for k in range(3):
            d = s_now[k] - self.last_s[k]
            self.cum[k] += d - TRACK_LENGTH_WRAP if d > 75. else (d + TRACK_LENGTH_WRAP if d < -75. else d)
        self.last_s = s_now
        reward = relative_progress(self.cum, learning_car) - prev
        next_observation = rl_observation(self.ms, learning_car)
        episode_end = self.step_count >= self.episode_steps
        metrics = {"offroad": int(sum(abs(m.e) > 0.55 for m in ms)), "contact": int(any(c > 0. for c in contact)),
                   "learning_recovery": bool(abs(ms[learning_car].e) > 0.55)}
        if episode_end:
            self.reset()
        return next_observation, float(reward), episode_end, metrics

    _SNAP = ("step_count", "cum", "ms", "last_s")

    def snapshot(self):
        w = self.world
        return copy.deepcopy({
            "rng_state": self.rng.bit_generator.state, **{k: getattr(self, k) for k in self._SNAP},
            "env": [(e.state, e._step, e.sim.batch_buffer) for e in w.envs], "obs": w.obs,
            "gen_last_i": [g.last_i for g in w.generators]})

    def restore(self, snap):
        snap = copy.deepcopy(snap)
        self.rng.bit_generator.state = snap["rng_state"]
        for k in self._SNAP:
            setattr(self, k, snap[k])
        for e, (st, step, buf) in zip(self.world.envs, snap["env"]):
            e.state, e._step, e.sim.batch_buffer = st, step, buf
        self.world.obs = snap["obs"]
        for g, li in zip(self.world.generators, snap["gen_last_i"]):
            g.last_i = li
