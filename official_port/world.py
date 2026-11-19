"""Official numerical world: three OffroadCar envs (jax DBM, delay 1), per-car WaypointGenerator
measurement, offroad recovery, throttle boost and contact velocity decay, in the official tick order."""
from dataclasses import dataclass
import os

import jax.numpy as jnp
import numpy as np

from car_dynamics.controllers_jax import WaypointGenerator
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams

DT = 0.1
DT_torch = 0.1
DELAY = 1
H = 9  # WaypointGenerator horizon: H+1 = 10 reference rows
EP_LEN = 500  # ticks per race
TRACK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "params-num.yaml")  # 'num' in the name selects the official path branch
TRACK_LENGTH_WRAP = 150.087  # lap length used by the official feature wrap (75 m threshold)

# Race 0 uses the module-level reset poses; later races use start_poses(ep) (official reset block).
INITIAL_POSES = ([3., 5., -np.pi/2.-0.72], [0., 0., -np.pi/2.-0.5], [-2., -6., -np.pi/2.-0.5])


def start_group(ep):
    return 1 if ep < 34 else (2 if ep < 67 else 3)  # G1 ego last, G2 ego middle, G3 ego first


def start_poses(ep):
    """Official reset poses per car (ego, opp, opp1) and the official reset call order."""
    if ep < 34:
        return [[3., 5., -np.pi/2.-0.72], [-2., -6., -np.pi/2.-0.5], [0., 0., -np.pi/2.-0.5]], (0, 2, 1)
    if ep < 67:
        return [[0., 0., -np.pi/2.-0.5], [-2., -6., -np.pi/2.-0.72], [3., 5., -np.pi/2.-0.72]], (2, 0, 1)
    return [[-2., -6., -np.pi/2.-0.72], [0., 0., -np.pi/2.-0.5], [3., 5., -np.pi/2.-0.72]], (2, 1, 0)


@dataclass
class CarMeasure:
    """One car's per-tick measurement, with the exact objects the official callback computes."""
    px: float
    py: float
    psi: float
    vx: float  # body-frame
    vy: float
    omega: float
    s: object  # raceline-generator arc length (jax scalar), m
    e: object  # signed lateral distance to the generator path, m
    target: object  # (H+1) x 5 jax rows: x, y, heading, curvature, speed
    lookaheads: list  # curvature column as Python floats (IBR input)
    theta_diff: object  # reference heading minus body heading, wrapped
    curv: object
    curv_lookahead: object


class World:
    def __init__(self):
        params = [DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=DELAY) for _ in range(3)]
        self.dynamics = [DynamicBicycleModel(p) for p in params]
        for d in self.dynamics:
            d.reset()
        self.generators = [WaypointGenerator(TRACK_YAML, DT, H, 2.), WaypointGenerator(TRACK_YAML, DT, H, 1.),
                           WaypointGenerator(TRACK_YAML, DT, H, 1.)]
        self.envs = [OffroadCar({}, d) for d in self.dynamics]
        self.obs = [env.reset(pose=p) for env, p in zip(self.envs, INITIAL_POSES)]  # last env.reset/step return per car
        self.raceline = self.generators[0].raceline
        self.raceline_dev = self.generators[0].raceline_dev

    def reset(self, ep):
        for g in self.generators:
            g.last_i = -1
        poses, order = start_poses(ep)
        for k in order:
            self.obs[k] = self.envs[k].reset(pose=poses[k])

    def measure(self):
        states = [env.obs_state().tolist() for env in self.envs]
        out = []
        for k, (env, gen, st) in enumerate(zip(self.envs, self.generators, states)):
            px, py, psi, vx, vy, omega = st
            target, _, s, e = gen.generate(jnp.array(self.obs[k][:5]), dt=DT_torch, mu_factor=1., body_speed=vx)
            # Official asymmetry kept for bit equality: the ego heading comes from the numpy copy of its
            # target rows, the opponents' from the jax rows (different scalar types in theta_diff).
            theta = np.array(target)[0, 2] if k == 0 else target[0, 2]
            theta_diff = np.arctan2(np.sin(theta-psi), np.cos(theta-psi))
            out.append(CarMeasure(px, py, psi, vx, vy, omega, s, e, target, target[:, 3].tolist(), theta_diff,
                                  target[0, 3], target[-1, 3]))
        return out

    def recover_and_step(self, k, m, steer, throttle, override=None):
        """Official recovery (|e| > .55) and heading throttle boost on car k, then one env step.
        override (IBR) replaces the MPC command after the recovery has modified the state."""
        env = self.envs[k]
        e = m.e
        if abs(e) > 0.55:
            env.state.vx *= np.exp(-3*(abs(e)-0.55))
            env.state.vy *= np.exp(-3*(abs(e)-0.55))
            env.state.psi += (1-np.exp(-(abs(e)-0.55)))*(m.theta_diff)
            steer += (-np.sign(e) - steer)*(1-np.exp(-3*(abs(e)-0.55)))
        if abs(m.theta_diff) > 1.:
            throttle += 0.2
        action = np.array([throttle, steer]) if override is None else np.array([override[0], override[1]])
        self.obs[k], _, _, _ = env.step(action)
        return action

    def apply_contacts(self, ms, ctrl):
        """Official contact rule from pre-step poses: overlap cost c, rear car velocity x exp(-20c), front x exp(-5c)."""
        pairs = ((0, 1), (0, 2), (1, 2))
        costs = [ctrl.has_collided(ms[a].px, ms[a].py, ms[a].psi, ms[b].px, ms[b].py, ms[b].psi) for a, b in pairs]
        for (a, b), c in zip(pairs, costs):
            d = ms[b].s - ms[a].s
            if d < -75.:
                d += 150.
            if d > 75.:
                d -= 150.
            fa, fb = (-20, -5) if d > 0. else (-5, -20)  # b ahead: a is the rear car
            sa, sb = self.envs[a].state, self.envs[b].state
            sa.vx *= np.exp(fa*c)
            sa.vy *= np.exp(fa*c)
            sb.vx *= np.exp(fb*c)
            sb.vy *= np.exp(fb*c)
        return [float(c) for c in costs]


def features(ms, obs):
    """First 24 entries of the official 39-feature state_obs (before the 15 joint parameters).
    Velocities come from the post-step env returns (obs), as in the official update block."""
    f = [ms[0].s, ms[1].s, ms[2].s, ms[0].e, ms[1].e, ms[2].e]
    for k in range(3):
        f += [ms[k].theta_diff, obs[k][3], obs[k][4], obs[k][5]]
    f += [ms[0].curv, ms[1].curv, ms[2].curv, ms[0].curv_lookahead, ms[1].curv_lookahead, ms[2].curv_lookahead]
    return f
