"""Per-car high-level policies. Every car drives the official MPC reference (controller.py) with five
parameters in official coordinates (sf1, sf2, lookahead, speed, blocking); the MPC call receives
lookahead*2 and speed**2. Policies differ in how those parameters evolve and whether the command is
replaced (IBR). Parameter state lives in the policy object for the whole session, as on the official
CarNode (an alpharacer opponent's beliefs about the other two cars persist across races)."""
import os

import numpy as np
import torch

from .ibr import mpc_ibr
from .vendor.model_arch import SimpleModel
from .world import EP_LEN, TRACK_LENGTH_WRAP

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "weights", "p_models_rel")
GRAD_RATE = 0.00001  # official grad_rate
N_STEPS_PER_ITER = 6
BOUNDS = [(0.1, 0.5), (0.1, 0.5), (0.15, 0.5), (0.85, 1.1), (0., 1.)]  # official clip ranges per parameter
INIT_EGO = [0.4, 0.2, 0.35, 0.93, 0.]  # CarNode constructor values (race 0)
INIT_OPP = [0.3, 0.2, 0.25, 0.87, 0.]
# Strategy -> (weight file, S): online updates run while tick i < EP_LEN - S. The official ego uses the
# myopic model with i < 450 (S = 50); opponent variants as in the official --opp1/--opp2 choices.
ALPHA_MODELS = {
    "alpharacer": ("model_multi_myopic_mpc.pth", 50),
    "ours-low_data": ("model_multi_small_myopic_mpc.pth", 50),
    "ours-low_p": ("model_multi_myopic_s_mpc.pth", 10),
    "ours-high_p": ("model_multi_mpc.pth", 250),
}


def load_model(fname):
    fs = 128
    model = SimpleModel(39, [3*fs, 3*fs, 3*64], 1)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS, fname), map_location="cpu"))
    model.eval()
    return model


def model_input(state_obs):
    """Official 39-feature tensor: progress relative to the ego (opp1 - opp, opp - ego, opp1 - ego), lap-wrapped."""
    X = torch.tensor(np.array([state_obs])).float()
    X[:,0] = float(state_obs[2] - state_obs[1])
    X[:,1] -= float(state_obs[0])
    X[:,2] -= float(state_obs[0])
    W = TRACK_LENGTH_WRAP
    X[:,0] = (X[:,0]>75.)*(X[:,0]-W) + (X[:,0]<-75.)*(X[:,0]+W) + (X[:,0]<=75.)*(X[:,0]>=-75.)*X[:,0]
    X[:,1] = (X[:,1]>75.)*(X[:,1]-W) + (X[:,1]<-75.)*(X[:,1]+W) + (X[:,1]<=75.)*(X[:,1]>=-75.)*X[:,1]
    X[:,2] = (X[:,2]>75.)*(X[:,2]-W) + (X[:,2]<-75.)*(X[:,2]+W) + (X[:,2]<=75.)*(X[:,2]>=-75.)*X[:,2]
    return X


def grad_step(model, feats, joint):
    """One official ascent step on all 15 joint parameters (rows: ego, opp, opp1), clipped per parameter."""
    X = model_input(feats + joint)
    X = torch.autograd.Variable(X, requires_grad=True)
    model.zero_grad()
    preds = model(X)
    grad = torch.autograd.grad(preds[0,0],X, retain_graph=True)[0].data
    out = []
    for j in range(15):
        lo, hi = BOUNDS[j % 5]
        idx = -15 + j
        out.append(max(lo,min(hi,X[0,idx].item() + GRAD_RATE*grad[0,idx].item())))
    return out


class Policy:
    """Fixed parameters for the race; the official 'mpc' opponent (random draw per race)."""
    name = "mpc"

    def __init__(self, car, init):
        self.car = car
        self.own = list(init)

    def params(self):
        return self.own

    def start_race(self, ep, draws, ego_beliefs):
        """draws: 3x5 official per-race draws (row = car); ego_beliefs: the ego's 2 belief rows."""
        self.own = list(draws[self.car])

    def action_override(self, ms, raceline):
        return None

    def update(self, i, feats):
        pass

    def before_tick(self, session, ms):
        """Called after the tick-start measurement, before any car's MPC call (LLM decisions)."""

    def after_tick(self, session, ms, contact):
        """Called after the contact rule with the three pair costs (0,1), (0,2), (1,2)."""

    def record(self):
        return {"params": list(map(float, self.params()))}


class FixedParams(Policy):
    """Constant parameters in every race (noros FIXED_EGO semantics: draws still consumed)."""
    name = "fixed"

    def __init__(self, car, params):
        super().__init__(car, params)
        self.fixed = list(params)

    def start_race(self, ep, draws, ego_beliefs):
        self.own = list(self.fixed)


class AlphaRacer(Policy):
    """Official alpharacer: 3x5 joint parameters (own row real, other rows beliefs), six gradient ascent
    steps of the learned value model per tick while i < EP_LEN - S."""

    def __init__(self, car, strategy):
        fname, self.S = ALPHA_MODELS[strategy]
        self.name = strategy
        self.model = load_model(fname)
        self.car = car
        self.joint = INIT_EGO + INIT_OPP + INIT_OPP  # flat 15, official initial values of every slot

    def params(self):
        return self.joint[5*self.car:5*self.car+5]

    def start_race(self, ep, draws, ego_beliefs):
        if self.car == 0:  # the ego redraws its own row and both beliefs
            self.joint = list(draws[0]) + list(ego_beliefs[0]) + list(ego_beliefs[1])
        else:  # an opponent redraws only its own row; its beliefs carry over (official)
            self.joint[5*self.car:5*self.car+5] = list(draws[self.car])

    def update(self, i, feats):
        if not i < EP_LEN - self.S:  # ego: i < 450 (S = 50); opponents: i < 500 - S
            return
        for _ in range(N_STEPS_PER_ITER):
            self.joint = grad_step(self.model, feats, self.joint)

    def record(self):
        return {"params": list(map(float, self.params())), "joint": list(map(float, self.joint))}


class IBR(Policy):
    """Official IBR opponent: commands from mpc_ibr against the nearer of the other two cars in s.
    Its own parameters (random draw) still drive the official MPC call, whose command is discarded."""
    name = "ibr"

    def action_override(self, ms, raceline):
        k = self.car
        assert k in (1, 2), "official IBR is defined for opponents only"
        j = 3 - k
        m = ms[k]
        dists = np.sqrt((raceline[:,0]-m.px)**2 + (raceline[:,1]-m.py)**2)
        closest_idx = np.argmin(dists)

        def x0(c):
            return [ms[c].s, ms[c].e, -ms[c].theta_diff, ms[c].vx] + ms[c].lookaheads + [raceline[closest_idx,2]]*10

        rival = 0 if abs(ms[0].s - m.s) < abs(m.s - ms[j].s) else j
        _, u = mpc_ibr(x0(rival), x0(k))
        return u


def make_policy(car, spec, fixed_params=None, weight_path=None, llm=None, world=None):
    if spec == "mpc":
        return Policy(car, INIT_EGO if car == 0 else INIT_OPP)
    if spec == "fixed":
        assert fixed_params is not None and len(fixed_params) == 5
        return FixedParams(car, fixed_params)
    if spec in ALPHA_MODELS:
        if car == 0:
            assert spec == "alpharacer", "the ego alpharacer is the official myopic model"
        else:
            assert spec != "alpharacer", "opponent alpharacer variants: ours-low_data, ours-low_p, ours-high_p"
        return AlphaRacer(car, spec)
    if spec == "ibr":
        return IBR(car, INIT_OPP)
    if spec == "llm":
        from .llm import LLMPolicy
        return LLMPolicy(car, llm, world)
    if spec == "rl":
        from .rl import RLPolicy
        assert weight_path, "rl needs a trained actor checkpoint"
        return RLPolicy(car, INIT_EGO if car == 0 else INIT_OPP, weight_path)
    raise ValueError(f"unknown policy {spec!r}")
