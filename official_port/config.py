"""Session configuration for the official-logic race."""
from dataclasses import asdict, dataclass, field
from typing import Optional

OPPONENTS = ("mpc", "ours-low_data", "ours-low_p", "ours-high_p", "ibr", "rl")
EGOS = ("alpharacer", "fixed", "mpc", "rl", "llm")


@dataclass
class RaceConfig:
    opp1: str = "mpc"  # car 1 (official --opp1)
    opp2: str = "mpc"  # car 2 (official --opp2)
    ego: str = "alpharacer"
    fixed_params: Optional[list] = None  # ego 'fixed': (sf1, sf2, lookahead, speed, blocking), official coordinates
    rl_weights: Optional[str] = None  # actor checkpoint for every rl car
    llm: Optional[dict] = None  # ego 'llm' settings (official_port/llm.py)
    seed: Optional[int] = 0  # per-race draw stream (official run.py SEED); None = unseeded
    n_races: int = 101  # official session: races 0..100
    start_ep: int = 0  # > 0: jump to this race's start group with one reset (branch.py semantics), not the official sequence
    save_ticks: bool = True  # per-tick trace arrays
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.opp1 in OPPONENTS and self.opp2 in OPPONENTS, (self.opp1, self.opp2)
        assert self.ego in EGOS, self.ego
        assert (self.ego == "fixed") == (self.fixed_params is not None)
        assert ("rl" in (self.ego, self.opp1, self.opp2)) == (self.rl_weights is not None)
        assert (self.ego == "llm") == (self.llm is not None)
        assert self.n_races >= 1 and 0 <= self.start_ep

    def to_dict(self):
        return asdict(self)
