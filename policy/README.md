# Policy packages

- `alpharacer/`: online alpha-RACER policy, network, training pipeline, data,
  and weights.
- `llm/`: LLM policy and its prompts.
- `self_play_rl/`: self-play RL policy, trainer, and weights.
- `ibr/`: local IBR implementations.
- `mpc.py`, `base.py`, and `utils.py`: shared policy infrastructure.

Training artifacts live with the policy that consumes them; there is no
repository-wide `train/` package.
