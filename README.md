# racer-LLM

Research code for testing whether an LLM can select the five alpha-RACER
parameters `(q, zeta, s1, s2, s3)` while a constrained MPC retains throttle and
steering authority.

## Repository map

- `protocols/`: frozen experimental substrate and validation.
- `policy/`: each learned policy together with its training code and weights.
- `simulation_core/`: environment, observations, recording, and trace I/O.
- `evaluation/`: reusable evaluation and preflight programs.
- `track_csv/`: registered tracks and racelines.
- `visualization/`: trace plotting, animation, reports, and Unity replay tools.
- `docs/ai_agent_docs/`: indexed historical context for a new AI agent.
- `unity/`: replay project source; generated Unity caches are not versioned.

Read `AGENTS.md`, then `protocols/README.md`, before changing experiments.
Validate the frozen substrate with:

```bash
.venv/bin/python protocols/validate_k_constrained_v1.py
```

Run local interface checks with:

```bash
.venv/bin/python -m evaluation.local_preflight \
  --output-dir /tmp/racer_llm_preflight --track-seconds 0.1
```
