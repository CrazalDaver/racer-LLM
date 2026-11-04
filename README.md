# racer-LLM

Racing simulation for studying high-level decisions through the alpha-RACER
five-parameter interface `(q, zeta, s1, s2, s3)`.

This is an **in-progress DBM development update**. The environment executes a
six-state dynamic bicycle model; the MPC predicts four-state kinematic motion
in centerline Frenet coordinates. The substrate is still under review and is
not presented as an exact reproduction of the published experiments.

## Setup and a fixed-parameter run

Use Python 3.9 or later with a virtual environment. The pinned requirements
reflect the current local environment; package availability depends on the
Python version and platform.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --policies static_alpharacer static_alpharacer static_alpharacer \
    --vehicle-model dbm --track-name berlin --total-time 5
```

Run `python main.py --help` for available options. Registered tracks are
`berlin`, `modena`, and `rounded_rectangle`.

## Code layout

- `track/`: centerline CSVs, offline raceline optimization, generated racelines,
  and shared runtime projection/geometry.
- `vehicle_model.py`: vehicle equations and numerical integration.
- `mpc_controller.py`: constrained Frenet MPC definition and solution.
- `policy/`: reference construction and fixed, alpha-potential, LLM and IBR policies.
- `simulation.py`, `simulation_core/`: observation, action, environment stepping
  and trace recording.
- `policy/alpharacer/train/`: rollout, value and potential training pipeline.
- `visualization/`: trace plots, animations and HTML reports.

Fixed-parameter simulation has regression coverage. The inherited training,
LLM and IBR paths still require validation under the current DBM setting.
There is no frozen experiment protocol at this stage; runs retain configuration
and Git provenance. Local agent notes and disposable checks are not published.

## Source

The implementation builds on the
[alpha-RACER paper](https://people.eecs.berkeley.edu/~sastry/pubs/Pdfs%20of%202025/Kalariaa-RACER2025.pdf)
and its [official code](https://github.com/sastry-group/alpha-RACER), with local
controller and simulation adaptations. Soft track/collision constraints and
model mismatch mean successful MPC solves do not guarantee physical safety.
