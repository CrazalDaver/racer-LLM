# Local raceline generation

`optimize_raceline.py` adapts the geometry pipeline from
[TUMFTM/global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization)
to this project's registered tracks and local vehicle model.

The default is iterative minimum-curvature optimization at 0.2 m geometry
spacing, followed by a closed forward/backward speed profile using the same
throttle acceleration model, released MPC steering/friction proxy, and global
speed limit as the simulator. It intentionally does not import the upstream
minimum-time double-track/tire model.

The default keeps a documented 5 cm center-point margin inside the configured
track boundary. A separate numerical buffer of half the output sample spacing
absorbs periodic-spline interpolation overshoot; the exported path is then
checked against the requested 5 cm physical margin. The dense iterative solve
is required for the released `Sa=0.34` steering map; the coarser single-QP
solution does not satisfy its actual curvature bound.

Install the optional upstream dependency in the project virtual environment:

```bash
.venv/bin/pip install trajectory-planning-helpers
```

The package pins an obsolete `quadprog` build. On macOS/ARM, replace it with:

```bash
.venv/bin/pip install --upgrade --force-reinstall quadprog==0.1.13
```

Generate the recommended Berlin candidate:

```bash
.venv/bin/python train/raceline/optimize_raceline.py --track berlin
```

Outputs are written under `track_csv/generated/` as a simulator-compatible CSV,
a JSON provenance/metrics file, and a diagnostic PNG. Existing registered
racelines are not overwritten.

The validated Berlin candidate has been promoted to the registered path
`track_csv/berlin_2018_raceline_with_speeds.csv`. The earlier upstream-derived
CSV is retained as `track_csv/berlin_2018_raceline_with_speeds_history.csv`.
The superseded candidate generated with the erroneous local `Sa=0.80` setting
is retained as
`track_csv/berlin_2018_raceline_with_speeds_local_sa080_history.csv`.
Running this script again still writes a candidate under `generated/`, so a new
optimization cannot silently replace the registered baseline.

The final choice must be made with a closed-loop single-car MPC comparison;
offline estimated lap time alone is not sufficient.
