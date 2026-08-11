# Local alpha-RACER training

This pipeline retains alpha-RACER's two-stage idea while replacing its ROS
dynamics, hard-coded track length, and implicit triplet data layout with this
repository's simulator and explicit unilateral-deviation groups.

The complete cached pipeline is intentionally kept in one readable module.
Run it from the repository root:

```bash
.venv/bin/python -m train.alpha_racer.alpha_potential_training \
  --groups 100 --duration 12
```

The first stage runs four races per group: one baseline joint parameter vector,
then one deviation for each of the three cars. The second stage fits each car's
discounted local return. The third stage enforces

`P(s, theta_i', theta_-i) - P(s, theta) = V_i(s, theta_i', theta_-i) - V_i(s, theta)`.

The default reward is progress minus configurable off-road and collision
penalties. The exact weights, track, dynamics, horizon, discount, and seed are
stored in both the dataset and checkpoint metadata.

Existing rollout data, all three value checkpoints, and the final potential
checkpoint are reused automatically. Use `--force-rollouts`, `--force-values`,
or `--force-potential` to rebuild a stage. Rebuilding an upstream stage also
rebuilds its dependent downstream stages.

Locally generated artifacts live under `train/data/alpha_racer/` and
`train/weights/alpha_racer/`. Released checkpoints live under
`train/weights/alpha_racer/upstream/`; the runtime automatically adapts their
legacy input coordinates while exposing only paper theta=(q,zeta,s1,s2,s3).

To evaluate a local model, set `SimParams.POTENTIAL_FUNCTION_WEIGHT_PATH` to
`train/weights/alpha_racer/potential_local.pth`. The runtime loader accepts
both the new metadata-bearing checkpoint and the original raw state dict.

The same choice can be made without editing configuration:

```bash
# Released checkpoint (legacy input adaptation is automatic)
./main.py --policies alpharacer alpharacer alpharacer \
  --alpharacer-weight train/weights/alpha_racer/upstream/potential_myopic_mpc.pth

# Locally trained checkpoint (paper coordinates are detected from metadata)
./main.py --policies alpharacer alpharacer alpharacer \
  --alpharacer-weight train/weights/alpha_racer/potential_local.pth
```

Both cases optimize normalized paper coordinates in `[0,1]^15`, reconstruct
theta=`(q,zeta,s1,s2,s3)` with the shared config bounds, and record the chosen
checkpoint path and detected parameterization in the simulation trace.
