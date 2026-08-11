# Released alpha-RACER checkpoints

These are the unmodified weight files released by the alpha-RACER authors,
renamed for readability:

| Local filename | Released filename |
| --- | --- |
| `potential_full_horizon_mpc.pth` | `model_multi_mpc.pth` |
| `potential_myopic_mpc.pth` | `model_multi_myopic_mpc.pth` |
| `potential_myopic_variant1_mpc.pth` | `model_multi_myopic1_mpc.pth` |
| `potential_short_horizon_mpc.pth` | `model_multi_myopic_s_mpc.pth` |
| `potential_small_dataset_myopic_mpc.pth` | `model_multi_small_myopic_mpc.pth` |

They are raw state dictionaries and were trained with transformed parameter
features `(sf1, sf2, lookahead_factor, speed_factor, blocking)`. The runtime
recognizes raw checkpoints as upstream models and converts the repository's
canonical paper parameters `(q, zeta, s1, s2, s3)` at the network boundary.
New local checkpoints carry explicit `parameterization` metadata and require
no conversion.
