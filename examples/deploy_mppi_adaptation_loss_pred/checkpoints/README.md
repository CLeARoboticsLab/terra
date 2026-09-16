# Example checkpoints (dummy — not for deployment)

The directories in this folder are **Orbax training checkpoints** committed so the
deployment example can run without access to your training cluster or robot logs.

| Directory | Model | Training |
|-----------|--------|----------|
| `behavior_cloning/` | BC policy (`behavior_cloning_with_pose_hist`) | 2 optimizer steps on synthetic dummy data |
| `world_model/` | Dynamics / world model (`world_model`) | 2 optimizer steps on synthetic dummy data |

**These weights are not trained on real trajectories and must not be used on hardware.**
They exist only to demonstrate checkpoint layout, `load_model_from_ckpt`, and the
control/adaptation API.

Each checkpoint directory contains:

- `config.yaml` — full Hydra training config (model, system, scaler struct, etc.)
- Orbax step folders (e.g. `2/`) with saved parameters

To regenerate these artifacts from scratch:

```bash
python scripts/generate_example_deploy_checkpoints.py
```

To use your own models, train with `scripts/train.py` and pass
`--bc-checkpoint` / `--world-model-checkpoint` to the example script.
