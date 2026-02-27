# sr_ghn_jax

## Running experiments

Run experiments from the repo root so imports resolve:

```bash
python -m experiments.cartpole_switch
python -m experiments.ant_brax --backend spring
python -m experiments.gymnax_generic --env-id CartPole-v1 --children-per-parent 2
python -m experiments.brax_generic --env-id ant --backend spring --children-per-parent 2
python -m experiments.mujoco_playground_generic --env-id humanoid:run --children-per-parent 2
```

For `mujoco_playground` runs, the final artifacts directory now also includes:
- `champion_rollout.mp4`: automatic rollout recording of the final population champion.
- `champion_rollout.json`: champion index/fitness and recording metadata.
