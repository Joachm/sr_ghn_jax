# sr_ghn_jax

Centralized experiment configuration now lives under `experiment_configs/`. For the canonical preset names, resolved-config behavior, and recommended launch patterns, see [docs/experiment_configuration_reference.md](/Users/jwin/Library/CloudStorage/OneDrive-ITU/desktop/projects/sr_ghn_jax/docs/experiment_configuration_reference.md).

## Running experiments

Run experiments from the repo root so imports resolve:

```bash
python -m experiments.cartpole_switch
python -m experiments.ant_brax --backend spring
python -m experiments.gymnax_generic --env-id CartPole-v1 --children-per-parent 2
python -m experiments.brax_generic --env-id ant --backend spring --children-per-parent 2
python -m experiments.mujoco_playground_generic --env-id humanoid:run --children-per-parent 2
python -m experiments.mujoco_playground_suite --env-id humanoid:run
python -m experiments.nonstationary_gymnax --variant cartpole_flip --baseline srghn_full
python -m experiments.nonstationary_gymnax --variant cartpole_flip --optimizer-family evosax --evosax-algo OpenES
python -m experiments.nonstationary_brax --variant brax_direction_switch --baseline srghn_full
python -m experiments.adaptation_compare --suite cartpole --evosax-algos OpenES CMA_ES
python plot_adaptation_results.py --input adaptation_compare_cartpole.pkl
bash run_gymnax_nonstationary_suite.sh --skip-existing
python -m experiments.gymnax_nonstationary_suite --optimizer-family evosax --evosax-algo OpenES --skip-existing
bash run_gymnax_minatar_suite.sh --skip-existing
python -m experiments.gymnax_minatar_suite --optimizer-family evosax --evosax-algo OpenES --skip-existing
python -m baselines.ppo.suite --skip-existing
```

MuJoCo Playground runs now also save a replayable solution artifact:

```bash
python -m experiments.mujoco_playground_generic --env-id cheetah:run
python render_mujoco_playground_solution.py \
  --solution mujoco_playground_cheetah_run_solution.pkl \
  --output cheetah_run.mp4
```

To run the same stationary MuJoCo Playground environment 10 times with different seeds in one command:

```bash
bash run_mujoco_playground_suite.sh --env-id humanoid:run
```
