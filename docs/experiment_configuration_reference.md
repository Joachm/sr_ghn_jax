# Experiment Configuration Reference

This repo now uses a centralized training-config layer in `experiment_configs/`.

## Common Conventions

- Training entrypoints keep their existing commands, but config resolution now follows:
  `family defaults -> task preset -> baseline preset -> run preset -> CLI overrides`
- Training scripts support `--print-config` to show the resolved config without running training.
- Run artifacts include a `resolved_config` payload alongside legacy fields.

## Training Families

### Evolutionary Control

- Entry points: `experiments/gymnax_generic.py`, `experiments/brax_generic.py`, `experiments/nonstationary_gymnax.py`, `experiments/nonstationary_brax.py`, `experiments/mujoco_playground_generic.py`
- Run presets: `default`, `debug`
- Baselines: `srghn_full`, `frozen_mutation`, `fixed_lr`, `no_self_reference`
- Nonstationary Gymnax task presets: `cartpole_flip`, `cartpole_flip_revert`, `cartpole_repeated`, `suite_default`, `minatar_suite`
- Nonstationary Brax task presets: `brax_direction_switch`, `brax_speed_target_switch`
- Recommended command:
  `python experiments/nonstationary_gymnax.py --env-id CartPole-v1 --task-preset cartpole_flip --baseline srghn_full --print-config`

### Meta-Learning

- Entry points: `meta_sine_srghn.py`, `meta_brax_heading.py`
- Meta-sine run presets: `default`, `fast`
- Meta-Brax run presets: `default`, `fast`
- Recommended command:
  `python meta_brax_heading.py --conditions srghn_full --run-preset fast --print-config`

### PPO

- Entry point: `baselines/ppo/suite.py`
- Run presets: `default`, `debug`
- Task preset: `suite_default`
- Recommended command:
  `python baselines/ppo/suite.py --run-preset debug --print-config`
