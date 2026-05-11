from __future__ import annotations

from experiment_configs.control import (
    BASELINE_PRESET_NAMES,
    CONTROL_RUN_PRESET_NAMES,
    NONSTATIONARY_BRAX_TASK_PRESET_NAMES,
    NONSTATIONARY_GYMNAX_TASK_PRESET_NAMES,
)
from experiment_configs.meta import META_BRAX_RUN_PRESET_NAMES, META_SINE_RUN_PRESET_NAMES
from experiment_configs.ppo import PPO_RUN_PRESET_NAMES


def render_experiment_catalog_markdown() -> str:
    return """# Experiment Configuration Reference

This repo now uses a centralized training-config layer in `experiment_configs/`.

## Common Conventions

- Training entrypoints keep their existing commands, but config resolution now follows:
  `family defaults -> task preset -> baseline preset -> run preset -> CLI overrides`
- Training scripts support `--print-config` to show the resolved config without running training.
- Run artifacts include a `resolved_config` payload alongside legacy fields.

## Training Families

### Evolutionary Control

- Entry points: `experiments/gymnax_generic.py`, `experiments/brax_generic.py`, `experiments/nonstationary_gymnax.py`, `experiments/nonstationary_brax.py`, `experiments/mujoco_playground_generic.py`
- Run presets: `""" + "`, `".join(CONTROL_RUN_PRESET_NAMES) + """`
- Baselines: `""" + "`, `".join(BASELINE_PRESET_NAMES) + """`
- Nonstationary Gymnax task presets: `""" + "`, `".join(NONSTATIONARY_GYMNAX_TASK_PRESET_NAMES) + """`
- Nonstationary Brax task presets: `""" + "`, `".join(NONSTATIONARY_BRAX_TASK_PRESET_NAMES) + """`
- Recommended command:
  `python experiments/nonstationary_gymnax.py --env-id CartPole-v1 --task-preset cartpole_flip --baseline srghn_full --print-config`

### Meta-Learning

- Entry points: `meta_sine_srghn.py`, `meta_brax_heading.py`
- Meta-sine run presets: `""" + "`, `".join(META_SINE_RUN_PRESET_NAMES) + """`
- Meta-Brax run presets: `""" + "`, `".join(META_BRAX_RUN_PRESET_NAMES) + """`
- Recommended command:
  `python meta_brax_heading.py --conditions srghn_full --run-preset fast --print-config`

### PPO

- Entry point: `baselines/ppo/suite.py`
- Run presets: `""" + "`, `".join(PPO_RUN_PRESET_NAMES) + """`
- Task preset: `suite_default`
- Recommended command:
  `python baselines/ppo/suite.py --run-preset debug --print-config`
"""
