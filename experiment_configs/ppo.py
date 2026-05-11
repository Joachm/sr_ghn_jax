from __future__ import annotations

import inspect
from typing import Any

from baselines.ppo.config import make_ppo_config_nonstationary_gymnax
from experiments._adaptation import gymnax_suite_shift_windows


PPO_RUN_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "debug": {
        "num_generations": 4,
        "episode_horizon": 32,
        "episodes_per_eval": 1,
        "num_envs": 4,
        "rollout_length": 16,
        "num_minibatches": 1,
        "update_epochs": 1,
    },
}
PPO_RUN_PRESET_NAMES = tuple(PPO_RUN_PRESETS)


def _builder_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, param in inspect.signature(make_ppo_config_nonstationary_gymnax).parameters.items():
        if param.default is inspect._empty:
            continue
        defaults[name] = param.default
    return defaults


def resolve_ppo_nonstationary_gymnax_config(
    env_id: str,
    *,
    task_preset: str | None = "suite_default",
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    preset_name = "default" if run_preset in (None, "") else run_preset
    if preset_name not in PPO_RUN_PRESETS:
        raise ValueError(f"Unknown PPO run preset: {preset_name}")
    if task_preset not in (None, "", "suite_default"):
        raise ValueError(f"Unknown PPO task preset: {task_preset}")
    values = _builder_defaults()
    values["shift_windows"] = gymnax_suite_shift_windows(env_id)
    values.update(PPO_RUN_PRESETS[preset_name])
    values.update(overrides or {})
    return make_ppo_config_nonstationary_gymnax(env_id, **values)
