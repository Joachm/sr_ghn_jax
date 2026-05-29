from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetaSineConfig:
    seed: int = 0
    outer_generations: int = 300
    meta_batch_size: int = 8
    test_task_batch_size: int = 128
    outer_pop_size: int = 32
    outer_children_per_parent: int = 1
    inner_pop_size: int = 8
    inner_children_per_parent: int = 1
    inner_generations: int = 3
    support_k: int = 5
    query_k: int = 25
    amplitude_min: float = 0.1
    amplitude_max: float = 5.0
    phase_max: float = math.pi
    x_min: float = -5.0
    x_max: float = 5.0
    policy_hidden_dims: tuple[int, ...] = (32, 32)
    embedding_dim: int = 32
    gnn_hidden_dim: int = 32
    gnn_steps_policy: int = 4
    gnn_steps_self: int = 4
    stoch_coeff_dim: int = 32
    parameter_block_size: int = 64
    mutation_block_ratio: float = 0.25
    mutation_rate_head_dim: int = 4
    clip_params: tuple[float, float] = (-20.0, 20.0)
    clip_std: tuple[float, float] = (0.0, 2.0)
    clip_update: tuple[float, float] = (-0.25, 0.25)
    const_noise_std: float = 1e-3
    vector_ga_sigma: float = 0.05
    vector_ga_init_scale: float = 0.1
    wandb_project: str | None = "meta_sine_srghn"
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_log_plots: bool = False
    outer_evosax_sigma_init: float | None = 0.05
    inner_evosax_sigma_init: float | None = 0.05


@dataclass(frozen=True)
class MetaBraxConfig:
    env_id: str = "ant"
    brax_backend: str | None = "spring"
    seed: int = 0
    outer_generations: int = 1500
    meta_batch_size: int = 12
    heldout_task_batch_size: int = 16
    outer_pop_size: int = 16
    outer_children_per_parent: int = 2
    inner_pop_size: int = 4
    inner_children_per_parent: int = 2
    inner_generations: int = 4
    support_episodes: int = 2
    query_episodes: int = 2
    episode_horizon: int = 1000
    policy_hidden_dims: tuple[int, ...] = (32, 32, 32)
    embedding_dim: int = 32
    gnn_hidden_dim: int = 32
    gnn_steps_policy: int = 10
    gnn_steps_self: int = 10
    stoch_coeff_dim: int = 32
    parameter_block_size: int = 32*32
    mutation_block_ratio: float = 1.
    mutation_rate_head_dim: int = 5
    clip_params: tuple[float, float] = (-20.0, 20.0)
    clip_std: tuple[float, float] = (0.0, 2.0)
    clip_update: tuple[float, float] = (-0.1, 0.1)
    const_noise_std: float = 1e-3
    baseline_fixed_mutation_lr: float = 0.02
    outer_evosax_sigma_init: float | None = 0.05
    inner_evosax_sigma_init: float | None = 0.05
    outer_evosax_std_decay: float | None = 1.0
    inner_evosax_std_decay: float | None = 1.0
    obs_norm_clip: float = 5.0
    obs_norm_eps: float = 1e-8
    wandb_project: str | None = "meta_brax_heading"
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_log_plots: bool = False


META_SINE_BASE_DEFAULTS: dict[str, Any] = {
    **MetaSineConfig().__dict__,
    "outer_generations": 1800,
    "mutation_block_ratio": 1.0,
    "wandb_project": "meta_sine_srghn_3",
}
META_BRAX_BASE_DEFAULTS: dict[str, Any] = dict(MetaBraxConfig().__dict__)

META_SINE_RUN_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {
        "outer_pop_size": 16,
        "outer_generations": 100,
        "meta_batch_size": 8,
        "test_task_batch_size": 64,
        "inner_pop_size": 6,
        "inner_generations": 2,
        "policy_hidden_dims": (16, 16),
        "embedding_dim": 16,
        "gnn_hidden_dim": 16,
        "gnn_steps_policy": 3,
        "gnn_steps_self": 3,
        "stoch_coeff_dim": 16,
        "parameter_block_size": 32,
        "mutation_rate_head_dim": 2,
    },
}
META_BRAX_RUN_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {
        "outer_generations": 2,
        "meta_batch_size": 6,
        "heldout_task_batch_size": 8,
        "outer_pop_size": 2,
        "inner_pop_size": 2,
        "inner_generations": 1,
        "support_episodes": 1,
        "query_episodes": 1,
        "episode_horizon": 32,
    },
}
META_SINE_RUN_PRESET_NAMES = tuple(META_SINE_RUN_PRESETS)
META_BRAX_RUN_PRESET_NAMES = tuple(META_BRAX_RUN_PRESETS)


def _build_config(base: dict[str, Any], presets: dict[str, dict[str, Any]], run_preset: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    preset_name = "default" if run_preset in (None, "") else run_preset
    if preset_name not in presets:
        raise ValueError(f"Unknown run preset: {preset_name}")
    values = dict(base)
    values.update(presets[preset_name])
    values.update(overrides)
    return values


def build_meta_sine_config(*, run_preset: str | None = None, overrides: dict[str, Any] | None = None) -> MetaSineConfig:
    return MetaSineConfig(**_build_config(META_SINE_BASE_DEFAULTS, META_SINE_RUN_PRESETS, run_preset, overrides or {}))


def build_meta_brax_config(*, run_preset: str | None = None, overrides: dict[str, Any] | None = None) -> MetaBraxConfig:
    return MetaBraxConfig(**_build_config(META_BRAX_BASE_DEFAULTS, META_BRAX_RUN_PRESETS, run_preset, overrides or {}))
