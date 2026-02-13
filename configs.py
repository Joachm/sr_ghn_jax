from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    task_name: str
    seed: int
    pop_size: int
    children_per_parent: int
    num_generations: int
    episode_horizon: int
    episodes_per_eval: int
    embedding_dim: int
    gnn_hidden_dim: int
    gnn_steps_policy: int
    gnn_steps_self: int
    stoch_coeff_dim: int
    stoch_max_out: int
    mutation_rate_head_dim: int
    clip_params: tuple[float, float]
    clip_std: tuple[float, float]
    clip_update: tuple[float, float]
    const_noise_std: float
    mutation_scale_alpha: float
    mutation_norm_eps: float
    child_factor: float
    switch_gen_start: int | None
    switch_gen_end: int | None
    switch_rule: str | None
    env_backend: str
    env_id: str
    brax_backend: str | None


def make_config_cartpole_switch() -> ExperimentConfig:
    return ExperimentConfig(
        task_name="cartpole_switch",
        seed=0,
        pop_size=10,
        children_per_parent=2,
        num_generations=1500,
        episode_horizon=500,
        episodes_per_eval=1,
        embedding_dim=32,
        gnn_hidden_dim=32,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        stoch_max_out=1024,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-1.0, 1.0),
        const_noise_std=0.001,
        mutation_scale_alpha=0.05,
        mutation_norm_eps=1e-8,
        child_factor=0.0,
        switch_gen_start=600,
        switch_gen_end=1200,
        switch_rule="cartpole_flip",
        env_backend="gymnax",
        env_id="CartPole-v1",
        brax_backend=None,
    )


def make_config_ant_brax() -> ExperimentConfig:
    return ExperimentConfig(
        task_name="ant_brax",
        seed=0,
        pop_size=50,
        children_per_parent=2,
        num_generations=1000,
        episode_horizon=1000,
        episodes_per_eval=1,
        embedding_dim=32,
        gnn_hidden_dim=32,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        stoch_max_out=1024,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-.1, .1),
        const_noise_std=0.001,
        mutation_scale_alpha=0.05,
        mutation_norm_eps=1e-8,
        child_factor=0.0,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        env_backend="brax",
        env_id="ant",
        brax_backend=None,
    )


def make_config_gymnax_generic(
    env_id: str,
    *,
    seed: int = 0,
    pop_size: int = 30,
    num_generations: int = 300,
    episode_horizon: int = 500,
    children_per_parent: int = 2,
    episodes_per_eval: int = 1,
    child_factor: float = 0.0,
) -> ExperimentConfig:
    return ExperimentConfig(
        task_name="gymnax_generic",
        seed=seed,
        pop_size=pop_size,
        children_per_parent=children_per_parent,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        embedding_dim=32,
        gnn_hidden_dim=32,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        stoch_max_out=1024,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-1.0, 1.0),
        const_noise_std=0.001,
        mutation_scale_alpha=0.05,
        mutation_norm_eps=1e-8,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        env_backend="gymnax",
        env_id=env_id,
        brax_backend=None,
    )


def make_config_brax_generic(
    env_id: str,
    *,
    seed: int = 0,
    pop_size: int = 50,
    num_generations: int = 1000,
    episode_horizon: int = 1000,
    children_per_parent: int = 2,
    episodes_per_eval: int = 1,
    child_factor: float = 0.0,
    brax_backend: str | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        task_name="brax_generic",
        seed=seed,
        pop_size=pop_size,
        children_per_parent=children_per_parent,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        embedding_dim=32,
        gnn_hidden_dim=32,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        stoch_max_out=1024,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-0.1, 0.1),
        const_noise_std=0.001,
        mutation_scale_alpha=0.05,
        mutation_norm_eps=1e-8,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        env_backend="brax",
        env_id=env_id,
        brax_backend=brax_backend,
    )


def make_config_mujoco_playground_generic(
    env_id: str,
    *,
    seed: int = 0,
    pop_size: int = 200,
    num_generations: int = 20000,
    episode_horizon: int = 1000,
    children_per_parent: int = 4,
    episodes_per_eval: int = 1,
    child_factor: float = 0.0,
) -> ExperimentConfig:
    return ExperimentConfig(
        task_name="mujoco_playground_generic",
        seed=seed,
        pop_size=pop_size,
        children_per_parent=children_per_parent,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        embedding_dim=32,
        gnn_hidden_dim=32,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        stoch_max_out=1024,
        mutation_rate_head_dim=2,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-0.1, 0.1),
        const_noise_std=0.001,
        mutation_scale_alpha=0.05,
        mutation_norm_eps=1e-8,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        env_backend="mujoco_playground",
        env_id=env_id,
        brax_backend=None,
    )
