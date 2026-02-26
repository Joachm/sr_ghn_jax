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
    gnn_aggregation: str
    policy_hidden_dims: tuple[int, ...]
    freeze_stoch_output_head: bool
    stoch_coeff_dim: int
    policy_head_max_out: int | None
    self_shard_size: int
    shard_graph_mode: str
    self_update_mode: str
    shard_residual_scale: float
    stoch_cov_rank: int
    stoch_cov_scale: float
    mutation_rate_head_dim: int
    clip_params: tuple[float, float]
    self_reg_mode: str
    self_weight_decay: float
    self_weight_norm_mode: str
    self_weight_norm_target: float | None
    self_weight_norm_eps: float
    clip_std: tuple[float, float]
    clip_update: tuple[float, float]
    const_noise_std: float
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
        gnn_aggregation="sum",
        policy_hidden_dims=(32,),
        freeze_stoch_output_head=True,
        stoch_coeff_dim=32,
        policy_head_max_out=None,
        self_shard_size=1024,
        shard_graph_mode="sibling_chain",
        self_update_mode="local",
        shard_residual_scale=0.25,
        stoch_cov_rank=0,
        stoch_cov_scale=0.1,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        self_reg_mode="none",
        self_weight_decay=0.0,
        self_weight_norm_mode="per_layer",
        self_weight_norm_target=150.0,
        self_weight_norm_eps=1e-8,
        clip_std=(0.0, 2.0),
        clip_update=(-1.0, 1.0),
        const_noise_std=0.001,
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
        gnn_aggregation="sum",
        policy_hidden_dims=(32, 32, 32),
        freeze_stoch_output_head=True,
        stoch_coeff_dim=32,
        policy_head_max_out=None,
        self_shard_size=1024,
        shard_graph_mode="sibling_chain",
        self_update_mode="local",
        shard_residual_scale=0.25,
        stoch_cov_rank=0,
        stoch_cov_scale=0.1,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        self_reg_mode="none",
        self_weight_decay=0.0,
        self_weight_norm_mode="per_layer",
        self_weight_norm_target=150.0,
        self_weight_norm_eps=1e-8,
        clip_std=(0.0, 2.0),
        clip_update=(-.1, .1),
        const_noise_std=0.001,
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
    gnn_aggregation: str = "sum",
    policy_hidden_dims: tuple[int, ...] = (32,),
    freeze_stoch_output_head: bool = True,
    self_reg_mode: str = "none",
    self_weight_decay: float = 0.0,
    policy_head_max_out: int | None = None,
    self_shard_size: int = 1024,
    shard_graph_mode: str = "sibling_chain",
    self_update_mode: str = "local",
    shard_residual_scale: float = 0.25,
    stoch_cov_rank: int = 0,
    stoch_cov_scale: float = 0.1,
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
        gnn_aggregation=gnn_aggregation,
        policy_hidden_dims=policy_hidden_dims,
        freeze_stoch_output_head=freeze_stoch_output_head,
        stoch_coeff_dim=32,
        policy_head_max_out=policy_head_max_out,
        self_shard_size=self_shard_size,
        shard_graph_mode=shard_graph_mode,
        self_update_mode=self_update_mode,
        shard_residual_scale=shard_residual_scale,
        stoch_cov_rank=stoch_cov_rank,
        stoch_cov_scale=stoch_cov_scale,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        self_reg_mode=self_reg_mode,
        self_weight_decay=self_weight_decay,
        self_weight_norm_mode="per_layer",
        self_weight_norm_target=150.0,
        self_weight_norm_eps=1e-8,
        clip_std=(0.0, 2.0),
        clip_update=(-1.0, 1.0),
        const_noise_std=0.001,
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
    gnn_aggregation: str = "sum",
    policy_hidden_dims: tuple[int, ...] = (32, 32, 32),
    freeze_stoch_output_head: bool = True,
    self_reg_mode: str = "none",
    self_weight_decay: float = 0.0,
    policy_head_max_out: int | None = None,
    self_shard_size: int = 1024,
    shard_graph_mode: str = "sibling_chain",
    self_update_mode: str = "local",
    shard_residual_scale: float = 0.25,
    stoch_cov_rank: int = 0,
    stoch_cov_scale: float = 0.1,
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
        gnn_aggregation=gnn_aggregation,
        policy_hidden_dims=policy_hidden_dims,
        freeze_stoch_output_head=freeze_stoch_output_head,
        stoch_coeff_dim=32,
        policy_head_max_out=policy_head_max_out,
        self_shard_size=self_shard_size,
        shard_graph_mode=shard_graph_mode,
        self_update_mode=self_update_mode,
        shard_residual_scale=shard_residual_scale,
        stoch_cov_rank=stoch_cov_rank,
        stoch_cov_scale=stoch_cov_scale,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        self_reg_mode=self_reg_mode,
        self_weight_decay=self_weight_decay,
        self_weight_norm_mode="per_layer",
        self_weight_norm_target=150.0,
        self_weight_norm_eps=1e-8,
        clip_std=(0.0, 2.0),
        clip_update=(-0.1, 0.1),
        const_noise_std=0.001,
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
    pop_size: int = 50,
    num_generations: int = 1000,
    episode_horizon: int = 1000,
    children_per_parent: int = 2,
    episodes_per_eval: int = 1,
    child_factor: float = 0.0,
    gnn_aggregation: str = "sum",
    policy_hidden_dims: tuple[int, ...] = (32, 32, 32),
    freeze_stoch_output_head: bool = True,
    self_reg_mode: str = "none",
    self_weight_decay: float = 0.0,
    policy_head_max_out: int | None = None,
    self_shard_size: int = 1024,
    shard_graph_mode: str = "sibling_chain",
    self_update_mode: str = "local",
    shard_residual_scale: float = 0.25,
    stoch_cov_rank: int = 0,
    stoch_cov_scale: float = 0.1,
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
        gnn_aggregation=gnn_aggregation,
        policy_hidden_dims=policy_hidden_dims,
        freeze_stoch_output_head=freeze_stoch_output_head,
        stoch_coeff_dim=32,
        policy_head_max_out=policy_head_max_out,
        self_shard_size=self_shard_size,
        shard_graph_mode=shard_graph_mode,
        self_update_mode=self_update_mode,
        shard_residual_scale=shard_residual_scale,
        stoch_cov_rank=stoch_cov_rank,
        stoch_cov_scale=stoch_cov_scale,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        self_reg_mode=self_reg_mode,
        self_weight_decay=self_weight_decay,
        self_weight_norm_mode="per_layer",
        self_weight_norm_target=150.0,
        self_weight_norm_eps=1e-8,
        clip_std=(0.0, 2.0),
        clip_update=(-0.1, 0.1),
        const_noise_std=0.001,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        env_backend="mujoco_playground",
        env_id=env_id,
        brax_backend=None,
    )
