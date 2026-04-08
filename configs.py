from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShiftWindowConfig:
    start_gen: int
    end_gen: int | None
    rule: str
    target_value: float = 0.0


BASELINE_FULL = "srghn_full"
BASELINE_FROZEN_MUTATION = "frozen_mutation"
BASELINE_FIXED_LR = "fixed_lr"
BASELINE_NO_SELF_REFERENCE = "no_self_reference"

GYMNAX_SUITE_DEFAULT_POP_SIZE = 50
GYMNAX_SUITE_DEFAULT_CHILDREN_PER_PARENT = 4
GYMNAX_SUITE_DEFAULT_NUM_GENERATIONS = 1500
GYMNAX_SUITE_DEFAULT_EVALS_PER_GENERATION = GYMNAX_SUITE_DEFAULT_POP_SIZE * (1 + GYMNAX_SUITE_DEFAULT_CHILDREN_PER_PARENT)


def gymnax_control_pop_size(children_per_parent: int) -> int:
    return GYMNAX_SUITE_DEFAULT_POP_SIZE * (1 + children_per_parent)


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
    parameter_block_size: int
    mutation_block_ratio: float
    mutation_rate_head_dim: int
    clip_params: tuple[float, float]
    clip_std: tuple[float, float]
    clip_update: tuple[float, float]
    const_noise_std: float
    child_factor: float
    switch_gen_start: int | None
    switch_gen_end: int | None
    switch_rule: str | None
    shift_windows: tuple[ShiftWindowConfig, ...]
    optimizer_family: str
    baseline_name: str
    evosax_algo: str | None
    evosax_sigma_init: float | None
    policy_architecture: str
    policy_conv_channels: tuple[int, ...]
    policy_conv_kernel_sizes: tuple[tuple[int, int], ...]
    policy_conv_strides: tuple[tuple[int, int], ...]
    policy_hidden_dims: tuple[int, ...]
    mutation_exclude_modules: tuple[str, ...]
    fixed_mutation_lr: float | None
    wandb_project: str | None
    wandb_group: str | None
    wandb_name: str | None
    env_backend: str
    env_id: str
    brax_backend: str | None
    obs_norm_clip: float
    obs_norm_eps: float


@dataclass(frozen=True)
class EvosaxControlConfig:
    task_name: str
    seed: int
    pop_size: int
    children_per_parent: int
    num_generations: int
    episode_horizon: int
    episodes_per_eval: int
    shift_windows: tuple[ShiftWindowConfig, ...]
    optimizer_family: str
    evosax_algo: str
    evosax_sigma_init: float | None
    policy_architecture: str
    policy_conv_channels: tuple[int, ...]
    policy_conv_kernel_sizes: tuple[tuple[int, int], ...]
    policy_conv_strides: tuple[tuple[int, int], ...]
    policy_hidden_dims: tuple[int, ...]
    wandb_project: str | None
    wandb_group: str | None
    wandb_name: str | None
    env_backend: str
    env_id: str
    brax_backend: str | None
    obs_norm_clip: float
    obs_norm_eps: float


def make_config_cartpole_switch(
    *,
    parameter_block_size: int = 64,
    mutation_block_ratio: float = 0.125,
) -> ExperimentConfig:
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
        parameter_block_size=parameter_block_size,
        mutation_block_ratio=mutation_block_ratio,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-1.0, 1.0),
        const_noise_std=0.001,
        child_factor=0.0,
        switch_gen_start=600,
        switch_gen_end=1200,
        switch_rule="cartpole_flip",
        shift_windows=(ShiftWindowConfig(600, 1200, "cartpole_flip"),),
        optimizer_family="srghn",
        baseline_name=BASELINE_FULL,
        evosax_algo=None,
        evosax_sigma_init=None,
        policy_architecture="mlp",
        policy_conv_channels=(),
        policy_conv_kernel_sizes=(),
        policy_conv_strides=(),
        policy_hidden_dims=(32,),
        mutation_exclude_modules=(),
        fixed_mutation_lr=None,
        wandb_project=None,
        wandb_group=None,
        wandb_name=None,
        env_backend="gymnax",
        env_id="CartPole-v1",
        brax_backend=None,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-8,
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
    parameter_block_size: int = 1024,
    mutation_block_ratio: float = 1.,
    optimizer_family: str = "srghn",
    evosax_algo: str | None = None,
    evosax_sigma_init: float | None = None,
    policy_architecture: str = "mlp",
    policy_conv_channels: tuple[int, ...] = (),
    policy_conv_kernel_sizes: tuple[tuple[int, int], ...] = (),
    policy_conv_strides: tuple[tuple[int, int], ...] = (),
    policy_hidden_dims: tuple[int, ...] = (32, 32),
) -> ExperimentConfig:
    return ExperimentConfig(
        task_name="gymnax_generic",
        seed=seed,
        pop_size=pop_size,
        children_per_parent=children_per_parent,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        embedding_dim=64,
        gnn_hidden_dim=64,
        gnn_steps_policy=10,
        gnn_steps_self=10,
        stoch_coeff_dim=32,
        parameter_block_size=parameter_block_size,
        mutation_block_ratio=mutation_block_ratio,
        mutation_rate_head_dim=5,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-.1, .1),
        const_noise_std=0.001,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        shift_windows=(),
        optimizer_family=optimizer_family,
        baseline_name=BASELINE_FULL,
        evosax_algo=evosax_algo,
        evosax_sigma_init=evosax_sigma_init,
        policy_architecture=policy_architecture,
        policy_conv_channels=policy_conv_channels,
        policy_conv_kernel_sizes=policy_conv_kernel_sizes,
        policy_conv_strides=policy_conv_strides,
        policy_hidden_dims=policy_hidden_dims,
        mutation_exclude_modules=(),
        fixed_mutation_lr=None,
        wandb_project=None,
        wandb_group=None,
        wandb_name=None,
        env_backend="gymnax",
        env_id=env_id,
        brax_backend=None,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-8,
    )


def make_evosax_control_config_gymnax_generic(
    env_id: str,
    *,
    seed: int = 0,
    pop_size: int = 30,
    num_generations: int = 300,
    episode_horizon: int = 500,
    children_per_parent: int = 2,
    episodes_per_eval: int = 1,
    evosax_algo: str,
    evosax_sigma_init: float | None = None,
    policy_architecture: str = "mlp",
    policy_conv_channels: tuple[int, ...] = (),
    policy_conv_kernel_sizes: tuple[tuple[int, int], ...] = (),
    policy_conv_strides: tuple[tuple[int, int], ...] = (),
    policy_hidden_dims: tuple[int, ...] = (32, 32),
    shift_windows: tuple[ShiftWindowConfig, ...] = (),
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
) -> EvosaxControlConfig:
    return EvosaxControlConfig(
        task_name="gymnax_generic",
        seed=seed,
        pop_size=pop_size,
        children_per_parent=children_per_parent,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        shift_windows=shift_windows,
        optimizer_family="evosax",
        evosax_algo=evosax_algo,
        evosax_sigma_init=evosax_sigma_init,
        policy_architecture=policy_architecture,
        policy_conv_channels=policy_conv_channels,
        policy_conv_kernel_sizes=policy_conv_kernel_sizes,
        policy_conv_strides=policy_conv_strides,
        policy_hidden_dims=policy_hidden_dims,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_name=wandb_name,
        env_backend="gymnax",
        env_id=env_id,
        brax_backend=None,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-8,
    )


def make_config_mujoco_playground_generic(
    env_id: str,
    *,
    seed: int = 1,
    pop_size: int = 200,
    num_generations: int = 10000,
    episode_horizon: int = 1000,
    children_per_parent: int = 8,
    episodes_per_eval: int = 2,
    child_factor: float = 0.0,
    parameter_block_size: int = 4096,
    mutation_block_ratio: float = 1.,
    optimizer_family: str = "srghn",
    evosax_algo: str | None = None,
    evosax_sigma_init: float | None = None,
    policy_hidden_dims: tuple[int, ...] = (128, 64, 64),
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
        parameter_block_size=parameter_block_size,
        mutation_block_ratio=mutation_block_ratio,
        mutation_rate_head_dim=2,
        clip_params=(-20.0, 20.0),
        clip_std=(0.0, 2.0),
        clip_update=(-0.1, 0.1),
        const_noise_std=0.001,
        child_factor=child_factor,
        switch_gen_start=None,
        switch_gen_end=None,
        switch_rule=None,
        shift_windows=(),
        optimizer_family=optimizer_family,
        baseline_name=BASELINE_FULL,
        evosax_algo=evosax_algo,
        evosax_sigma_init=evosax_sigma_init,
        policy_architecture="mlp",
        policy_conv_channels=(),
        policy_conv_kernel_sizes=(),
        policy_conv_strides=(),
        policy_hidden_dims=policy_hidden_dims,
        mutation_exclude_modules=(),
        fixed_mutation_lr=None,
        #wandb_project=None,
        wandb_project=env_id,
        wandb_group=None,
        wandb_name=None,
        env_backend="mujoco_playground",
        env_id=env_id,
        brax_backend=None,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-8,
    )


def _with_nonstationary_overrides(
    config: ExperimentConfig,
    *,
    shift_windows: tuple[ShiftWindowConfig, ...],
    baseline_name: str = BASELINE_FULL,
    optimizer_family: str | None = None,
    evosax_algo: str | None = None,
    evosax_sigma_init: float | None = None,
    fixed_mutation_lr: float | None = None,
    mutation_exclude_modules: tuple[str, ...] = (),
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
) -> ExperimentConfig:
    return config.__class__(
        **{
            **config.__dict__,
            "shift_windows": shift_windows,
            "optimizer_family": config.optimizer_family if optimizer_family is None else optimizer_family,
            "baseline_name": baseline_name,
            "evosax_algo": config.evosax_algo if evosax_algo is None else evosax_algo,
            "evosax_sigma_init": config.evosax_sigma_init if evosax_sigma_init is None else evosax_sigma_init,
            "fixed_mutation_lr": fixed_mutation_lr,
            "mutation_exclude_modules": mutation_exclude_modules,
            "wandb_project": wandb_project,
            "wandb_group": wandb_group,
            "wandb_name": wandb_name,
        }
    )


def baseline_overrides(
    baseline_name: str,
    *,
    fixed_mutation_lr: float | None = 0.05,
) -> dict:
    if baseline_name == BASELINE_FULL:
        return {
            "baseline_name": baseline_name,
            "mutation_exclude_modules": (),
            "fixed_mutation_lr": None,
        }
    if baseline_name == BASELINE_FROZEN_MUTATION:
        return {
            "baseline_name": baseline_name,
            "mutation_exclude_modules": ("stoch",),
            "fixed_mutation_lr": None,
        }
    if baseline_name == BASELINE_FIXED_LR:
        return {
            "baseline_name": baseline_name,
            "mutation_exclude_modules": (),
            "fixed_mutation_lr": fixed_mutation_lr,
        }
    if baseline_name == BASELINE_NO_SELF_REFERENCE:
        return {
            "baseline_name": baseline_name,
            "mutation_exclude_modules": (
                "self_node_emb",
                "self_context_emb",
                "self_feat_proj",
                "encoder_self",
                "stoch",
            ),
            "fixed_mutation_lr": None,
        }
    raise ValueError(f"Unknown baseline_name: {baseline_name}")


def make_config_nonstationary_gymnax(
    env_id: str,
    *,
    seed: int = 0,
    pop_size: int | None = None,
    num_generations: int = GYMNAX_SUITE_DEFAULT_NUM_GENERATIONS,
    episode_horizon: int = 500,
    children_per_parent: int = 4,
    episodes_per_eval: int = 1,
    parameter_block_size: int = 1024*4,
    mutation_block_ratio: float = 1.,
    shift_windows: tuple[ShiftWindowConfig, ...] = (ShiftWindowConfig(600, 1200, "cartpole_flip"),),
    optimizer_family: str = "srghn",
    baseline_name: str = BASELINE_FULL,
    evosax_algo: str | None = None,
    evosax_sigma_init: float | None = None,
    policy_architecture: str = "mlp",
    policy_conv_channels: tuple[int, ...] = (),
    policy_conv_kernel_sizes: tuple[tuple[int, int], ...] = (),
    policy_conv_strides: tuple[tuple[int, int], ...] = (),
    policy_hidden_dims: tuple[int, ...] = (32, 32),
    fixed_mutation_lr: float | None = 0.01,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
) -> ExperimentConfig | EvosaxControlConfig:
    resolved_pop_size = (
        gymnax_control_pop_size(children_per_parent)
        if optimizer_family == "evosax" and pop_size is None
        else GYMNAX_SUITE_DEFAULT_POP_SIZE
        if pop_size is None
        else pop_size
    )
    if optimizer_family == "evosax":
        if not evosax_algo:
            raise ValueError("evosax_algo must be set when optimizer_family='evosax'.")
        return make_evosax_control_config_gymnax_generic(
            env_id,
            seed=seed,
            pop_size=resolved_pop_size,
            num_generations=num_generations,
            episode_horizon=episode_horizon,
            children_per_parent=children_per_parent,
            episodes_per_eval=episodes_per_eval,
            evosax_algo=evosax_algo,
            evosax_sigma_init=evosax_sigma_init,
            policy_architecture=policy_architecture,
            policy_conv_channels=policy_conv_channels,
            policy_conv_kernel_sizes=policy_conv_kernel_sizes,
            policy_conv_strides=policy_conv_strides,
            policy_hidden_dims=policy_hidden_dims,
            shift_windows=shift_windows,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            wandb_name=wandb_name,
        )
    config = make_config_gymnax_generic(
        env_id,
        seed=seed,
        pop_size=resolved_pop_size,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        children_per_parent=children_per_parent,
        episodes_per_eval=episodes_per_eval,
        parameter_block_size=parameter_block_size,
        mutation_block_ratio=mutation_block_ratio,
        optimizer_family=optimizer_family,
        evosax_algo=evosax_algo,
        evosax_sigma_init=evosax_sigma_init,
        policy_architecture=policy_architecture,
        policy_conv_channels=policy_conv_channels,
        policy_conv_kernel_sizes=policy_conv_kernel_sizes,
        policy_conv_strides=policy_conv_strides,
        policy_hidden_dims=policy_hidden_dims,
    )
    return _with_nonstationary_overrides(
        config,
        shift_windows=shift_windows,
        optimizer_family=optimizer_family,
        evosax_algo=evosax_algo,
        evosax_sigma_init=evosax_sigma_init,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_name=wandb_name,
        **baseline_overrides(baseline_name, fixed_mutation_lr=fixed_mutation_lr),
    )


