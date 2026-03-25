from __future__ import annotations

from dataclasses import dataclass

from configs import ShiftWindowConfig
from experiments._adaptation import gymnax_suite_shift_windows


PPO_GYMNAX_SUITE_ENVIRONMENTS = (
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
)


@dataclass(frozen=True)
class PPOConfig:
    task_name: str
    seed: int
    num_generations: int
    episode_horizon: int
    episodes_per_eval: int
    policy_hidden_dims: tuple[int, ...]
    shift_windows: tuple[ShiftWindowConfig, ...]
    optimizer_family: str
    baseline_name: str
    wandb_project: str | None
    wandb_group: str | None
    wandb_name: str | None
    env_backend: str
    env_id: str
    brax_backend: str | None
    obs_norm_clip: float
    obs_norm_eps: float
    num_envs: int
    rollout_length: int
    num_minibatches: int
    update_epochs: int
    learning_rate: float
    clip_eps: float
    gae_lambda: float
    gamma: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float


def make_ppo_config_nonstationary_gymnax(
    env_id: str,
    *,
    seed: int = 0,
    num_generations: int = 1500,
    episode_horizon: int = 500,
    episodes_per_eval: int = 1,
    policy_hidden_dims: tuple[int, ...] = (32, 32),
    shift_windows: tuple[ShiftWindowConfig, ...] | None = None,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    num_envs: int = 32,
    rollout_length: int = 128,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    learning_rate: float = 3e-4,
    clip_eps: float = 0.2,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> PPOConfig:
    resolved_shift_windows = gymnax_suite_shift_windows(env_id) if shift_windows is None else shift_windows
    return PPOConfig(
        task_name="gymnax_generic",
        seed=seed,
        num_generations=num_generations,
        episode_horizon=episode_horizon,
        episodes_per_eval=episodes_per_eval,
        policy_hidden_dims=policy_hidden_dims,
        shift_windows=resolved_shift_windows,
        optimizer_family="ppo",
        baseline_name="ppo",
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_name=wandb_name,
        env_backend="gymnax",
        env_id=env_id,
        brax_backend=None,
        obs_norm_clip=5.0,
        obs_norm_eps=1e-8,
        num_envs=num_envs,
        rollout_length=rollout_length,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
        learning_rate=learning_rate,
        clip_eps=clip_eps,
        gae_lambda=gae_lambda,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
    )
