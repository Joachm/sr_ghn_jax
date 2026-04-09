from __future__ import annotations

from configs import ShiftWindowConfig


DEFAULT_BASELINES = (
    "srghn_full",
    "frozen_mutation",
    "fixed_lr",
    "no_self_reference",
)


GYMNAX_MINATAR_SUITE_ENVIRONMENTS = (
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
)


def gymnax_shift_windows(variant: str) -> tuple[ShiftWindowConfig, ...]:
    if variant == "cartpole_flip":
        return (ShiftWindowConfig(600, 1200, "cartpole_flip"),)
    if variant == "cartpole_flip_revert":
        return (
            ShiftWindowConfig(400, 700, "cartpole_flip"),
            ShiftWindowConfig(900, 1200, "cartpole_flip"),
        )
    if variant == "cartpole_repeated":
        return (
            ShiftWindowConfig(300, 500, "cartpole_flip"),
            ShiftWindowConfig(700, 900, "cartpole_flip"),
            ShiftWindowConfig(1100, 1300, "cartpole_flip"),
        )
    raise ValueError(f"Unknown Gymnax adaptation variant: {variant}")


def gymnax_suite_shift_windows(env_id: str) -> tuple[ShiftWindowConfig, ...]:
    if env_id == "CartPole-v1":
        return (
            ShiftWindowConfig(400, 700, "cartpole_flip"),
            ShiftWindowConfig(900, 1200, "cartpole_flip"),
        )
    if env_id in {"Acrobot-v1", "MountainCar-v0"}:
        return (
            ShiftWindowConfig(400, 700, "discrete_reverse"),
            ShiftWindowConfig(900, 1200, "discrete_reverse"),
        )
    if env_id == "MountainCarContinuous-v0":
        return (
            ShiftWindowConfig(400, 700, "continuous_action_flip"),
            ShiftWindowConfig(900, 1200, "continuous_action_flip"),
        )
    if env_id == "Pendulum-v1":
        return (
            ShiftWindowConfig(400, 700, "pendulum_obs_flip"),
            ShiftWindowConfig(900, 1200, "pendulum_obs_flip"),
        )
    raise ValueError(f"Unsupported Gymnax suite env: {env_id}")


def gymnax_minatar_suite_shift_windows(env_id: str) -> tuple[ShiftWindowConfig, ...]:
    if env_id in GYMNAX_MINATAR_SUITE_ENVIRONMENTS:
        return (
            ShiftWindowConfig(3200*2, 2*5600, "discrete_reverse"),
            ShiftWindowConfig(2*7200, 2*11200, "discrete_reverse"),
        )
    raise ValueError(f"Unsupported Gymnax MinAtar suite env: {env_id}")


def brax_shift_windows(variant: str, *, target_speed: float = 1.0) -> tuple[ShiftWindowConfig, ...]:
    if variant == "brax_direction_switch":
        return (ShiftWindowConfig(400, None, "brax_direction_switch"),)
    if variant == "brax_speed_target_switch":
        return (ShiftWindowConfig(400, None, "brax_speed_target_switch", target_value=target_speed),)
    raise ValueError(f"Unknown Brax adaptation variant: {variant}")
