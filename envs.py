from __future__ import annotations

from math import prod
from typing import Tuple

import jax
import jax.numpy as jnp


def _infer_obs_dim(env) -> int:
    if hasattr(env, "observation_size"):
        return int(env.observation_size)
    if hasattr(env, "obs_size"):
        return int(env.obs_size)
    obs_space = getattr(env, "observation_space", None)
    if callable(obs_space):
        try:
            obs_space = obs_space()
        except TypeError:
            pass
    if obs_space is not None and hasattr(obs_space, "shape"):
        return int(prod(obs_space.shape))
    raise ValueError("Unable to infer observation size from environment.")


def _infer_action_info(env) -> tuple[int, bool, tuple[int, ...], jnp.ndarray | None, jnp.ndarray | None]:
    action_space = getattr(env, "action_space", None)
    if callable(action_space):
        try:
            action_space = action_space()
        except TypeError:
            pass

    if action_space is not None:
        if hasattr(action_space, "nvec"):
            raise ValueError("MultiDiscrete action spaces are not supported.")
        if hasattr(action_space, "n"):
            return int(action_space.n), True, (), None, None
        if hasattr(action_space, "shape"):
            act_dim = int(prod(action_space.shape))
            action_low = jnp.asarray(action_space.low) if hasattr(action_space, "low") else None
            action_high = jnp.asarray(action_space.high) if hasattr(action_space, "high") else None
            return act_dim, False, action_space.shape, action_low, action_high

    if hasattr(env, "action_size"):
        act_dim = int(env.action_size)
        return act_dim, False, (act_dim,), None, None

    raise ValueError("Unable to infer action space from environment.")


def _load_mujoco_playground_env(env_id: str):
    from mujoco_playground import registry as suite

    if ":" in env_id:
        domain, task = env_id.split(":", 1)
        return suite.load(domain, task)
    if "/" in env_id:
        domain, task = env_id.split("/", 1)
        return suite.load(domain, task)
    if hasattr(suite, "load"):
        try:
            return suite.load(env_id)
        except TypeError:
            pass
    if hasattr(suite, "make"):
        return suite.make(env_id)
    raise ValueError("Unsupported mujoco_playground suite API.")


def make_env(config):
    if config.env_backend == "gymnax":
        import gymnax

        env, env_params = gymnax.make(config.env_id)
        obs_dim = int(prod(env.observation_space(env_params).shape))
        action_space = env.action_space(env_params)
        if hasattr(action_space, "nvec"):
            raise ValueError("MultiDiscrete action spaces are not supported.")
        if hasattr(action_space, "n"):
            act_dim = int(action_space.n)
            is_discrete = True
            action_shape = ()
            action_low = None
            action_high = None
        else:
            act_dim = int(prod(action_space.shape))
            is_discrete = False
            action_shape = action_space.shape
            action_low = jnp.asarray(action_space.low)
            action_high = jnp.asarray(action_space.high)
        return env, env_params, obs_dim, act_dim, is_discrete, action_shape, action_low, action_high

    if config.env_backend == "brax":
        from brax import envs

        if config.brax_backend is None:
            env = envs.create(config.env_id)
        else:
            env = envs.create(config.env_id, backend=config.brax_backend)
        env_params = None
        obs_dim = int(env.observation_size)
        act_dim = int(env.action_size)
        return env, env_params, obs_dim, act_dim, False, (act_dim,), None, None

    if config.env_backend == "mujoco_playground":
        env = _load_mujoco_playground_env(config.env_id)
        env_params = None
        obs_dim = _infer_obs_dim(env)
        act_dim, is_discrete, action_shape, action_low, action_high = _infer_action_info(env)
        return env, env_params, obs_dim, act_dim, is_discrete, action_shape, action_low, action_high

    raise ValueError(f"Unknown env_backend: {config.env_backend}")


def _window_active(gen: jnp.ndarray, start: int, end: int | None) -> jnp.ndarray:
    if end is None:
        return gen >= start
    return jnp.logical_and(gen >= start, gen <= end)


def iter_shift_windows(config) -> tuple:
    windows = getattr(config, "shift_windows", ())
    if windows:
        return windows
    if getattr(config, "switch_gen_start", None) is None:
        return ()
    try:
        from configs import ShiftWindowConfig
    except Exception:
        return ()
    return (
        ShiftWindowConfig(
            start_gen=config.switch_gen_start,
            end_gen=config.switch_gen_end,
            rule=config.switch_rule or "cartpole_flip",
        ),
    )


def active_shift_mask(gen: jnp.ndarray, config, rule: str) -> jnp.ndarray:
    windows = iter_shift_windows(config)
    if not windows:
        return jnp.asarray(False)
    active = jnp.asarray(False)
    for window in windows:
        if window.rule != rule:
            continue
        active = jnp.logical_or(active, _window_active(gen, window.start_gen, window.end_gen))
    return active


def map_observation_for_shifts(obs: jnp.ndarray, gen: jnp.ndarray, config) -> jnp.ndarray:
    out = jnp.asarray(obs)
    for window in iter_shift_windows(config):
        active = _window_active(gen, window.start_gen, window.end_gen)
        if window.rule == "pendulum_obs_flip":
            flipped = out.at[0].set(out[1]).at[1].set(out[0]).at[2].set(-out[2])
            out = jnp.where(active, flipped, out)
    return out


def map_action_for_shifts(
    action: jnp.ndarray,
    gen: jnp.ndarray,
    config,
    *,
    act_dim: int | None = None,
    is_discrete: bool | None = None,
) -> jnp.ndarray:
    out = action
    for window in iter_shift_windows(config):
        active = _window_active(gen, window.start_gen, window.end_gen)
        if window.rule == "cartpole_flip":
            out = jnp.where(active, 1 - out, out)
        elif window.rule == "discrete_reverse":
            if is_discrete and act_dim is not None:
                candidate = (act_dim - 1) - out
                out = jnp.where(active, candidate, out)
        elif window.rule == "continuous_action_flip":
            if is_discrete is False:
                out = jnp.where(active, -out, out)
    return out


def map_action_for_switch(action: jnp.ndarray, gen: jnp.ndarray, config) -> jnp.ndarray:
    return map_action_for_shifts(action, gen, config)


def _extract_progress_metric(next_state) -> jnp.ndarray | None:
    metrics = getattr(next_state, "metrics", None)
    if metrics is None:
        return None
    for key in (
        "x_velocity",
        "reward_forward",
        "forward_reward",
        "reward_linvel",
        "velocity_x",
    ):
        if key in metrics:
            return jnp.asarray(metrics[key], dtype=jnp.float32)
    return None


def apply_reward_shifts(
    reward: jnp.ndarray,
    next_state,
    gen: jnp.ndarray,
    config,
) -> jnp.ndarray:
    out = reward
    for window in iter_shift_windows(config):
        active = _window_active(gen, window.start_gen, window.end_gen)
        if window.rule == "reward_negate":
            out = jnp.where(active, -out, out)
        elif window.rule == "brax_direction_switch":
            progress = _extract_progress_metric(next_state)
            candidate = -out if progress is None else -progress
            out = jnp.where(active, candidate.astype(out.dtype), out)
        elif window.rule == "brax_speed_target_switch":
            progress = _extract_progress_metric(next_state)
            if progress is None:
                candidate = -jnp.abs(out - jnp.asarray(window.target_value, dtype=out.dtype))
            else:
                candidate = -jnp.abs(progress - jnp.asarray(window.target_value, dtype=progress.dtype))
            out = jnp.where(active, candidate.astype(out.dtype), out)
        elif window.rule in {
            "cartpole_flip",
            "discrete_reverse",
            "continuous_action_flip",
            "pendulum_obs_flip",
        }:
            continue
        else:
            raise ValueError(f"Unknown shift rule: {window.rule}")
    return out
