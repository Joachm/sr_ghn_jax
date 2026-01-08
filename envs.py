from __future__ import annotations

from math import prod
from typing import Tuple

import jax
import jax.numpy as jnp


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

    raise ValueError(f"Unknown env_backend: {config.env_backend}")


def map_action_for_switch(action: jnp.ndarray, gen: jnp.ndarray, config) -> jnp.ndarray:
    if config.switch_gen_start is None:
        return action

    start = config.switch_gen_start
    end = config.switch_gen_end
    if end is None:
        in_window = gen >= start
    else:
        in_window = jnp.logical_and(gen >= start, gen <= end)

    if config.switch_rule == "cartpole_flip":
        flipped = 1 - action
        return jnp.where(in_window, flipped, action)

    return action
