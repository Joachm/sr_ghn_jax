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

    domain_task: tuple[str, str] | None = None
    if ":" in env_id:
        domain, task = env_id.split(":", 1)
        domain_task = (domain, task)
    elif "/" in env_id:
        domain, task = env_id.split("/", 1)
        domain_task = (domain, task)

    attempted: list[str] = []

    def _try_call(fn, *args):
        attempted.append(f"{fn.__name__}{args}")
        try:
            return fn(*args)
        except (TypeError, ValueError, KeyError):
            return None

    if hasattr(suite, "load"):
        load_fn = suite.load
        if domain_task is not None:
            domain, task = domain_task
            env = _try_call(load_fn, domain, task)
            if env is not None:
                return env
            env = _try_call(load_fn, task)
            if env is not None:
                return env
        env = _try_call(load_fn, env_id)
        if env is not None:
            return env

    if hasattr(suite, "make"):
        make_fn = suite.make
        if domain_task is not None:
            domain, task = domain_task
            env = _try_call(make_fn, domain, task)
            if env is not None:
                return env
            env = _try_call(make_fn, task)
            if env is not None:
                return env
        env = _try_call(make_fn, env_id)
        if env is not None:
            return env

    attempted_desc = ", ".join(attempted) if attempted else "<none>"
    raise ValueError(
        "Unable to load MuJoCo Playground env "
        f"{env_id!r}. Tried: {attempted_desc}. "
        "Use a valid registry name (e.g., 'CheetahRun') or a domain/task id if your suite version supports it."
    )


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
