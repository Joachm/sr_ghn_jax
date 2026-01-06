from __future__ import annotations

import jax
import jax.numpy as jnp

from .envs import make_env, map_action_for_switch
from .policy import apply_policy
from .srghn import make_policy


def rollout_episode(policy_params, key: jax.random.KeyArray, gen: jnp.ndarray, config) -> jnp.ndarray:
    env, env_params, _, _, is_discrete, action_shape = make_env(config)

    if config.env_backend == "gymnax":
        key, key_reset = jax.random.split(key, 2)
        obs, state = env.reset(key_reset, env_params)

        def step_fn(carry, _):
            obs_t, state_t, done_t, key_t = carry
            key_t, key_step = jax.random.split(key_t, 2)
            action = apply_policy(policy_params, obs_t, is_discrete=is_discrete)
            action = map_action_for_switch(action, gen, config)
            if is_discrete:
                action = jnp.asarray(action, dtype=jnp.int32)
            else:
                action = action.reshape(action_shape)
            next_obs, next_state, reward, done, _ = env.step(key_step, state_t, action, env_params)
            reward = reward * (1.0 - done_t.astype(reward.dtype))
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done, key_t), reward

    else:
        state = env.reset(key)
        obs = state.obs

        def step_fn(carry, _):
            obs_t, state_t, done_t, key_t = carry
            key_t, key_step = jax.random.split(key_t, 2)
            action = apply_policy(policy_params, obs_t, is_discrete=is_discrete)
            action = map_action_for_switch(action, gen, config)
            if not is_discrete:
                action = action.reshape(action_shape)
            next_state = env.step(state_t, action)
            next_obs = next_state.obs
            reward = next_state.reward
            done = next_state.done
            reward = reward * (1.0 - done_t.astype(reward.dtype))
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done, key_t), reward

    init_done = jnp.array(False)
    (final_obs, final_state, final_done, final_key), rewards = jax.lax.scan(
        step_fn,
        (obs, state, init_done, key),
        None,
        length=config.episode_horizon,
    )
    return jnp.sum(rewards)


def evaluate_individual(srghn, key: jax.random.KeyArray, gen: jnp.ndarray, config) -> jnp.ndarray:
    policy_params = make_policy(srghn)
    keys = jax.random.split(key, config.episodes_per_eval)
    returns = jax.vmap(lambda k: rollout_episode(policy_params, k, gen, config))(keys)
    return jnp.mean(returns)
