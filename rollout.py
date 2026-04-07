from __future__ import annotations

import jax
import jax.numpy as jnp

from envs import apply_reward_shifts, make_env, map_action_for_shifts, map_observation_for_shifts
from obs_norm import flatten_observation, normalize_obs
from policy import apply_policy
from policy_vectors import unflatten_policy_vector
from srghn import make_policy


def rollout_episode(
    policy_params,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    obs_norm_state=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    env, env_params, obs_dim, act_dim, is_discrete, action_shape, action_low, action_high = make_env(config)
    obs_dtype = jnp.float32

    if config.env_backend == "gymnax":
        key, key_reset = jax.random.split(key, 2)
        obs, state = env.reset(key_reset, env_params)

        def step_fn(carry, _):
            obs_t, state_t, done_t, key_t, obs_sum_t, obs_sq_sum_t, obs_count_t = carry
            key_t, key_step = jax.random.split(key_t, 2)
            obs_flat = flatten_observation(obs_t)
            active = jnp.asarray(~done_t, dtype=obs_dtype)
            obs_sum_t = obs_sum_t + active * obs_flat
            obs_sq_sum_t = obs_sq_sum_t + active * jnp.square(obs_flat)
            obs_count_t = obs_count_t + active
            obs_shifted = map_observation_for_shifts(obs_t, gen, config)
            obs_in = normalize_obs(obs_shifted, obs_norm_state, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
            action = apply_policy(policy_params, obs_in, config, is_discrete=is_discrete)
            action = map_action_for_shifts(action, gen, config, act_dim=act_dim, is_discrete=is_discrete)
            if is_discrete:
                action = jnp.asarray(action, dtype=jnp.int32)
            else:
                action = action.reshape(action_shape)
                if action_low is not None:
                    action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

            def do_step(_):
                next_obs, next_state, reward, done, _ = env.step(key_step, state_t, action, env_params)
                reward = jnp.asarray(reward, dtype=jnp.float32)
                reward = apply_reward_shifts(reward, next_state, gen, config)
                done = jnp.asarray(done, dtype=jnp.bool_)
                return next_obs, next_state, reward, done

            def skip_step(_):
                zero = jnp.zeros((), dtype=jnp.float32)
                return obs_t, state_t, zero, done_t

            next_obs, next_state, reward, done = jax.lax.cond(done_t, skip_step, do_step, operand=None)
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done, key_t, obs_sum_t, obs_sq_sum_t, obs_count_t), reward

    else:
        state = env.reset(key)
        obs = state.obs

        def step_fn(carry, _):
            obs_t, state_t, done_t, key_t, obs_sum_t, obs_sq_sum_t, obs_count_t = carry
            key_t, key_step = jax.random.split(key_t, 2)
            obs_flat = flatten_observation(obs_t)
            active = jnp.asarray(~done_t, dtype=obs_dtype)
            obs_sum_t = obs_sum_t + active * obs_flat
            obs_sq_sum_t = obs_sq_sum_t + active * jnp.square(obs_flat)
            obs_count_t = obs_count_t + active
            obs_shifted = map_observation_for_shifts(obs_t, gen, config)
            obs_in = normalize_obs(obs_shifted, obs_norm_state, clip=config.obs_norm_clip, eps=config.obs_norm_eps)
            action = apply_policy(policy_params, obs_in, config, is_discrete=is_discrete)
            action = map_action_for_shifts(action, gen, config, act_dim=act_dim, is_discrete=is_discrete)
            if is_discrete:
                action = jnp.asarray(action, dtype=jnp.int32)
            else:
                action = action.reshape(action_shape)
                if action_low is not None:
                    action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

            def do_step(_):
                next_state = env.step(state_t, action)
                reward = jnp.asarray(next_state.reward, dtype=jnp.float32)
                reward = apply_reward_shifts(reward, next_state, gen, config)
                done = jnp.asarray(next_state.done, dtype=jnp.bool_)
                return next_state.obs, next_state, reward, done

            def skip_step(_):
                zero = jnp.zeros((), dtype=jnp.float32)
                return obs_t, state_t, zero, done_t

            next_obs, next_state, reward, done = jax.lax.cond(done_t, skip_step, do_step, operand=None)
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done, key_t, obs_sum_t, obs_sq_sum_t, obs_count_t), reward

    init_done = jnp.array(False)
    init_obs_sum = jnp.zeros((obs_dim,), dtype=obs_dtype)
    init_obs_sq_sum = jnp.zeros((obs_dim,), dtype=obs_dtype)
    init_obs_count = jnp.zeros((), dtype=obs_dtype)
    (_, _, _, _, obs_sum, obs_sq_sum, obs_count), rewards = jax.lax.scan(
        step_fn,
        (obs, state, init_done, key, init_obs_sum, init_obs_sq_sum, init_obs_count),
        None,
        length=config.episode_horizon,
    )
    return jnp.sum(rewards), obs_sum, obs_sq_sum, obs_count


def evaluate_individual(
    srghn,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    obs_norm_state=None,
) -> jnp.ndarray:
    policy_params = make_policy(srghn)
    keys = jax.random.split(key, config.episodes_per_eval)
    returns, _, _, _ = jax.vmap(lambda k: rollout_episode(policy_params, k, gen, config, obs_norm_state))(keys)
    return jnp.mean(returns)


def evaluate_individual_with_obs_stats(
    srghn,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    obs_norm_state=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    policy_params = make_policy(srghn)
    keys = jax.random.split(key, config.episodes_per_eval)
    returns, obs_sum, obs_sq_sum, obs_count = jax.vmap(
        lambda k: rollout_episode(policy_params, k, gen, config, obs_norm_state)
    )(keys)
    return jnp.mean(returns), jnp.sum(obs_sum, axis=0), jnp.sum(obs_sq_sum, axis=0), jnp.sum(obs_count, axis=0)


def evaluate_policy_vector(
    policy_vector: jnp.ndarray,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    policy_spec,
    obs_norm_state=None,
) -> jnp.ndarray:
    policy_params = unflatten_policy_vector(policy_vector, policy_spec)
    keys = jax.random.split(key, config.episodes_per_eval)
    returns, _, _, _ = jax.vmap(lambda k: rollout_episode(policy_params, k, gen, config, obs_norm_state))(keys)
    return jnp.mean(returns)


def evaluate_policy_vector_with_obs_stats(
    policy_vector: jnp.ndarray,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    policy_spec,
    obs_norm_state=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    policy_params = unflatten_policy_vector(policy_vector, policy_spec)
    keys = jax.random.split(key, config.episodes_per_eval)
    returns, obs_sum, obs_sq_sum, obs_count = jax.vmap(
        lambda k: rollout_episode(policy_params, k, gen, config, obs_norm_state)
    )(keys)
    return jnp.mean(returns), jnp.sum(obs_sum, axis=0), jnp.sum(obs_sq_sum, axis=0), jnp.sum(obs_count, axis=0)
