from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax

from baselines.ppo.networks import (
    apply_actor_critic,
    categorical_entropy,
    categorical_log_prob,
    categorical_sample_and_log_prob,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_sample_and_log_prob,
    init_actor_critic_params,
)
from envs import apply_reward_shifts, iter_shift_windows, make_env, map_action_for_shifts, map_observation_for_shifts
from obs_norm import ObsNormState, init_obs_norm, normalize_obs, update_obs_norm


ZERO_COMPAT_METRICS = (
    "diversity",
    "population_mutation_rate_mean",
    "population_mutation_rate_std",
    "population_mutation_rate_max",
    "population_mutation_block_fraction_mean",
    "population_mutation_blocks_selected_mean",
    "population_mutation_total_blocks_mean",
    "population_update_rms_mean",
    "population_self_distance_rms_mean",
    "elite_mutation_rate_mean",
    "elite_mutation_rate_std",
    "elite_mutation_rate_max",
    "elite_mutation_block_fraction_mean",
    "elite_mutation_blocks_selected_mean",
    "elite_mutation_total_blocks_mean",
    "elite_update_rms_mean",
    "elite_self_distance_rms_mean",
)


@jax.tree_util.register_pytree_node_class
@dataclass
class PPOTrainState:
    params: Any
    opt_state: Any
    key: jax.random.KeyArray
    obs_norm: ObsNormState
    obs: jnp.ndarray
    env_state: Any

    def tree_flatten(self):
        return (self.params, self.opt_state, self.key, self.obs_norm, self.obs, self.env_state), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, opt_state, key, obs_norm, obs, env_state = children
        return cls(params=params, opt_state=opt_state, key=key, obs_norm=obs_norm, obs=obs, env_state=env_state)


def _wandb_log(metrics_dict: dict, gen_idx: int) -> None:
    try:
        import wandb
    except Exception:
        return
    if wandb.run is None:
        return
    payload = {"gen": int(gen_idx)}
    for key, value in metrics_dict.items():
        payload[key] = float(value)
    wandb.log(payload)


def _active_shift_windows(gen: jnp.ndarray, config) -> jnp.ndarray:
    active_windows = 0
    for window in iter_shift_windows(config):
        if window.end_gen is None:
            active = gen >= window.start_gen
        else:
            active = jnp.logical_and(gen >= window.start_gen, gen <= window.end_gen)
        active_windows = active_windows + active.astype(jnp.int32)
    return active_windows.astype(jnp.float32)


def _broadcast_batch_mask(mask: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    reshape = (mask.shape[0],) + (1,) * max(value.ndim - 1, 0)
    return mask.reshape(reshape)


def _tree_where(mask: jnp.ndarray, true_tree, false_tree):
    return jax.tree_util.tree_map(
        lambda true_value, false_value: jnp.where(
            _broadcast_batch_mask(mask, true_value),
            true_value,
            false_value,
        ),
        true_tree,
        false_tree,
    )


def _policy_obs_single(obs: jnp.ndarray, gen: jnp.ndarray, config, obs_norm_state: ObsNormState) -> jnp.ndarray:
    shifted_obs = map_observation_for_shifts(obs, gen, config)
    return normalize_obs(shifted_obs, obs_norm_state, clip=config.obs_norm_clip, eps=config.obs_norm_eps)


def _policy_obs_batch(obs: jnp.ndarray, gen: jnp.ndarray, config, obs_norm_state: ObsNormState) -> jnp.ndarray:
    return jax.vmap(lambda single_obs: _policy_obs_single(single_obs, gen, config, obs_norm_state))(obs)


def _sample_actions(policy_output: jnp.ndarray, params, *, is_discrete: bool, key: jax.random.KeyArray):
    keys = jax.random.split(key, policy_output.shape[0])
    if is_discrete:
        return jax.vmap(categorical_sample_and_log_prob)(policy_output, keys)
    log_std = params["log_std"]
    return jax.vmap(lambda mean, sample_key: gaussian_sample_and_log_prob(mean, log_std, sample_key))(policy_output, keys)


def _log_prob_and_entropy(policy_output: jnp.ndarray, params, actions: jnp.ndarray, *, is_discrete: bool):
    if is_discrete:
        return categorical_log_prob(policy_output, actions), categorical_entropy(policy_output)
    log_std = params["log_std"]
    log_prob = jax.vmap(lambda mean, action: gaussian_log_prob(mean, log_std, action))(policy_output, actions)
    entropy = jnp.broadcast_to(gaussian_entropy(log_std), log_prob.shape)
    return log_prob, entropy


def _deterministic_action(policy_output: jnp.ndarray, *, is_discrete: bool) -> jnp.ndarray:
    if is_discrete:
        return jnp.argmax(policy_output, axis=-1).astype(jnp.int32)
    return policy_output


def _policy_action_to_env_action(
    raw_action: jnp.ndarray,
    gen: jnp.ndarray,
    config,
    *,
    act_dim: int,
    is_discrete: bool,
    action_shape: tuple[int, ...],
    action_low: jnp.ndarray | None,
    action_high: jnp.ndarray | None,
) -> jnp.ndarray:
    shifted_action = map_action_for_shifts(raw_action, gen, config, act_dim=act_dim, is_discrete=is_discrete)
    if is_discrete:
        return jnp.asarray(shifted_action, dtype=jnp.int32)
    env_action = jnp.asarray(shifted_action, dtype=jnp.float32).reshape(action_shape)
    if action_low is not None and action_high is not None:
        env_action = action_low + (env_action + 1.0) * 0.5 * (action_high - action_low)
    return env_action


def _evaluate_policy(
    params,
    key: jax.random.KeyArray,
    gen: jnp.ndarray,
    config,
    env,
    env_params,
    *,
    act_dim: int,
    is_discrete: bool,
    action_shape: tuple[int, ...],
    action_low: jnp.ndarray | None,
    action_high: jnp.ndarray | None,
    obs_norm_state: ObsNormState,
) -> jnp.ndarray:
    def rollout_once(rollout_key: jax.random.KeyArray) -> jnp.ndarray:
        key_reset, key_loop = jax.random.split(rollout_key, 2)
        reset_obs, reset_state = env.reset(key_reset, env_params)

        def step_fn(carry, _):
            obs_t, state_t, done_t, key_t = carry
            key_t, key_step = jax.random.split(key_t, 2)
            obs_in = _policy_obs_single(obs_t, gen, config, obs_norm_state)
            policy_output, _ = apply_actor_critic(params, obs_in)
            raw_action = _deterministic_action(policy_output, is_discrete=is_discrete)
            env_action = _policy_action_to_env_action(
                raw_action,
                gen,
                config,
                act_dim=act_dim,
                is_discrete=is_discrete,
                action_shape=action_shape,
                action_low=action_low,
                action_high=action_high,
            )

            def do_step(_):
                next_obs, next_state, reward, done, _ = env.step(key_step, state_t, env_action, env_params)
                reward = jnp.asarray(reward, dtype=jnp.float32)
                reward = apply_reward_shifts(reward, next_state, gen, config)
                done = jnp.asarray(done, dtype=jnp.bool_)
                return next_obs, next_state, reward, done

            def skip_step(_):
                zero = jnp.zeros((), dtype=jnp.float32)
                return obs_t, state_t, zero, done_t

            next_obs, next_state, reward, done = jax.lax.cond(done_t, skip_step, do_step, operand=None)
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done, key_t), reward

        (_, _, _, _), rewards = jax.lax.scan(
            step_fn,
            (reset_obs, reset_state, jnp.asarray(False), key_loop),
            None,
            length=config.episode_horizon,
        )
        return jnp.sum(rewards)

    keys = jax.random.split(key, config.episodes_per_eval)
    returns = jax.vmap(rollout_once)(keys)
    return jnp.mean(returns)


def _compute_advantages(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    values: jnp.ndarray,
    last_values: jnp.ndarray,
    config,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    next_values = jnp.concatenate([values[1:], last_values[None, :]], axis=0)
    done_mask = dones.astype(jnp.float32)

    def scan_fn(gae, inputs):
        reward_t, done_t, value_t, next_value_t = inputs
        non_terminal = 1.0 - done_t
        delta = reward_t + config.gamma * next_value_t * non_terminal - value_t
        gae = delta + config.gamma * config.gae_lambda * non_terminal * gae
        return gae, gae

    _, reversed_advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_values),
        (
            rewards[::-1],
            done_mask[::-1],
            values[::-1],
            next_values[::-1],
        ),
    )
    advantages = reversed_advantages[::-1]
    returns = advantages + values
    return advantages, returns


def _flatten_batch(rollout: dict[str, jnp.ndarray], advantages: jnp.ndarray, returns: jnp.ndarray) -> dict[str, jnp.ndarray]:
    batch = {
        "obs": rollout["obs"].reshape((-1, rollout["obs"].shape[-1])),
        "actions": rollout["actions"].reshape((-1,) + rollout["actions"].shape[2:]),
        "log_probs": rollout["log_probs"].reshape((-1,)),
        "advantages": advantages.reshape((-1,)),
        "returns": returns.reshape((-1,)),
    }
    if batch["actions"].ndim == 2 and batch["actions"].shape[1] == 1:
        batch["actions"] = batch["actions"].reshape((-1, 1))
    return batch


def _ppo_loss(params, batch: dict[str, jnp.ndarray], config, *, is_discrete: bool):
    policy_output, values = apply_actor_critic(params, batch["obs"])
    new_log_probs, entropy = _log_prob_and_entropy(policy_output, params, batch["actions"], is_discrete=is_discrete)
    ratios = jnp.exp(new_log_probs - batch["log_probs"])
    clipped_ratios = jnp.clip(ratios, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratios * batch["advantages"], clipped_ratios * batch["advantages"]))
    value_loss = 0.5 * jnp.mean(jnp.square(values - batch["returns"]))
    entropy_mean = jnp.mean(entropy)
    total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
    approx_kl = jnp.mean(batch["log_probs"] - new_log_probs)
    return total_loss, {
        "actor_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
    }


def _collect_rollout(
    params,
    key: jax.random.KeyArray,
    obs_norm_state: ObsNormState,
    obs: jnp.ndarray,
    env_state,
    gen: jnp.ndarray,
    config,
    env,
    env_params,
    *,
    act_dim: int,
    is_discrete: bool,
    action_shape: tuple[int, ...],
    action_low: jnp.ndarray | None,
    action_high: jnp.ndarray | None,
):
    def step_fn(carry, _):
        carry_key, carry_obs, carry_env_state = carry
        carry_key, key_sample, key_step, key_reset = jax.random.split(carry_key, 4)
        obs_sum = jnp.sum(carry_obs, axis=0)
        obs_sq_sum = jnp.sum(jnp.square(carry_obs), axis=0)
        obs_count = jnp.asarray(carry_obs.shape[0], dtype=jnp.float32)
        policy_obs = _policy_obs_batch(carry_obs, gen, config, obs_norm_state)
        policy_output, values = apply_actor_critic(params, policy_obs)
        raw_actions, log_probs = _sample_actions(policy_output, params, is_discrete=is_discrete, key=key_sample)
        env_actions = jax.vmap(
            lambda action: _policy_action_to_env_action(
                action,
                gen,
                config,
                act_dim=act_dim,
                is_discrete=is_discrete,
                action_shape=action_shape,
                action_low=action_low,
                action_high=action_high,
            )
        )(raw_actions)
        step_keys = jax.random.split(key_step, config.num_envs)
        next_obs, next_env_state, reward, done, _ = jax.vmap(
            lambda step_key, single_state, single_action: env.step(step_key, single_state, single_action, env_params)
        )(step_keys, carry_env_state, env_actions)
        reward = jax.vmap(
            lambda reward_value, next_state: apply_reward_shifts(
                jnp.asarray(reward_value, dtype=jnp.float32),
                next_state,
                gen,
                config,
            )
        )(reward, next_env_state)
        done = jnp.asarray(done, dtype=jnp.bool_)
        reset_keys = jax.random.split(key_reset, config.num_envs)
        reset_obs, reset_env_state = jax.vmap(lambda reset_key: env.reset(reset_key, env_params))(reset_keys)
        carry_next_obs = _tree_where(done, reset_obs, next_obs)
        carry_next_env_state = _tree_where(done, reset_env_state, next_env_state)
        transition = {
            "obs": policy_obs,
            "actions": raw_actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": jnp.asarray(reward, dtype=jnp.float32),
            "dones": done,
        }
        stats = {
            "obs_sum": obs_sum,
            "obs_sq_sum": obs_sq_sum,
            "obs_count": obs_count,
        }
        return (carry_key, carry_next_obs, carry_next_env_state), (transition, stats)

    (next_key, next_obs, next_env_state), (rollout, stats) = jax.lax.scan(
        step_fn,
        (key, obs, env_state),
        None,
        length=config.rollout_length,
    )
    aggregated_stats = {
        "obs_sum": jnp.sum(stats["obs_sum"], axis=0),
        "obs_sq_sum": jnp.sum(stats["obs_sq_sum"], axis=0),
        "obs_count": jnp.sum(stats["obs_count"], axis=0),
    }
    return next_key, next_obs, next_env_state, rollout, aggregated_stats


def run_ppo(config):
    if config.env_backend != "gymnax":
        raise ValueError("PPO baseline only supports Gymnax environments.")

    batch_size = config.num_envs * config.rollout_length
    if batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * rollout_length must be divisible by num_minibatches.")
    minibatch_size = batch_size // config.num_minibatches

    env, env_params, obs_dim, act_dim, is_discrete, action_shape, action_low, action_high = make_env(config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )

    key = jax.random.key(config.seed)
    key_params, key_reset, key_loop = jax.random.split(key, 3)
    params = init_actor_critic_params(
        key_params,
        obs_dim,
        act_dim,
        tuple(config.policy_hidden_dims),
        is_discrete=is_discrete,
    )
    opt_state = optimizer.init(params)
    reset_keys = jax.random.split(key_reset, config.num_envs)
    initial_obs, initial_env_state = jax.vmap(lambda reset_key: env.reset(reset_key, env_params))(reset_keys)
    initial_state = PPOTrainState(
        params=params,
        opt_state=opt_state,
        key=key_loop,
        obs_norm=init_obs_norm(obs_dim),
        obs=initial_obs,
        env_state=initial_env_state,
    )

    try:
        import wandb

        if config.wandb_project and wandb.run is None:
            wandb.init(
                project=config.wandb_project,
                group=config.wandb_group,
                name=config.wandb_name or f"{config.env_id}-seed{config.seed}",
                config=dict(config.__dict__),
            )
    except Exception:
        wandb = None

    def update_step(state: PPOTrainState, gen: jnp.ndarray):
        key_rollout, key_train, key_eval, key_next = jax.random.split(state.key, 4)
        _rollout_key, next_obs, next_env_state, rollout, rollout_stats = _collect_rollout(
            state.params,
            key_rollout,
            state.obs_norm,
            state.obs,
            state.env_state,
            gen,
            config,
            env,
            env_params,
            act_dim=act_dim,
            is_discrete=is_discrete,
            action_shape=action_shape,
            action_low=action_low,
            action_high=action_high,
        )
        last_policy_obs = _policy_obs_batch(next_obs, gen, config, state.obs_norm)
        _, last_values = apply_actor_critic(state.params, last_policy_obs)
        advantages, returns = _compute_advantages(
            rollout["rewards"],
            rollout["dones"],
            rollout["values"],
            last_values,
            config,
        )
        normalized_advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        batch = _flatten_batch(rollout, normalized_advantages, returns)

        def epoch_step(carry, epoch_key):
            params_t, opt_state_t = carry
            perm = jax.random.permutation(epoch_key, batch_size)
            shuffled = jax.tree_util.tree_map(lambda value: value[perm], batch)
            minibatches = jax.tree_util.tree_map(
                lambda value: value.reshape((config.num_minibatches, minibatch_size) + value.shape[1:]),
                shuffled,
            )

            def minibatch_step(inner_carry, minibatch):
                params_inner, opt_state_inner = inner_carry

                def loss_fn(loss_params):
                    return _ppo_loss(loss_params, minibatch, config, is_discrete=is_discrete)

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_inner)
                updates, opt_state_inner = optimizer.update(grads, opt_state_inner, params_inner)
                params_inner = optax.apply_updates(params_inner, updates)
                metrics = {
                    "loss": loss,
                    "actor_loss": aux["actor_loss"],
                    "value_loss": aux["value_loss"],
                    "entropy": aux["entropy"],
                    "approx_kl": aux["approx_kl"],
                }
                return (params_inner, opt_state_inner), metrics

            (params_t, opt_state_t), minibatch_metrics = jax.lax.scan(
                minibatch_step,
                (params_t, opt_state_t),
                minibatches,
            )
            epoch_metrics = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), minibatch_metrics)
            return (params_t, opt_state_t), epoch_metrics

        epoch_keys = jax.random.split(key_train, config.update_epochs)
        (updated_params, updated_opt_state), epoch_metrics = jax.lax.scan(
            epoch_step,
            (state.params, state.opt_state),
            epoch_keys,
        )
        train_metrics = jax.tree_util.tree_map(lambda value: jnp.mean(value, axis=0), epoch_metrics)
        updated_obs_norm = update_obs_norm(
            state.obs_norm,
            rollout_stats["obs_sum"],
            rollout_stats["obs_sq_sum"],
            rollout_stats["obs_count"],
        )
        is_last_gen = gen == jnp.asarray(config.num_generations - 1, dtype=gen.dtype)
        next_obs_norm = jax.tree_util.tree_map(
            lambda updated, current: jnp.where(is_last_gen, current, updated),
            updated_obs_norm,
            state.obs_norm,
        )
        eval_return = _evaluate_policy(
            updated_params,
            key_eval,
            gen,
            config,
            env,
            env_params,
            act_dim=act_dim,
            is_discrete=is_discrete,
            action_shape=action_shape,
            action_low=action_low,
            action_high=action_high,
            obs_norm_state=next_obs_norm,
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        metrics = {
            "fitness_mean": eval_return,
            "fitness_best": eval_return,
            "fitness_min": eval_return,
            "fitness_std": zero,
            "fitness_median": eval_return,
            "active_shift_windows": _active_shift_windows(gen, config),
            "actor_loss": train_metrics["actor_loss"],
            "value_loss": train_metrics["value_loss"],
            "entropy": train_metrics["entropy"],
            "approx_kl": train_metrics["approx_kl"],
            "loss": train_metrics["loss"],
        }
        for metric_name in ZERO_COMPAT_METRICS:
            metrics[metric_name] = zero
        jax.debug.callback(_wandb_log, metrics, gen)
        next_state = PPOTrainState(
            params=updated_params,
            opt_state=updated_opt_state,
            key=key_next,
            obs_norm=next_obs_norm,
            obs=next_obs,
            env_state=next_env_state,
        )
        return next_state, metrics

    generations = jnp.arange(config.num_generations, dtype=jnp.int32)
    run_impl = jax.jit(lambda state, gens: jax.lax.scan(update_step, state, gens))
    final_state, metrics = run_impl(initial_state, generations)

    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass
    return final_state, metrics
