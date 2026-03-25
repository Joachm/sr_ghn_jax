from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from envs import iter_shift_windows, make_env
from experiments.evosax_adapter import EvosaxStrategyAdapter
from obs_norm import init_obs_norm, update_obs_norm
from experiments.policy_vectors import policy_num_dims, unflatten_policy_vector, zero_policy_vector
from rollout import rollout_episode


@dataclass
class EvosaxState:
    population: jnp.ndarray
    key: jax.random.KeyArray
    obs_norm: Any
    strategy_state: Any
    fitness: jnp.ndarray


def _vector_population_diversity(pop: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(pop, dtype=jnp.float32)
    n = x.shape[0]
    sum_sq = jnp.sum(x * x, axis=1)
    sum_sq_total = jnp.sum(sum_sq)
    sum_vec = jnp.sum(x, axis=0)
    sum_vec_sq = jnp.sum(sum_vec * sum_vec)
    denom = n * (n - 1)
    pairwise_sq = 2.0 * (n * sum_sq_total - sum_vec_sq) / jnp.maximum(denom, 1)
    rms_dist = jnp.sqrt(jnp.maximum(pairwise_sq, 0.0))
    param_count = jnp.maximum(x.shape[1], 1)
    scaled = rms_dist / jnp.sqrt(param_count)
    return jnp.where(denom > 0, scaled, 0.0)


def _zero_mutation_metrics(prefix: str) -> dict:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return {
        f"{prefix}_mutation_rate_mean": zero,
        f"{prefix}_mutation_rate_std": zero,
        f"{prefix}_mutation_rate_max": zero,
        f"{prefix}_mutation_block_fraction_mean": zero,
        f"{prefix}_mutation_blocks_selected_mean": zero,
        f"{prefix}_mutation_total_blocks_mean": zero,
        f"{prefix}_update_rms_mean": zero,
        f"{prefix}_self_distance_rms_mean": zero,
    }


def _compute_vector_metrics(pop: jnp.ndarray, fitness: jnp.ndarray) -> dict:
    metrics = {
        "fitness_mean": jnp.mean(fitness),
        "fitness_best": jnp.max(fitness),
        "fitness_min": jnp.min(fitness),
        "fitness_std": jnp.std(fitness),
        "fitness_median": jnp.median(fitness),
        "diversity": _vector_population_diversity(pop),
    }
    metrics.update(_zero_mutation_metrics("population"))
    metrics.update(_zero_mutation_metrics("elite"))
    return metrics


def _evaluate_policy_vector_with_obs_stats(
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
        lambda rollout_key: rollout_episode(policy_params, rollout_key, gen, config, obs_norm_state)
    )(keys)
    return jnp.mean(returns), jnp.sum(obs_sum, axis=0), jnp.sum(obs_sq_sum, axis=0), jnp.sum(obs_count, axis=0)


def _active_shift_windows(gen: jnp.ndarray, config) -> jnp.ndarray:
    active_windows = 0
    for window in iter_shift_windows(config):
        if window.end_gen is None:
            active = gen >= window.start_gen
        else:
            active = jnp.logical_and(gen >= window.start_gen, gen <= window.end_gen)
        active_windows = active_windows + active.astype(jnp.int32)
    return active_windows.astype(jnp.float32)


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


def run_evosax(key: jax.random.KeyArray, config, policy_spec):
    expected_dims = policy_num_dims(policy_spec)
    solution = zero_policy_vector(policy_spec)
    if solution.shape[0] != expected_dims:
        raise ValueError("Policy vector dimensionality does not match the policy spec.")
    adapter = EvosaxStrategyAdapter(config, solution=solution)
    key_init, key_loop = jax.random.split(key, 2)
    strategy_state = adapter.init(key_init)
    _, _, obs_dim, _, _, _, _, _ = make_env(config)
    state = EvosaxState(
        population=jnp.tile(solution[None, :], (config.pop_size, 1)),
        key=key_loop,
        obs_norm=init_obs_norm(obs_dim),
        strategy_state=strategy_state,
        fitness=jnp.zeros((config.pop_size,), dtype=jnp.float32),
    )

    metrics_history = []
    for gen_idx in range(config.num_generations):
        gen = jnp.asarray(gen_idx, dtype=jnp.int32)
        key_eval, key_ask, key_next = jax.random.split(state.key, 3)
        population, ask_state = adapter.ask(key_ask, state.strategy_state)
        eval_keys = jax.random.split(key_eval, config.pop_size)
        fitness, obs_sum, obs_sq_sum, obs_count = jax.vmap(
            lambda vec, eval_key: _evaluate_policy_vector_with_obs_stats(
                vec,
                eval_key,
                gen,
                config,
                policy_spec,
                state.obs_norm,
            )
        )(population, eval_keys)
        next_strategy_state = adapter.tell(population, fitness, ask_state)
        metrics = _compute_vector_metrics(population, fitness)
        metrics["active_shift_windows"] = _active_shift_windows(gen, config)
        _wandb_log(metrics, gen_idx)

        updated_obs_norm = update_obs_norm(
            state.obs_norm,
            jnp.sum(obs_sum, axis=0),
            jnp.sum(obs_sq_sum, axis=0),
            jnp.sum(obs_count, axis=0),
        )
        next_obs_norm = state.obs_norm if gen_idx == config.num_generations - 1 else updated_obs_norm
        state = EvosaxState(
            population=population,
            key=key_next,
            obs_norm=next_obs_norm,
            strategy_state=next_strategy_state,
            fitness=fitness,
        )
        metrics_history.append(metrics)

    stacked_metrics = {
        metric_name: jnp.stack([generation_metrics[metric_name] for generation_metrics in metrics_history])
        for metric_name in metrics_history[0]
    }
    return state, stacked_metrics
