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


@jax.tree_util.register_pytree_node_class
@dataclass
class EvosaxState:
    population: jnp.ndarray
    key: jax.random.KeyArray
    obs_norm: Any
    strategy_state: Any
    fitness: jnp.ndarray
    pending_obs_sum: jnp.ndarray
    pending_obs_sq_sum: jnp.ndarray
    pending_obs_count: jnp.ndarray
    has_pending_eval: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.population,
            self.key,
            self.obs_norm,
            self.strategy_state,
            self.fitness,
            self.pending_obs_sum,
            self.pending_obs_sq_sum,
            self.pending_obs_count,
            self.has_pending_eval,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        population, key, obs_norm, strategy_state, fitness, pending_obs_sum, pending_obs_sq_sum, pending_obs_count, has_pending_eval = children
        return cls(
            population=population,
            key=key,
            obs_norm=obs_norm,
            strategy_state=strategy_state,
            fitness=fitness,
            pending_obs_sum=pending_obs_sum,
            pending_obs_sq_sum=pending_obs_sq_sum,
            pending_obs_count=pending_obs_count,
            has_pending_eval=has_pending_eval,
        )


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


def _evaluate_population_with_obs_stats(population, key_eval, gen, config, policy_spec, obs_norm_state):
    eval_keys = jax.random.split(key_eval, config.pop_size)
    return jax.vmap(
        lambda vec, eval_key: _evaluate_policy_vector_with_obs_stats(
            vec,
            eval_key,
            gen,
            config,
            policy_spec,
            obs_norm_state,
        )
    )(population, eval_keys)


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


def _run_evosax_impl(init_state: EvosaxState, gens: jnp.ndarray, config, policy_spec, adapter):
    def step_fn(state: EvosaxState, gen: jnp.ndarray):
        key_eval, key_ask, key_tell, key_next = jax.random.split(state.key, 4)

        def use_pending_eval(_):
            return (
                state.population,
                state.fitness,
                state.pending_obs_sum,
                state.pending_obs_sq_sum,
                state.pending_obs_count,
                state.strategy_state,
            )

        def ask_and_evaluate(_):
            population, ask_state = adapter.ask(key_ask, state.strategy_state)
            fitness, obs_sum, obs_sq_sum, obs_count = _evaluate_population_with_obs_stats(
                population,
                key_eval,
                gen,
                config,
                policy_spec,
                state.obs_norm,
            )
            next_strategy_state = adapter.tell(key_tell, population, fitness, ask_state)
            return (
                population,
                fitness,
                jnp.sum(obs_sum, axis=0),
                jnp.sum(obs_sq_sum, axis=0),
                jnp.sum(obs_count, axis=0),
                next_strategy_state,
            )

        population, fitness, obs_sum, obs_sq_sum, obs_count, next_strategy_state = jax.lax.cond(
            state.has_pending_eval,
            use_pending_eval,
            ask_and_evaluate,
            operand=None,
        )
        metrics = _compute_vector_metrics(population, fitness)
        metrics["active_shift_windows"] = _active_shift_windows(gen, config)
        jax.debug.callback(_wandb_log, metrics, gen)

        updated_obs_norm = update_obs_norm(
            state.obs_norm,
            obs_sum,
            obs_sq_sum,
            obs_count,
        )
        is_last_gen = gen == jnp.asarray(config.num_generations - 1, dtype=gen.dtype)
        next_obs_norm = jax.tree_util.tree_map(
            lambda updated, current: jnp.where(is_last_gen, current, updated),
            updated_obs_norm,
            state.obs_norm,
        )
        next_state = EvosaxState(
            population=population,
            key=key_next,
            obs_norm=next_obs_norm,
            strategy_state=next_strategy_state,
            fitness=fitness,
            pending_obs_sum=jnp.zeros_like(state.pending_obs_sum),
            pending_obs_sq_sum=jnp.zeros_like(state.pending_obs_sq_sum),
            pending_obs_count=jnp.zeros_like(state.pending_obs_count),
            has_pending_eval=jnp.asarray(False),
        )
        return next_state, metrics

    return jax.lax.scan(step_fn, init_state, gens)


def run_evosax(key: jax.random.KeyArray, config, policy_spec):
    expected_dims = policy_num_dims(policy_spec)
    solution = zero_policy_vector(policy_spec)
    if solution.shape[0] != expected_dims:
        raise ValueError("Policy vector dimensionality does not match the policy spec.")
    adapter = EvosaxStrategyAdapter(config, solution=solution)
    _, _, obs_dim, _, _, _, _, _ = make_env(config)
    initial_obs_norm = init_obs_norm(obs_dim)
    key_init, key_bootstrap, key_eval0, key_loop = jax.random.split(key, 4)
    zero_obs_sum = jnp.zeros((obs_dim,), dtype=jnp.float32)
    zero_obs_sq_sum = jnp.zeros((obs_dim,), dtype=jnp.float32)
    zero_obs_count = jnp.asarray(0.0, dtype=jnp.float32)

    if adapter.requires_population_init:
        bootstrap_population = adapter.sample_initial_population(key_bootstrap)
        bootstrap_fitness, bootstrap_obs_sum, bootstrap_obs_sq_sum, bootstrap_obs_count = (
            _evaluate_population_with_obs_stats(
                bootstrap_population,
                key_eval0,
                jnp.asarray(0, dtype=jnp.int32),
                config,
                policy_spec,
                initial_obs_norm,
            )
        )
        strategy_state = adapter.init(key_init, bootstrap_population, bootstrap_fitness)
        init_population = bootstrap_population
        init_fitness = bootstrap_fitness
        pending_obs_sum = jnp.sum(bootstrap_obs_sum, axis=0)
        pending_obs_sq_sum = jnp.sum(bootstrap_obs_sq_sum, axis=0)
        pending_obs_count = jnp.sum(bootstrap_obs_count, axis=0)
        has_pending_eval = jnp.asarray(True)
    else:
        strategy_state = adapter.init(key_init)
        init_population = jnp.tile(solution[None, :], (config.pop_size, 1))
        init_fitness = jnp.zeros((config.pop_size,), dtype=jnp.float32)
        pending_obs_sum = zero_obs_sum
        pending_obs_sq_sum = zero_obs_sq_sum
        pending_obs_count = zero_obs_count
        has_pending_eval = jnp.asarray(False)

    init_state = EvosaxState(
        population=init_population,
        key=key_loop,
        obs_norm=initial_obs_norm,
        strategy_state=strategy_state,
        fitness=init_fitness,
        pending_obs_sum=pending_obs_sum,
        pending_obs_sq_sum=pending_obs_sq_sum,
        pending_obs_count=pending_obs_count,
        has_pending_eval=has_pending_eval,
    )
    gens = jnp.arange(config.num_generations, dtype=jnp.int32)
    run_impl = jax.jit(lambda state, generations: _run_evosax_impl(state, generations, config, policy_spec, adapter))
    return run_impl(init_state, gens)
