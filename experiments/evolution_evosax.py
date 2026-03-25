from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from envs import iter_shift_windows, make_env
from experiments.evosax_adapter import EvosaxStrategyAdapter
from metrics import compute_vector_metrics
from obs_norm import init_obs_norm, update_obs_norm
from experiments.policy_vectors import zero_policy_vector
from rollout import evaluate_policy_vector_with_obs_stats


@dataclass
class EvosaxState:
    population: jnp.ndarray
    key: jax.random.KeyArray
    obs_norm: Any
    strategy_state: Any
    fitness: jnp.ndarray


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
    solution = zero_policy_vector(policy_spec)
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
            lambda vec, eval_key: evaluate_policy_vector_with_obs_stats(
                vec,
                eval_key,
                gen,
                config,
                policy_spec,
                state.obs_norm,
            )
        )(population, eval_keys)
        next_strategy_state = adapter.tell(population, fitness, ask_state)
        metrics = compute_vector_metrics(population, fitness)
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
