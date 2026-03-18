from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from srghn import MutationMetadata


def _ravel_individual(indiv) -> jnp.ndarray:
    vec, _ = ravel_pytree(eqx.filter(indiv, eqx.is_array))
    return vec


def population_diversity(pop) -> jnp.ndarray:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    pop_size = jax.tree_util.tree_leaves(pop_arr)[0].shape[0]

    def _select_individual(idx):
        pop_i = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
        return eqx.combine(pop_i, pop_static)

    idxs = jnp.arange(pop_size)
    x = jax.vmap(lambda i: _ravel_individual(_select_individual(i)))(idxs)
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


def compute_metrics(pop, fitness) -> dict:
    return {
        "fitness_mean": jnp.mean(fitness),
        "fitness_best": jnp.max(fitness),
        "fitness_min": jnp.min(fitness),
        "fitness_std": jnp.std(fitness),
        "fitness_median": jnp.median(fitness),
        "diversity": population_diversity(pop),
    }


def _metadata_stats(prefix: str, metadata: MutationMetadata) -> dict:
    return {
        f"{prefix}_mutation_rate_mean": jnp.mean(metadata.mutation_rate_mean),
        f"{prefix}_mutation_rate_std": jnp.std(metadata.mutation_rate_mean),
        f"{prefix}_mutation_rate_max": jnp.max(metadata.mutation_rate_max),
        f"{prefix}_mutation_block_fraction_mean": jnp.mean(metadata.mutation_block_fraction),
        f"{prefix}_mutation_blocks_selected_mean": jnp.mean(metadata.mutation_blocks_selected.astype(jnp.float32)),
        f"{prefix}_mutation_total_blocks_mean": jnp.mean(metadata.mutation_total_blocks.astype(jnp.float32)),
        f"{prefix}_update_rms_mean": jnp.mean(metadata.update_rms),
        f"{prefix}_self_distance_rms_mean": jnp.mean(metadata.self_distance_rms),
    }


def compute_experiment_metrics(
    pop,
    fitness,
    parent_metadata: MutationMetadata,
    elite_metadata: MutationMetadata,
) -> dict:
    metrics = compute_metrics(pop, fitness)
    metrics.update(_metadata_stats("population", parent_metadata))
    metrics.update(_metadata_stats("elite", elite_metadata))
    return metrics
