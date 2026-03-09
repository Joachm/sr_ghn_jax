from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rollout import evaluate_individual


def select_individual(pop, idx: int):
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    indiv_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
    return eqx.combine(indiv_arr, pop_static)


def _population_size(pop) -> int:
    pop_arr, _ = eqx.partition(pop, eqx.is_array)
    first_leaf = jax.tree_util.tree_leaves(pop_arr)[0]
    return int(first_leaf.shape[0])


def evaluate_population(
    pop,
    config,
    *,
    key: jax.random.KeyArray,
    gen: int | None = None,
    obs_norm_state=None,
) -> jnp.ndarray:
    pop_size = _population_size(pop)
    keys = jax.random.split(key, pop_size)
    gen_idx = config.num_generations - 1 if gen is None else gen
    fitness = []
    for i, eval_key in enumerate(keys):
        indiv = select_individual(pop, i)
        score = evaluate_individual(
            indiv,
            eval_key,
            jnp.asarray(gen_idx, dtype=jnp.int32),
            config,
            obs_norm_state=obs_norm_state,
        )
        fitness.append(score)
    return jnp.stack(fitness)


def select_best_individual(pop, config, *, key: jax.random.KeyArray, gen: int | None = None, obs_norm_state=None):
    fitness = evaluate_population(pop, config, key=key, gen=gen, obs_norm_state=obs_norm_state)
    return select_best_individual_from_fitness(pop, fitness)


def select_best_individual_from_fitness(pop, fitness):
    best_index = int(jnp.argmax(fitness))
    best_fitness = float(fitness[best_index])
    best_individual = select_individual(pop, best_index)
    return best_individual, best_index, best_fitness, fitness


def build_solution_artifact(
    *,
    config,
    individual,
    obs_norm_state=None,
    best_index: int | None = None,
    best_fitness: float | None = None,
    population_fitness: Any | None = None,
    metrics: Any | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 2,
        "config": config,
        "individual": individual,
        "obs_norm_state": obs_norm_state,
        "best_index": best_index,
        "best_fitness": best_fitness,
        "population_fitness": population_fitness,
        "metrics": metrics,
    }


def save_solution_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)


def load_solution_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict) or artifact.get("format_version") not in {1, 2}:
        raise ValueError("Unsupported or invalid solution artifact format.")
    if "individual" not in artifact or "config" not in artifact:
        raise ValueError("Solution artifact is missing required fields.")
    return artifact
