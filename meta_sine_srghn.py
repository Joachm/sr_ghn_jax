from __future__ import annotations

"""Minimal SR-GHN meta-learning experiment on few-shot sine regression.

This script is intentionally self-contained and avoids the Gymnax/Brax entrypoints,
so it can run even though some of the repo's RL config helpers are currently missing.

Experiment summary
------------------
- Task family: sine regression A * sin(x + phase)
- Inner adaptation: hill-climbing in parameter space using the SR-GHN's learned
  self-mutation operator (or a fixed-lr / frozen-mutator / vector-GA baseline)
- Outer meta-training: population-based evolution over meta-initializers
- Metrics: meta-train query fitness over generations and held-out adaptation curves

Recommended first run
---------------------
python meta_sine_srghn.py --fast --baselines full fixed_lr vector_ga

A more serious run
------------------
python meta_sine_srghn.py \
  --baselines full fixed_lr frozen_mutator vector_ga \
  --outer-generations 400 \
  --pop-size 32 \
  --meta-batch-size 16 \
  --inner-steps 3 \
  --inner-children 4 \
  --support-k 5 \
  --query-k 25
"""

import argparse
import math
import pickle
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from math import prod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from gnn import GraphEncoder
from graphs import make_chain_graph, make_policy_hierarchical_graph, make_self_hierarchical_graph
from hypernets import DeterministicHead, StochasticHyper
from metrics import compute_experiment_metrics, compute_vector_metrics
from specs import ParamNodeSpec, _policy_metadata, srghn_self_spec
from srghn import SRGHN, make_policy, mutate_with_metadata, mutation_metadata
from experiments.policy_vectors import (
    init_policy_vector_population,
    policy_num_dims,
    unflatten_policy_vector,
)


@jax.tree_util.register_pytree_node_class
@dataclass
class MetaTaskBatch:
    support_x: jnp.ndarray  # [T, K_s, 1]
    support_y: jnp.ndarray  # [T, K_s]
    query_x: jnp.ndarray  # [T, K_q, 1]
    query_y: jnp.ndarray  # [T, K_q]

    def tree_flatten(self):
        children = (self.support_x, self.support_y, self.query_x, self.query_y)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class SRGHNMetaState:
    pop: SRGHN
    key: jax.Array
    pop_fitness: jnp.ndarray

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key, pop_fitness = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness)


@jax.tree_util.register_pytree_node_class
@dataclass
class VectorMetaState:
    pop: jnp.ndarray
    key: jax.Array
    pop_fitness: jnp.ndarray

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key, pop_fitness = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness)


@dataclass(frozen=True)
class MetaSineConfig:
    seed: int = 0
    baseline_name: str = "full"

    # Outer meta-evolution.
    pop_size: int = 32
    outer_children_per_parent: int = 1
    outer_generations: int = 300
    meta_batch_size: int = 8
    test_task_batch_size: int = 128

    # Inner adaptation.
    inner_steps: int = 3
    inner_children: int = 4
    support_k: int = 5
    query_k: int = 25

    # Task distribution.
    amplitude_min: float = 0.1
    amplitude_max: float = 5.0
    phase_max: float = math.pi
    x_min: float = -5.0
    x_max: float = 5.0

    # Policy architecture (regression MLP).
    policy_hidden_dims: tuple[int, ...] = (32, 32)

    # SR-GHN architecture.
    embedding_dim: int = 32
    gnn_hidden_dim: int = 32
    gnn_steps_policy: int = 4
    gnn_steps_self: int = 4
    stoch_coeff_dim: int = 32
    parameter_block_size: int = 64
    mutation_block_ratio: float = 0.25
    mutation_rate_head_dim: int = 4
    clip_params: tuple[float, float] = (-20.0, 20.0)
    clip_std: tuple[float, float] = (0.0, 2.0)
    clip_update: tuple[float, float] = (-0.25, 0.25)
    const_noise_std: float = 1e-3

    # Baseline knobs.
    mutation_exclude_modules: tuple[str, ...] = ()
    fixed_mutation_lr: float | None = None
    vector_ga_sigma: float = 0.05
    vector_ga_init_scale: float = 0.1


def _sizes_from_shapes(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(int(prod(shape)) for shape in shapes)


def regression_policy_spec(hidden_dims: tuple[int, ...]) -> ParamNodeSpec:
    shapes: list[tuple[int, ...]] = []
    prev = 1
    for hidden in hidden_dims:
        shapes.extend(((hidden, prev), (hidden,)))
        prev = hidden
    shapes.extend(((1, prev), (1,)))
    shapes_tuple = tuple(shapes)
    sizes = _sizes_from_shapes(shapes_tuple)
    node_features, group_ids, parent_ids = _policy_metadata(shapes_tuple)
    return ParamNodeSpec(
        shapes=shapes_tuple,
        sizes=sizes,
        max_size=max(sizes) if sizes else 0,
        num_nodes=len(shapes_tuple),
        node_features=node_features,
        module_names=("policy",) * len(shapes_tuple),
        group_ids=group_ids,
        parent_ids=parent_ids,
        context_index=None,
    )


def _empty_self_spec() -> ParamNodeSpec:
    return ParamNodeSpec((), (), 0, 0, (), (), (), (), None)


def build_template_srghn(
    num_self_nodes: int,
    policy_spec: ParamNodeSpec,
    cfg: MetaSineConfig,
    *,
    key: jax.Array,
) -> SRGHN:
    if cfg.embedding_dim != cfg.gnn_hidden_dim:
        raise ValueError("cfg.embedding_dim must equal cfg.gnn_hidden_dim.")

    (
        k_self_emb,
        k_self_ctx,
        k_policy_emb,
        k_self_feat,
        k_policy_feat,
        k_enc_self,
        k_enc_pol,
        k_stoch,
        k_det,
    ) = jax.random.split(key, 9)

    self_node_emb = 0.1 * jax.random.normal(k_self_emb, (num_self_nodes, cfg.embedding_dim), dtype=jnp.float32)
    self_context_emb = 0.1 * jax.random.normal(k_self_ctx, (cfg.embedding_dim,), dtype=jnp.float32)
    policy_node_emb = 0.1 * jax.random.normal(
        k_policy_emb, (policy_spec.num_nodes, cfg.embedding_dim), dtype=jnp.float32
    )

    feat_dim = len(policy_spec.node_features[0])
    self_feat_proj = eqx.nn.Linear(feat_dim, cfg.gnn_hidden_dim, use_bias=True, key=k_self_feat)
    policy_feat_proj = eqx.nn.Linear(feat_dim, cfg.gnn_hidden_dim, use_bias=True, key=k_policy_feat)

    encoder_self = GraphEncoder(cfg.gnn_hidden_dim, cfg.gnn_steps_self, key=k_enc_self)
    encoder_policy = GraphEncoder(cfg.gnn_hidden_dim, cfg.gnn_steps_policy, key=k_enc_pol)
    stoch = StochasticHyper(
        in_dim=cfg.gnn_hidden_dim,
        hidden_dim=cfg.gnn_hidden_dim,
        coeff_dim=cfg.stoch_coeff_dim,
        block_size=cfg.parameter_block_size,
        mutation_block_ratio=cfg.mutation_block_ratio,
        mutation_rate_head_dim=cfg.mutation_rate_head_dim,
        clip_std=cfg.clip_std,
        clip_update=cfg.clip_update,
        const_noise_std=cfg.const_noise_std,
        key=k_stoch,
    )
    det = DeterministicHead(
        cfg.gnn_hidden_dim,
        cfg.gnn_hidden_dim,
        cfg.parameter_block_size,
        key=k_det,
    )

    return SRGHN(
        self_node_emb=self_node_emb,
        self_context_emb=self_context_emb,
        policy_node_emb=policy_node_emb,
        self_feat_proj=self_feat_proj,
        policy_feat_proj=policy_feat_proj,
        encoder_self=encoder_self,
        encoder_policy=encoder_policy,
        stoch=stoch,
        det=det,
        self_graph=make_chain_graph(num_self_nodes, bidir=True),
        policy_graph=make_policy_hierarchical_graph(policy_spec.group_ids, bidir=True),
        self_spec=_empty_self_spec(),
        policy_spec=policy_spec,
        clip_params=cfg.clip_params,
    )


def build_srghn_graphs_and_specs(cfg: MetaSineConfig):
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
    setup_key = jax.random.PRNGKey(cfg.seed)

    temp = build_template_srghn(1, policy_spec, cfg, key=setup_key)
    provisional_self_spec = srghn_self_spec(temp)
    final = build_template_srghn(provisional_self_spec.num_nodes, policy_spec, cfg, key=setup_key)
    self_spec = srghn_self_spec(final)

    self_graph = make_self_hierarchical_graph(
        self_spec.group_ids,
        self_spec.parent_ids,
        context_index=self_spec.context_index,
        bidir=True,
    )
    policy_graph = make_policy_hierarchical_graph(policy_spec.group_ids, bidir=True)
    return (self_graph, policy_graph), (self_spec, policy_spec)


def init_srghn_individual(
    key: jax.Array,
    cfg: MetaSineConfig,
    graphs: tuple[Any, Any],
    specs: tuple[ParamNodeSpec, ParamNodeSpec],
) -> SRGHN:
    self_graph, policy_graph = graphs
    self_spec, policy_spec = specs
    indiv = build_template_srghn(self_spec.num_nodes, policy_spec, cfg, key=key)
    return eqx.tree_at(
        lambda m: (m.self_graph, m.policy_graph, m.self_spec),
        indiv,
        (self_graph, policy_graph, self_spec),
    )


def init_srghn_population(
    key: jax.Array,
    cfg: MetaSineConfig,
    graphs: tuple[Any, Any],
    specs: tuple[ParamNodeSpec, ParamNodeSpec],
) -> SRGHN:
    keys = jax.random.split(key, cfg.pop_size)
    return eqx.filter_vmap(lambda k: init_srghn_individual(k, cfg, graphs, specs))(keys)


def sample_sine_tasks(key: jax.Array, cfg: MetaSineConfig, batch_size: int) -> MetaTaskBatch:
    k_amp, k_phase, k_support, k_query = jax.random.split(key, 4)
    amp = jax.random.uniform(
        k_amp,
        (batch_size, 1, 1),
        minval=cfg.amplitude_min,
        maxval=cfg.amplitude_max,
        dtype=jnp.float32,
    )
    phase = jax.random.uniform(
        k_phase,
        (batch_size, 1, 1),
        minval=0.0,
        maxval=cfg.phase_max,
        dtype=jnp.float32,
    )
    support_x = jax.random.uniform(
        k_support,
        (batch_size, cfg.support_k, 1),
        minval=cfg.x_min,
        maxval=cfg.x_max,
        dtype=jnp.float32,
    )
    query_x = jax.random.uniform(
        k_query,
        (batch_size, cfg.query_k, 1),
        minval=cfg.x_min,
        maxval=cfg.x_max,
        dtype=jnp.float32,
    )
    support_y = jnp.squeeze(amp * jnp.sin(support_x + phase), axis=-1)
    query_y = jnp.squeeze(amp * jnp.sin(query_x + phase), axis=-1)
    return MetaTaskBatch(support_x=support_x, support_y=support_y, query_x=query_x, query_y=query_y)


def regression_forward(params: tuple[jnp.ndarray, ...], x: jnp.ndarray) -> jnp.ndarray:
    h = x
    num_layers = len(params) // 2
    for layer in range(num_layers):
        w = params[2 * layer]
        b = params[2 * layer + 1]
        h = h @ w.T + b
        if layer != num_layers - 1:
            h = jnp.tanh(h)
    return jnp.squeeze(h, axis=-1)


def regression_mse(params: tuple[jnp.ndarray, ...], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = regression_forward(params, x)
    return jnp.mean(jnp.square(pred - y))


def srghn_support_fitness(indiv: SRGHN, sx: jnp.ndarray, sy: jnp.ndarray) -> jnp.ndarray:
    return -regression_mse(make_policy(indiv), sx, sy)


def srghn_query_mse(indiv: SRGHN, qx: jnp.ndarray, qy: jnp.ndarray) -> jnp.ndarray:
    return regression_mse(make_policy(indiv), qx, qy)


def vector_support_fitness(
    policy_vector: jnp.ndarray,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    policy_spec: ParamNodeSpec,
) -> jnp.ndarray:
    params = unflatten_policy_vector(policy_vector, policy_spec)
    return -regression_mse(params, sx, sy)


def vector_query_mse(
    policy_vector: jnp.ndarray,
    qx: jnp.ndarray,
    qy: jnp.ndarray,
    policy_spec: ParamNodeSpec,
) -> jnp.ndarray:
    params = unflatten_policy_vector(policy_vector, policy_spec)
    return regression_mse(params, qx, qy)


def _stack_parent_with_children(parent: SRGHN, children: SRGHN) -> SRGHN:
    parent_arr, _ = eqx.partition(parent, eqx.is_array)
    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p[None, ...], c], axis=0), parent_arr, child_arr)
    return eqx.combine(all_arr, child_static)


def _select_srghn(pop: SRGHN, idx: int | jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    indiv_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
    return eqx.combine(indiv_arr, pop_static)


def srghn_adapt_one_step(
    indiv: SRGHN,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    cfg: MetaSineConfig,
) -> tuple[SRGHN, jnp.ndarray]:
    child_keys = jax.random.split(key, cfg.inner_children)
    mutation_kwargs = {
        "excluded_modules": cfg.mutation_exclude_modules,
        "fixed_mutation_lr": cfg.fixed_mutation_lr,
    }
    children, _ = eqx.filter_vmap(lambda child_key: mutate_with_metadata(indiv, child_key, **mutation_kwargs))(child_keys)
    candidates = _stack_parent_with_children(indiv, children)
    support_fitness = eqx.filter_vmap(lambda cand: srghn_support_fitness(cand, sx, sy))(candidates)
    best_idx = jnp.argmax(support_fitness)
    return _select_srghn(candidates, best_idx), support_fitness[best_idx]


def srghn_adapt(
    indiv: SRGHN,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    cfg: MetaSineConfig,
) -> SRGHN:
    def step_fn(carry, _):
        current, rng = carry
        rng, step_key = jax.random.split(rng)
        nxt, _ = srghn_adapt_one_step(current, step_key, sx, sy, cfg)
        return (nxt, rng), None

    (final, _), _ = jax.lax.scan(step_fn, (indiv, key), None, length=cfg.inner_steps)
    return final


def srghn_meta_fitness(indiv: SRGHN, key: jax.Array, tasks: MetaTaskBatch, cfg: MetaSineConfig) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.support_x.shape[0])

    def per_task(sx, sy, qx, qy, task_key):
        adapted = srghn_adapt(indiv, task_key, sx, sy, cfg)
        return -srghn_query_mse(adapted, qx, qy)

    task_fitness = jax.vmap(per_task)(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, task_keys)
    return jnp.mean(task_fitness)


def srghn_task_curve(
    indiv: SRGHN,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    qx: jnp.ndarray,
    qy: jnp.ndarray,
    cfg: MetaSineConfig,
) -> jnp.ndarray:
    initial_mse = srghn_query_mse(indiv, qx, qy)

    def step_fn(carry, _):
        current, rng = carry
        rng, step_key = jax.random.split(rng)
        nxt, _ = srghn_adapt_one_step(current, step_key, sx, sy, cfg)
        mse = srghn_query_mse(nxt, qx, qy)
        return (nxt, rng), mse

    (_, _), mses = jax.lax.scan(step_fn, (indiv, key), None, length=cfg.inner_steps)
    return jnp.concatenate([initial_mse[None], mses], axis=0)


def vector_ga_adapt_one_step(
    policy_vector: jnp.ndarray,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    cfg: MetaSineConfig,
    policy_spec: ParamNodeSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    noise = cfg.vector_ga_sigma * jax.random.normal(
        key,
        (cfg.inner_children, policy_vector.shape[0]),
        dtype=policy_vector.dtype,
    )
    children = policy_vector[None, :] + noise
    candidates = jnp.concatenate([policy_vector[None, :], children], axis=0)
    fitness = jax.vmap(lambda cand: vector_support_fitness(cand, sx, sy, policy_spec))(candidates)
    best_idx = jnp.argmax(fitness)
    return candidates[best_idx], fitness[best_idx]


def vector_ga_adapt(
    policy_vector: jnp.ndarray,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    cfg: MetaSineConfig,
    policy_spec: ParamNodeSpec,
) -> jnp.ndarray:
    def step_fn(carry, _):
        current, rng = carry
        rng, step_key = jax.random.split(rng)
        nxt, _ = vector_ga_adapt_one_step(current, step_key, sx, sy, cfg, policy_spec)
        return (nxt, rng), None

    (final, _), _ = jax.lax.scan(step_fn, (policy_vector, key), None, length=cfg.inner_steps)
    return final


def vector_ga_meta_fitness(
    policy_vector: jnp.ndarray,
    key: jax.Array,
    tasks: MetaTaskBatch,
    cfg: MetaSineConfig,
    policy_spec: ParamNodeSpec,
) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.support_x.shape[0])

    def per_task(sx, sy, qx, qy, task_key):
        adapted = vector_ga_adapt(policy_vector, task_key, sx, sy, cfg, policy_spec)
        return -vector_query_mse(adapted, qx, qy, policy_spec)

    task_fitness = jax.vmap(per_task)(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, task_keys)
    return jnp.mean(task_fitness)


def vector_ga_task_curve(
    policy_vector: jnp.ndarray,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    qx: jnp.ndarray,
    qy: jnp.ndarray,
    cfg: MetaSineConfig,
    policy_spec: ParamNodeSpec,
) -> jnp.ndarray:
    initial_mse = vector_query_mse(policy_vector, qx, qy, policy_spec)

    def step_fn(carry, _):
        current, rng = carry
        rng, step_key = jax.random.split(rng)
        nxt, _ = vector_ga_adapt_one_step(current, step_key, sx, sy, cfg, policy_spec)
        mse = vector_query_mse(nxt, qx, qy, policy_spec)
        return (nxt, rng), mse

    (_, _), mses = jax.lax.scan(step_fn, (policy_vector, key), None, length=cfg.inner_steps)
    return jnp.concatenate([initial_mse[None], mses], axis=0)


def srghn_outer_step(state: SRGHNMetaState, cfg: MetaSineConfig) -> tuple[SRGHNMetaState, dict[str, jnp.ndarray]]:
    key_next, key_tasks, key_eval, key_children, key_probe = jax.random.split(state.key, 5)
    tasks = sample_sine_tasks(key_tasks, cfg, cfg.meta_batch_size)

    mutation_kwargs = {
        "excluded_modules": cfg.mutation_exclude_modules,
        "fixed_mutation_lr": cfg.fixed_mutation_lr,
    }
    probe_keys = jax.random.split(key_probe, cfg.pop_size)
    parent_metadata = eqx.filter_vmap(
        lambda indiv, probe_key: mutation_metadata(indiv, probe_key, **mutation_kwargs)
    )(state.pop, probe_keys)

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(cfg.pop_size), cfg.outer_children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(key_children, cfg.pop_size * cfg.outer_children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, child_key: mutate_with_metadata(indiv, child_key, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)

    num_candidates = cfg.pop_size * (1 + cfg.outer_children_per_parent)
    eval_keys = jax.random.split(key_eval, num_candidates)
    all_fitness = eqx.filter_vmap(lambda indiv, eval_key: srghn_meta_fitness(indiv, eval_key, tasks, cfg))(
        all_candidates, eval_keys
    )

    select_idx = jnp.argsort(all_fitness)[-cfg.pop_size :]
    next_arr = jax.tree_util.tree_map(lambda x: x[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_val, child_val: jnp.concatenate([parent_val, child_val], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda x: x[select_idx], candidate_metadata)

    metrics = compute_experiment_metrics(state.pop, all_fitness, parent_metadata, elite_metadata)
    metrics["query_mse_best"] = -metrics["fitness_best"]
    metrics["query_mse_mean"] = -metrics["fitness_mean"]
    return SRGHNMetaState(pop=next_pop, key=key_next, pop_fitness=next_fitness), metrics


def vector_outer_step(
    state: VectorMetaState,
    cfg: MetaSineConfig,
    policy_spec: ParamNodeSpec,
) -> tuple[VectorMetaState, dict[str, jnp.ndarray]]:
    key_next, key_tasks, key_eval, key_children = jax.random.split(state.key, 4)
    tasks = sample_sine_tasks(key_tasks, cfg, cfg.meta_batch_size)

    parent_idx = jnp.repeat(jnp.arange(cfg.pop_size), cfg.outer_children_per_parent)
    parent_rep = state.pop[parent_idx]
    child_noise = cfg.vector_ga_sigma * jax.random.normal(
        key_children,
        (cfg.pop_size * cfg.outer_children_per_parent, state.pop.shape[1]),
        dtype=state.pop.dtype,
    )
    children = parent_rep + child_noise
    all_candidates = jnp.concatenate([state.pop, children], axis=0)

    eval_keys = jax.random.split(key_eval, all_candidates.shape[0])
    all_fitness = jax.vmap(
        lambda vec, eval_key: vector_ga_meta_fitness(vec, eval_key, tasks, cfg, policy_spec)
    )(all_candidates, eval_keys)

    select_idx = jnp.argsort(all_fitness)[-cfg.pop_size :]
    next_pop = all_candidates[select_idx]
    next_fitness = all_fitness[select_idx]

    metrics = compute_vector_metrics(state.pop, all_fitness)
    metrics["query_mse_best"] = -metrics["fitness_best"]
    metrics["query_mse_mean"] = -metrics["fitness_mean"]
    return VectorMetaState(pop=next_pop, key=key_next, pop_fitness=next_fitness), metrics


def _history_to_host(history: dict[str, jnp.ndarray]) -> dict[str, Any]:
    return {k: jax.device_get(v) for k, v in history.items()}


def _curve_summary(curves: jnp.ndarray) -> dict[str, Any]:
    curves_host = jax.device_get(curves)
    return {
        "all": curves_host,
        "mean": curves_host.mean(axis=0),
        "std": curves_host.std(axis=0),
        "stderr": curves_host.std(axis=0) / math.sqrt(max(curves_host.shape[0], 1)),
    }


def make_baseline_cfg(base_cfg: MetaSineConfig, baseline_name: str, fixed_lr: float) -> MetaSineConfig:
    if baseline_name == "full":
        return replace(base_cfg, baseline_name=baseline_name, mutation_exclude_modules=(), fixed_mutation_lr=None)
    if baseline_name == "fixed_lr":
        return replace(
            base_cfg,
            baseline_name=baseline_name,
            mutation_exclude_modules=(),
            fixed_mutation_lr=fixed_lr,
        )
    if baseline_name == "frozen_mutator":
        return replace(
            base_cfg,
            baseline_name=baseline_name,
            mutation_exclude_modules=("stoch",),
            fixed_mutation_lr=None,
        )
    if baseline_name == "no_self_reference":
        return replace(
            base_cfg,
            baseline_name=baseline_name,
            mutation_exclude_modules=(
                "self_node_emb",
                "self_context_emb",
                "self_feat_proj",
                "encoder_self",
                "stoch",
            ),
            fixed_mutation_lr=None,
        )
    if baseline_name == "vector_ga":
        return replace(base_cfg, baseline_name=baseline_name)
    raise ValueError(f"Unknown baseline_name: {baseline_name}")


def run_srghn_baseline(cfg: MetaSineConfig) -> dict[str, Any]:
    graphs, specs = build_srghn_graphs_and_specs(cfg)
    self_graph, policy_graph = graphs
    self_spec, policy_spec = specs

    init_key = jax.random.PRNGKey(cfg.seed)
    key_pop, key_loop = jax.random.split(init_key)
    init_pop = init_srghn_population(key_pop, cfg, (self_graph, policy_graph), (self_spec, policy_spec))
    init_state = SRGHNMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.pop_size,), dtype=jnp.float32),
    )

    @jax.jit
    def run_impl(state: SRGHNMetaState):
        return jax.lax.scan(lambda carry, _: srghn_outer_step(carry, cfg), state, None, length=cfg.outer_generations)

    t0 = time.perf_counter()
    final_state, history = run_impl(init_state)
    jax.block_until_ready(history["fitness_best"])
    train_seconds = time.perf_counter() - t0

    best_idx = int(jax.device_get(jnp.argmax(final_state.pop_fitness)))
    best_indiv = _select_srghn(final_state.pop, best_idx)

    heldout_tasks = sample_sine_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg, cfg.test_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.test_task_batch_size)

    @jax.jit
    def eval_curves(indiv: SRGHN, tasks: MetaTaskBatch, keys: jax.Array):
        return jax.vmap(
            lambda sx, sy, qx, qy, task_key: srghn_task_curve(indiv, task_key, sx, sy, qx, qy, cfg)
        )(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, keys)

    t1 = time.perf_counter()
    curves = eval_curves(best_indiv, heldout_tasks, heldout_keys)
    jax.block_until_ready(curves)
    eval_seconds = time.perf_counter() - t1

    return {
        "kind": "srghn",
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _curve_summary(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "best_index": best_idx,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }


def run_vector_ga_baseline(cfg: MetaSineConfig) -> dict[str, Any]:
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
    num_dims = policy_num_dims(policy_spec)

    init_key = jax.random.PRNGKey(cfg.seed)
    key_pop, key_loop = jax.random.split(init_key)
    init_pop = init_policy_vector_population(
        key_pop,
        cfg.pop_size,
        policy_spec,
        scale=cfg.vector_ga_init_scale,
    )
    init_state = VectorMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.pop_size,), dtype=jnp.float32),
    )

    @jax.jit
    def run_impl(state: VectorMetaState):
        return jax.lax.scan(
            lambda carry, _: vector_outer_step(carry, cfg, policy_spec),
            state,
            None,
            length=cfg.outer_generations,
        )

    t0 = time.perf_counter()
    final_state, history = run_impl(init_state)
    jax.block_until_ready(history["fitness_best"])
    train_seconds = time.perf_counter() - t0

    best_idx = int(jax.device_get(jnp.argmax(final_state.pop_fitness)))
    best_vec = final_state.pop[best_idx]

    heldout_tasks = sample_sine_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg, cfg.test_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.test_task_batch_size)

    @jax.jit
    def eval_curves(policy_vector: jnp.ndarray, tasks: MetaTaskBatch, keys: jax.Array):
        return jax.vmap(
            lambda sx, sy, qx, qy, task_key: vector_ga_task_curve(
                policy_vector,
                task_key,
                sx,
                sy,
                qx,
                qy,
                cfg,
                policy_spec,
            )
        )(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, keys)

    t1 = time.perf_counter()
    curves = eval_curves(best_vec, heldout_tasks, heldout_keys)
    jax.block_until_ready(curves)
    eval_seconds = time.perf_counter() - t1

    return {
        "kind": "vector_ga",
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
        "policy_num_dims": int(num_dims),
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _curve_summary(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "best_index": best_idx,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }


def maybe_save_plots(output_stem: Path, results: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for baseline_name, payload in results.items():
        hist = payload["train_history"]
        ax.plot(hist["query_mse_best"], label=baseline_name)
    ax.set_title("Meta-train best query MSE")
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Query MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_train.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for baseline_name, payload in results.items():
        curve = payload["adaptation_curve_query_mse"]
        xs = list(range(len(curve["mean"])))
        mean = curve["mean"]
        stderr = curve["stderr"]
        ax.plot(xs, mean, label=baseline_name)
        ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_title("Held-out adaptation curve")
    ax.set_xlabel("Inner mutation step")
    ax.set_ylabel("Query MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_adaptation.png"), dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-learning with SR-GHNs on sine regression.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=("full", "fixed_lr", "vector_ga"),
        choices=("full", "fixed_lr", "frozen_mutator", "no_self_reference", "vector_ga"),
        help="Baselines to run.",
    )
    parser.add_argument("--output", default="meta_sine_srghn_results.pkl")
    parser.add_argument("--plot", action="store_true", help="Also save PNG plots next to the pickle.")
    parser.add_argument("--fast", action="store_true", help="Use a smaller debug/smoke-test configuration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--outer-children-per-parent", type=int, default=1)
    parser.add_argument("--outer-generations", type=int, default=300)
    parser.add_argument("--meta-batch-size", type=int, default=8)
    parser.add_argument("--test-task-batch-size", type=int, default=128)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--inner-children", type=int, default=4)
    parser.add_argument("--support-k", type=int, default=5)
    parser.add_argument("--query-k", type=int, default=25)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=(32, 32))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-steps-policy", type=int, default=4)
    parser.add_argument("--gnn-steps-self", type=int, default=4)
    parser.add_argument("--stoch-coeff-dim", type=int, default=32)
    parser.add_argument("--parameter-block-size", type=int, default=64)
    parser.add_argument("--mutation-block-ratio", type=float, default=0.25)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=4)
    parser.add_argument("--const-noise-std", type=float, default=1e-3)
    parser.add_argument("--fixed-lr", type=float, default=0.05, help="Used by the fixed_lr baseline.")
    parser.add_argument("--vector-ga-sigma", type=float, default=0.05)
    parser.add_argument("--vector-ga-init-scale", type=float, default=0.1)
    return parser.parse_args()


def make_base_cfg(args: argparse.Namespace) -> MetaSineConfig:
    cfg = MetaSineConfig(
        seed=args.seed,
        pop_size=args.pop_size,
        outer_children_per_parent=args.outer_children_per_parent,
        outer_generations=args.outer_generations,
        meta_batch_size=args.meta_batch_size,
        test_task_batch_size=args.test_task_batch_size,
        inner_steps=args.inner_steps,
        inner_children=args.inner_children,
        support_k=args.support_k,
        query_k=args.query_k,
        policy_hidden_dims=tuple(args.policy_hidden_dims),
        embedding_dim=args.embedding_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_steps_policy=args.gnn_steps_policy,
        gnn_steps_self=args.gnn_steps_self,
        stoch_coeff_dim=args.stoch_coeff_dim,
        parameter_block_size=args.parameter_block_size,
        mutation_block_ratio=args.mutation_block_ratio,
        mutation_rate_head_dim=args.mutation_rate_head_dim,
        const_noise_std=args.const_noise_std,
        vector_ga_sigma=args.vector_ga_sigma,
        vector_ga_init_scale=args.vector_ga_init_scale,
    )
    if args.fast:
        cfg = replace(
            cfg,
            pop_size=16,
            outer_generations=100,
            meta_batch_size=8,
            test_task_batch_size=64,
            inner_steps=2,
            inner_children=3,
            policy_hidden_dims=(16, 16),
            embedding_dim=16,
            gnn_hidden_dim=16,
            gnn_steps_policy=3,
            gnn_steps_self=3,
            stoch_coeff_dim=16,
            parameter_block_size=32,
            mutation_rate_head_dim=2,
        )
    return cfg


def main() -> None:
    args = parse_args()
    base_cfg = make_base_cfg(args)

    results: dict[str, Any] = {}
    for baseline_name in args.baselines:
        cfg = make_baseline_cfg(base_cfg, baseline_name, fixed_lr=args.fixed_lr)
        print(f"[run] baseline={baseline_name}", flush=True)
        if baseline_name == "vector_ga":
            payload = run_vector_ga_baseline(cfg)
        else:
            payload = run_srghn_baseline(cfg)
        results[baseline_name] = payload

        curve = payload["adaptation_curve_query_mse"]["mean"]
        print(
            f"[done] baseline={baseline_name} "
            f"train_s={payload['train_seconds']:.2f} eval_s={payload['eval_seconds']:.2f} "
            f"heldout_mse_steps={curve.tolist()}",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_config": asdict(base_cfg),
        "results": results,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}")

    if args.plot:
        maybe_save_plots(output_path.with_suffix(""), results)
        print(f"[saved] plots near {output_path.with_suffix('')}")


if __name__ == "__main__":
    main()
