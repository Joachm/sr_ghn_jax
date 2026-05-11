from __future__ import annotations

"""Generic paired-loop sine-regression meta-learning harness for SR-GHN and vector controls.

This version makes the *inner loop generic* and lets you pair the same search family in
both loops when desired:

- SR-GHN self / self            : search_object=srghn, outer=srghn_self, inner=srghn_self
- CMA-ES / CMA-ES              : search_object=vector, outer=evosax(CMA_ES), inner=evosax(CMA_ES)
- OpenES / OpenES              : search_object=vector, outer=evosax(Open_ES), inner=evosax(Open_ES)
- Gaussian GA / Gaussian GA    : search_object=vector, outer=gaussian, inner=gaussian

The main intended use is the fully self-referential SR-GHN condition, with vector-based
ES/GA controls available through the same task/evaluation protocol.
"""

import argparse
import math
import pickle
import time
from dataclasses import asdict, dataclass
from functools import partial
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from gnn import GraphEncoder
from graphs import make_chain_graph, make_policy_hierarchical_graph, make_self_hierarchical_graph
from hypernets import DeterministicHead, StochasticHyper
from metrics import compute_experiment_metrics, compute_vector_metrics, zero_mutation_metrics
from specs import ParamNodeSpec, _policy_metadata, srghn_self_spec_from_layout
from srghn import SRGHN, make_policy, mutate_with_metadata, mutation_metadata
from experiments.evosax_adapter import EvosaxStrategyAdapter, fitness_for_evosax
from experiments.policy_vectors import (
    init_policy_vector_population,
    policy_num_dims,
    unflatten_policy_vector,
    zero_policy_vector,
)
from experiment_configs import MetaSineConfig, build_meta_sine_config, print_resolved_config, resolved_config_payload


@jax.tree_util.register_pytree_node_class
@dataclass
class MetaTaskBatch:
    support_x: jnp.ndarray
    support_y: jnp.ndarray
    query_x: jnp.ndarray
    query_y: jnp.ndarray

    def tree_flatten(self):
        return (self.support_x, self.support_y, self.query_x, self.query_y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass
class SRGHNMetaState:
    pop: SRGHN
    key: jax.Array
    pop_fitness: jnp.ndarray
    best_fitness: jnp.ndarray
    best_indiv: SRGHN

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness, self.best_fitness, self.best_indiv), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key, pop_fitness, best_fitness, best_indiv = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness, best_fitness=best_fitness, best_indiv=best_indiv)


@jax.tree_util.register_pytree_node_class
@dataclass
class VectorGAMetaState:
    pop: jnp.ndarray
    key: jax.Array
    pop_fitness: jnp.ndarray
    best_fitness: jnp.ndarray
    best_solution: jnp.ndarray

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness, self.best_fitness, self.best_solution), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key, pop_fitness, best_fitness, best_solution = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness, best_fitness=best_fitness, best_solution=best_solution)


@jax.tree_util.register_pytree_node_class
@dataclass
class VectorEvosaxMetaState:
    key: jax.Array
    population: jnp.ndarray
    fitness: jnp.ndarray
    strategy_state: Any
    best_fitness: jnp.ndarray
    best_solution: jnp.ndarray

    def tree_flatten(self):
        return (
            self.key,
            self.population,
            self.fitness,
            self.strategy_state,
            self.best_fitness,
            self.best_solution,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        key, population, fitness, strategy_state, best_fitness, best_solution = children
        return cls(
            key=key,
            population=population,
            fitness=fitness,
            strategy_state=strategy_state,
            best_fitness=best_fitness,
            best_solution=best_solution,
        )


@jax.tree_util.register_pytree_node_class
@dataclass
class LocalEvosaxState:
    strategy_state: Any
    population: jnp.ndarray
    fitness: jnp.ndarray
    best_solution: jnp.ndarray
    best_fitness: jnp.ndarray
    needs_initial_shift: jnp.ndarray
    initial_center: jnp.ndarray
    key: jax.Array

    def tree_flatten(self):
        return (
            self.strategy_state,
            self.population,
            self.fitness,
            self.best_solution,
            self.best_fitness,
            self.needs_initial_shift,
            self.initial_center,
            self.key,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    search_object: str
    outer_optimizer: str
    inner_optimizer: str
    outer_evosax_algo: str | None = None
    inner_evosax_algo: str | None = None
    mutation_exclude_modules: tuple[str, ...] = ()
    fixed_mutation_lr: float | None = None


# --------------------------------------------------------------------------------------
# Specs / initialization
# --------------------------------------------------------------------------------------


def _sizes_from_shapes(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(int(prod(shape)) for shape in shapes)


def _safe_wandb_import():
    try:
        import wandb
    except Exception:
        return None
    return wandb


def wandb_config_payload(
    cfg: MetaSineConfig,
    cond: ConditionSpec,
    *,
    policy_spec: ParamNodeSpec | None = None,
    self_spec: ParamNodeSpec | None = None,
) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["condition"] = asdict(cond)
    if policy_spec is not None:
        payload["policy_spec_shapes"] = [tuple(shape) for shape in policy_spec.shapes]
        payload["policy_num_dims"] = int(policy_num_dims(policy_spec))
    if self_spec is not None:
        payload["self_spec_shapes"] = [tuple(shape) for shape in self_spec.shapes]
        payload["self_num_nodes"] = int(self_spec.num_nodes)
    return payload


def _wandb_run_name(cfg: MetaSineConfig, cond: ConditionSpec) -> str:
    prefix = cfg.wandb_name or "meta_sine_srghn"
    return f"{prefix}-{cond.name}"


def _wandb_init_run(
    cfg: MetaSineConfig,
    cond: ConditionSpec,
    *,
    policy_spec: ParamNodeSpec | None = None,
    self_spec: ParamNodeSpec | None = None,
):
    if cfg.wandb_project is None:
        return None
    wandb = _safe_wandb_import()
    if wandb is None:
        return None
    try:
        if wandb.run is None:
            return wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                name=_wandb_run_name(cfg, cond),
                config=wandb_config_payload(cfg, cond, policy_spec=policy_spec, self_spec=self_spec),
            )
    except Exception:
        return None
    return wandb.run


def _wandb_log_metrics(metrics: dict[str, Any], step: int, *, prefix: str = "train") -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    payload = {f"{prefix}/{key}": float(value) for key, value in metrics.items()}
    payload["gen"] = int(step)
    try:
        wandb.log(payload, step=int(step))
    except Exception:
        pass


def _wandb_log_summary(
    cfg: MetaSineConfig,
    cond: ConditionSpec,
    payload: dict[str, Any],
) -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return

    history = payload["train_history"]
    curve = payload["adaptation_curve_query_mse"]
    final_population_fitness = jnp.asarray(payload["final_population_fitness"])
    summary_payload = {
        "final/train_fitness_best": float(history["fitness_best"][-1]),
        "final/train_fitness_mean": float(history["fitness_mean"][-1]),
        "final/train_query_mse_best": float(history["query_mse_best"][-1]),
        "final/train_query_mse_mean": float(history["query_mse_mean"][-1]),
        "final/train_diversity": float(history["diversity"][-1]),
        "final/final_population_fitness_best": float(jnp.max(final_population_fitness)),
        "final/final_population_fitness_mean": float(jnp.mean(final_population_fitness)),
        "final/heldout_curve_pre_mse_mean": float(curve["mean"][0]),
        "final/heldout_curve_post_mse_mean": float(curve["mean"][-1]),
        "final/heldout_curve_stderr_post_mse": float(curve["stderr"][-1]),
        "final/heldout_curve_improvement_mean": float(curve["mean"][0] - curve["mean"][-1]),
        "final/heldout_curve_length": int(len(curve["mean"])),
    }
    try:
        wandb.log(summary_payload, step=int(cfg.outer_generations))
    except Exception:
        pass

    if not cfg.wandb_log_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["query_mse_best"], label="best")
    axes[0].plot(history["query_mse_mean"], label="mean")
    axes[0].set_title("Outer meta-train")
    axes[0].set_xlabel("Outer generation")
    axes[0].set_ylabel("Query MSE")
    axes[0].legend()

    xs = list(range(len(curve["mean"])))
    axes[1].plot(xs, curve["mean"], label="mean")
    axes[1].fill_between(xs, curve["mean"] - curve["stderr"], curve["mean"] + curve["stderr"], alpha=0.2)
    axes[1].set_title("Held-out adaptation")
    axes[1].set_xlabel("Inner generation")
    axes[1].set_ylabel("Query MSE")
    axes[1].legend()
    fig.suptitle(cond.name)
    fig.tight_layout()

    try:
        wandb.log({"plots/summary": wandb.Image(fig)}, step=int(cfg.outer_generations))
    except Exception:
        pass
    finally:
        plt.close(fig)


def _wandb_finish_run() -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass



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



def _srghn_self_layout(
    policy_spec: ParamNodeSpec,
    cfg: MetaSineConfig,
) -> tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]:
    if cfg.embedding_dim != cfg.gnn_hidden_dim:
        raise ValueError("cfg.embedding_dim must equal cfg.gnn_hidden_dim.")

    feat_dim = len(policy_spec.node_features[0])
    hidden_dim = cfg.gnn_hidden_dim
    coeff_dim = cfg.stoch_coeff_dim
    block_size = cfg.parameter_block_size
    mutation_rate_head_dim = cfg.mutation_rate_head_dim

    leaf_entries: list[tuple[tuple[str, ...], tuple[int, ...]]] = [
        (("self_context_emb",), (cfg.embedding_dim,)),
        (("policy_node_emb",), (policy_spec.num_nodes, cfg.embedding_dim)),
        (("self_feat_proj", "weight"), (hidden_dim, feat_dim)),
        (("self_feat_proj", "bias"), (hidden_dim,)),
        (("policy_feat_proj", "weight"), (hidden_dim, feat_dim)),
        (("policy_feat_proj", "bias"), (hidden_dim,)),
        (("encoder_self", "msg", "weight"), (hidden_dim, hidden_dim)),
        (("encoder_self", "msg", "bias"), (hidden_dim,)),
        (("encoder_self", "gru", "weight_ih"), (3 * hidden_dim, hidden_dim)),
        (("encoder_self", "gru", "weight_hh"), (3 * hidden_dim, hidden_dim)),
        (("encoder_self", "gru", "bias"), (3 * hidden_dim,)),
        (("encoder_self", "gru", "bias_n"), (hidden_dim,)),
        (("encoder_self", "rel_emb"), (8, hidden_dim)),
        (("encoder_policy", "msg", "weight"), (hidden_dim, hidden_dim)),
        (("encoder_policy", "msg", "bias"), (hidden_dim,)),
        (("encoder_policy", "gru", "weight_ih"), (3 * hidden_dim, hidden_dim)),
        (("encoder_policy", "gru", "weight_hh"), (3 * hidden_dim, hidden_dim)),
        (("encoder_policy", "gru", "bias"), (3 * hidden_dim,)),
        (("encoder_policy", "gru", "bias_n"), (hidden_dim,)),
        (("encoder_policy", "rel_emb"), (8, hidden_dim)),
        (("stoch", "trunk", "lin1", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "trunk", "lin2", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "global_proj", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "global_proj", "bias"), (hidden_dim,)),
        (("stoch", "child_mu_head", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "child_mu_head", "bias"), (hidden_dim,)),
        (("stoch", "child_logstd_head", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "child_logstd_head", "bias"), (hidden_dim,)),
        (("stoch", "pos_proj", "weight"), (hidden_dim, 6)),
        (("stoch", "pos_proj", "bias"), (hidden_dim,)),
        (("stoch", "block_proj", "weight"), (hidden_dim, hidden_dim)),
        (("stoch", "block_proj", "bias"), (hidden_dim,)),
        (("stoch", "score_head", "weight"), (1, hidden_dim)),
        (("stoch", "score_head", "bias"), (1,)),
        (("stoch", "std_head", "weight"), (coeff_dim, hidden_dim)),
        (("stoch", "lr_head", "weight"), (mutation_rate_head_dim, hidden_dim)),
        (("stoch", "lr_head", "bias"), (mutation_rate_head_dim,)),
        (("stoch", "basis"), (coeff_dim, block_size)),
        (("det", "context_proj", "weight"), (hidden_dim, hidden_dim)),
        (("det", "context_proj", "bias"), (hidden_dim,)),
        (("det", "pos_proj", "weight"), (hidden_dim, 6)),
        (("det", "pos_proj", "bias"), (hidden_dim,)),
        (("det", "block_proj", "weight"), (hidden_dim, hidden_dim)),
        (("det", "block_proj", "bias"), (hidden_dim,)),
        (("det", "out_proj", "weight"), (block_size, hidden_dim)),
        (("det", "out_proj", "bias"), (block_size,)),
    ]

    num_self_nodes = 1 + len(leaf_entries)
    return (
        (("self_node_emb",), (num_self_nodes, cfg.embedding_dim)),
        *leaf_entries,
    )


def _srghn_self_spec(policy_spec: ParamNodeSpec, cfg: MetaSineConfig) -> ParamNodeSpec:
    return srghn_self_spec_from_layout(_srghn_self_layout(policy_spec, cfg))


def build_srghn_module(
    self_spec: ParamNodeSpec,
    policy_spec: ParamNodeSpec,
    cfg: MetaSineConfig,
    *,
    key: jax.Array,
    self_graph: Any | None = None,
    policy_graph: Any | None = None,
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

    self_node_emb = 0.1 * jax.random.normal(k_self_emb, (self_spec.num_nodes, cfg.embedding_dim), dtype=jnp.float32)
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

    if self_graph is None:
        self_graph = make_chain_graph(self_spec.num_nodes, bidir=True)
    if policy_graph is None:
        policy_graph = make_policy_hierarchical_graph(policy_spec.group_ids, bidir=True)

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
        self_graph=self_graph,
        policy_graph=policy_graph,
        self_spec=self_spec,
        policy_spec=policy_spec,
        clip_params=cfg.clip_params,
    )



def build_srghn_graphs_and_specs(cfg: MetaSineConfig):
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
    self_spec = _srghn_self_spec(policy_spec, cfg)

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
    return build_srghn_module(
        self_spec,
        policy_spec,
        cfg,
        key=key,
        self_graph=self_graph,
        policy_graph=policy_graph,
    )



def init_srghn_population(
    key: jax.Array,
    pop_size: int,
    cfg: MetaSineConfig,
    graphs: tuple[Any, Any],
    specs: tuple[ParamNodeSpec, ParamNodeSpec],
) -> SRGHN:
    keys = jax.random.split(key, pop_size)
    return eqx.filter_vmap(lambda k: init_srghn_individual(k, cfg, graphs, specs))(keys)


@partial(jax.jit, static_argnums=(0, 1))
def run_srghn_compiled(cfg: MetaSineConfig, cond: ConditionSpec, key: jax.Array):
    graphs, specs = build_srghn_graphs_and_specs(cfg)
    self_graph, policy_graph = graphs
    self_spec, policy_spec = specs

    key_pop, key_loop = jax.random.split(key)
    init_pop = init_srghn_population(
        key_pop,
        cfg.outer_pop_size,
        cfg,
        (self_graph, policy_graph),
        (self_spec, policy_spec),
    )
    init_best = _select_srghn(init_pop, 0)
    init_state = SRGHNMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.outer_pop_size,), dtype=jnp.float32),
        best_fitness=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_indiv=init_best,
    )

    gens = jnp.arange(cfg.outer_generations, dtype=jnp.int32)
    final_state, history = jax.lax.scan(
        lambda carry, gen: srghn_outer_step(carry, gen, cfg, cond),
        init_state,
        gens,
    )

    heldout_tasks = sample_sine_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg, cfg.test_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.test_task_batch_size)

    curves = jax.vmap(
        lambda sx, sy, qx, qy, task_key: srghn_task_curve(final_state.best_indiv, task_key, sx, sy, qx, qy, cfg, cond)
    )(heldout_tasks.support_x, heldout_tasks.support_y, heldout_tasks.query_x, heldout_tasks.query_y, heldout_keys)
    return final_state, history, curves


# --------------------------------------------------------------------------------------
# Tasks / regression helpers
# --------------------------------------------------------------------------------------


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



def vector_support_fitness(policy_vector: jnp.ndarray, sx: jnp.ndarray, sy: jnp.ndarray, policy_spec: ParamNodeSpec) -> jnp.ndarray:
    params = unflatten_policy_vector(policy_vector, policy_spec)
    return -regression_mse(params, sx, sy)



def vector_query_mse(policy_vector: jnp.ndarray, qx: jnp.ndarray, qy: jnp.ndarray, policy_spec: ParamNodeSpec) -> jnp.ndarray:
    params = unflatten_policy_vector(policy_vector, policy_spec)
    return regression_mse(params, qx, qy)


# --------------------------------------------------------------------------------------
# Tree helpers / metrics helpers
# --------------------------------------------------------------------------------------


def _stack_parent_with_children(parent: SRGHN, children: SRGHN) -> SRGHN:
    parent_arr, _ = eqx.partition(parent, eqx.is_array)
    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p[None, ...], c], axis=0), parent_arr, child_arr)
    return eqx.combine(all_arr, child_static)



def _select_srghn(pop: SRGHN, idx: int | jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    indiv_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
    return eqx.combine(indiv_arr, pop_static)



def _repeat_srghn(indiv: SRGHN, repeats: int) -> SRGHN:
    arr, static = eqx.partition(indiv, eqx.is_array)
    repeated = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (repeats,) + x.shape), arr)
    return eqx.combine(repeated, static)



def _summary_from_curves(curves: jnp.ndarray) -> dict[str, Any]:
    curves_host = jax.device_get(curves)
    return {
        "all": curves_host,
        "mean": curves_host.mean(axis=0),
        "std": curves_host.std(axis=0),
        "stderr": curves_host.std(axis=0) / math.sqrt(max(curves_host.shape[0], 1)),
    }



def _history_to_host(history: dict[str, jnp.ndarray]) -> dict[str, Any]:
    return {k: jax.device_get(v) for k, v in history.items()}


# --------------------------------------------------------------------------------------
# Condition parsing
# --------------------------------------------------------------------------------------


PRESET_CONDITIONS: dict[str, ConditionSpec] = {
    "srghn_self_self": ConditionSpec(
        name="srghn_self_self",
        search_object="srghn",
        outer_optimizer="srghn_self",
        inner_optimizer="srghn_self",
    ),
    "srghn_fixedlr_self_self": ConditionSpec(
        name="srghn_fixedlr_self_self",
        search_object="srghn",
        outer_optimizer="srghn_self",
        inner_optimizer="srghn_self",
        fixed_mutation_lr=0.05,
    ),
    "srghn_frozenmut_self_self": ConditionSpec(
        name="srghn_frozenmut_self_self",
        search_object="srghn",
        outer_optimizer="srghn_self",
        inner_optimizer="srghn_self",
        mutation_exclude_modules=("stoch",),
    ),
    "vector_ga_ga": ConditionSpec(
        name="vector_ga_ga",
        search_object="vector",
        outer_optimizer="gaussian",
        inner_optimizer="gaussian",
    ),
    "cma_es_cma_es": ConditionSpec(
        name="cma_es_cma_es",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="CMA_ES",
        inner_evosax_algo="CMA_ES",
    ),
    "sep_cma_es_sep_cma_es": ConditionSpec(
        name="sep_cma_es_sep_cma_es",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="Sep_CMA_ES",
        inner_evosax_algo="Sep_CMA_ES",
    ),
    "open_es_open_es": ConditionSpec(
        name="open_es_open_es",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="Open_ES",
        inner_evosax_algo="Open_ES",
    ),
    "simple_ga_simple_ga": ConditionSpec(
        name="simple_ga_simple_ga",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="SimpleGA",
        inner_evosax_algo="SimpleGA",
    ),
    "samr_ga_samr_ga": ConditionSpec(
        name="samr_ga_samr_ga",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="SAMR_GA",
        inner_evosax_algo="SAMR_GA",
    ),
    "gesmr_ga_gesmr_ga": ConditionSpec(
        name="gesmr_ga_gesmr_ga",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="GESMR_GA",
        inner_evosax_algo="GESMR_GA",
    ),
    "pgpe_pgpe": ConditionSpec(
        name="pgpe_pgpe",
        search_object="vector",
        outer_optimizer="evosax",
        inner_optimizer="evosax",
        outer_evosax_algo="PGPE",
        inner_evosax_algo="PGPE",
    ),
}



def validate_condition(cond: ConditionSpec) -> ConditionSpec:
    if cond.search_object not in ("srghn", "vector"):
        raise ValueError(f"Unsupported search_object '{cond.search_object}' for {cond.name}.")
    if cond.outer_optimizer not in ("srghn_self", "gaussian", "evosax"):
        raise ValueError(f"Unsupported outer optimizer '{cond.outer_optimizer}' for {cond.name}.")
    if cond.inner_optimizer not in ("srghn_self", "gaussian", "evosax"):
        raise ValueError(f"Unsupported inner optimizer '{cond.inner_optimizer}' for {cond.name}.")
    if cond.search_object == "srghn":
        if cond.outer_optimizer != "srghn_self" or cond.inner_optimizer != "srghn_self":
            raise ValueError(f"SR-GHN conditions currently support only srghn_self inner/outer search: {cond.name}")
    if cond.search_object == "vector":
        if cond.outer_optimizer == "evosax" and not cond.outer_evosax_algo:
            raise ValueError(f"Vector evosax outer loop needs outer_evosax_algo: {cond.name}")
        if cond.inner_optimizer == "evosax" and not cond.inner_evosax_algo:
            raise ValueError(f"Vector evosax inner loop needs inner_evosax_algo: {cond.name}")
    return cond



def parse_condition_spec(text: str) -> ConditionSpec:
    if text in PRESET_CONDITIONS:
        return PRESET_CONDITIONS[text]
    if "=" not in text:
        raise ValueError(
            f"Unknown condition '{text}'. Use a preset name or a custom spec like "
            "name=search_object,outer_optimizer,inner_optimizer,outer_algo,inner_algo"
        )
    name, rhs = text.split("=", 1)
    parts = [part.strip() for part in rhs.split(",")]
    if len(parts) < 3:
        raise ValueError(
            "Custom condition specs need at least 3 comma-separated fields: "
            "search_object,outer_optimizer,inner_optimizer"
        )
    while len(parts) < 5:
        parts.append("")
    search_object, outer_optimizer, inner_optimizer, outer_algo, inner_algo = parts[:5]
    cond = ConditionSpec(
        name=name.strip(),
        search_object=search_object,
        outer_optimizer=outer_optimizer,
        inner_optimizer=inner_optimizer,
        outer_evosax_algo=outer_algo or None,
        inner_evosax_algo=inner_algo or None,
    )
    return validate_condition(cond)


# --------------------------------------------------------------------------------------
# SR-GHN self-search, shared by inner and outer loops
# --------------------------------------------------------------------------------------



def _srghn_mutation_kwargs(cond: ConditionSpec) -> dict[str, Any]:
    return {
        "excluded_modules": cond.mutation_exclude_modules,
        "fixed_mutation_lr": cond.fixed_mutation_lr,
    }



def srghn_population_generation(
    pop: SRGHN,
    key: jax.Array,
    pop_size: int,
    children_per_parent: int,
    fitness_fn,
    cond: ConditionSpec,
):
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key)
    probe_keys = jax.random.split(probe_key, pop_size)
    parent_metadata = eqx.filter_vmap(lambda indiv, k: mutation_metadata(indiv, k, **mutation_kwargs))(pop, probe_keys)

    if children_per_parent <= 0:
        all_candidates = pop
        child_metadata = parent_metadata
        all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)
        select_idx = jnp.argsort(all_fitness)[-pop_size:]
        next_pop = _select_batch_srghn(all_candidates, select_idx)
        next_fitness = all_fitness[select_idx]
        elite_metadata = jax.tree_util.tree_map(lambda x: x[select_idx], parent_metadata)
        metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
        return next_pop, next_fitness, metrics

    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(pop_size), children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, pop_size * children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, k: mutate_with_metadata(indiv, k, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)

    select_idx = jnp.argsort(all_fitness)[-pop_size:]
    next_arr = jax.tree_util.tree_map(lambda x: x[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_val, child_val: jnp.concatenate([parent_val, child_val], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda x: x[select_idx], candidate_metadata)
    metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
    return next_pop, next_fitness, metrics



def _select_batch_srghn(pop: SRGHN, idx: jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    batch_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
    return eqx.combine(batch_arr, pop_static)



def init_srghn_local_population(indiv: SRGHN, key: jax.Array, cfg: MetaSineConfig, cond: ConditionSpec) -> SRGHN:
    if cfg.inner_pop_size <= 1:
        return _repeat_srghn(indiv, 1)
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    child_keys = jax.random.split(key, cfg.inner_pop_size - 1)
    children, _ = eqx.filter_vmap(lambda k: mutate_with_metadata(indiv, k, **mutation_kwargs))(child_keys)
    return _stack_parent_with_children(indiv, children)



def srghn_adapt(indiv: SRGHN, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, cfg: MetaSineConfig, cond: ConditionSpec):
    key_init, key_loop = jax.random.split(key)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return srghn_support_fitness(candidate, sx, sy)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        return (next_pop, rng, next_fitness), best_indiv

    if cfg.inner_generations <= 0:
        best0 = _select_srghn(init_pop, jnp.argmax(init_fitness))
        return best0

    (final_pop, _rng, final_fitness), best_seq = jax.lax.scan(
        step_fn,
        (init_pop, key_loop, init_fitness),
        None,
        length=cfg.inner_generations,
    )
    del final_pop, final_fitness
    return _select_srghn(best_seq, -1)



def srghn_task_curve(indiv: SRGHN, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, qx: jnp.ndarray, qy: jnp.ndarray, cfg: MetaSineConfig, cond: ConditionSpec) -> jnp.ndarray:
    key_init, key_loop = jax.random.split(key)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return srghn_support_fitness(candidate, sx, sy)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)
    best0 = _select_srghn(init_pop, jnp.argmax(init_fitness))
    initial_mse = srghn_query_mse(best0, qx, qy)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        mse = srghn_query_mse(best_indiv, qx, qy)
        return (next_pop, rng, next_fitness), mse

    if cfg.inner_generations <= 0:
        return initial_mse[None]

    (_, _, _), mses = jax.lax.scan(step_fn, (init_pop, key_loop, init_fitness), None, length=cfg.inner_generations)
    return jnp.concatenate([initial_mse[None], mses], axis=0)


# --------------------------------------------------------------------------------------
# Vector Gaussian search, shared by inner and outer loops
# --------------------------------------------------------------------------------------



def vector_gaussian_generation(pop: jnp.ndarray, key: jax.Array, cfg: MetaSineConfig, pop_size: int, children_per_parent: int, fitness_fn):
    if children_per_parent <= 0:
        all_candidates = pop
    else:
        parent_idx = jnp.repeat(jnp.arange(pop_size), children_per_parent)
        parents = pop[parent_idx]
        noise = cfg.vector_ga_sigma * jax.random.normal(
            key,
            (pop_size * children_per_parent, pop.shape[1]),
            dtype=pop.dtype,
        )
        children = parents + noise
        all_candidates = jnp.concatenate([pop, children], axis=0)
    fitness = jax.vmap(fitness_fn)(all_candidates)
    select_idx = jnp.argsort(fitness)[-pop_size:]
    next_pop = all_candidates[select_idx]
    next_fitness = fitness[select_idx]
    metrics = compute_vector_metrics(pop, fitness)
    return next_pop, next_fitness, metrics



def init_vector_local_population(center: jnp.ndarray, key: jax.Array, cfg: MetaSineConfig) -> jnp.ndarray:
    if cfg.inner_pop_size <= 1:
        return center[None, :]
    noise = cfg.vector_ga_sigma * jax.random.normal(
        key,
        (cfg.inner_pop_size - 1, center.shape[0]),
        dtype=center.dtype,
    )
    children = center[None, :] + noise
    return jnp.concatenate([center[None, :], children], axis=0)



def vector_gaussian_adapt(center: jnp.ndarray, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, cfg: MetaSineConfig, policy_spec: ParamNodeSpec) -> jnp.ndarray:
    key_init, key_loop = jax.random.split(key)
    init_pop = init_vector_local_population(center, key_init, cfg)
    support_fitness = partial(vector_support_fitness, sx=sx, sy=sy, policy_spec=policy_spec)
    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = vector_gaussian_generation(
            pop,
            step_key,
            cfg,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
        )
        best_vec = next_pop[jnp.argmax(next_fitness)]
        return (next_pop, rng, next_fitness), best_vec

    if cfg.inner_generations <= 0:
        return init_pop[jnp.argmax(init_fitness)]

    (_, _, _), best_seq = jax.lax.scan(step_fn, (init_pop, key_loop, init_fitness), None, length=cfg.inner_generations)
    return best_seq[-1]



def vector_gaussian_task_curve(center: jnp.ndarray, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, qx: jnp.ndarray, qy: jnp.ndarray, cfg: MetaSineConfig, policy_spec: ParamNodeSpec) -> jnp.ndarray:
    key_init, key_loop = jax.random.split(key)
    init_pop = init_vector_local_population(center, key_init, cfg)
    support_fitness = partial(vector_support_fitness, sx=sx, sy=sy, policy_spec=policy_spec)
    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)
    best0 = init_pop[jnp.argmax(init_fitness)]
    initial_mse = vector_query_mse(best0, qx, qy, policy_spec)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = vector_gaussian_generation(
            pop,
            step_key,
            cfg,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
        )
        best_vec = next_pop[jnp.argmax(next_fitness)]
        mse = vector_query_mse(best_vec, qx, qy, policy_spec)
        return (next_pop, rng, next_fitness), mse

    if cfg.inner_generations <= 0:
        return initial_mse[None]

    (_, _, _), mses = jax.lax.scan(step_fn, (init_pop, key_loop, init_fitness), None, length=cfg.inner_generations)
    return jnp.concatenate([initial_mse[None], mses], axis=0)


# --------------------------------------------------------------------------------------
# Vector evosax search, shared by inner and outer loops
# --------------------------------------------------------------------------------------



def make_evosax_adapter(pop_size: int, algo_name: str, sigma_init: float | None, num_dims: int) -> EvosaxStrategyAdapter:
    cfg = SimpleNamespace(pop_size=pop_size, evosax_algo=algo_name, evosax_sigma_init=sigma_init)
    return EvosaxStrategyAdapter(cfg, solution=jnp.zeros((num_dims,), dtype=jnp.float32))



def _sigma_from_adapter(adapter: EvosaxStrategyAdapter, fallback: float = 1.0) -> float:
    params = getattr(adapter, "params", None)
    for field_name in ("sigma_init", "init_std", "std_init", "sigma"):
        if params is not None and hasattr(params, field_name):
            try:
                return float(jax.device_get(getattr(params, field_name)))
            except Exception:
                continue
    return fallback



def init_local_evosax_state(center: jnp.ndarray, key: jax.Array, adapter: EvosaxStrategyAdapter, fitness_fn) -> LocalEvosaxState:
    key_init, key_boot = jax.random.split(key)
    center = jnp.asarray(center, dtype=jnp.float32)
    pop_size = adapter.pop_size
    initial_population = jnp.tile(center[None, :], (pop_size, 1))
    initial_fitness = jax.vmap(fitness_fn)(initial_population)

    if adapter.init_signature == ("key", "mean", "params"):
        strategy_state = adapter.strategy.init(key_init, center, adapter.params)
        needs_initial_shift = jnp.asarray(False)
        population = initial_population
        fitness = initial_fitness
    elif adapter.init_signature == ("key", "params"):
        strategy_state = adapter.strategy.init(key_init, adapter.params)
        needs_initial_shift = jnp.asarray(True)
        population = initial_population
        fitness = initial_fitness
    elif adapter.init_signature == ("key", "population", "fitness", "params"):
        sigma = jnp.asarray(_sigma_from_adapter(adapter, fallback=1.0), dtype=jnp.float32)
        bootstrap = center[None, :] + sigma * jax.random.normal(key_boot, (pop_size, center.shape[0]), dtype=jnp.float32)
        bootstrap_fitness = jax.vmap(fitness_fn)(bootstrap)
        strategy_state = adapter.strategy.init(key_init, bootstrap, fitness_for_evosax(bootstrap_fitness), adapter.params)
        needs_initial_shift = jnp.asarray(False)
        population = bootstrap
        fitness = bootstrap_fitness
    else:
        raise TypeError(f"Unsupported evosax init signature for local adaptation: {adapter.init_signature}")

    best_idx = jnp.argmax(fitness)
    return LocalEvosaxState(
        strategy_state=strategy_state,
        population=population,
        fitness=fitness,
        best_solution=population[best_idx],
        best_fitness=fitness[best_idx],
        needs_initial_shift=needs_initial_shift,
        initial_center=center,
        key=key,
    )



def local_evosax_generation(state: LocalEvosaxState, adapter: EvosaxStrategyAdapter, fitness_fn) -> LocalEvosaxState:
    key_ask, key_tell, key_next = jax.random.split(state.key, 3)
    raw_population, ask_state = adapter.ask(key_ask, state.strategy_state)
    population = jax.lax.cond(
        state.needs_initial_shift,
        lambda _: raw_population + state.initial_center[None, :],
        lambda _: raw_population,
        operand=None,
    )
    fitness = jax.vmap(fitness_fn)(population)
    next_strategy_state = adapter.tell(key_tell, population, fitness, ask_state)
    best_idx = jnp.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    improved = best_fitness > state.best_fitness
    running_best_solution = jax.lax.cond(
        improved,
        lambda _: best_solution,
        lambda _: state.best_solution,
        operand=None,
    )
    running_best_fitness = jnp.maximum(state.best_fitness, best_fitness)
    return LocalEvosaxState(
        strategy_state=next_strategy_state,
        population=population,
        fitness=fitness,
        best_solution=running_best_solution,
        best_fitness=running_best_fitness,
        needs_initial_shift=jnp.asarray(False),
        initial_center=state.initial_center,
        key=key_next,
    )



def vector_evosax_adapt(center: jnp.ndarray, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, cfg: MetaSineConfig, policy_spec: ParamNodeSpec, adapter: EvosaxStrategyAdapter) -> jnp.ndarray:
    fitness_fn = partial(vector_support_fitness, sx=sx, sy=sy, policy_spec=policy_spec)
    init_state = init_local_evosax_state(center, key, adapter, fitness_fn)

    def step_fn(state, _):
        next_state = local_evosax_generation(state, adapter, fitness_fn)
        return next_state, next_state.best_solution

    if cfg.inner_generations <= 0:
        return init_state.best_solution

    final_state, best_seq = jax.lax.scan(step_fn, init_state, None, length=cfg.inner_generations)
    del final_state
    return best_seq[-1]



def vector_evosax_task_curve(center: jnp.ndarray, key: jax.Array, sx: jnp.ndarray, sy: jnp.ndarray, qx: jnp.ndarray, qy: jnp.ndarray, cfg: MetaSineConfig, policy_spec: ParamNodeSpec, adapter: EvosaxStrategyAdapter) -> jnp.ndarray:
    fitness_fn = partial(vector_support_fitness, sx=sx, sy=sy, policy_spec=policy_spec)
    init_state = init_local_evosax_state(center, key, adapter, fitness_fn)
    initial_mse = vector_query_mse(init_state.best_solution, qx, qy, policy_spec)

    def step_fn(state, _):
        next_state = local_evosax_generation(state, adapter, fitness_fn)
        mse = vector_query_mse(next_state.best_solution, qx, qy, policy_spec)
        return next_state, mse

    if cfg.inner_generations <= 0:
        return initial_mse[None]

    _, mses = jax.lax.scan(step_fn, init_state, None, length=cfg.inner_generations)
    return jnp.concatenate([initial_mse[None], mses], axis=0)


# --------------------------------------------------------------------------------------
# Generic meta-fitness / outer loops
# --------------------------------------------------------------------------------------



def srghn_meta_fitness(indiv: SRGHN, key: jax.Array, tasks: MetaTaskBatch, cfg: MetaSineConfig, cond: ConditionSpec) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.support_x.shape[0])

    def per_task(sx, sy, qx, qy, task_key):
        adapted = srghn_adapt(indiv, task_key, sx, sy, cfg, cond)
        return -srghn_query_mse(adapted, qx, qy)

    return jnp.mean(jax.vmap(per_task)(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, task_keys))



def vector_meta_fitness(
    center: jnp.ndarray,
    key: jax.Array,
    tasks: MetaTaskBatch,
    cfg: MetaSineConfig,
    cond: ConditionSpec,
    policy_spec: ParamNodeSpec,
    inner_adapter: EvosaxStrategyAdapter | None,
) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.support_x.shape[0])

    def per_task(sx, sy, qx, qy, task_key):
        if cond.inner_optimizer == "gaussian":
            adapted = vector_gaussian_adapt(center, task_key, sx, sy, cfg, policy_spec)
        elif cond.inner_optimizer == "evosax":
            if inner_adapter is None:
                raise ValueError("inner_adapter must be provided for evosax inner adaptation.")
            adapted = vector_evosax_adapt(center, task_key, sx, sy, cfg, policy_spec, inner_adapter)
        else:
            raise ValueError(f"Unsupported vector inner optimizer: {cond.inner_optimizer}")
        return -vector_query_mse(adapted, qx, qy, policy_spec)

    return jnp.mean(jax.vmap(per_task)(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, task_keys))



def vector_task_curve(
    center: jnp.ndarray,
    key: jax.Array,
    sx: jnp.ndarray,
    sy: jnp.ndarray,
    qx: jnp.ndarray,
    qy: jnp.ndarray,
    cfg: MetaSineConfig,
    cond: ConditionSpec,
    policy_spec: ParamNodeSpec,
    inner_adapter: EvosaxStrategyAdapter | None,
) -> jnp.ndarray:
    if cond.inner_optimizer == "gaussian":
        return vector_gaussian_task_curve(center, key, sx, sy, qx, qy, cfg, policy_spec)
    if cond.inner_optimizer == "evosax":
        if inner_adapter is None:
            raise ValueError("inner_adapter must be provided for evosax inner adaptation.")
        return vector_evosax_task_curve(center, key, sx, sy, qx, qy, cfg, policy_spec, inner_adapter)
    raise ValueError(f"Unsupported vector inner optimizer: {cond.inner_optimizer}")



def srghn_outer_step(state: SRGHNMetaState, gen: jnp.ndarray, cfg: MetaSineConfig, cond: ConditionSpec):
    key_next, key_tasks, key_eval, key_evolve = jax.random.split(state.key, 4)
    tasks = sample_sine_tasks(key_tasks, cfg, cfg.meta_batch_size)
    eval_keys = jax.random.split(key_eval, cfg.outer_pop_size * (1 + cfg.outer_children_per_parent))

    # Need unique eval keys per candidate.
    def batched_fitness(indiv: SRGHN, eval_key: jax.Array) -> jnp.ndarray:
        return srghn_meta_fitness(indiv, eval_key, tasks, cfg, cond)

    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key_evolve)
    probe_keys = jax.random.split(probe_key, cfg.outer_pop_size)
    parent_metadata = eqx.filter_vmap(lambda indiv, k: mutation_metadata(indiv, k, **mutation_kwargs))(state.pop, probe_keys)

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(cfg.outer_pop_size), cfg.outer_children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, cfg.outer_pop_size * cfg.outer_children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, k: mutate_with_metadata(indiv, k, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(batched_fitness)(all_candidates, eval_keys)

    select_idx = jnp.argsort(all_fitness)[-cfg.outer_pop_size:]
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
    jax.debug.callback(partial(_wandb_log_metrics, prefix="train"), metrics, gen)

    gen_best_idx = jnp.argmax(next_fitness)
    gen_best = _select_srghn(next_pop, gen_best_idx)
    gen_best_fit = next_fitness[gen_best_idx]
    improved = gen_best_fit > state.best_fitness
    best_indiv = jax.lax.cond(improved, lambda _: gen_best, lambda _: state.best_indiv, operand=None)
    best_fitness = jnp.maximum(state.best_fitness, gen_best_fit)

    return SRGHNMetaState(
        pop=next_pop,
        key=key_next,
        pop_fitness=next_fitness,
        best_fitness=best_fitness,
        best_indiv=best_indiv,
    ), metrics



def vector_gaussian_outer_step(state: VectorGAMetaState, gen: jnp.ndarray, cfg: MetaSineConfig, cond: ConditionSpec, policy_spec: ParamNodeSpec, inner_adapter: EvosaxStrategyAdapter | None):
    key_next, key_tasks, key_eval, key_children = jax.random.split(state.key, 4)
    tasks = sample_sine_tasks(key_tasks, cfg, cfg.meta_batch_size)

    def fitness_fn(vec: jnp.ndarray, eval_key: jax.Array) -> jnp.ndarray:
        return vector_meta_fitness(vec, eval_key, tasks, cfg, cond, policy_spec, inner_adapter)

    parent_idx = jnp.repeat(jnp.arange(cfg.outer_pop_size), cfg.outer_children_per_parent)
    parent_rep = state.pop[parent_idx]
    child_noise = cfg.vector_ga_sigma * jax.random.normal(
        key_children,
        (cfg.outer_pop_size * cfg.outer_children_per_parent, state.pop.shape[1]),
        dtype=state.pop.dtype,
    )
    children = parent_rep + child_noise
    all_candidates = jnp.concatenate([state.pop, children], axis=0)

    eval_keys = jax.random.split(key_eval, all_candidates.shape[0])
    all_fitness = jax.vmap(fitness_fn)(all_candidates, eval_keys)
    select_idx = jnp.argsort(all_fitness)[-cfg.outer_pop_size:]
    next_pop = all_candidates[select_idx]
    next_fitness = all_fitness[select_idx]

    metrics = compute_vector_metrics(state.pop, all_fitness)
    metrics["query_mse_best"] = -metrics["fitness_best"]
    metrics["query_mse_mean"] = -metrics["fitness_mean"]
    jax.debug.callback(partial(_wandb_log_metrics, prefix="train"), metrics, gen)

    gen_best_idx = jnp.argmax(next_fitness)
    gen_best_solution = next_pop[gen_best_idx]
    gen_best_fitness = next_fitness[gen_best_idx]
    improved = gen_best_fitness > state.best_fitness
    best_solution = jax.lax.cond(improved, lambda _: gen_best_solution, lambda _: state.best_solution, operand=None)
    best_fitness = jnp.maximum(state.best_fitness, gen_best_fitness)

    return VectorGAMetaState(
        pop=next_pop,
        key=key_next,
        pop_fitness=next_fitness,
        best_fitness=best_fitness,
        best_solution=best_solution,
    ), metrics



def init_outer_evosax_state(key: jax.Array, cfg: MetaSineConfig, adapter: EvosaxStrategyAdapter) -> VectorEvosaxMetaState:
    key_init, key_boot, key_state = jax.random.split(key, 3)
    zero = zero_policy_vector(regression_policy_spec(cfg.policy_hidden_dims))
    pop_size = adapter.pop_size

    if adapter.init_signature == ("key", "mean", "params"):
        strategy_state = adapter.strategy.init(key_init, zero, adapter.params)
        population = jnp.tile(zero[None, :], (pop_size, 1))
        fitness = -jnp.inf * jnp.ones((pop_size,), dtype=jnp.float32)
    elif adapter.init_signature == ("key", "params"):
        strategy_state = adapter.strategy.init(key_init, adapter.params)
        population = jnp.tile(zero[None, :], (pop_size, 1))
        fitness = -jnp.inf * jnp.ones((pop_size,), dtype=jnp.float32)
    elif adapter.init_signature == ("key", "population", "fitness", "params"):
        sigma = jnp.asarray(_sigma_from_adapter(adapter, fallback=1.0), dtype=jnp.float32)
        population = zero[None, :] + sigma * jax.random.normal(key_boot, (pop_size, zero.shape[0]), dtype=jnp.float32)
        boot_fitness = jnp.zeros((pop_size,), dtype=jnp.float32)
        strategy_state = adapter.strategy.init(key_init, population, fitness_for_evosax(boot_fitness), adapter.params)
        fitness = -jnp.inf * jnp.ones((pop_size,), dtype=jnp.float32)
    else:
        raise TypeError(f"Unsupported evosax outer init signature: {adapter.init_signature}")

    return VectorEvosaxMetaState(
        key=key_state,
        population=population,
        fitness=fitness,
        strategy_state=strategy_state,
        best_fitness=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_solution=zero,
    )



def vector_evosax_outer_step(state: VectorEvosaxMetaState, gen: jnp.ndarray, cfg: MetaSineConfig, cond: ConditionSpec, policy_spec: ParamNodeSpec, inner_adapter: EvosaxStrategyAdapter | None, outer_adapter: EvosaxStrategyAdapter):
    key_tasks, key_ask, key_eval, key_tell, key_next = jax.random.split(state.key, 5)
    tasks = sample_sine_tasks(key_tasks, cfg, cfg.meta_batch_size)

    population, ask_state = outer_adapter.ask(key_ask, state.strategy_state)
    eval_keys = jax.random.split(key_eval, population.shape[0])

    def fitness_fn(vec: jnp.ndarray, eval_key: jax.Array) -> jnp.ndarray:
        return vector_meta_fitness(vec, eval_key, tasks, cfg, cond, policy_spec, inner_adapter)

    fitness = jax.vmap(fitness_fn)(population, eval_keys)
    next_strategy_state = outer_adapter.tell(key_tell, population, fitness, ask_state)

    metrics = compute_vector_metrics(population, fitness)
    metrics["query_mse_best"] = -metrics["fitness_best"]
    metrics["query_mse_mean"] = -metrics["fitness_mean"]
    jax.debug.callback(partial(_wandb_log_metrics, prefix="train"), metrics, gen)

    gen_best_idx = jnp.argmax(fitness)
    gen_best_solution = population[gen_best_idx]
    gen_best_fitness = fitness[gen_best_idx]
    improved = gen_best_fitness > state.best_fitness
    best_solution = jax.lax.cond(improved, lambda _: gen_best_solution, lambda _: state.best_solution, operand=None)
    best_fitness = jnp.maximum(state.best_fitness, gen_best_fitness)

    return VectorEvosaxMetaState(
        key=key_next,
        population=population,
        fitness=fitness,
        strategy_state=next_strategy_state,
        best_fitness=best_fitness,
        best_solution=best_solution,
    ), metrics


# --------------------------------------------------------------------------------------
# Condition runners
# --------------------------------------------------------------------------------------



def run_srghn_condition(cfg: MetaSineConfig, cond: ConditionSpec) -> dict[str, Any]:
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
    self_spec = _srghn_self_spec(policy_spec, cfg)

    _wandb_init_run(cfg, cond, policy_spec=policy_spec, self_spec=self_spec)

    init_key = jax.random.PRNGKey(cfg.seed)
    t0 = time.perf_counter()
    final_state, history, curves = run_srghn_compiled(cfg, cond, init_key)
    jax.block_until_ready(history["fitness_best"])
    jax.block_until_ready(curves)
    train_seconds = time.perf_counter() - t0
    champion = final_state.best_indiv
    eval_seconds = 0.0
    summary_payload = {
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
    }
    _wandb_log_summary(cfg, cond, summary_payload)
    _wandb_finish_run()

    return {
        "kind": "srghn",
        "condition": asdict(cond),
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "champion": champion,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }



def run_vector_condition(cfg: MetaSineConfig, cond: ConditionSpec) -> dict[str, Any]:
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
    num_dims = policy_num_dims(policy_spec)
    inner_adapter = None
    outer_adapter = None
    if cond.inner_optimizer == "evosax":
        inner_adapter = make_evosax_adapter(cfg.inner_pop_size, cond.inner_evosax_algo or "", cfg.inner_evosax_sigma_init, num_dims)
    if cond.outer_optimizer == "evosax":
        outer_adapter = make_evosax_adapter(cfg.outer_pop_size, cond.outer_evosax_algo or "", cfg.outer_evosax_sigma_init, num_dims)

    _wandb_init_run(cfg, cond, policy_spec=policy_spec)

    init_key = jax.random.PRNGKey(cfg.seed)
    key_pop, key_loop = jax.random.split(init_key)

    if cond.outer_optimizer == "gaussian":
        init_pop = init_policy_vector_population(key_pop, cfg.outer_pop_size, policy_spec, scale=cfg.vector_ga_init_scale)
        init_state = VectorGAMetaState(
            pop=init_pop,
            key=key_loop,
            pop_fitness=-jnp.inf * jnp.ones((cfg.outer_pop_size,), dtype=jnp.float32),
            best_fitness=jnp.asarray(-jnp.inf, dtype=jnp.float32),
            best_solution=init_pop[0],
        )

        @jax.jit
        def run_impl(state: VectorGAMetaState):
            gens = jnp.arange(cfg.outer_generations, dtype=jnp.int32)
            return jax.lax.scan(
                lambda carry, gen: vector_gaussian_outer_step(carry, gen, cfg, cond, policy_spec, inner_adapter),
                state,
                gens,
            )

    elif cond.outer_optimizer == "evosax":
        if outer_adapter is None:
            raise ValueError("outer_adapter must be provided for evosax outer optimization.")
        init_state = init_outer_evosax_state(key_loop, cfg, outer_adapter)

        @jax.jit
        def run_impl(state: VectorEvosaxMetaState):
            gens = jnp.arange(cfg.outer_generations, dtype=jnp.int32)
            return jax.lax.scan(
                lambda carry, gen: vector_evosax_outer_step(carry, gen, cfg, cond, policy_spec, inner_adapter, outer_adapter),
                state,
                gens,
            )
    else:
        raise ValueError(f"Unsupported vector outer optimizer: {cond.outer_optimizer}")

    t0 = time.perf_counter()
    final_state, history = run_impl(init_state)
    jax.block_until_ready(history["fitness_best"])
    train_seconds = time.perf_counter() - t0

    champion = final_state.best_solution
    final_fitness = final_state.fitness
    heldout_tasks = sample_sine_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg, cfg.test_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.test_task_batch_size)

    @jax.jit
    def eval_curves(policy_vector: jnp.ndarray, tasks: MetaTaskBatch, keys: jax.Array):
        return jax.vmap(
            lambda sx, sy, qx, qy, task_key: vector_task_curve(
                policy_vector,
                task_key,
                sx,
                sy,
                qx,
                qy,
                cfg,
                cond,
                policy_spec,
                inner_adapter,
            )
        )(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, keys)

    t1 = time.perf_counter()
    curves = eval_curves(champion, heldout_tasks, heldout_keys)
    jax.block_until_ready(curves)
    eval_seconds = time.perf_counter() - t1
    summary_payload = {
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_fitness),
    }
    _wandb_log_summary(cfg, cond, summary_payload)
    _wandb_finish_run()

    return {
        "kind": "vector",
        "condition": asdict(cond),
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
        "policy_num_dims": int(num_dims),
        "train_history": _history_to_host(history),
        "adaptation_curve_query_mse": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_fitness),
        "champion": jax.device_get(champion),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }



def run_condition(cfg: MetaSineConfig, cond: ConditionSpec) -> dict[str, Any]:
    cond = validate_condition(cond)
    if cond.search_object == "srghn":
        return run_srghn_condition(cfg, cond)
    if cond.search_object == "vector":
        return run_vector_condition(cfg, cond)
    raise ValueError(f"Unsupported search_object: {cond.search_object}")


# --------------------------------------------------------------------------------------
# Plotting / CLI
# --------------------------------------------------------------------------------------



def maybe_save_plots(output_stem: Path, results: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        hist = payload["train_history"]
        ax.plot(hist["query_mse_best"], label=name)
    ax.set_title("Meta-train best query MSE")
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Query MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_train.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        curve = payload["adaptation_curve_query_mse"]
        xs = list(range(len(curve["mean"])))
        mean = curve["mean"]
        stderr = curve["stderr"]
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_title("Held-out adaptation curve")
    ax.set_xlabel("Inner generation")
    ax.set_ylabel("Query MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_adaptation.png"), dpi=180)
    plt.close(fig)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic paired-loop sine meta-learning harness.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=("srghn_self_self", "cma_es_cma_es", "open_es_open_es"),
        help="Preset names or custom specs. Custom format: name=search_object,outer_optimizer,inner_optimizer,outer_algo,inner_algo",
    )
    parser.add_argument("--output", default="meta_sine_srghn_results.pkl")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-preset", default="default", choices=("default", "fast"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outer-generations", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--test-task-batch-size", type=int, default=None)
    parser.add_argument("--outer-pop-size", type=int, default=None)
    parser.add_argument("--outer-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-pop-size", type=int, default=None)
    parser.add_argument("--inner-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-generations", type=int, default=None)
    parser.add_argument("--support-k", type=int, default=None)
    parser.add_argument("--query-k", type=int, default=None)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--gnn-hidden-dim", type=int, default=None)
    parser.add_argument("--gnn-steps-policy", type=int, default=None)
    parser.add_argument("--gnn-steps-self", type=int, default=None)
    parser.add_argument("--stoch-coeff-dim", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=None)
    parser.add_argument("--const-noise-std", type=float, default=None)
    parser.add_argument("--vector-ga-sigma", type=float, default=None)
    parser.add_argument("--vector-ga-init-scale", type=float, default=None)
    parser.add_argument("--outer-evosax-sigma-init", type=float, default=None)
    parser.add_argument("--inner-evosax-sigma-init", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)



def make_base_cfg(args: argparse.Namespace) -> MetaSineConfig:
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "outer_generations": args.outer_generations,
            "meta_batch_size": args.meta_batch_size,
            "test_task_batch_size": args.test_task_batch_size,
            "outer_pop_size": args.outer_pop_size,
            "outer_children_per_parent": args.outer_children_per_parent,
            "inner_pop_size": args.inner_pop_size,
            "inner_children_per_parent": args.inner_children_per_parent,
            "inner_generations": args.inner_generations,
            "support_k": args.support_k,
            "query_k": args.query_k,
            "policy_hidden_dims": None if args.policy_hidden_dims is None else tuple(args.policy_hidden_dims),
            "embedding_dim": args.embedding_dim,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "gnn_steps_policy": args.gnn_steps_policy,
            "gnn_steps_self": args.gnn_steps_self,
            "stoch_coeff_dim": args.stoch_coeff_dim,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
            "mutation_rate_head_dim": args.mutation_rate_head_dim,
            "const_noise_std": args.const_noise_std,
            "vector_ga_sigma": args.vector_ga_sigma,
            "vector_ga_init_scale": args.vector_ga_init_scale,
            "outer_evosax_sigma_init": args.outer_evosax_sigma_init,
            "inner_evosax_sigma_init": args.inner_evosax_sigma_init,
            "wandb_project": None if args.no_wandb else args.wandb_project,
            "wandb_group": args.wandb_group,
            "wandb_name": args.wandb_name,
            "wandb_log_plots": args.wandb_log_plots,
        }.items()
        if value is not None
    }
    if args.no_wandb:
        overrides["wandb_project"] = None
    run_preset = "fast" if args.fast else args.run_preset
    return build_meta_sine_config(run_preset=run_preset, overrides=overrides)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(args)
    if args.print_config:
        print_resolved_config(base_cfg, family="meta_sine", run_preset="fast" if args.fast else args.run_preset)
        return 0
    conditions = [validate_condition(parse_condition_spec(text)) for text in args.conditions]

    results: dict[str, Any] = {}
    for cond in conditions:
        print(
            f"[run] {cond.name} search={cond.search_object} outer={cond.outer_optimizer}:{cond.outer_evosax_algo} inner={cond.inner_optimizer}:{cond.inner_evosax_algo}",
            flush=True,
        )
        payload = run_condition(base_cfg, cond)
        results[cond.name] = payload
        curve = payload["adaptation_curve_query_mse"]["mean"]
        print(
            f"[done] {cond.name} train_s={payload['train_seconds']:.2f} eval_s={payload['eval_seconds']:.2f} heldout_mse={curve.tolist()}",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_sine",
            run_preset="fast" if args.fast else args.run_preset,
        ),
        "conditions": [asdict(cond) for cond in conditions],
        "results": results,
    }
    with output_path.open("wb") as f:
        pickle.dump(save_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}")

    if args.plot:
        maybe_save_plots(output_path.with_suffix(""), results)
        print(f"[saved] plots near {output_path.with_suffix('')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
