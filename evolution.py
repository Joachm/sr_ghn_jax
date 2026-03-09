from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from envs import make_env
from graphs import GraphSpec
from hypernets import DeterministicHead, StochasticHyper
from metrics import compute_metrics
from rollout import evaluate_individual
from srghn import SRGHN, mutate
from gnn import GraphEncoder
from specs import ParamNodeSpec


@jax.tree_util.register_pytree_node_class
@dataclass
class EvoState:
    pop: SRGHN
    key: jax.random.KeyArray

    def tree_flatten(self):
        children = (self.pop, self.key)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key = children
        return cls(pop=pop, key=key)


def _unpack_graphs(graphs: Any) -> tuple[GraphSpec, GraphSpec]:
    if hasattr(graphs, "self_graph") and hasattr(graphs, "policy_graph"):
        return graphs.self_graph, graphs.policy_graph
    if isinstance(graphs, (tuple, list)) and len(graphs) == 2:
        return graphs[0], graphs[1]
    raise ValueError("graphs must provide (self_graph, policy_graph).")


def _unpack_specs(specs: Any) -> tuple[ParamNodeSpec, ParamNodeSpec]:
    if hasattr(specs, "self_spec") and hasattr(specs, "policy_spec"):
        return specs.self_spec, specs.policy_spec
    if isinstance(specs, (tuple, list)) and len(specs) == 2:
        return specs[0], specs[1]
    raise ValueError("specs must provide (self_spec, policy_spec).")


def _init_single(key: jax.random.KeyArray, config, graphs, specs) -> SRGHN:
    self_graph, policy_graph = _unpack_graphs(graphs)
    self_spec, policy_spec = _unpack_specs(specs)

    if config.embedding_dim != config.gnn_hidden_dim:
        raise ValueError("config.embedding_dim must equal config.gnn_hidden_dim.")

    key, k_self_emb, k_self_ctx, k_policy_emb, k_self_feat, k_policy_feat, k_enc_self, k_enc_pol, k_stoch, k_det = (
        jax.random.split(key, 10)
    )

    self_node_emb = 0.1 * jax.random.normal(k_self_emb, (self_spec.num_nodes, config.embedding_dim))
    self_context_emb = 0.1 * jax.random.normal(k_self_ctx, (config.embedding_dim,))
    policy_node_emb = 0.1 * jax.random.normal(k_policy_emb, (policy_spec.num_nodes, config.embedding_dim))
    self_feat_proj = eqx.nn.Linear(len(self_spec.node_features[0]), config.gnn_hidden_dim, use_bias=True, key=k_self_feat)
    policy_feat_proj = eqx.nn.Linear(
        len(policy_spec.node_features[0]), config.gnn_hidden_dim, use_bias=True, key=k_policy_feat
    )

    encoder_self = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_self, key=k_enc_self)
    encoder_policy = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_policy, key=k_enc_pol)

    stoch = StochasticHyper(
        in_dim=config.gnn_hidden_dim,
        hidden_dim=config.gnn_hidden_dim,
        coeff_dim=config.stoch_coeff_dim,
        block_size=config.parameter_block_size,
        mutation_block_ratio=config.mutation_block_ratio,
        mutation_rate_head_dim=config.mutation_rate_head_dim,
        clip_std=config.clip_std,
        clip_update=config.clip_update,
        const_noise_std=config.const_noise_std,
        key=k_stoch,
    )

    det = DeterministicHead(
        config.gnn_hidden_dim,
        config.gnn_hidden_dim,
        config.parameter_block_size,
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
        self_graph=self_graph,
        policy_graph=policy_graph,
        self_spec=self_spec,
        policy_spec=policy_spec,
        clip_params=config.clip_params,
    )


def init_population(key: jax.random.KeyArray, config, graphs, specs) -> SRGHN:
    keys = jax.random.split(key, config.pop_size)
    return eqx.filter_vmap(lambda k: _init_single(k, config, graphs, specs))(keys)


def evo_step(state: EvoState, gen: jnp.int32, config) -> tuple[EvoState, dict]:
    key, key_eval, key_children = jax.random.split(state.key, 3)
    def _select_individual(pop, idx):
        pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
        pop_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
        return eqx.combine(pop_arr, pop_static)

    def _wandb_log(metrics_dict, gen_idx):
        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return
        wandb.log(
            {
                "gen": int(gen_idx),
                "fitness_mean": float(metrics_dict["fitness_mean"]),
                "fitness_best": float(metrics_dict["fitness_best"]),
                "diversity": float(metrics_dict["diversity"]),
            }
        )

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    num_children = config.pop_size * config.children_per_parent
    child_keys = jax.random.split(key_children, num_children)
    parent_idx = jnp.repeat(jnp.arange(config.pop_size), config.children_per_parent)
    parents_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parents_rep_arr, pop_static)
    children = eqx.filter_vmap(mutate)(parents_rep, child_keys)

    children_arr, children_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), pop_arr, children_arr)
    all_candidates = eqx.combine(all_arr, children_static)

    all_count = config.pop_size + num_children
    eval_keys = jax.random.split(key_eval, all_count)
    all_idxs = jnp.arange(all_count)
    all_fitness = jax.vmap(
        lambda i, k: evaluate_individual(_select_individual(all_candidates, i), k, gen, config)
    )(all_idxs, eval_keys)
    parent_fitness = all_fitness[: config.pop_size]
    child_fitness = all_fitness[config.pop_size :]
    child_fitness = child_fitness.reshape(config.pop_size, config.children_per_parent)
    blended_parent = (1.0 - config.child_factor) * parent_fitness + config.child_factor * jnp.mean(
        child_fitness, axis=1
    )
    selection_fitness = jnp.concatenate([blended_parent, child_fitness.reshape(-1)], axis=0)
    # Log raw environment fitness, not the blended selection fitness.
    metrics = compute_metrics(state.pop, all_fitness)
    jax.debug.callback(_wandb_log, metrics, gen)
    select_idx = jnp.argsort(selection_fitness)[-config.pop_size :]
    next_arr = jax.tree_util.tree_map(lambda x: x[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, children_static)

    return EvoState(pop=next_pop, key=key), metrics


def run(key: jax.random.KeyArray, config, graphs, specs):
    key_init, key_loop = jax.random.split(key, 2)
    init_pop = init_population(key_init, config, graphs, specs)
    init_state = EvoState(pop=init_pop, key=key_loop)
    gens = jnp.arange(config.num_generations, dtype=jnp.int32)

    def step_fn(state, gen):
        return evo_step(state, gen, config)

    return jax.lax.scan(step_fn, init_state, gens)


run_jit = jax.jit(run, static_argnames=("config",))
