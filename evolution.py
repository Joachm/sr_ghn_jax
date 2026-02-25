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
from srghn import SRGHN, mutate_with_stats
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

    key, k_self_emb, k_policy_emb, k_enc_self, k_enc_pol, k_stoch, k_det = jax.random.split(key, 7)

    self_node_emb = jax.random.normal(k_self_emb, (self_spec.num_nodes, config.embedding_dim))
    policy_node_emb = jax.random.normal(k_policy_emb, (policy_spec.num_nodes, config.embedding_dim))

    encoder_self = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_self, key=k_enc_self)
    encoder_policy = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_policy, key=k_enc_pol)

    stoch = StochasticHyper(
        in_dim=config.gnn_hidden_dim,
        hidden_dim=config.gnn_hidden_dim,
        coeff_dim=config.stoch_coeff_dim,
        max_out=self_spec.max_size,
        mutation_rate_head_dim=config.mutation_rate_head_dim,
        clip_std=config.clip_std,
        clip_update=config.clip_update,
        const_noise_std=config.const_noise_std,
        key=k_stoch,
    )

    det = DeterministicHead(config.gnn_hidden_dim, policy_spec.max_size, key=k_det)

    return SRGHN(
        self_node_emb=self_node_emb,
        policy_node_emb=policy_node_emb,
        encoder_self=encoder_self,
        encoder_policy=encoder_policy,
        stoch=stoch,
        det=det,
        self_graph=self_graph,
        policy_graph=policy_graph,
        self_spec=self_spec,
        policy_spec=policy_spec,
        clip_params=config.clip_params,
        self_reg_mode=config.self_reg_mode,
        self_weight_decay=config.self_weight_decay,
        self_weight_norm_mode=config.self_weight_norm_mode,
        self_weight_norm_target=config.self_weight_norm_target,
        self_weight_norm_eps=config.self_weight_norm_eps,
        freeze_stoch_output_head=config.freeze_stoch_output_head,
    )


def init_population(key: jax.random.KeyArray, config, graphs, specs) -> SRGHN:
    keys = jax.random.split(key, config.pop_size)
    return eqx.filter_vmap(lambda k: _init_single(k, config, graphs, specs))(keys)


def _sumsq_per_individual(module) -> jnp.ndarray:
    filtered = eqx.filter(module, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(filtered)
    if not leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    n = leaves[0].shape[0]
    out = jnp.zeros((n,), dtype=jnp.float32)
    for leaf in leaves:
        axes = tuple(range(1, leaf.ndim))
        out = out + jnp.sum(jnp.square(leaf), axis=axes)
    return out


def _l2_per_individual(module) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(_sumsq_per_individual(module), 0.0))


def _delta_l2_per_individual(parent_module, child_module) -> jnp.ndarray:
    p = eqx.filter(parent_module, eqx.is_array)
    c = eqx.filter(child_module, eqx.is_array)
    p_leaves = jax.tree_util.tree_leaves(p)
    c_leaves = jax.tree_util.tree_leaves(c)
    if not p_leaves:
        return jnp.zeros((0,), dtype=jnp.float32)
    n = p_leaves[0].shape[0]
    out = jnp.zeros((n,), dtype=jnp.float32)
    for p_leaf, c_leaf in zip(p_leaves, c_leaves):
        delta = c_leaf - p_leaf
        axes = tuple(range(1, delta.ndim))
        out = out + jnp.sum(jnp.square(delta), axis=axes)
    return jnp.sqrt(jnp.maximum(out, 0.0))


def _fraction_at_param_bounds(module, low: float, high: float) -> jnp.ndarray:
    filtered = eqx.filter(module, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(filtered)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    hit_count = jnp.array(0.0, dtype=jnp.float32)
    total_count = jnp.array(0.0, dtype=jnp.float32)
    for leaf in leaves:
        hit = jnp.logical_or(leaf <= low, leaf >= high)
        hit_count = hit_count + jnp.sum(hit.astype(jnp.float32))
        total_count = total_count + leaf.size
    return hit_count / jnp.maximum(total_count, 1.0)


def _stack_component_vectors(component) -> jnp.ndarray:
    filtered = eqx.filter(component, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(filtered)
    if not leaves:
        return jnp.zeros((0, 1), dtype=jnp.float32)
    n = leaves[0].shape[0]
    blocks = [leaf.reshape((n, -1)) for leaf in leaves]
    return jnp.concatenate(blocks, axis=1)


def _diversity_from_matrix(x: jnp.ndarray) -> jnp.ndarray:
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
        payload = {"gen": int(gen_idx)}
        for k, v in metrics_dict.items():
            payload[k] = float(v)
        wandb.log(payload)

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    num_children = config.pop_size * config.children_per_parent
    child_keys = jax.random.split(key_children, num_children)
    parent_idx = jnp.repeat(jnp.arange(config.pop_size), config.children_per_parent)
    parents_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parents_rep_arr, pop_static)
    children, child_mut_stats = eqx.filter_vmap(mutate_with_stats)(parents_rep, child_keys)

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
    child_mean_per_parent = jnp.mean(child_fitness, axis=1)
    blended_parent = (1.0 - config.child_factor) * parent_fitness + config.child_factor * jnp.mean(
        child_fitness, axis=1
    )
    selection_fitness = jnp.concatenate([blended_parent, child_fitness.reshape(-1)], axis=0)
    select_idx = jnp.argsort(selection_fitness)[-config.pop_size :]
    selected_fitness = selection_fitness[select_idx]

    mutation_delta_l2 = _delta_l2_per_individual(parents_rep, children)
    parent_l2 = _l2_per_individual(parents_rep)
    mutation_delta_rel_l2 = mutation_delta_l2 / jnp.maximum(parent_l2, 1e-8)

    delta_encoder_self_l2 = _delta_l2_per_individual(parents_rep.encoder_self, children.encoder_self)
    delta_encoder_policy_l2 = _delta_l2_per_individual(parents_rep.encoder_policy, children.encoder_policy)
    delta_stoch_l2 = _delta_l2_per_individual(parents_rep.stoch, children.stoch)
    delta_det_l2 = _delta_l2_per_individual(parents_rep.det, children.det)

    pop_param_norms = _l2_per_individual(state.pop)
    clip_params_fraction = _fraction_at_param_bounds(state.pop, config.clip_params[0], config.clip_params[1])
    policy_component = (state.pop.policy_node_emb, state.pop.encoder_policy, state.pop.det)
    self_component = (state.pop.self_node_emb, state.pop.encoder_self, state.pop.stoch)
    diversity_policy_component = _diversity_from_matrix(_stack_component_vectors(policy_component))
    diversity_self_component = _diversity_from_matrix(_stack_component_vectors(self_component))

    # Log raw environment fitness, not the blended selection fitness, and add diagnostics.
    metrics = compute_metrics(state.pop, all_fitness)
    metrics.update(
        {
            "parent_fitness_mean": jnp.mean(parent_fitness),
            "child_fitness_mean": jnp.mean(child_fitness),
            "child_parent_delta_mean": jnp.mean(child_mean_per_parent - parent_fitness),
            "selected_fitness_mean": jnp.mean(selected_fitness),
            "selection_threshold": jnp.min(selected_fitness),
            "elite_turnover": jnp.mean((select_idx >= config.pop_size).astype(jnp.float32)),
            "mutation_delta_l2": jnp.mean(mutation_delta_l2),
            "mutation_delta_rel_l2": jnp.mean(mutation_delta_rel_l2),
            "mutation_delta_encoder_self_l2": jnp.mean(delta_encoder_self_l2),
            "mutation_delta_encoder_policy_l2": jnp.mean(delta_encoder_policy_l2),
            "mutation_delta_stoch_l2": jnp.mean(delta_stoch_l2),
            "mutation_delta_det_l2": jnp.mean(delta_det_l2),
            "lr_mean": jnp.mean(child_mut_stats["lr_mean"]),
            "lr_std": jnp.mean(child_mut_stats["lr_std"]),
            "std_head_mean": jnp.mean(child_mut_stats["std_head_mean"]),
            "std_head_std": jnp.mean(child_mut_stats["std_head_std"]),
            "update_clip_fraction": jnp.mean(child_mut_stats["update_clip_fraction"]),
            "weight_norm_pre": jnp.mean(child_mut_stats["weight_norm_pre"]),
            "weight_norm_post": jnp.mean(child_mut_stats["weight_norm_post"]),
            "weight_norm_scale": jnp.mean(child_mut_stats["weight_norm_scale"]),
            "param_norm_mean": jnp.mean(pop_param_norms),
            "param_norm_std": jnp.std(pop_param_norms),
            "param_clip_fraction": clip_params_fraction,
            "diversity_policy_component": diversity_policy_component,
            "diversity_self_component": diversity_self_component,
        }
    )
    jax.debug.callback(_wandb_log, metrics, gen)
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
