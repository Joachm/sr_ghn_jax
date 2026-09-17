from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from envs import iter_shift_windows, make_env
from graphs import GraphSpec
from hypernets import DeterministicHead, StochasticHyper
from metrics import compute_experiment_metrics
from obs_norm import ObsNormState, init_obs_norm, update_obs_norm
from rollout import evaluate_individual_with_obs_stats
from srghn import SRGHN, mutation_metadata, mutate_with_metadata
from gnn import GraphEncoder
from specs import ParamNodeSpec


@jax.tree_util.register_pytree_node_class
@dataclass
class EvoState:
    pop: SRGHN
    key: jax.random.KeyArray
    obs_norm: ObsNormState
    pop_fitness: jnp.ndarray

    def tree_flatten(self):
        children = (self.pop, self.key, self.obs_norm, self.pop_fitness)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pop, key, obs_norm, pop_fitness = children
        return cls(pop=pop, key=key, obs_norm=obs_norm, pop_fitness=pop_fitness)


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
    replacement_mode = getattr(config, "srghn_replacement_mode", "elitist_union")
    if replacement_mode not in ("elitist_union", "generational"):
        raise ValueError(f"Unknown srghn_replacement_mode: {replacement_mode}")
    if replacement_mode == "generational" and (config.children_per_parent <= 0 or config.pop_size % config.children_per_parent):
        raise ValueError("Generational SR-GHN requires positive children_per_parent dividing pop_size.")

    key, key_eval, key_evolve = jax.random.split(state.key, 3)

    def _wandb_log(metrics_dict, gen_idx):
        try:
            import wandb
        except Exception:
            return
        if wandb.run is None:
            return
        wandb.log({"gen": int(gen_idx), **{name: float(value) for name, value in metrics_dict.items()}})

    def _select_individual(pop, idx):
        pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
        pop_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
        return eqx.combine(pop_arr, pop_static)

    def _generational_step():
        mutation_kwargs = {
            "excluded_modules": config.mutation_exclude_modules,
            "fixed_mutation_lr": config.fixed_mutation_lr,
        }
        def evaluate(population, eval_key):
            count = config.pop_size
            keys = jax.random.split(eval_key, count)
            fitness, obs_sum, obs_sq_sum, obs_count = jax.vmap(
                lambda i, k: evaluate_individual_with_obs_stats(
                    _select_individual(population, i), k, gen, config, state.obs_norm
                )
            )(jnp.arange(count), keys)
            return fitness, obs_sum, obs_sq_sum, obs_count

        def initial(_):
            probe_keys = jax.random.split(key_evolve, config.pop_size)
            metadata = eqx.filter_vmap(
                lambda indiv, probe_key: mutation_metadata(indiv, probe_key, **mutation_kwargs)
            )(state.pop, probe_keys)
            fitness, obs_sum, obs_sq_sum, obs_count = evaluate(state.pop, key_eval)
            return state.pop, fitness, metadata, obs_sum, obs_sq_sum, obs_count

        def offspring(_):
            num_reproducers = config.pop_size // config.children_per_parent
            repro_idx = jnp.argsort(state.pop_fitness)[-num_reproducers:]
            pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
            parent_idx = jnp.repeat(repro_idx, config.children_per_parent)
            parents_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
            parents = eqx.combine(parents_arr, pop_static)
            child_keys = jax.random.split(key_evolve, config.pop_size)
            children, child_metadata = eqx.filter_vmap(
                lambda indiv, child_key: mutate_with_metadata(indiv, child_key, **mutation_kwargs)
            )(parents, child_keys)
            fitness, obs_sum, obs_sq_sum, obs_count = evaluate(children, key_eval)
            return children, fitness, child_metadata, obs_sum, obs_sq_sum, obs_count

        next_pop, evaluated_fitness, metadata, obs_sum, obs_sq_sum, obs_count = jax.lax.cond(
            gen == 0, initial, offspring, operand=None
        )
        metrics = compute_experiment_metrics(next_pop, evaluated_fitness, metadata, metadata)
        active_windows = 0
        for window in iter_shift_windows(config):
            active = gen >= window.start_gen if window.end_gen is None else jnp.logical_and(gen >= window.start_gen, gen <= window.end_gen)
            active_windows = active_windows + active.astype(jnp.int32)
        metrics["active_shift_windows"] = jnp.asarray(active_windows, dtype=jnp.float32)
        jax.debug.callback(_wandb_log, metrics, gen)
        updated_obs_norm = update_obs_norm(
            state.obs_norm,
            jnp.sum(obs_sum, axis=0), jnp.sum(obs_sq_sum, axis=0), jnp.sum(obs_count, axis=0)
        )
        next_obs_norm = jax.tree_util.tree_map(
            lambda updated, current: jnp.where(gen == config.num_generations - 1, current, updated),
            updated_obs_norm, state.obs_norm
        )
        return EvoState(next_pop, key, next_obs_norm, evaluated_fitness), metrics

    if replacement_mode == "generational":
        return _generational_step()

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    num_children = config.pop_size * config.children_per_parent
    parent_probe_keys = jax.random.split(key_evolve, config.pop_size)
    child_seed = jax.random.fold_in(key_evolve, 1)
    child_keys = jax.random.split(child_seed, num_children)
    parent_idx = jnp.repeat(jnp.arange(config.pop_size), config.children_per_parent)
    parents_rep_arr = jax.tree_util.tree_map(lambda x: x[parent_idx], pop_arr)
    parents_rep = eqx.combine(parents_rep_arr, pop_static)
    baseline_kwargs = {"excluded_modules": config.mutation_exclude_modules, "fixed_mutation_lr": config.fixed_mutation_lr}
    parent_metadata = eqx.filter_vmap(lambda indiv, probe_key: mutation_metadata(indiv, probe_key, **baseline_kwargs))(state.pop, parent_probe_keys)
    children, child_metadata = eqx.filter_vmap(lambda indiv, child_key: mutate_with_metadata(indiv, child_key, **baseline_kwargs))(parents_rep, child_keys)
    children_arr, children_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), pop_arr, children_arr)
    all_candidates = eqx.combine(all_arr, children_static)
    all_count = config.pop_size + num_children
    eval_keys = jax.random.split(key_eval, all_count)
    all_fitness, all_obs_sum, all_obs_sq_sum, all_obs_count = jax.vmap(lambda i, k: evaluate_individual_with_obs_stats(_select_individual(all_candidates, i), k, gen, config, state.obs_norm))(jnp.arange(all_count), eval_keys)
    parent_fitness = all_fitness[: config.pop_size]
    child_fitness = all_fitness[config.pop_size:].reshape(config.pop_size, config.children_per_parent)
    blended_parent = (1.0 - config.child_factor) * parent_fitness + config.child_factor * jnp.mean(child_fitness, axis=1)
    selection_fitness = jnp.concatenate([blended_parent, child_fitness.reshape(-1)], axis=0)
    select_idx = jnp.argsort(selection_fitness)[-config.pop_size:]
    next_arr = jax.tree_util.tree_map(lambda x: x[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, children_static)
    next_pop_fitness = all_fitness[select_idx]
    parent_meta_arr = eqx.filter(parent_metadata, eqx.is_array)
    child_meta_arr, child_meta_static = eqx.partition(child_metadata, eqx.is_array)
    candidate_meta_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p, c], axis=0), parent_meta_arr, child_meta_arr)
    elite_metadata = eqx.combine(jax.tree_util.tree_map(lambda x: x[select_idx], candidate_meta_arr), child_meta_static)
    metrics = compute_experiment_metrics(state.pop, all_fitness, parent_metadata, elite_metadata)
    active_windows = 0
    for window in iter_shift_windows(config):
        active = gen >= window.start_gen if window.end_gen is None else jnp.logical_and(gen >= window.start_gen, gen <= window.end_gen)
        active_windows = active_windows + active.astype(jnp.int32)
    metrics["active_shift_windows"] = jnp.asarray(active_windows, dtype=jnp.float32)
    jax.debug.callback(_wandb_log, metrics, gen)
    updated_obs_norm = update_obs_norm(state.obs_norm, jnp.sum(all_obs_sum[select_idx], axis=0), jnp.sum(all_obs_sq_sum[select_idx], axis=0), jnp.sum(all_obs_count[select_idx], axis=0))
    next_obs_norm = jax.tree_util.tree_map(lambda updated, current: jnp.where(gen == config.num_generations - 1, current, updated), updated_obs_norm, state.obs_norm)
    return EvoState(pop=next_pop, key=key, obs_norm=next_obs_norm, pop_fitness=next_pop_fitness), metrics


def run(key: jax.random.KeyArray, config, graphs, specs):
    key_init, key_loop = jax.random.split(key, 2)
    init_pop = init_population(key_init, config, graphs, specs)
    _, _, obs_dim, _, _, _, _, _ = make_env(config)
    init_state = EvoState(
        pop=init_pop,
        key=key_loop,
        obs_norm=init_obs_norm(obs_dim),
        pop_fitness=jnp.zeros((config.pop_size,), dtype=jnp.float32),
    )
    gens = jnp.arange(config.num_generations, dtype=jnp.int32)

    def step_fn(state, gen):
        return evo_step(state, gen, config)

    return jax.lax.scan(step_fn, init_state, gens)


run_jit = jax.jit(run, static_argnames=("config",))
