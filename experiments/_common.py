from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from evolution import run_jit
from graphs import GraphSpec, make_chain_graph, make_parallel_shard_graph
from gnn import GraphEncoder
from hypernets import DeterministicHead, StochasticHyper
from rollout import evaluate_individual
from specs import ParamNodeSpec, policy_spec_for_task, srghn_self_spec
from srghn import SRGHN
from visualization import record_mujoco_playground_rollout


@dataclass(frozen=True)
class GraphBundle:
    self_graph: GraphSpec
    policy_graph: GraphSpec


@dataclass(frozen=True)
class SpecBundle:
    self_spec: ParamNodeSpec
    policy_spec: ParamNodeSpec


def _build_template_srghn(
    num_self_nodes: int,
    policy_spec: ParamNodeSpec,
    policy_head_out: int,
    self_shard_size: int,
    config,
    key,
) -> SRGHN:
    if config.embedding_dim != config.gnn_hidden_dim:
        raise ValueError("config.embedding_dim must equal config.gnn_hidden_dim.")

    key, k_self_emb, k_policy_emb, k_enc_self, k_enc_pol, k_stoch, k_det = jax.random.split(key, 7)
    self_node_emb = jax.random.normal(k_self_emb, (num_self_nodes, config.embedding_dim))
    policy_node_emb = jax.random.normal(k_policy_emb, (policy_spec.num_nodes, config.embedding_dim))

    encoder_self = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_self, key=k_enc_self)
    encoder_policy = GraphEncoder(config.gnn_hidden_dim, config.gnn_steps_policy, key=k_enc_pol)

    stoch = StochasticHyper(
        in_dim=config.gnn_hidden_dim,
        hidden_dim=config.gnn_hidden_dim,
        coeff_dim=config.stoch_coeff_dim,
        max_out=self_shard_size,
        mutation_rate_head_dim=config.mutation_rate_head_dim,
        cov_rank=config.stoch_cov_rank,
        cov_scale=config.stoch_cov_scale,
        clip_std=config.clip_std,
        clip_update=config.clip_update,
        const_noise_std=config.const_noise_std,
        key=k_stoch,
    )

    det = DeterministicHead(config.gnn_hidden_dim, policy_head_out, key=k_det)

    return SRGHN(
        self_node_emb=self_node_emb,
        policy_node_emb=policy_node_emb,
        encoder_self=encoder_self,
        encoder_policy=encoder_policy,
        stoch=stoch,
        det=det,
        self_graph=make_chain_graph(num_self_nodes, bidir=True),
        policy_graph=make_chain_graph(policy_spec.num_nodes, bidir=True),
        self_spec=ParamNodeSpec((), (), (), 0, 0, (), ()),
        policy_spec=policy_spec,
        clip_params=config.clip_params,
        self_reg_mode=config.self_reg_mode,
        self_weight_decay=config.self_weight_decay,
        self_weight_norm_mode=config.self_weight_norm_mode,
        self_weight_norm_target=config.self_weight_norm_target,
        self_weight_norm_eps=config.self_weight_norm_eps,
        shard_residual_scale=config.shard_residual_scale,
        freeze_stoch_output_head=config.freeze_stoch_output_head,
    )


def build_graphs_and_specs(config) -> tuple[GraphBundle, SpecBundle]:
    key = jax.random.key(config.seed)
    policy_spec = policy_spec_for_task(config, shard_size=None)
    policy_head_out = policy_spec.max_size if config.policy_head_max_out is None else int(config.policy_head_max_out)
    if policy_head_out < policy_spec.max_size:
        raise ValueError(
            "policy_head_max_out must be at least the largest policy parameter size "
            f"({policy_spec.max_size}), got {policy_head_out}."
        )
    self_shard_size = int(config.self_shard_size)
    if self_shard_size <= 0:
        raise ValueError("self_shard_size must be positive.")
    if float(config.shard_residual_scale) < 0.0:
        raise ValueError("shard_residual_scale must be non-negative.")
    shard_graph_mode = str(getattr(config, "shard_graph_mode", "dense"))
    valid_shard_graph_modes = ("dense", "sibling_chain", "hub")
    if shard_graph_mode not in valid_shard_graph_modes:
        raise ValueError(
            f"Invalid shard_graph_mode={shard_graph_mode!r}. "
            f"Expected one of {valid_shard_graph_modes}."
        )

    num_self_nodes = 1
    self_spec = None
    max_iters = 200
    for _ in range(max_iters):
        temp_srghn = _build_template_srghn(
            num_self_nodes, policy_spec, policy_head_out, self_shard_size, config, key
        )
        provisional_spec = srghn_self_spec(temp_srghn, self_shard_size)
        if provisional_spec.num_nodes == num_self_nodes:
            self_spec = provisional_spec
            break
        num_self_nodes = provisional_spec.num_nodes

    if self_spec is None:
        raise ValueError(
            "Failed to converge on self graph size for SRGHN parameters "
            f"after {max_iters} iterations. Last estimate: {num_self_nodes}."
        )

    final_srghn = _build_template_srghn(
        num_self_nodes, policy_spec, policy_head_out, self_shard_size, config, key
    )
    self_spec = srghn_self_spec(final_srghn, self_shard_size)
    if self_spec.num_nodes != num_self_nodes:
        raise ValueError("Self graph size did not stabilize; increase iteration budget.")

    graphs = GraphBundle(
        self_graph=make_parallel_shard_graph(
            self_spec.shard_param_idxs, bidir=True, mode=shard_graph_mode
        ),
        policy_graph=make_parallel_shard_graph(
            policy_spec.shard_param_idxs, bidir=True, mode=shard_graph_mode
        ),
    )
    specs = SpecBundle(self_spec=self_spec, policy_spec=policy_spec)
    return graphs, specs


def _sanitize_component(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    cleaned_str = "".join(cleaned).strip("-")
    return cleaned_str or "run"


def _make_run_dir(config, root_dir: str) -> str:
    task = _sanitize_component(config.task_name)
    env_id = _sanitize_component(config.env_id)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{task}_{env_id}_seed{config.seed}_{timestamp}"
    run_dir = os.path.join(root_dir, base)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(root_dir, f"{base}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _metrics_to_numpy(metrics: dict) -> dict:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), metrics)


def _select_individual(pop: SRGHN, idx: int | jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    pop_arr = jax.tree_util.tree_map(lambda x: x[idx], pop_arr)
    return eqx.combine(pop_arr, pop_static)


def _record_champion_rollout(config, final_state, run_dir: str) -> None:
    if config.env_backend != "mujoco_playground":
        return

    print("Evaluating final population for champion rollout...")
    gen = int(config.num_generations)
    key = jax.random.key(config.seed + 1_000_003)
    eval_key, key_candidates = jax.random.split(key, 2)
    candidate_keys = jax.random.split(key_candidates, config.pop_size)
    fitness_values = []
    gen_arr = jnp.asarray(gen, dtype=jnp.int32)
    for i in range(config.pop_size):
        fitness_i = evaluate_individual(
            _select_individual(final_state.pop, i),
            candidate_keys[i],
            gen_arr,
            config,
        )
        fitness_values.append(fitness_i)
    fitness = jnp.stack(fitness_values)

    champion_idx = int(jnp.argmax(fitness))
    champion_fitness = float(fitness[champion_idx])
    champion = _select_individual(final_state.pop, champion_idx)
    rollout_path = os.path.join(run_dir, "champion_rollout.gif")
    rollout_stats = record_mujoco_playground_rollout(
        champion,
        config,
        rollout_path,
        key=eval_key,
        gen=gen,
    )
    metadata = {
        "champion_index": champion_idx,
        "champion_fitness_eval": champion_fitness,
        **rollout_stats,
    }
    metadata_path = os.path.join(run_dir, "champion_rollout.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Saved champion rollout to {rollout_path}")


def save_artifacts(config, final_state, metrics, root_dir: str = "artifacts") -> str:
    run_dir = _make_run_dir(config, root_dir)
    model_path = os.path.join(run_dir, "model.eqx")
    metrics_path = os.path.join(run_dir, "metrics.npz")
    config_path = os.path.join(run_dir, "config.json")

    eqx.tree_serialise_leaves(model_path, final_state.pop)
    np.savez(metrics_path, **_metrics_to_numpy(metrics))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)

    return run_dir


def run_experiment(config):
    graphs, specs = build_graphs_and_specs(config)
    key = jax.random.key(config.seed)
    try:
        import wandb

        if wandb.run is None:
            wandb.init(project="srghn_jax", name=config.env_id, config=config.__dict__)
    except Exception:
        wandb = None
    graph_tuple = (graphs.self_graph, graphs.policy_graph)
    spec_tuple = (specs.self_spec, specs.policy_spec)
    final_state, metrics = run_jit(key, config, graph_tuple, spec_tuple)
    run_dir = save_artifacts(config, final_state, metrics)
    try:
        _record_champion_rollout(config, final_state, run_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"Champion rollout recording skipped: {type(exc).__name__}: {exc}")
    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass
    print(f"Saved artifacts to {run_dir}")
    return final_state, metrics
