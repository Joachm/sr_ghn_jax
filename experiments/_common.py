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
from graphs import GraphSpec, make_chain_graph
from gnn import GraphEncoder
from hypernets import DeterministicHead, StochasticHyper
from specs import ParamNodeSpec, policy_spec_for_task, srghn_self_spec
from srghn import SRGHN


@dataclass(frozen=True)
class GraphBundle:
    self_graph: GraphSpec
    policy_graph: GraphSpec


@dataclass(frozen=True)
class SpecBundle:
    self_spec: ParamNodeSpec
    policy_spec: ParamNodeSpec


def _build_template_srghn(num_self_nodes: int, policy_spec: ParamNodeSpec, config, key) -> SRGHN:
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
        max_out=1,
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
        self_graph=make_chain_graph(num_self_nodes, bidir=True),
        policy_graph=make_chain_graph(policy_spec.num_nodes, bidir=True),
        self_spec=ParamNodeSpec((), (), (), 0, 0, (), ()),
        policy_spec=policy_spec,
        clip_params=config.clip_params,
    )


def build_graphs_and_specs(config) -> tuple[GraphBundle, SpecBundle]:
    key = jax.random.key(config.seed)
    policy_spec = policy_spec_for_task(config, config.stoch_max_out)

    temp_srghn = _build_template_srghn(1, policy_spec, config, key)
    provisional_spec = srghn_self_spec(temp_srghn, config.stoch_max_out)
    num_self_nodes = provisional_spec.num_nodes

    final_srghn = _build_template_srghn(num_self_nodes, policy_spec, config, key)
    self_spec = srghn_self_spec(final_srghn, config.stoch_max_out)

    graphs = GraphBundle(
        self_graph=make_chain_graph(self_spec.num_nodes, bidir=True),
        policy_graph=make_chain_graph(policy_spec.num_nodes, bidir=True),
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
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass
    print(f"Saved artifacts to {run_dir}")
    return final_state, metrics
