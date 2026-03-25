from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from evolution import run_jit
from experiments.evosax_adapter import EvosaxStrategyAdapter
from experiments.evolution_evosax import run_evosax
from experiments.policy_vectors import policy_num_dims, zero_policy_vector
from graphs import GraphSpec, make_chain_graph, make_policy_hierarchical_graph, make_self_hierarchical_graph
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


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def default_wandb_project(config) -> str:
    if getattr(config, "optimizer_family", "srghn") == "evosax":
        algo = getattr(config, "evosax_algo", None) or "unknown"
        return f"sr-ghn_control_{_safe_name(algo)}"
    return "srghn_jax"


def wandb_config_payload(config, policy_spec=None) -> dict:
    payload = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    if getattr(config, "optimizer_family", "srghn") != "evosax":
        return payload
    sigma_override = payload.pop("evosax_sigma_init", None)
    payload["evosax_sigma_init_override"] = sigma_override
    if policy_spec is None:
        return payload
    solution = zero_policy_vector(policy_spec)
    adapter = EvosaxStrategyAdapter(config, solution=solution)
    payload["policy_num_dims"] = int(policy_num_dims(policy_spec))
    payload["evosax_effective_params"] = adapter.effective_params_dict()
    payload["evosax_init_signature"] = list(adapter.init_signature)
    payload["evosax_requires_population_init"] = bool(adapter.requires_population_init)
    return payload


def _build_template_srghn(num_self_nodes: int, policy_spec: ParamNodeSpec, config, key) -> SRGHN:
    if config.embedding_dim != config.gnn_hidden_dim:
        raise ValueError("config.embedding_dim must equal config.gnn_hidden_dim.")

    key, k_self_emb, k_self_ctx, k_policy_emb, k_self_feat, k_policy_feat, k_enc_self, k_enc_pol, k_stoch, k_det = (
        jax.random.split(key, 10)
    )
    self_node_emb = 0.1 * jax.random.normal(k_self_emb, (num_self_nodes, config.embedding_dim))
    self_context_emb = 0.1 * jax.random.normal(k_self_ctx, (config.embedding_dim,))
    policy_node_emb = 0.1 * jax.random.normal(k_policy_emb, (policy_spec.num_nodes, config.embedding_dim))
    self_feat_proj = eqx.nn.Linear(
        len(policy_spec.node_features[0]), config.gnn_hidden_dim, use_bias=True, key=k_self_feat
    )
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
        self_graph=make_chain_graph(num_self_nodes, bidir=True),
        policy_graph=make_policy_hierarchical_graph(policy_spec.group_ids, bidir=True),
        self_spec=ParamNodeSpec((), (), 0, 0, (), (), (), (), None),
        policy_spec=policy_spec,
        clip_params=config.clip_params,
    )


def build_graphs_and_specs(config) -> tuple[GraphBundle, SpecBundle]:
    key = jax.random.key(config.seed)
    policy_spec = policy_spec_for_task(config)

    temp_srghn = _build_template_srghn(1, policy_spec, config, key)
    provisional_spec = srghn_self_spec(temp_srghn)
    num_self_nodes = provisional_spec.num_nodes

    final_srghn = _build_template_srghn(num_self_nodes, policy_spec, config, key)
    self_spec = srghn_self_spec(final_srghn)

    graphs = GraphBundle(
        self_graph=make_self_hierarchical_graph(
            self_spec.group_ids,
            self_spec.parent_ids,
            context_index=self_spec.context_index,
            bidir=True,
        ),
        policy_graph=make_policy_hierarchical_graph(policy_spec.group_ids, bidir=True),
    )
    specs = SpecBundle(self_spec=self_spec, policy_spec=policy_spec)
    return graphs, specs


def run_experiment(config):
    key = jax.random.key(config.seed)
    graphs = None
    specs = None
    if config.optimizer_family == "srghn":
        graphs, specs = build_graphs_and_specs(config)
    else:
        specs = SpecBundle(self_spec=ParamNodeSpec((), (), 0, 0, (), (), (), (), None), policy_spec=policy_spec_for_task(config))
    try:
        import wandb

        if wandb.run is None:
            wandb.init(
                project=config.wandb_project or default_wandb_project(config),
                group=config.wandb_group,
                name=config.wandb_name or config.env_id,
                config=wandb_config_payload(config, specs.policy_spec),
            )
    except Exception:
        wandb = None
    if config.optimizer_family == "srghn":
        graph_tuple = (graphs.self_graph, graphs.policy_graph)
        spec_tuple = (specs.self_spec, specs.policy_spec)
        final_state, metrics = run_jit(key, config, graph_tuple, spec_tuple)
    elif config.optimizer_family == "evosax":
        final_state, metrics = run_evosax(key, config, specs.policy_spec)
    else:
        raise ValueError(f"Unknown optimizer_family: {config.optimizer_family}")
    try:
        if wandb is not None:
            wandb.finish()
    except Exception:
        pass
    return final_state, metrics
