from __future__ import annotations

from math import ceil

import equinox as eqx
import jax
import jax.numpy as jnp

from gnn import GraphEncoder
from graphs import GraphSpec
from hypernets import DeterministicHead, StochasticHyper
from specs import ParamNodeSpec, _srghn_filter_spec


class SRGHN(eqx.Module):
    self_node_emb: jnp.ndarray
    self_context_emb: jnp.ndarray
    policy_node_emb: jnp.ndarray
    self_feat_proj: eqx.nn.Linear
    policy_feat_proj: eqx.nn.Linear
    encoder_self: GraphEncoder
    encoder_policy: GraphEncoder
    stoch: StochasticHyper
    det: DeterministicHead
    self_graph: GraphSpec
    policy_graph: GraphSpec
    self_spec: ParamNodeSpec = eqx.field(static=True)
    policy_spec: ParamNodeSpec = eqx.field(static=True)
    clip_params: tuple[float, float] = eqx.field(static=True)


def _num_blocks(size: int, block_size: int) -> int:
    if size <= 0:
        return 0
    return ceil(size / block_size)


def _dense_delta_from_blocks(
    size: int,
    block_size: int,
    block_ids: jnp.ndarray,
    block_updates: jnp.ndarray,
    dtype,
) -> jnp.ndarray:
    if size <= 0:
        return jnp.zeros((0,), dtype=dtype)

    num_blocks = _num_blocks(size, block_size)
    delta_blocks = jnp.zeros((num_blocks, block_size), dtype=dtype)
    if block_ids.size:
        delta_blocks = delta_blocks.at[block_ids].add(block_updates.astype(dtype))
    return delta_blocks.reshape(-1)[:size]


def _node_inputs(
    spec: ParamNodeSpec,
    node_offset: jnp.ndarray,
    feat_proj: eqx.nn.Linear,
    *,
    context_index: int | None = None,
    context_emb: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if spec.num_nodes == 0:
        return node_offset
    features = jnp.asarray(spec.node_features, dtype=node_offset.dtype)
    projected = jax.vmap(feat_proj)(features)
    inputs = projected + node_offset
    if context_index is not None and context_emb is not None:
        inputs = inputs.at[context_index].add(context_emb.astype(inputs.dtype))
    return inputs


def make_policy(srghn: SRGHN) -> tuple[jnp.ndarray, ...]:
    h0 = _node_inputs(srghn.policy_spec, srghn.policy_node_emb, srghn.policy_feat_proj)
    h = srghn.encoder_policy(h0, srghn.policy_graph)
    shapes = srghn.policy_spec.shapes
    sizes = srghn.policy_spec.sizes

    outputs = []
    for i, shape in enumerate(shapes):
        vec = srghn.det(h[i], sizes[i])
        outputs.append(vec.reshape(shape))
    return tuple(outputs)


def mutate(srghn: SRGHN, key: jax.random.KeyArray) -> SRGHN:
    h0 = _node_inputs(
        srghn.self_spec,
        srghn.self_node_emb,
        srghn.self_feat_proj,
        context_index=srghn.self_spec.context_index,
        context_emb=srghn.self_context_emb,
    )
    h = srghn.encoder_self(h0, srghn.self_graph)
    filter_spec = _srghn_filter_spec(srghn)
    arr_tree, static_tree = eqx.partition(srghn, filter_spec)
    flat, treedef = jax.tree_util.tree_flatten(arr_tree)
    idxs = [i for i, leaf in enumerate(flat) if leaf is not None]
    leaves = [flat[i] for i in idxs]
    sizes = srghn.self_spec.sizes
    shapes = srghn.self_spec.shapes
    num_nodes = srghn.self_spec.num_nodes

    context_idx = srghn.self_spec.context_index
    if context_idx is None:
        raise ValueError("self_spec.context_index must be set for self mutation.")

    keys = jax.random.split(key, num_nodes + 1)
    child_ctx = srghn.stoch.sample_child_context(h[context_idx], keys[0])

    new_leaves = []
    for i, leaf in enumerate(leaves):
        block_ids, block_updates, _ = srghn.stoch(h[i], child_ctx, sizes[i], keys[i + 1])
        upd = _dense_delta_from_blocks(
            sizes[i],
            srghn.stoch.block_size,
            block_ids,
            block_updates,
            leaf.dtype,
        )
        new_leaf = leaf + upd.reshape(shapes[i])
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    return eqx.combine(new_arr_tree, static_tree)
