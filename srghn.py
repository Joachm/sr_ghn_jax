from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from gnn import GraphEncoder
from graphs import GraphSpec
from hypernets import DeterministicHead, StochasticHyper
from specs import ParamNodeSpec, _srghn_filter_spec


class SRGHN(eqx.Module):
    self_node_emb: jnp.ndarray
    policy_node_emb: jnp.ndarray
    encoder_self: GraphEncoder
    encoder_policy: GraphEncoder
    stoch: StochasticHyper
    det: DeterministicHead
    self_graph: GraphSpec = eqx.field(static=True)
    policy_graph: GraphSpec = eqx.field(static=True)
    self_spec: ParamNodeSpec = eqx.field(static=True)
    policy_spec: ParamNodeSpec = eqx.field(static=True)
    clip_params: tuple[float, float] = eqx.field(static=True)


def make_policy(srghn: SRGHN) -> tuple[jnp.ndarray, ...]:
    h = srghn.encoder_policy(srghn.policy_node_emb, srghn.policy_graph)
    shapes = srghn.policy_spec.shapes
    sizes = srghn.policy_spec.sizes
    num_nodes = srghn.policy_spec.num_nodes
    max_size = srghn.policy_spec.max_size

    def fori_body(i, out_mat):
        vec = srghn.det(h[i], max_size)
        return out_mat.at[i].set(vec)

    out_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    out_mat = jax.lax.fori_loop(0, num_nodes, fori_body, out_mat)

    outputs = []
    for i, shape in enumerate(shapes):
        vec = out_mat[i, : sizes[i]]
        outputs.append(vec.reshape(shape))
    return tuple(outputs)


def mutate(srghn: SRGHN, key: jax.random.KeyArray) -> SRGHN:
    h = srghn.encoder_self(srghn.self_node_emb, srghn.self_graph)
    filter_spec = _srghn_filter_spec(srghn)
    arr_tree, static_tree = eqx.partition(srghn, filter_spec)
    flat, treedef = jax.tree_util.tree_flatten(arr_tree)
    idxs = [i for i, leaf in enumerate(flat) if leaf is not None]
    leaves = [flat[i] for i in idxs]
    sizes = srghn.self_spec.sizes
    shapes = srghn.self_spec.shapes
    num_nodes = srghn.self_spec.num_nodes
    max_size = srghn.self_spec.max_size

    keys = jax.random.split(key, num_nodes)

    def fori_body(i, out_mat):
        upd, _ = srghn.stoch(h[i], max_size, keys[i])
        return out_mat.at[i].set(upd)

    out_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    out_mat = jax.lax.fori_loop(0, num_nodes, fori_body, out_mat)

    new_leaves = []
    for i, leaf in enumerate(leaves):
        upd = out_mat[i, : sizes[i]]
        new_leaf = leaf + upd.reshape(shapes[i])
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    return eqx.combine(new_arr_tree, static_tree)
