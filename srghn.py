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
    mutation_scale_alpha: float = eqx.field(static=True)
    mutation_norm_eps: float = eqx.field(static=True)

def _param_offsets(param_sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets = []
    total = 0
    for size in param_sizes:
        offsets.append(total)
        total += size
    return tuple(offsets)


def _assemble_params(out_mat: jnp.ndarray, spec: ParamNodeSpec) -> list[jnp.ndarray]:
    param_sizes = spec.param_sizes
    param_offsets = _param_offsets(param_sizes)
    total_size = param_offsets[-1] + param_sizes[-1] if param_sizes else 0
    flat = jnp.zeros((total_size,), dtype=out_mat.dtype)
    for i in range(spec.num_nodes):
        size = spec.sizes[i]
        if size == 0:
            continue
        param_idx = spec.shard_param_idxs[i]
        start = spec.shard_starts[i]
        global_start = param_offsets[param_idx] + start
        flat = flat.at[global_start : global_start + size].set(out_mat[i, :size])

    outputs = []
    for idx, shape in enumerate(spec.shapes):
        start = param_offsets[idx]
        size = param_sizes[idx]
        vec = flat[start : start + size]
        outputs.append(vec.reshape(shape))
    return outputs


def make_policy(srghn: SRGHN) -> tuple[jnp.ndarray, ...]:
    h = srghn.encoder_policy(srghn.policy_node_emb, srghn.policy_graph)
    num_nodes = srghn.policy_spec.num_nodes
    max_size = srghn.policy_spec.max_size

    def fori_body(i, out_mat):
        vec = srghn.det(h[i], max_size)
        return out_mat.at[i].set(vec)

    out_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    out_mat = jax.lax.fori_loop(0, num_nodes, fori_body, out_mat)

    outputs = _assemble_params(out_mat, srghn.policy_spec)
    return tuple(outputs)


def mutate(srghn: SRGHN, key: jax.random.KeyArray) -> SRGHN:
    h = srghn.encoder_self(srghn.self_node_emb, srghn.self_graph)
    filter_spec = _srghn_filter_spec(srghn)
    arr_tree, static_tree = eqx.partition(srghn, filter_spec)
    flat, treedef = jax.tree_util.tree_flatten(arr_tree)
    idxs = [i for i, leaf in enumerate(flat) if leaf is not None]
    leaves = [flat[i] for i in idxs]
    num_nodes = srghn.self_spec.num_nodes
    max_size = srghn.self_spec.max_size

    keys = jax.random.split(key, num_nodes)

    def fori_body(i, out_mat):
        upd, _ = srghn.stoch(h[i], max_size, keys[i])
        return out_mat.at[i].set(upd)

    out_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    out_mat = jax.lax.fori_loop(0, num_nodes, fori_body, out_mat)

    updates = _assemble_params(out_mat, srghn.self_spec)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        raw_update = updates[i]
        param_rms = jnp.sqrt(jnp.mean(jnp.square(leaf)) + srghn.mutation_norm_eps)
        update_rms = jnp.sqrt(jnp.mean(jnp.square(raw_update)) + srghn.mutation_norm_eps)
        scaled_update = raw_update * (param_rms / update_rms) * srghn.mutation_scale_alpha
        new_leaf = leaf + scaled_update
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    return eqx.combine(new_arr_tree, static_tree)
