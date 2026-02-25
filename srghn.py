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
    self_weight_norm_mode: str = eqx.field(static=True)
    self_weight_norm_target: float | None = eqx.field(static=True)
    self_weight_norm_eps: float = eqx.field(static=True)
    freeze_stoch_output_head: bool = eqx.field(static=True)

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


def _global_l2_norm(leaves: list[jnp.ndarray]) -> jnp.ndarray:
    sq_sum = jnp.array(0.0, dtype=jnp.float32)
    for leaf in leaves:
        sq_sum = sq_sum + jnp.sum(jnp.square(leaf))
    return jnp.sqrt(jnp.maximum(sq_sum, 0.0))


def _leaf_l2_norm(leaf: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(leaf)), 0.0))


def _normalize_leaves_global_l2(
    leaves: list[jnp.ndarray], target_norm: float | None, eps: float
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pre_norm = _global_l2_norm(leaves)
    if target_norm is None:
        return leaves, pre_norm, pre_norm, jnp.array(1.0, dtype=pre_norm.dtype)

    target = jnp.asarray(target_norm, dtype=pre_norm.dtype)
    scale = jnp.minimum(1.0, target / jnp.maximum(pre_norm, eps))
    normalized = [leaf * scale for leaf in leaves]
    post_norm = _global_l2_norm(normalized)
    return normalized, pre_norm, post_norm, scale


def _normalize_leaves_per_layer_l2(
    leaves: list[jnp.ndarray], target_norm: float | None, eps: float
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if not leaves:
        zero = jnp.array(0.0, dtype=jnp.float32)
        one = jnp.array(1.0, dtype=jnp.float32)
        return leaves, zero, zero, one

    pre_norms = []
    post_norms = []
    scales = []
    normalized = []

    for leaf in leaves:
        pre = _leaf_l2_norm(leaf)
        if target_norm is None:
            scale = jnp.array(1.0, dtype=pre.dtype)
        else:
            target = jnp.asarray(target_norm, dtype=pre.dtype)
            scale = jnp.minimum(1.0, target / jnp.maximum(pre, eps))
        out = leaf * scale
        post = _leaf_l2_norm(out)

        pre_norms.append(pre)
        post_norms.append(post)
        scales.append(scale)
        normalized.append(out)

    return (
        normalized,
        jnp.mean(jnp.stack(pre_norms)),
        jnp.mean(jnp.stack(post_norms)),
        jnp.mean(jnp.stack(scales)),
    )


def _normalize_leaves(
    leaves: list[jnp.ndarray], mode: str, target_norm: float | None, eps: float
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if mode == "global":
        return _normalize_leaves_global_l2(leaves, target_norm, eps)
    if mode == "per_layer":
        return _normalize_leaves_per_layer_l2(leaves, target_norm, eps)
    raise ValueError(f"Unknown self weight normalization mode: {mode}")


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
        new_leaf = leaf + updates[i]
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)
    new_leaves, _, _, _ = _normalize_leaves(
        new_leaves,
        srghn.self_weight_norm_mode,
        srghn.self_weight_norm_target,
        srghn.self_weight_norm_eps,
    )

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    return eqx.combine(new_arr_tree, static_tree)


def mutate_with_stats(srghn: SRGHN, key: jax.random.KeyArray) -> tuple[SRGHN, dict]:
    h = srghn.encoder_self(srghn.self_node_emb, srghn.self_graph)
    filter_spec = _srghn_filter_spec(srghn)
    arr_tree, static_tree = eqx.partition(srghn, filter_spec)
    flat, treedef = jax.tree_util.tree_flatten(arr_tree)
    idxs = [i for i, leaf in enumerate(flat) if leaf is not None]
    leaves = [flat[i] for i in idxs]
    num_nodes = srghn.self_spec.num_nodes
    max_size = srghn.self_spec.max_size

    keys = jax.random.split(key, num_nodes)

    def fori_body(i, carry):
        out_mat, lrs, std_means, std_stds, clip_fracs = carry
        upd, lr, std_mean, std_std, clip_fraction = srghn.stoch.with_stats(h[i], max_size, keys[i])
        out_mat = out_mat.at[i].set(upd)
        lrs = lrs.at[i].set(lr)
        std_means = std_means.at[i].set(std_mean)
        std_stds = std_stds.at[i].set(std_std)
        clip_fracs = clip_fracs.at[i].set(clip_fraction)
        return out_mat, lrs, std_means, std_stds, clip_fracs

    init = (
        jnp.zeros((num_nodes, max_size), dtype=h.dtype),
        jnp.zeros((num_nodes,), dtype=h.dtype),
        jnp.zeros((num_nodes,), dtype=h.dtype),
        jnp.zeros((num_nodes,), dtype=h.dtype),
        jnp.zeros((num_nodes,), dtype=h.dtype),
    )
    out_mat, lrs, std_means, std_stds, clip_fracs = jax.lax.fori_loop(0, num_nodes, fori_body, init)

    updates = _assemble_params(out_mat, srghn.self_spec)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        new_leaf = leaf + updates[i]
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)
    new_leaves, weight_norm_pre, weight_norm_post, weight_norm_scale = _normalize_leaves(
        new_leaves,
        srghn.self_weight_norm_mode,
        srghn.self_weight_norm_target,
        srghn.self_weight_norm_eps,
    )

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    new_srghn = eqx.combine(new_arr_tree, static_tree)

    stats = {
        "lr_mean": jnp.mean(lrs),
        "lr_std": jnp.std(lrs),
        "std_head_mean": jnp.mean(std_means),
        "std_head_std": jnp.mean(std_stds),
        "update_clip_fraction": jnp.mean(clip_fracs),
        "weight_norm_pre": weight_norm_pre,
        "weight_norm_post": weight_norm_post,
        "weight_norm_scale": weight_norm_scale,
    }
    return new_srghn, stats
