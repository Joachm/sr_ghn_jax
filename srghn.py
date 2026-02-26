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
    self_reg_mode: str = eqx.field(static=True)
    self_weight_decay: float = eqx.field(static=True)
    self_weight_norm_mode: str = eqx.field(static=True)
    self_weight_norm_target: float | None = eqx.field(static=True)
    self_weight_norm_eps: float = eqx.field(static=True)
    shard_residual_scale: float = eqx.field(static=True)
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


def _mean_leaf_l2_norm(leaves: list[jnp.ndarray]) -> jnp.ndarray:
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    norms = [_leaf_l2_norm(leaf) for leaf in leaves]
    return jnp.mean(jnp.stack(norms))


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


def _decay_leaves(
    leaves: list[jnp.ndarray], decay: float
) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    keep = jnp.clip(1.0 - jnp.asarray(decay, dtype=jnp.float32), 0.0, 1.0)
    pre_norm = _mean_leaf_l2_norm(leaves)
    decayed = [leaf * keep for leaf in leaves]
    post_norm = _mean_leaf_l2_norm(decayed)
    return decayed, pre_norm, post_norm, keep


def _identity_regularization(leaves: list[jnp.ndarray]) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    pre_norm = _mean_leaf_l2_norm(leaves)
    scale = jnp.array(1.0, dtype=pre_norm.dtype)
    return leaves, pre_norm, pre_norm, scale


def _regularize_leaves(srghn: SRGHN, leaves: list[jnp.ndarray]) -> tuple[list[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if srghn.self_reg_mode == "weight_norm":
        return _normalize_leaves(
            leaves,
            srghn.self_weight_norm_mode,
            srghn.self_weight_norm_target,
            srghn.self_weight_norm_eps,
        )
    if srghn.self_reg_mode == "weight_decay":
        if srghn.self_weight_decay < 0.0 or srghn.self_weight_decay > 1.0:
            raise ValueError("self_weight_decay must be in [0, 1].")
        return _decay_leaves(leaves, srghn.self_weight_decay)
    if srghn.self_reg_mode == "none":
        return _identity_regularization(leaves)
    raise ValueError(f"Unknown self regularization mode: {srghn.self_reg_mode}")


def make_policy(srghn: SRGHN) -> tuple[jnp.ndarray, ...]:
    h = srghn.encoder_policy(srghn.policy_node_emb, srghn.policy_graph)
    num_nodes = srghn.policy_spec.num_nodes
    max_size = srghn.policy_spec.max_size
    det_out = int(srghn.det.linear.weight.shape[0])
    if det_out < max_size:
        raise ValueError(
            "Deterministic head output dimension is smaller than required policy node size: "
            f"{det_out} < {max_size}."
        )

    def fori_body(i, out_mat):
        vec = srghn.det(h[i], max_size)
        return out_mat.at[i].set(vec)

    out_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    out_mat = jax.lax.fori_loop(0, num_nodes, fori_body, out_mat)

    outputs = _assemble_params(out_mat, srghn.policy_spec)
    return tuple(outputs)


def _sample_group_latents(
    spec: ParamNodeSpec, cov_rank: int, key: jax.random.KeyArray, dtype: jnp.dtype
) -> tuple[jnp.ndarray, jnp.ndarray]:
    shard_group_idxs = jnp.asarray(spec.shard_param_idxs, dtype=jnp.int32)
    num_groups = len(spec.param_sizes)
    if cov_rank <= 0:
        return shard_group_idxs, jnp.zeros((num_groups, 0), dtype=dtype)
    return shard_group_idxs, jax.random.normal(key, (num_groups, cov_rank), dtype=dtype)


def _group_counts(spec: ParamNodeSpec, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    shard_group_idxs = jnp.asarray(spec.shard_param_idxs, dtype=jnp.int32)
    num_groups = len(spec.param_sizes)
    counts = jnp.zeros((num_groups,), dtype=dtype).at[shard_group_idxs].add(
        jnp.ones((spec.num_nodes,), dtype=dtype)
    )
    return shard_group_idxs, counts


def _mean_pairwise_sibling_cosine(
    out_mat: jnp.ndarray, shard_group_idxs: jnp.ndarray, group_counts: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    if out_mat.shape[0] == 0:
        return jnp.array(0.0, dtype=jnp.float32)
    norms = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(out_mat), axis=1), 0.0))
    unit = out_mat / jnp.maximum(norms[:, None], eps)
    num_groups = group_counts.shape[0]
    group_sum = jnp.zeros((num_groups, out_mat.shape[1]), dtype=out_mat.dtype).at[shard_group_idxs].add(unit)
    group_sq = jnp.sum(jnp.square(group_sum), axis=1)
    k = group_counts
    pair_counts = 0.5 * k * jnp.maximum(k - 1.0, 0.0)
    pair_cos = (group_sq - k) / jnp.maximum(k * (k - 1.0), 1.0)
    pair_cos = jnp.clip(pair_cos, -1.0, 1.0)
    total_pairs = jnp.sum(pair_counts)
    return jnp.where(
        total_pairs > 0,
        jnp.sum(pair_cos * pair_counts) / total_pairs,
        jnp.array(0.0, dtype=out_mat.dtype),
    )


def _assembled_update_norm_stats(
    updates: list[jnp.ndarray], group_counts: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if not updates:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return zero, zero, zero
    norms = jnp.stack([_leaf_l2_norm(leaf) for leaf in updates])
    per_shard = norms / jnp.maximum(group_counts, 1.0)
    x = norms
    y = group_counts.astype(norms.dtype)
    x_centered = x - jnp.mean(x)
    y_centered = y - jnp.mean(y)
    corr_denom = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.square(x_centered)) * jnp.sum(jnp.square(y_centered)), 0.0)
    )
    corr = jnp.where(corr_denom > 0, jnp.sum(x_centered * y_centered) / corr_denom, jnp.array(0.0, dtype=x.dtype))
    return jnp.mean(norms), jnp.mean(per_shard), corr


def mutate(srghn: SRGHN, key: jax.random.KeyArray) -> SRGHN:
    h = srghn.encoder_self(srghn.self_node_emb, srghn.self_graph)
    filter_spec = _srghn_filter_spec(srghn)
    arr_tree, static_tree = eqx.partition(srghn, filter_spec)
    flat, treedef = jax.tree_util.tree_flatten(arr_tree)
    idxs = [i for i, leaf in enumerate(flat) if leaf is not None]
    leaves = [flat[i] for i in idxs]
    num_nodes = srghn.self_spec.num_nodes
    max_size = srghn.self_spec.max_size

    key_local, key_group, key_cov = jax.random.split(key, 3)
    shard_group_idxs, group_counts = _group_counts(srghn.self_spec, h.dtype)
    num_groups = len(srghn.self_spec.param_sizes)
    local_keys = jax.random.split(key_local, num_nodes)
    group_keys = jax.random.split(key_group, num_groups)
    _, group_latents = _sample_group_latents(srghn.self_spec, srghn.stoch.cov_rank, key_cov, h.dtype)

    group_h_sum = jnp.zeros((num_groups, h.shape[-1]), dtype=h.dtype).at[shard_group_idxs].add(h)
    group_h = group_h_sum / jnp.maximum(group_counts[:, None], 1.0)
    group_stds, group_lrs, _, _ = jax.vmap(srghn.stoch.scales_from_hidden)(group_h)

    def local_body(i, out_mat):
        group_idx = shard_group_idxs[i]
        upd, _ = srghn.stoch(
            h[i],
            max_size,
            local_keys[i],
            group_latent=group_latents[group_idx],
            forced_std=group_stds[group_idx],
            forced_lr=group_lrs[group_idx],
        )
        return out_mat.at[i].set(upd)

    local_mat = jnp.zeros((num_nodes, max_size), dtype=h.dtype)
    local_mat = jax.lax.fori_loop(0, num_nodes, local_body, local_mat)

    def group_body(group_idx, out_mat):
        upd, _ = srghn.stoch(
            group_h[group_idx],
            max_size,
            group_keys[group_idx],
            group_latent=group_latents[group_idx],
            forced_std=group_stds[group_idx],
            forced_lr=group_lrs[group_idx],
        )
        return out_mat.at[group_idx].set(upd)

    group_mat = jnp.zeros((num_groups, max_size), dtype=h.dtype)
    group_mat = jax.lax.fori_loop(0, num_groups, group_body, group_mat)

    group_local_sum = jnp.zeros((num_groups, max_size), dtype=h.dtype).at[shard_group_idxs].add(local_mat)
    group_local_mean = group_local_sum / jnp.maximum(group_counts[:, None], 1.0)
    residual_mat = local_mat - group_local_mean[shard_group_idxs]
    out_preclip = group_mat[shard_group_idxs] + srghn.shard_residual_scale * residual_mat
    out_mat = jnp.clip(out_preclip, srghn.stoch.clip_update[0], srghn.stoch.clip_update[1])

    updates = _assemble_params(out_mat, srghn.self_spec)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        new_leaf = leaf + updates[i]
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)
    new_leaves, _, _, _ = _regularize_leaves(srghn, new_leaves)

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

    key_local, key_group, key_cov = jax.random.split(key, 3)
    shard_group_idxs, group_counts = _group_counts(srghn.self_spec, h.dtype)
    num_groups = len(srghn.self_spec.param_sizes)
    local_keys = jax.random.split(key_local, num_nodes)
    group_keys = jax.random.split(key_group, num_groups)
    _, group_latents = _sample_group_latents(srghn.self_spec, srghn.stoch.cov_rank, key_cov, h.dtype)

    group_h_sum = jnp.zeros((num_groups, h.shape[-1]), dtype=h.dtype).at[shard_group_idxs].add(h)
    group_h = group_h_sum / jnp.maximum(group_counts[:, None], 1.0)
    group_stds, group_lrs, group_std_means, group_std_stds = jax.vmap(srghn.stoch.scales_from_hidden)(group_h)

    def local_body(i, carry):
        out_mat, clip_fracs = carry
        group_idx = shard_group_idxs[i]
        upd, _, _, _, clip_fraction = srghn.stoch.with_stats(
            h[i],
            max_size,
            local_keys[i],
            group_latent=group_latents[group_idx],
            forced_std=group_stds[group_idx],
            forced_lr=group_lrs[group_idx],
        )
        out_mat = out_mat.at[i].set(upd)
        clip_fracs = clip_fracs.at[i].set(clip_fraction)
        return out_mat, clip_fracs

    local_init = (
        jnp.zeros((num_nodes, max_size), dtype=h.dtype),
        jnp.zeros((num_nodes,), dtype=h.dtype),
    )
    local_mat, local_clip_fracs = jax.lax.fori_loop(0, num_nodes, local_body, local_init)

    def group_body(group_idx, carry):
        out_mat, clip_fracs = carry
        upd, _, _, _, clip_fraction = srghn.stoch.with_stats(
            group_h[group_idx],
            max_size,
            group_keys[group_idx],
            group_latent=group_latents[group_idx],
            forced_std=group_stds[group_idx],
            forced_lr=group_lrs[group_idx],
        )
        out_mat = out_mat.at[group_idx].set(upd)
        clip_fracs = clip_fracs.at[group_idx].set(clip_fraction)
        return out_mat, clip_fracs

    group_init = (
        jnp.zeros((num_groups, max_size), dtype=h.dtype),
        jnp.zeros((num_groups,), dtype=h.dtype),
    )
    group_mat, group_clip_fracs = jax.lax.fori_loop(0, num_groups, group_body, group_init)

    group_local_sum = jnp.zeros((num_groups, max_size), dtype=h.dtype).at[shard_group_idxs].add(local_mat)
    group_local_mean = group_local_sum / jnp.maximum(group_counts[:, None], 1.0)
    residual_mat = local_mat - group_local_mean[shard_group_idxs]
    out_preclip = group_mat[shard_group_idxs] + srghn.shard_residual_scale * residual_mat
    out_mat = jnp.clip(out_preclip, srghn.stoch.clip_update[0], srghn.stoch.clip_update[1])
    final_hit_clip = jnp.logical_or(
        out_preclip <= srghn.stoch.clip_update[0], out_preclip >= srghn.stoch.clip_update[1]
    )
    final_clip_fracs = jnp.mean(final_hit_clip.astype(h.dtype), axis=1)
    lrs = group_lrs[shard_group_idxs]
    std_means = group_std_means[shard_group_idxs]
    std_stds = group_std_stds[shard_group_idxs]

    updates = _assemble_params(out_mat, srghn.self_spec)
    new_leaves = []
    for i, leaf in enumerate(leaves):
        new_leaf = leaf + updates[i]
        new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
        new_leaves.append(new_leaf)
    new_leaves, weight_norm_pre, weight_norm_post, weight_norm_scale = _regularize_leaves(srghn, new_leaves)

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    new_srghn = eqx.combine(new_arr_tree, static_tree)

    sibling_cosine_mean = _mean_pairwise_sibling_cosine(out_mat, shard_group_idxs, group_counts)
    assembled_norm_mean, assembled_norm_per_shard_mean, assembled_norm_shard_corr = _assembled_update_norm_stats(
        updates, group_counts
    )
    group_base_norm_mean = jnp.mean(jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(group_mat[shard_group_idxs]), axis=1), 0.0)))
    residual_norm_mean = jnp.mean(jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(residual_mat), axis=1), 0.0)))
    residual_to_group_norm = residual_norm_mean / jnp.maximum(group_base_norm_mean, 1e-8)

    stats = {
        "lr_mean": jnp.mean(lrs),
        "lr_std": jnp.std(lrs),
        "std_head_mean": jnp.mean(std_means),
        "std_head_std": jnp.mean(std_stds),
        "update_clip_fraction": jnp.mean(final_clip_fracs),
        "local_update_clip_fraction": jnp.mean(local_clip_fracs),
        "group_update_clip_fraction": jnp.mean(group_clip_fracs),
        "shard_sibling_cosine_mean": sibling_cosine_mean,
        "assembled_update_norm_mean": assembled_norm_mean,
        "assembled_update_norm_per_shard_mean": assembled_norm_per_shard_mean,
        "assembled_update_norm_shard_corr": assembled_norm_shard_corr,
        "group_base_norm_mean": group_base_norm_mean,
        "residual_norm_mean": residual_norm_mean,
        "residual_to_group_norm": residual_to_group_norm,
        "self_reg_pre_norm": weight_norm_pre,
        "self_reg_post_norm": weight_norm_post,
        "self_reg_scale": weight_norm_scale,
        "weight_norm_pre": weight_norm_pre,
        "weight_norm_post": weight_norm_post,
        "weight_norm_scale": weight_norm_scale,
    }
    return new_srghn, stats
