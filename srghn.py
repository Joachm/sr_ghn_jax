from __future__ import annotations

from dataclasses import dataclass
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


@jax.tree_util.register_pytree_node_class
@dataclass
class MutationMetadata:
    mutation_rate_mean: jnp.ndarray
    mutation_rate_max: jnp.ndarray
    mutation_block_fraction: jnp.ndarray
    mutation_blocks_selected: jnp.ndarray
    mutation_total_blocks: jnp.ndarray
    update_rms: jnp.ndarray
    self_distance_rms: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.mutation_rate_mean,
            self.mutation_rate_max,
            self.mutation_block_fraction,
            self.mutation_blocks_selected,
            self.mutation_total_blocks,
            self.update_rms,
            self.self_distance_rms,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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


def _zero_metadata(dtype) -> MutationMetadata:
    zero_f = jnp.asarray(0.0, dtype=dtype)
    zero_i = jnp.asarray(0, dtype=jnp.int32)
    return MutationMetadata(
        mutation_rate_mean=zero_f,
        mutation_rate_max=zero_f,
        mutation_block_fraction=zero_f,
        mutation_blocks_selected=zero_i,
        mutation_total_blocks=zero_i,
        update_rms=zero_f,
        self_distance_rms=zero_f,
    )


def _should_exclude(module_name: str, excluded_modules: tuple[str, ...]) -> bool:
    return module_name in excluded_modules


def mutate_with_metadata(
    srghn: SRGHN,
    key: jax.random.KeyArray,
    *,
    excluded_modules: tuple[str, ...] = (),
    fixed_mutation_lr: float | None = None,
    apply_updates: bool = True,
) -> tuple[SRGHN, MutationMetadata]:
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
    lr_values = []
    update_sq_sum = jnp.asarray(0.0, dtype=h.dtype)
    update_size = jnp.asarray(0.0, dtype=h.dtype)
    selected_blocks = jnp.asarray(0, dtype=jnp.int32)
    total_blocks = jnp.asarray(0, dtype=jnp.int32)
    for i, leaf in enumerate(leaves):
        if _should_exclude(srghn.self_spec.module_names[i], excluded_modules):
            upd = jnp.zeros((sizes[i],), dtype=leaf.dtype)
            new_leaf = leaf
        else:
            block_ids, block_updates, lr = srghn.stoch(
                h[i],
                child_ctx,
                sizes[i],
                keys[i + 1],
                fixed_lr=fixed_mutation_lr,
            )
            upd = _dense_delta_from_blocks(
                sizes[i],
                srghn.stoch.block_size,
                block_ids,
                block_updates,
                leaf.dtype,
            )
            total_blocks = total_blocks + jnp.asarray(_num_blocks(sizes[i], srghn.stoch.block_size), dtype=jnp.int32)
            selected_blocks = selected_blocks + jnp.asarray(block_ids.shape[0], dtype=jnp.int32)
            lr_values.append(lr.astype(h.dtype))
            update_sq_sum = update_sq_sum + jnp.sum(jnp.square(upd.astype(h.dtype)))
            update_size = update_size + jnp.asarray(max(sizes[i], 1), dtype=h.dtype)
            if apply_updates:
                new_leaf = leaf + upd.reshape(shapes[i])
                new_leaf = jnp.clip(new_leaf, srghn.clip_params[0], srghn.clip_params[1])
            else:
                new_leaf = leaf
        new_leaves.append(new_leaf)

    new_flat = list(flat)
    for out_idx, flat_idx in enumerate(idxs):
        new_flat[flat_idx] = new_leaves[out_idx]
    new_arr_tree = jax.tree_util.tree_unflatten(treedef, new_flat)
    next_srghn = eqx.combine(new_arr_tree, static_tree)

    if not lr_values:
        return next_srghn, _zero_metadata(h.dtype)

    lr_stack = jnp.stack(lr_values)
    update_rms = jnp.sqrt(update_sq_sum / jnp.maximum(update_size, 1.0))
    metadata = MutationMetadata(
        mutation_rate_mean=jnp.mean(lr_stack),
        mutation_rate_max=jnp.max(lr_stack),
        mutation_block_fraction=selected_blocks.astype(h.dtype) / jnp.maximum(total_blocks.astype(h.dtype), 1.0),
        mutation_blocks_selected=selected_blocks,
        mutation_total_blocks=total_blocks,
        update_rms=update_rms,
        self_distance_rms=update_rms,
    )
    return next_srghn, metadata


def mutation_metadata(
    srghn: SRGHN,
    key: jax.random.KeyArray,
    *,
    excluded_modules: tuple[str, ...] = (),
    fixed_mutation_lr: float | None = None,
) -> MutationMetadata:
    _, metadata = mutate_with_metadata(
        srghn,
        key,
        excluded_modules=excluded_modules,
        fixed_mutation_lr=fixed_mutation_lr,
        apply_updates=False,
    )
    return metadata


def mutate(srghn: SRGHN, key: jax.random.KeyArray) -> SRGHN:
    child, _ = mutate_with_metadata(srghn, key)
    return child
