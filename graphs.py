from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GraphSpec:
    src: jnp.ndarray  # int32 (E,)
    dst: jnp.ndarray  # int32 (E,)
    edge_type: jnp.ndarray  # int32 (E,)
    num_nodes: int
    num_edge_types: int

    def tree_flatten(self):
        children = (self.src, self.dst, self.edge_type)
        aux_data = (self.num_nodes, self.num_edge_types)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        src, dst, edge_type = children
        num_nodes, num_edge_types = aux_data
        return cls(src=src, dst=dst, edge_type=edge_type, num_nodes=num_nodes, num_edge_types=num_edge_types)


def _to_int32_array(values: Iterable[int]) -> jnp.ndarray:
    return jnp.array(list(values), dtype=jnp.int32)


def _empty_graph(num_edge_types: int = 1) -> GraphSpec:
    return GraphSpec(
        src=jnp.zeros((0,), dtype=jnp.int32),
        dst=jnp.zeros((0,), dtype=jnp.int32),
        edge_type=jnp.zeros((0,), dtype=jnp.int32),
        num_nodes=0,
        num_edge_types=num_edge_types,
    )


def _build_graph(num_nodes: int, edges: list[tuple[int, int, int]], num_edge_types: int) -> GraphSpec:
    if num_nodes <= 0:
        return _empty_graph(num_edge_types=num_edge_types)
    if not edges:
        return GraphSpec(
            src=jnp.zeros((0,), dtype=jnp.int32),
            dst=jnp.zeros((0,), dtype=jnp.int32),
            edge_type=jnp.zeros((0,), dtype=jnp.int32),
            num_nodes=num_nodes,
            num_edge_types=num_edge_types,
        )
    src, dst, edge_type = zip(*edges)
    return GraphSpec(
        src=_to_int32_array(src),
        dst=_to_int32_array(dst),
        edge_type=_to_int32_array(edge_type),
        num_nodes=num_nodes,
        num_edge_types=num_edge_types,
    )


def make_chain_graph(num_nodes: int, bidir: bool = True) -> GraphSpec:
    if num_nodes <= 0:
        return _empty_graph()
    edges: list[tuple[int, int, int]] = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1, 0))
        if bidir:
            edges.append((i + 1, i, 0))
    return _build_graph(num_nodes, edges, num_edge_types=1)


def make_block_graph(block_ids: tuple[int, ...], bidir: bool = True) -> GraphSpec:
    num_nodes = len(block_ids)
    if num_nodes == 0:
        return _empty_graph(num_edge_types=2 if bidir else 1)

    blocks = {}
    for idx, block_id in enumerate(block_ids):
        blocks.setdefault(block_id, []).append(idx)

    edges: list[tuple[int, int, int]] = []
    for block_nodes in blocks.values():
        for i in range(len(block_nodes) - 1):
            a = block_nodes[i]
            b = block_nodes[i + 1]
            edges.append((a, b, 0))
            if bidir:
                edges.append((b, a, 0))

    for a_block, b_block in zip(sorted(blocks.keys())[:-1], sorted(blocks.keys())[1:]):
        a_nodes = blocks[a_block]
        b_nodes = blocks[b_block]
        if a_nodes and b_nodes:
            a = a_nodes[-1]
            b = b_nodes[0]
            edges.append((a, b, 1))
            if bidir:
                edges.append((b, a, 1))

    return _build_graph(num_nodes, edges, num_edge_types=2)


def make_policy_hierarchical_graph(layer_ids: tuple[int, ...], bidir: bool = True) -> GraphSpec:
    num_nodes = len(layer_ids)
    if num_nodes == 0:
        return _empty_graph(num_edge_types=3 if bidir else 2)

    layers: dict[int, list[int]] = {}
    for idx, layer_id in enumerate(layer_ids):
        layers.setdefault(layer_id, []).append(idx)

    edges: list[tuple[int, int, int]] = []
    ordered_layers = sorted(layers.keys())

    for layer_nodes in layers.values():
        for i, src in enumerate(layer_nodes):
            for dst in layer_nodes[i + 1 :]:
                edges.append((src, dst, 0))
                if bidir:
                    edges.append((dst, src, 0))

    for cur_layer, next_layer in zip(ordered_layers[:-1], ordered_layers[1:]):
        for src in layers[cur_layer]:
            for dst in layers[next_layer]:
                edges.append((src, dst, 1))
                if bidir:
                    edges.append((dst, src, 2))

    return _build_graph(num_nodes, edges, num_edge_types=3 if bidir else 2)


def make_self_hierarchical_graph(
    group_ids: tuple[int, ...],
    parent_ids: tuple[int, ...],
    *,
    context_index: int | None = None,
    bidir: bool = True,
) -> GraphSpec:
    num_nodes = len(group_ids)
    if num_nodes == 0:
        return _empty_graph(num_edge_types=6 if bidir else 5)

    parent_nodes: dict[int, list[int]] = {}
    group_nodes: dict[int, list[int]] = {}
    for idx, parent_id in enumerate(parent_ids):
        if idx == context_index:
            continue
        parent_nodes.setdefault(parent_id, []).append(idx)
    for idx, group_id in enumerate(group_ids):
        if idx == context_index:
            continue
        group_nodes.setdefault(group_id, []).append(idx)

    edges: list[tuple[int, int, int]] = []

    for nodes in parent_nodes.values():
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                edges.append((src, dst, 0))
                if bidir:
                    edges.append((dst, src, 0))

    for nodes in group_nodes.values():
        for i, src in enumerate(nodes):
            for dst in nodes[i + 1 :]:
                if parent_ids[src] == parent_ids[dst]:
                    continue
                edges.append((src, dst, 1))
                if bidir:
                    edges.append((dst, src, 1))

    ordered_groups = sorted(group_nodes.keys())
    for cur_group, next_group in zip(ordered_groups[:-1], ordered_groups[1:]):
        src = group_nodes[cur_group][-1]
        dst = group_nodes[next_group][0]
        edges.append((src, dst, 2))
        if bidir:
            edges.append((dst, src, 3))

    if context_index is not None:
        for group_id in ordered_groups:
            rep = group_nodes[group_id][0]
            edges.append((context_index, rep, 4))
            if bidir:
                edges.append((rep, context_index, 5))

    return _build_graph(num_nodes, edges, num_edge_types=6 if bidir else 5)
