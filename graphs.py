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
    num_nodes: int

    def tree_flatten(self):
        children = (self.src, self.dst)
        aux_data = self.num_nodes
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        src, dst = children
        return cls(src=src, dst=dst, num_nodes=aux_data)


def _to_int32_array(values: Iterable[int]) -> jnp.ndarray:
    return jnp.array(list(values), dtype=jnp.int32)


def make_chain_graph(num_nodes: int, bidir: bool = True) -> GraphSpec:
    if num_nodes <= 0:
        return GraphSpec(src=jnp.zeros((0,), dtype=jnp.int32),
                         dst=jnp.zeros((0,), dtype=jnp.int32),
                         num_nodes=0)
    src = []
    dst = []
    for i in range(num_nodes - 1):
        src.append(i)
        dst.append(i + 1)
        if bidir:
            src.append(i + 1)
            dst.append(i)
    return GraphSpec(src=_to_int32_array(src),
                     dst=_to_int32_array(dst),
                     num_nodes=num_nodes)


def make_block_graph(block_ids: tuple[int, ...], bidir: bool = True) -> GraphSpec:
    num_nodes = len(block_ids)
    if num_nodes == 0:
        return GraphSpec(src=jnp.zeros((0,), dtype=jnp.int32),
                         dst=jnp.zeros((0,), dtype=jnp.int32),
                         num_nodes=0)

    blocks = {}
    for idx, block_id in enumerate(block_ids):
        blocks.setdefault(block_id, []).append(idx)

    src = []
    dst = []
    for block_nodes in blocks.values():
        for i in range(len(block_nodes) - 1):
            a = block_nodes[i]
            b = block_nodes[i + 1]
            src.append(a)
            dst.append(b)
            if bidir:
                src.append(b)
                dst.append(a)

    for a_block, b_block in zip(sorted(blocks.keys())[:-1], sorted(blocks.keys())[1:]):
        a_nodes = blocks[a_block]
        b_nodes = blocks[b_block]
        if a_nodes and b_nodes:
            a = a_nodes[-1]
            b = b_nodes[0]
            src.append(a)
            dst.append(b)
            if bidir:
                src.append(b)
                dst.append(a)

    return GraphSpec(src=_to_int32_array(src),
                     dst=_to_int32_array(dst),
                     num_nodes=num_nodes)
