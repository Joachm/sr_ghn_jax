from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from graphs import GraphSpec


class GraphEncoder(eqx.Module):
    msg: eqx.nn.Linear
    gru: eqx.nn.GRUCell
    rel_emb: jnp.ndarray
    max_edge_types: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, steps: int, *, key: jax.random.KeyArray, max_edge_types: int = 8):
        key_msg, key_gru, key_rel = jax.random.split(key, 3)
        self.msg = eqx.nn.Linear(hidden_dim, hidden_dim, key=key_msg, use_bias=True)
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key_gru)
        self.rel_emb = jax.random.normal(key_rel, (max_edge_types, hidden_dim)) * 0.05
        self.max_edge_types = max_edge_types
        self.steps = steps

    def __call__(self, h: jnp.ndarray, graph: GraphSpec) -> jnp.ndarray:
        num_nodes = graph.num_nodes
        src = graph.src
        dst = graph.dst
        edge_type = graph.edge_type

        def step_fn(_, h_t):
            rel = self.rel_emb[edge_type]
            messages = jax.vmap(self.msg)(h_t[src] + rel)
            agg = jnp.zeros((num_nodes, h_t.shape[-1]), dtype=h_t.dtype).at[dst].add(messages)
            counts = jnp.zeros((num_nodes,), dtype=h_t.dtype).at[dst].add(1.0)
            agg = agg / jnp.maximum(counts[:, None], 1.0)
            h_next = jax.vmap(self.gru)(agg, h_t)
            return h_next

        return jax.lax.fori_loop(0, self.steps, step_fn, h)
