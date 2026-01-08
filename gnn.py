from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from graphs import GraphSpec


class GraphEncoder(eqx.Module):
    msg: eqx.nn.Linear
    gru: eqx.nn.GRUCell
    steps: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, steps: int, *, key: jax.random.KeyArray):
        key_msg, key_gru = jax.random.split(key, 2)
        self.msg = eqx.nn.Linear(hidden_dim, hidden_dim, key=key_msg, use_bias=True)
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key_gru)
        self.steps = steps

    def __call__(self, h: jnp.ndarray, graph: GraphSpec) -> jnp.ndarray:
        num_nodes = graph.num_nodes
        src = graph.src
        dst = graph.dst

        def step_fn(_, h_t):
            messages = jax.vmap(self.msg)(h_t[src])
            agg = jnp.zeros((num_nodes, h_t.shape[-1]), dtype=h_t.dtype).at[dst].add(messages)
            h_next = jax.vmap(self.gru)(agg, h_t)
            return h_next

        return jax.lax.fori_loop(0, self.steps, step_fn, h)
