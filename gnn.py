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
    aggregation: str = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        steps: int,
        aggregation: str = "sum",
        *,
        key: jax.random.KeyArray,
    ):
        if aggregation not in ("sum", "mean"):
            raise ValueError(f"Unknown GNN aggregation mode: {aggregation!r}. Expected 'sum' or 'mean'.")
        key_msg, key_gru = jax.random.split(key, 2)
        self.msg = eqx.nn.Linear(hidden_dim, hidden_dim, key=key_msg, use_bias=True)
        self.gru = eqx.nn.GRUCell(hidden_dim, hidden_dim, key=key_gru)
        self.steps = steps
        self.aggregation = aggregation

    def __call__(self, h: jnp.ndarray, graph: GraphSpec) -> jnp.ndarray:
        num_nodes = graph.num_nodes
        src = graph.src
        dst = graph.dst
        if self.aggregation == "mean":
            in_deg = jnp.zeros((num_nodes,), dtype=h.dtype).at[dst].add(
                jnp.ones((dst.shape[0],), dtype=h.dtype)
            )
            in_deg = jnp.maximum(in_deg, 1.0)

        def step_fn(_, h_t):
            messages = jax.vmap(self.msg)(h_t[src])
            agg = jnp.zeros((num_nodes, h_t.shape[-1]), dtype=h_t.dtype).at[dst].add(messages)
            if self.aggregation == "mean":
                agg = agg / in_deg[:, None]
            h_next = jax.vmap(self.gru)(agg, h_t)
            return h_next

        return jax.lax.fori_loop(0, self.steps, step_fn, h)
