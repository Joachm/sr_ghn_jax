from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp


def _mlp_forward(params: Sequence[jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    x = jnp.ravel(obs)
    num_layers = len(params) // 2
    for i in range(num_layers):
        w = params[2 * i]
        b = params[2 * i + 1]
        x = x @ w.T + b
        if i != num_layers - 1:
            x = jnp.tanh(x)
    return x


def apply_policy(
    policy_params: Sequence[jnp.ndarray],
    obs: jnp.ndarray,
    *,
    is_discrete: bool,
) -> jnp.ndarray:
    logits = _mlp_forward(policy_params, obs)
    if is_discrete:
        return jnp.argmax(logits)
    return jnp.tanh(logits)
