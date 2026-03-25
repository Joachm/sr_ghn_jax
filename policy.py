from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


def _dense_forward(params: Sequence[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(params) // 2
    for i in range(num_layers):
        w = params[2 * i]
        b = params[2 * i + 1]
        x = x @ w.T + b
        if i != num_layers - 1:
            x = jnp.tanh(x)
    return x


def _mlp_forward(params: Sequence[jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    return _dense_forward(params, jnp.ravel(obs))


def _cnn_mlp_forward(params: Sequence[jnp.ndarray], obs: jnp.ndarray, config) -> jnp.ndarray:
    x = jnp.asarray(obs, dtype=jnp.float32)
    num_conv_layers = len(getattr(config, "policy_conv_channels", ()))
    for idx in range(num_conv_layers):
        weight = params[2 * idx]
        bias = params[2 * idx + 1]
        x = jax.lax.conv_general_dilated(
            x[jnp.newaxis, ...],
            weight,
            window_strides=getattr(config, "policy_conv_strides", ())[idx],
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )[0]
        x = jnp.tanh(x + bias)
    return _dense_forward(params[2 * num_conv_layers :], jnp.ravel(x))


def apply_policy(
    policy_params: Sequence[jnp.ndarray],
    obs: jnp.ndarray,
    config,
    *,
    is_discrete: bool,
) -> jnp.ndarray:
    policy_architecture = getattr(config, "policy_architecture", "mlp")
    if policy_architecture == "cnn_mlp":
        logits = _cnn_mlp_forward(policy_params, obs, config)
    elif policy_architecture == "mlp":
        logits = _mlp_forward(policy_params, obs)
    else:
        raise ValueError(f"Unknown policy_architecture: {policy_architecture}")
    if is_discrete:
        return jnp.argmax(logits)
    return logits
