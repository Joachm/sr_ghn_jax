from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _layer_scale(fan_in: int, *, gain: float) -> float:
    return gain / jnp.sqrt(max(fan_in, 1))


def _init_linear(key: jax.random.KeyArray, in_dim: int, out_dim: int, *, gain: float) -> dict[str, jnp.ndarray]:
    weight_scale = _layer_scale(in_dim, gain=gain)
    weight = weight_scale * jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight, "bias": bias}


def init_actor_critic_params(
    key: jax.random.KeyArray,
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple[int, ...],
    *,
    is_discrete: bool,
) -> dict[str, Any]:
    keys = jax.random.split(key, len(hidden_dims) + 2)
    trunk = []
    prev_dim = obs_dim
    for idx, hidden_dim in enumerate(hidden_dims):
        trunk.append(_init_linear(keys[idx], prev_dim, hidden_dim, gain=jnp.sqrt(2.0)))
        prev_dim = hidden_dim
    policy_head = _init_linear(keys[-2], prev_dim, act_dim, gain=0.01)
    value_head = _init_linear(keys[-1], prev_dim, 1, gain=1.0)
    params = {
        "trunk": tuple(trunk),
        "policy_head": policy_head,
        "value_head": value_head,
    }
    if not is_discrete:
        params["log_std"] = jnp.zeros((act_dim,), dtype=jnp.float32)
    return params


def _apply_linear(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["weight"].T + params["bias"]


def apply_actor_critic(params, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(obs, dtype=jnp.float32)
    for layer in params["trunk"]:
        x = jnp.tanh(_apply_linear(layer, x))
    policy_output = _apply_linear(params["policy_head"], x)
    value = _apply_linear(params["value_head"], x)[..., 0]
    return policy_output, value


def policy_output_dim(params) -> int:
    return int(params["policy_head"]["bias"].shape[0])


def categorical_sample_and_log_prob(logits: jnp.ndarray, key: jax.random.KeyArray) -> tuple[jnp.ndarray, jnp.ndarray]:
    action = jax.random.categorical(key, logits)
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[action]
    return action.astype(jnp.int32), log_prob


def categorical_log_prob(logits: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_probs, jnp.asarray(action, dtype=jnp.int32)[..., None], axis=-1)[..., 0]


def categorical_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


def gaussian_sample_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = jnp.exp(log_std)
    action = mean + std * jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return action, gaussian_log_prob(mean, log_std, action)


def gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    std = jnp.exp(log_std)
    z = (action - mean) / std
    log_unnormalized = -0.5 * jnp.square(z)
    log_normalizer = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(log_unnormalized - log_normalizer, axis=-1)


def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(log_std + 0.5 * (1.0 + jnp.log(2.0 * jnp.pi)), axis=-1)
