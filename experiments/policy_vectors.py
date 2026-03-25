from __future__ import annotations

import jax
import jax.numpy as jnp

from specs import ParamNodeSpec


def policy_num_dims(policy_spec: ParamNodeSpec) -> int:
    return int(sum(policy_spec.sizes))


def unflatten_policy_vector(vector: jnp.ndarray, policy_spec: ParamNodeSpec) -> tuple[jnp.ndarray, ...]:
    flat = jnp.ravel(jnp.asarray(vector, dtype=jnp.float32))
    params = []
    offset = 0
    for shape, size in zip(policy_spec.shapes, policy_spec.sizes):
        next_offset = offset + size
        params.append(flat[offset:next_offset].reshape(shape))
        offset = next_offset
    return tuple(params)


def flatten_policy_params(policy_params: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    if not policy_params:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate([jnp.ravel(jnp.asarray(param, dtype=jnp.float32)) for param in policy_params], axis=0)


def init_policy_vector_population(
    key: jax.random.KeyArray,
    pop_size: int,
    policy_spec: ParamNodeSpec,
    *,
    scale: float = 0.1,
) -> jnp.ndarray:
    num_dims = policy_num_dims(policy_spec)
    return scale * jax.random.normal(key, (pop_size, num_dims), dtype=jnp.float32)


def zero_policy_vector(policy_spec: ParamNodeSpec) -> jnp.ndarray:
    return jnp.zeros((policy_num_dims(policy_spec),), dtype=jnp.float32)
