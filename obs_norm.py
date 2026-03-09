from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ObsNormState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray

    def tree_flatten(self):
        return (self.mean, self.var, self.count), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        mean, var, count = children
        return cls(mean=mean, var=var, count=count)


def init_obs_norm(obs_dim: int, *, dtype=jnp.float32) -> ObsNormState:
    return ObsNormState(
        mean=jnp.zeros((obs_dim,), dtype=dtype),
        var=jnp.ones((obs_dim,), dtype=dtype),
        count=jnp.zeros((), dtype=dtype),
    )


def normalize_obs(obs: jnp.ndarray, state: ObsNormState | None, *, clip: float, eps: float) -> jnp.ndarray:
    if state is None:
        return obs
    flat_obs = jnp.ravel(jnp.asarray(obs, dtype=state.mean.dtype))
    normed = (flat_obs - state.mean) / jnp.sqrt(jnp.maximum(state.var, eps))
    normed = jnp.clip(normed, -clip, clip)
    return normed.reshape(obs.shape)


def update_obs_norm(
    state: ObsNormState,
    obs_sum: jnp.ndarray,
    obs_sq_sum: jnp.ndarray,
    obs_count: jnp.ndarray,
    *,
    min_var: float = 1e-6,
) -> ObsNormState:
    batch_count = jnp.asarray(obs_count, dtype=state.count.dtype)
    safe_batch_count = jnp.maximum(batch_count, 1.0)
    batch_mean = obs_sum / safe_batch_count
    batch_var = jnp.maximum(obs_sq_sum / safe_batch_count - jnp.square(batch_mean), min_var)

    total_count = state.count + batch_count
    safe_total_count = jnp.maximum(total_count, 1.0)
    delta = batch_mean - state.mean
    new_mean = state.mean + delta * (batch_count / safe_total_count)

    m_a = state.var * state.count
    m_b = batch_var * batch_count
    correction = jnp.square(delta) * state.count * batch_count / safe_total_count
    new_var = jnp.maximum((m_a + m_b + correction) / safe_total_count, min_var)

    has_batch = batch_count > 0
    return ObsNormState(
        mean=jnp.where(has_batch, new_mean, state.mean),
        var=jnp.where(has_batch, new_var, state.var),
        count=jnp.where(has_batch, total_count, state.count),
    )
