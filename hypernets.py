from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp


class _Trunk(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear

    def __init__(self, in_dim: int, hidden_dim: int, *, key: jax.random.KeyArray):
        k1, k2 = jax.random.split(key, 2)
        self.lin1 = eqx.nn.Linear(in_dim, hidden_dim, use_bias=False, key=k1)
        self.lin2 = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.lin1(x))
        x = jax.nn.relu(self.lin2(x))
        return x


class StochasticHyper(eqx.Module):
    trunk: _Trunk
    std_head: eqx.nn.Linear
    lr_head: eqx.nn.Linear
    basis: jnp.ndarray
    clip_std: tuple[float, float] = eqx.field(static=True)
    clip_update: tuple[float, float] = eqx.field(static=True)
    const_noise_std: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        coeff_dim: int,
        max_out: int,
        mutation_rate_head_dim: int,
        clip_std: tuple[float, float],
        clip_update: tuple[float, float],
        const_noise_std: float,
        *,
        key: jax.random.KeyArray,
    ):
        k_trunk, k_std, k_lr, k_basis = jax.random.split(key, 4)
        self.trunk = _Trunk(in_dim, hidden_dim, key=k_trunk)
        self.std_head = eqx.nn.Linear(hidden_dim, coeff_dim, use_bias=False, key=k_std)
        self.lr_head = eqx.nn.Linear(hidden_dim, mutation_rate_head_dim, use_bias=True, key=k_lr)
        basis_init = jax.nn.initializers.orthogonal()
        self.basis = basis_init(k_basis, (coeff_dim, max_out), jnp.float32)
        self.clip_std = clip_std
        self.clip_update = clip_update
        self.const_noise_std = const_noise_std

    def _sample_update(
        self, h_i: jnp.ndarray, out_dim: int, key: jax.random.KeyArray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        k_eps, k_noise = jax.random.split(key, 2)
        x = self.trunk(h_i)
        log_std = self.std_head(x)
        std = jnp.clip(jnp.exp(log_std), self.clip_std[0], self.clip_std[1])

        coeffs = jax.random.normal(k_eps, std.shape) * std

        w_out = coeffs @ self.basis
        clipped = jnp.clip(w_out, self.clip_update[0], self.clip_update[1])

        lr_logits = self.lr_head(x)
        lr = jnp.max(jax.nn.sigmoid(lr_logits))

        noise = jax.random.normal(k_noise, w_out.shape) * self.const_noise_std
        update = clipped * lr + noise
        hit_clip = jnp.logical_or(w_out <= self.clip_update[0], w_out >= self.clip_update[1])
        clip_fraction = jnp.mean(hit_clip.astype(jnp.float32))
        std_mean = jnp.mean(std)
        std_std = jnp.std(std)
        return update[:out_dim], lr, std_mean, std_std, clip_fraction

    def __call__(self, h_i: jnp.ndarray, out_dim: int, key: jax.random.KeyArray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (update_vec[out_dim], mutation_rate_scalar)."""
        update, lr, _, _, _ = self._sample_update(h_i, out_dim, key)
        return update, lr

    def with_stats(
        self, h_i: jnp.ndarray, out_dim: int, key: jax.random.KeyArray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._sample_update(h_i, out_dim, key)


class DeterministicHead(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, max_out: int, *, key: jax.random.KeyArray):
        self.linear = eqx.nn.Linear(in_dim, max_out, use_bias=True, key=key)

    def __call__(self, h_i: jnp.ndarray, out_dim: int) -> jnp.ndarray:
        return self.linear(h_i)[:out_dim]
