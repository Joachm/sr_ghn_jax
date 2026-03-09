from __future__ import annotations

from math import ceil

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


def _num_blocks(out_dim: int, block_size: int) -> int:
    if out_dim <= 0:
        return 0
    return ceil(out_dim / block_size)


def _block_position_features(num_blocks: int, block_size: int, out_dim: int, dtype) -> jnp.ndarray:
    if num_blocks <= 0:
        return jnp.zeros((0, 6), dtype=dtype)

    idx = jnp.arange(num_blocks, dtype=dtype)
    denom = jnp.asarray(max(num_blocks - 1, 1), dtype=dtype)
    out_dim_f = jnp.asarray(max(out_dim, 1), dtype=dtype)
    start = idx * jnp.asarray(block_size, dtype=dtype)
    end = jnp.minimum(start + jnp.asarray(block_size, dtype=dtype), out_dim_f)
    center = 0.5 * (start + end)
    block_fraction = (end - start) / jnp.asarray(block_size, dtype=dtype)
    inv_num_blocks = jnp.full((num_blocks,), 1.0 / float(max(num_blocks, 1)), dtype=dtype)
    return jnp.stack(
        (
            idx / denom,
            start / out_dim_f,
            center / out_dim_f,
            end / out_dim_f,
            block_fraction,
            inv_num_blocks,
        ),
        axis=-1,
    )


class StochasticHyper(eqx.Module):
    trunk: _Trunk
    global_proj: eqx.nn.Linear
    child_mu_head: eqx.nn.Linear
    child_logstd_head: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    block_proj: eqx.nn.Linear
    score_head: eqx.nn.Linear
    std_head: eqx.nn.Linear
    lr_head: eqx.nn.Linear
    basis: jnp.ndarray
    block_size: int = eqx.field(static=True)
    mutation_block_ratio: float = eqx.field(static=True)
    clip_std: tuple[float, float] = eqx.field(static=True)
    clip_update: tuple[float, float] = eqx.field(static=True)
    const_noise_std: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        coeff_dim: int,
        block_size: int,
        mutation_block_ratio: float,
        mutation_rate_head_dim: int,
        clip_std: tuple[float, float],
        clip_update: tuple[float, float],
        const_noise_std: float,
        *,
        key: jax.random.KeyArray,
    ):
        k_trunk, k_global, k_child_mu, k_child_std, k_pos, k_block, k_score, k_std, k_lr, k_basis = (
            jax.random.split(key, 10)
        )
        self.trunk = _Trunk(in_dim, hidden_dim, key=k_trunk)
        self.global_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=k_global)
        self.child_mu_head = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=k_child_mu)
        self.child_logstd_head = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=k_child_std)
        self.pos_proj = eqx.nn.Linear(6, hidden_dim, use_bias=True, key=k_pos)
        self.block_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=k_block)
        self.score_head = eqx.nn.Linear(hidden_dim, 1, use_bias=True, key=k_score)
        self.std_head = eqx.nn.Linear(hidden_dim, coeff_dim, use_bias=False, key=k_std)
        self.lr_head = eqx.nn.Linear(hidden_dim, mutation_rate_head_dim, use_bias=True, key=k_lr)
        basis_init = jax.nn.initializers.orthogonal()
        self.basis = basis_init(k_basis, (coeff_dim, block_size), jnp.float32)
        self.block_size = block_size
        self.mutation_block_ratio = mutation_block_ratio
        self.clip_std = clip_std
        self.clip_update = clip_update
        self.const_noise_std = const_noise_std

    def sample_child_context(self, h_ctx: jnp.ndarray, key: jax.random.KeyArray) -> jnp.ndarray:
        mu = self.child_mu_head(h_ctx)
        log_std = jnp.clip(self.child_logstd_head(h_ctx), -4.0, 1.0)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mu.shape, dtype=mu.dtype)
        return mu + eps * std

    def __call__(
        self,
        h_i: jnp.ndarray,
        child_ctx: jnp.ndarray,
        out_dim: int,
        key: jax.random.KeyArray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Returns (selected_block_ids, dense_updates_per_block, mutation_rate_scalar)."""
        if out_dim <= 0:
            empty_ids = jnp.zeros((0,), dtype=jnp.int32)
            empty_updates = jnp.zeros((0, self.block_size), dtype=h_i.dtype)
            return empty_ids, empty_updates, jnp.asarray(0.0, dtype=h_i.dtype)

        local_x = self.trunk(h_i)
        global_x = jax.nn.tanh(self.global_proj(child_ctx))
        x = jax.nn.relu(local_x + global_x)
        num_blocks = _num_blocks(out_dim, self.block_size)
        pos_features = _block_position_features(num_blocks, self.block_size, out_dim, x.dtype)
        pos_ctx = jax.vmap(self.pos_proj)(pos_features)
        block_ctx = jax.nn.relu(jax.vmap(self.block_proj)(pos_ctx + x))
        scores = jax.vmap(self.score_head)(block_ctx).squeeze(-1)

        num_selected = max(1, ceil(num_blocks * self.mutation_block_ratio))
        num_selected = min(num_selected, num_blocks)
        _, block_ids = jax.lax.top_k(scores, num_selected)

        selected_ctx = block_ctx[block_ids]
        lr_logits = self.lr_head(x)
        lr = jnp.max(jax.nn.sigmoid(lr_logits))

        k_eps, k_noise = jax.random.split(key, 2)
        log_std = jax.vmap(self.std_head)(selected_ctx)
        std = jnp.clip(jnp.exp(log_std), self.clip_std[0], self.clip_std[1])
        coeffs = jax.random.normal(k_eps, std.shape, dtype=std.dtype) * std
        w_blocks = coeffs @ self.basis
        clipped = jnp.clip(w_blocks, self.clip_update[0], self.clip_update[1])
        noise = jax.random.normal(k_noise, w_blocks.shape, dtype=w_blocks.dtype) * self.const_noise_std
        updates = clipped * lr + noise
        return block_ids.astype(jnp.int32), updates, lr


class DeterministicHead(eqx.Module):
    context_proj: eqx.nn.Linear
    pos_proj: eqx.nn.Linear
    block_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    block_size: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dim: int, block_size: int, *, key: jax.random.KeyArray):
        k_context, k_pos, k_block, k_out = jax.random.split(key, 4)
        self.context_proj = eqx.nn.Linear(in_dim, hidden_dim, use_bias=True, key=k_context)
        self.pos_proj = eqx.nn.Linear(6, hidden_dim, use_bias=True, key=k_pos)
        self.block_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=True, key=k_block)
        self.out_proj = eqx.nn.Linear(hidden_dim, block_size, use_bias=True, key=k_out)
        self.block_size = block_size

    def __call__(self, h_i: jnp.ndarray, out_dim: int) -> jnp.ndarray:
        if out_dim <= 0:
            return jnp.zeros((0,), dtype=h_i.dtype)

        num_blocks = _num_blocks(out_dim, self.block_size)
        pos_features = _block_position_features(num_blocks, self.block_size, out_dim, h_i.dtype)
        context = jax.nn.relu(self.context_proj(h_i))
        pos_ctx = jax.vmap(self.pos_proj)(pos_features)
        block_ctx = jax.nn.relu(jax.vmap(self.block_proj)(pos_ctx + context))
        blocks = jax.vmap(self.out_proj)(block_ctx)
        return blocks.reshape(-1)[:out_dim]
