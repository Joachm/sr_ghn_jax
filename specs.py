from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ParamNodeSpec:
    shapes: tuple[tuple[int, ...], ...]
    param_sizes: tuple[int, ...]
    sizes: tuple[int, ...]
    max_size: int
    num_nodes: int
    shard_param_idxs: tuple[int, ...]
    shard_starts: tuple[int, ...]

    def tree_flatten(self):
        children = ()
        aux_data = (
            self.shapes,
            self.param_sizes,
            self.sizes,
            self.max_size,
            self.num_nodes,
            self.shard_param_idxs,
            self.shard_starts,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            shapes,
            param_sizes,
            sizes,
            max_size,
            num_nodes,
            shard_param_idxs,
            shard_starts,
        ) = aux_data
        return cls(
            shapes=shapes,
            param_sizes=param_sizes,
            sizes=sizes,
            max_size=max_size,
            num_nodes=num_nodes,
            shard_param_idxs=shard_param_idxs,
            shard_starts=shard_starts,
        )


def _linear_param_shapes(layer_in: int, layer_out: int) -> Sequence[tuple[int, ...]]:
    return ((layer_out, layer_in), (layer_out,))


def _mlp_param_shapes(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []
    prev = input_dim
    for hidden in hidden_dims:
        shapes.extend(_linear_param_shapes(prev, hidden))
        prev = hidden
    shapes.extend(_linear_param_shapes(prev, output_dim))
    return tuple(shapes)


def _sizes_from_shapes(shapes: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(int(prod(shape)) for shape in shapes)


def _infer_obs_dim(env) -> int:
    if hasattr(env, "observation_size"):
        return int(env.observation_size)
    if hasattr(env, "obs_size"):
        return int(env.obs_size)
    obs_space = getattr(env, "observation_space", None)
    if callable(obs_space):
        try:
            obs_space = obs_space()
        except TypeError:
            pass
    if obs_space is not None and hasattr(obs_space, "shape"):
        return int(prod(obs_space.shape))
    raise ValueError("Unable to infer observation size from environment.")


def _infer_action_dim(env) -> int:
    action_space = getattr(env, "action_space", None)
    if callable(action_space):
        try:
            action_space = action_space()
        except TypeError:
            pass

    if action_space is not None:
        if hasattr(action_space, "nvec"):
            raise ValueError("MultiDiscrete action spaces are not supported.")
        if hasattr(action_space, "n"):
            return int(action_space.n)
        if hasattr(action_space, "shape"):
            return int(prod(action_space.shape))

    if hasattr(env, "action_size"):
        return int(env.action_size)

    raise ValueError("Unable to infer action size from environment.")


def _load_mujoco_playground_env(env_id: str):
    from mujoco_playground import registry as suite

    if ":" in env_id:
        domain, task = env_id.split(":", 1)
        return suite.load(domain, task)
    if "/" in env_id:
        domain, task = env_id.split("/", 1)
        return suite.load(domain, task)
    if hasattr(suite, "load"):
        try:
            return suite.load(env_id)
        except TypeError:
            pass
    if hasattr(suite, "make"):
        return suite.make(env_id)
    raise ValueError("Unsupported mujoco_playground suite API.")


def _make_sharded_spec(shapes: tuple[tuple[int, ...], ...], shard_size: int) -> ParamNodeSpec:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    param_sizes = _sizes_from_shapes(shapes)
    shard_param_idxs: list[int] = []
    shard_starts: list[int] = []
    shard_sizes: list[int] = []
    for param_idx, size in enumerate(param_sizes):
        for start in range(0, size, shard_size):
            end = min(size, start + shard_size)
            shard_param_idxs.append(param_idx)
            shard_starts.append(start)
            shard_sizes.append(end - start)
    return ParamNodeSpec(
        shapes=shapes,
        param_sizes=param_sizes,
        sizes=tuple(shard_sizes),
        max_size=shard_size,
        num_nodes=len(shard_sizes),
        shard_param_idxs=tuple(shard_param_idxs),
        shard_starts=tuple(shard_starts),
    )


def policy_spec_for_task(config, shard_size: int) -> ParamNodeSpec:
    task = config.task_name.lower()
    hidden_dims = tuple(int(x) for x in config.policy_hidden_dims)
    if task == "cartpole_switch":
        shapes = _mlp_param_shapes(4, hidden_dims, 2)
    elif task == "ant_brax":
        from brax import envs

        env = envs.create(config.env_id)
        obs_dim = int(env.observation_size)
        act_dim = int(env.action_size)
        shapes = _mlp_param_shapes(obs_dim, hidden_dims, act_dim)
    elif task == "brax_generic":
        from brax import envs

        if config.brax_backend is None:
            env = envs.create(config.env_id)
        else:
            env = envs.create(config.env_id, backend=config.brax_backend)
        obs_dim = int(env.observation_size)
        act_dim = int(env.action_size)
        shapes = _mlp_param_shapes(obs_dim, hidden_dims, act_dim)
    elif task == "lunarlander_switch":
        raise ValueError("LunarLander policy spec is not defined in the reference.")
    elif task == "gymnax_generic":
        import gymnax

        env, env_params = gymnax.make(config.env_id)
        obs_dim = int(prod(env.observation_space(env_params).shape))
        action_space = env.action_space(env_params)
        if hasattr(action_space, "nvec"):
            raise ValueError("MultiDiscrete action spaces are not supported.")
        if hasattr(action_space, "n"):
            act_dim = int(action_space.n)
        else:
            act_dim = int(prod(action_space.shape))
        shapes = _mlp_param_shapes(obs_dim, hidden_dims, act_dim)
    elif task == "mujoco_playground_generic":
        env = _load_mujoco_playground_env(config.env_id)
        obs_dim = _infer_obs_dim(env)
        act_dim = _infer_action_dim(env)
        shapes = _mlp_param_shapes(obs_dim, hidden_dims, act_dim)
    else:
        raise ValueError(f"Unknown task_name: {config.task_name}")

    return _make_sharded_spec(shapes, shard_size)


def _srghn_filter_spec(srghn_module: eqx.Module):
    filter_spec = jax.tree_util.tree_map(eqx.is_array, srghn_module)
    if getattr(srghn_module, "freeze_stoch_output_head", False):
        filter_spec = eqx.tree_at(lambda m: m.stoch.basis, filter_spec, False)
    return filter_spec


def srghn_self_spec(srghn_module: eqx.Module, shard_size: int) -> ParamNodeSpec:
    filter_spec = _srghn_filter_spec(srghn_module)
    filtered = eqx.filter(srghn_module, filter_spec)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(filtered) if eqx.is_array(leaf)]
    shapes = tuple(tuple(int(d) for d in leaf.shape) for leaf in leaves)
    return _make_sharded_spec(shapes, shard_size)
