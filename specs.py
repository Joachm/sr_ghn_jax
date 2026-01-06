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
    sizes: tuple[int, ...]
    max_size: int
    num_nodes: int

    def tree_flatten(self):
        children = ()
        aux_data = (self.shapes, self.sizes, self.max_size, self.num_nodes)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        shapes, sizes, max_size, num_nodes = aux_data
        return cls(shapes=shapes, sizes=sizes, max_size=max_size, num_nodes=num_nodes)


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


def policy_spec_for_task(config) -> ParamNodeSpec:
    task = config.task_name.lower()
    if task == "cartpole_switch":
        shapes = _mlp_param_shapes(4, (32,), 2)
    elif task == "ant_brax":
        from brax import envs

        env = envs.create(config.env_id)
        obs_dim = int(env.observation_size)
        act_dim = int(env.action_size)
        shapes = _mlp_param_shapes(obs_dim, (32, 32, 32), act_dim)
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
        shapes = _mlp_param_shapes(obs_dim, (32,), act_dim)
    else:
        raise ValueError(f"Unknown task_name: {config.task_name}")

    sizes = _sizes_from_shapes(shapes)
    max_size = max(sizes) if sizes else 0
    return ParamNodeSpec(shapes=shapes, sizes=sizes, max_size=max_size, num_nodes=len(shapes))


def _srghn_filter_spec(srghn_module: eqx.Module):
    filter_spec = jax.tree_util.tree_map(eqx.is_array, srghn_module)
    filter_spec = eqx.tree_at(lambda m: m.stoch.basis, filter_spec, False)
    return filter_spec


def srghn_self_spec(srghn_module: eqx.Module) -> ParamNodeSpec:
    filter_spec = _srghn_filter_spec(srghn_module)
    filtered = eqx.filter(srghn_module, filter_spec)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(filtered) if eqx.is_array(leaf)]
    shapes = tuple(tuple(int(d) for d in leaf.shape) for leaf in leaves)
    sizes = _sizes_from_shapes(shapes)
    max_size = max(sizes) if sizes else 0
    return ParamNodeSpec(shapes=shapes, sizes=sizes, max_size=max_size, num_nodes=len(shapes))
