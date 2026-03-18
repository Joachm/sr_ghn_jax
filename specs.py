from __future__ import annotations

from dataclasses import dataclass
from math import log1p, prod, tanh
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
    node_features: tuple[tuple[float, ...], ...]
    module_names: tuple[str, ...]
    group_ids: tuple[int, ...]
    parent_ids: tuple[int, ...]
    context_index: int | None

    def tree_flatten(self):
        children = ()
        aux_data = (
            self.shapes,
            self.sizes,
            self.max_size,
            self.num_nodes,
            self.node_features,
            self.module_names,
            self.group_ids,
            self.parent_ids,
            self.context_index,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        shapes, sizes, max_size, num_nodes, node_features, module_names, group_ids, parent_ids, context_index = aux_data
        return cls(
            shapes=shapes,
            sizes=sizes,
            max_size=max_size,
            num_nodes=num_nodes,
            node_features=node_features,
            module_names=module_names,
            group_ids=group_ids,
            parent_ids=parent_ids,
            context_index=context_index,
        )


def _shape_rows_cols(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return int(shape[0]), 1
    return int(shape[0]), int(prod(shape[1:]))


def _structural_feature_row(
    shape: tuple[int, ...],
    *,
    is_weight: bool,
    is_bias: bool,
    is_embedding: bool,
    is_self_side: bool,
    is_policy_side: bool,
    is_mutator_side: bool,
    depth_norm: float,
    group_norm: float,
) -> tuple[float, ...]:
    rows, cols = _shape_rows_cols(shape)
    size = int(prod(shape)) if shape else 1
    rank = len(shape)
    is_matrix = float(rank >= 2)
    is_vector = float(rank == 1)
    is_other = float(not (is_matrix or is_vector or is_embedding))
    aspect = tanh(log1p(rows) - log1p(cols))
    return (
        float(is_weight),
        float(is_bias),
        float(is_embedding),
        is_matrix,
        is_vector,
        is_other,
        float(is_self_side),
        float(is_policy_side),
        float(is_mutator_side),
        float(depth_norm),
        float(group_norm),
        float(log1p(rows) / 6.0),
        float(log1p(cols) / 6.0),
        float(log1p(size) / 10.0),
        float(rank / 4.0),
        float(aspect),
    )


def _policy_metadata(shapes: tuple[tuple[int, ...], ...]) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...], tuple[int, ...]]:
    num_nodes = len(shapes)
    if num_nodes == 0:
        return (), (), ()

    num_layers = max((num_nodes + 1) // 2, 1)
    features = []
    group_ids = []
    parent_ids = []
    for idx, shape in enumerate(shapes):
        layer_idx = idx // 2
        is_weight = idx % 2 == 0
        depth_norm = layer_idx / max(num_layers - 1, 1)
        group_norm = layer_idx / max(num_layers, 1)
        features.append(
            _structural_feature_row(
                shape,
                is_weight=is_weight,
                is_bias=not is_weight,
                is_embedding=False,
                is_self_side=False,
                is_policy_side=True,
                is_mutator_side=False,
                depth_norm=depth_norm,
                group_norm=group_norm,
            )
        )
        group_ids.append(layer_idx)
        parent_ids.append(layer_idx)
    return tuple(features), tuple(group_ids), tuple(parent_ids)


def _key_token_name(key) -> str:
    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "idx"):
        return str(key.idx)
    return str(key)


def _self_metadata(
    path_leaves: Sequence[tuple[tuple[object, ...], jnp.ndarray]],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    int | None,
]:
    entries: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    for path, leaf in path_leaves:
        if leaf is None or not eqx.is_array(leaf):
            continue
        tokens = tuple(_key_token_name(key) for key in path)
        shape = tuple(int(dim) for dim in leaf.shape)
        entries.append((tokens, shape))

    if not entries:
        return (), (), (), (), None

    top_level_ids: dict[str, int] = {}
    parent_ids_map: dict[str, int] = {}
    for tokens, _ in entries:
        top_name = tokens[0]
        parent_name = ".".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]
        top_level_ids.setdefault(top_name, len(top_level_ids))
        parent_ids_map.setdefault(parent_name, len(parent_ids_map))

    num_top_groups = max(len(top_level_ids), 1)
    features = []
    module_names = []
    group_ids = []
    parent_ids = []
    shapes = []
    context_index = None
    for tokens, shape in entries:
        top_name = tokens[0]
        last_name = tokens[-1]
        top_id = top_level_ids[top_name]
        parent_name = ".".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]
        parent_id = parent_ids_map[parent_name]
        depth_norm = (len(tokens) - 1) / 5.0
        group_norm = top_id / max(num_top_groups - 1, 1)
        is_weight = "weight" in last_name
        is_bias = "bias" in last_name
        is_embedding = top_name in {"self_node_emb", "policy_node_emb"} or "emb" in last_name
        is_self_side = top_name.startswith("self_") or top_name == "encoder_self"
        is_policy_side = top_name.startswith("policy_") or top_name == "encoder_policy"
        is_mutator_side = top_name in {"stoch", "det"}
        features.append(
            _structural_feature_row(
                shape,
                is_weight=is_weight,
                is_bias=is_bias,
                is_embedding=is_embedding,
                is_self_side=is_self_side,
                is_policy_side=is_policy_side,
                is_mutator_side=is_mutator_side,
                depth_norm=depth_norm,
                group_norm=group_norm,
            )
        )
        module_names.append(top_name)
        group_ids.append(top_id)
        parent_ids.append(parent_id)
        shapes.append(shape)
        if top_name == "self_context_emb":
            context_index = len(shapes) - 1

    return tuple(features), tuple(module_names), tuple(group_ids), tuple(parent_ids), tuple(shapes), context_index


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
    elif task == "brax_generic":
        from brax import envs

        if config.brax_backend is None:
            env = envs.create(config.env_id)
        else:
            env = envs.create(config.env_id, backend=config.brax_backend)
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
    elif task == "mujoco_playground_generic":
        env = _load_mujoco_playground_env(config.env_id)
        obs_dim = _infer_obs_dim(env)
        act_dim = _infer_action_dim(env)
        shapes = _mlp_param_shapes(obs_dim, (64, 64, 64), act_dim)
    else:
        raise ValueError(f"Unknown task_name: {config.task_name}")

    sizes = _sizes_from_shapes(shapes)
    max_size = max(sizes) if sizes else 0
    node_features, group_ids, parent_ids = _policy_metadata(shapes)
    return ParamNodeSpec(
        shapes=shapes,
        sizes=sizes,
        max_size=max_size,
        num_nodes=len(shapes),
        node_features=node_features,
        module_names=("policy",) * len(shapes),
        group_ids=group_ids,
        parent_ids=parent_ids,
        context_index=None,
    )


def _srghn_filter_spec(srghn_module: eqx.Module):
    filter_spec = jax.tree_util.tree_map(eqx.is_array, srghn_module)
    filter_spec = eqx.tree_at(
        lambda m: m.self_graph,
        filter_spec,
        jax.tree_util.tree_map(lambda _: False, srghn_module.self_graph),
    )
    filter_spec = eqx.tree_at(
        lambda m: m.policy_graph,
        filter_spec,
        jax.tree_util.tree_map(lambda _: False, srghn_module.policy_graph),
    )
    return filter_spec


def srghn_self_spec(srghn_module: eqx.Module) -> ParamNodeSpec:
    filter_spec = _srghn_filter_spec(srghn_module)
    filtered = eqx.filter(srghn_module, filter_spec)
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(filtered)
    node_features, module_names, group_ids, parent_ids, shapes, context_index = _self_metadata(path_leaves)
    sizes = _sizes_from_shapes(shapes)
    max_size = max(sizes) if sizes else 0
    return ParamNodeSpec(
        shapes=shapes,
        sizes=sizes,
        max_size=max_size,
        num_nodes=len(shapes),
        node_features=node_features,
        module_names=module_names,
        group_ids=group_ids,
        parent_ids=parent_ids,
        context_index=context_index,
    )
