from __future__ import annotations

import inspect
from typing import Any, Callable

from configs import (
    BASELINE_FIXED_LR,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FULL,
    BASELINE_NO_SELF_REFERENCE,
    make_config_ant_brax,
    make_config_brax_generic,
    make_config_gymnax_generic,
    make_config_mujoco_playground_generic,
    make_config_nonstationary_brax,
    make_config_nonstationary_gymnax,
)
from experiments._adaptation import (
    GYMNAX_MINATAR_SUITE_ENVIRONMENTS,
    brax_shift_windows,
    gymnax_minatar_suite_shift_windows,
    gymnax_shift_windows,
    gymnax_suite_shift_windows,
)


BASELINE_PRESET_NAMES = (
    BASELINE_FULL,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FIXED_LR,
    BASELINE_NO_SELF_REFERENCE,
)

CONTROL_RUN_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "debug": {
        "pop_size": 2,
        "num_generations": 2,
        "episode_horizon": 32,
        "children_per_parent": 1,
        "episodes_per_eval": 1,
    },
}
CONTROL_RUN_PRESET_NAMES = tuple(CONTROL_RUN_PRESETS)

NONSTATIONARY_GYMNAX_TASK_PRESET_NAMES = (
    "cartpole_flip",
    "cartpole_flip_revert",
    "cartpole_repeated",
    "suite_default",
    "minatar_suite",
)
NONSTATIONARY_BRAX_TASK_PRESET_NAMES = (
    "brax_direction_switch",
    "brax_speed_target_switch",
)

MINATAR_POLICY_PRESET = {
    "policy_architecture": "cnn_mlp",
    "policy_conv_channels": (8,),
    "policy_conv_kernel_sizes": ((3, 3),),
    "policy_conv_strides": ((1, 1),),
    "policy_hidden_dims": (128,),
}


def _builder_defaults(builder: Callable[..., Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, param in inspect.signature(builder).parameters.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if param.default is inspect._empty:
            continue
        defaults[name] = param.default
    return defaults


def _merge_layers(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)
    return merged


def _run_preset_overrides(name: str | None) -> dict[str, Any]:
    preset_name = "default" if name in (None, "") else name
    if preset_name not in CONTROL_RUN_PRESETS:
        raise ValueError(f"Unknown control run preset: {preset_name}")
    return dict(CONTROL_RUN_PRESETS[preset_name])


def _nonstationary_gymnax_task_overrides(name: str | None, env_id: str) -> dict[str, Any]:
    if name in (None, "", "cartpole_flip"):
        return {"shift_windows": gymnax_shift_windows("cartpole_flip")}
    if name in {"cartpole_flip_revert", "cartpole_repeated"}:
        return {"shift_windows": gymnax_shift_windows(name)}
    if name == "suite_default":
        return {"shift_windows": gymnax_suite_shift_windows(env_id)}
    if name == "minatar_suite":
        if env_id not in GYMNAX_MINATAR_SUITE_ENVIRONMENTS:
            raise ValueError(f"Unsupported MinAtar env: {env_id}")
        return {
            "shift_windows": gymnax_minatar_suite_shift_windows(env_id),
            **MINATAR_POLICY_PRESET,
        }
    raise ValueError(f"Unknown nonstationary Gymnax task preset: {name}")


def _nonstationary_brax_task_overrides(name: str | None, *, target_speed: float | None = None) -> dict[str, Any]:
    preset_name = "brax_direction_switch" if name in (None, "") else name
    if preset_name == "brax_direction_switch":
        return {"shift_windows": brax_shift_windows("brax_direction_switch")}
    if preset_name == "brax_speed_target_switch":
        speed = 1.0 if target_speed is None else target_speed
        return {
            "shift_windows": brax_shift_windows("brax_speed_target_switch", target_speed=speed),
        }
    raise ValueError(f"Unknown nonstationary Brax task preset: {preset_name}")


def resolve_gymnax_generic_config(
    env_id: str,
    *,
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_gymnax_generic),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_gymnax_generic(env_id, **kwargs)


def resolve_brax_generic_config(
    env_id: str,
    *,
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_brax_generic),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_brax_generic(env_id, **kwargs)


def resolve_ant_brax_config(
    *,
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_ant_brax),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_ant_brax(**kwargs)


def resolve_mujoco_playground_config(
    env_id: str,
    *,
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_mujoco_playground_generic),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_mujoco_playground_generic(env_id, **kwargs)


def resolve_nonstationary_gymnax_config(
    env_id: str,
    *,
    task_preset: str | None = None,
    run_preset: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_nonstationary_gymnax),
        _nonstationary_gymnax_task_overrides(task_preset, env_id),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_nonstationary_gymnax(env_id, **kwargs)


def resolve_nonstationary_brax_config(
    env_id: str,
    *,
    task_preset: str | None = None,
    run_preset: str | None = None,
    target_speed: float | None = None,
    overrides: dict[str, Any] | None = None,
):
    kwargs = _merge_layers(
        _builder_defaults(make_config_nonstationary_brax),
        _nonstationary_brax_task_overrides(task_preset, target_speed=target_speed),
        _run_preset_overrides(run_preset),
        overrides or {},
    )
    return make_config_nonstationary_brax(env_id, **kwargs)
