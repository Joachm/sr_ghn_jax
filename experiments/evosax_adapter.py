from __future__ import annotations

from dataclasses import is_dataclass, replace
import importlib
import inspect

import jax
import jax.numpy as jnp


def fitness_for_evosax(raw_fitness: jnp.ndarray) -> jnp.ndarray:
    return -jnp.asarray(raw_fitness, dtype=jnp.float32)


def _set_attr_like(obj, name: str, value):
    if obj is None or not hasattr(obj, name):
        return obj, False
    if hasattr(obj, "replace") and callable(obj.replace):
        return obj.replace(**{name: value}), True
    if is_dataclass(obj):
        return replace(obj, **{name: value}), True
    if hasattr(obj, "_replace") and callable(obj._replace):
        return obj._replace(**{name: value}), True
    try:
        setattr(obj, name, value)
        return obj, True
    except Exception:
        return obj, False


def _override_strategy_params(params, sigma_init: float | None):
    if sigma_init is None or params is None:
        return params
    for field_name in ("sigma_init", "init_std", "std_init", "sigma"):
        updated, changed = _set_attr_like(params, field_name, sigma_init)
        if changed:
            return updated
    return params


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _instantiate_strategy(strategy_cls, *, pop_size: int, solution: jnp.ndarray):
    constructor_attempts = (
        {"population_size": pop_size, "solution": solution},
        {"popsize": pop_size, "solution": solution},
        {"population_size": pop_size, "num_dims": int(solution.shape[0])},
        {"popsize": pop_size, "num_dims": int(solution.shape[0])},
    )
    for kwargs in constructor_attempts:
        try:
            return strategy_cls(**kwargs)
        except TypeError:
            continue
    raise TypeError(f"Unsupported constructor for evosax strategy {strategy_cls.__name__}.")


def _strategy_registry():
    try:
        evosax_mod = importlib.import_module("evosax")
        algorithms_mod = importlib.import_module("evosax.algorithms")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "evosax is required for optimizer_family='evosax'. Install it from requirements.txt."
        ) from exc

    registry = {}
    modules = [evosax_mod, algorithms_mod]
    for attr_name in dir(algorithms_mod):
        attr = getattr(algorithms_mod, attr_name)
        if inspect.ismodule(attr) and getattr(attr, "__name__", "").startswith("evosax.algorithms"):
            modules.append(attr)

    for module in modules:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not inspect.isclass(attr):
                continue
            has_init = hasattr(attr, "init") or hasattr(attr, "initialize")
            if not (has_init and hasattr(attr, "ask") and hasattr(attr, "tell")):
                continue
            registry[attr_name] = attr
            registry[attr_name.lower()] = attr
            registry[_normalize_name(attr_name)] = attr
    return registry


def available_evosax_algorithms() -> list[str]:
    registry = _strategy_registry()
    names = sorted({name for name in registry.keys() if name == name.lower() and "_" in name})
    return names


def resolve_evosax_strategy(name: str):
    registry = _strategy_registry()
    strategy_cls = registry.get(name) or registry.get(name.lower()) or registry.get(_normalize_name(name))
    if strategy_cls is None:
        known = available_evosax_algorithms()
        raise ValueError(f"Unsupported evosax algorithm '{name}'. Available strategies include: {', '.join(known[:20])}.")
    return strategy_cls


class EvosaxStrategyAdapter:
    def __init__(self, config, *, solution: jnp.ndarray):
        if not config.evosax_algo:
            raise ValueError("config.evosax_algo must be set when optimizer_family='evosax'.")
        strategy_cls = resolve_evosax_strategy(config.evosax_algo)
        self.strategy = _instantiate_strategy(strategy_cls, pop_size=config.pop_size, solution=solution)
        self.params = _override_strategy_params(getattr(self.strategy, "default_params", None), config.evosax_sigma_init)

    def init(self, key: jax.random.KeyArray):
        if hasattr(self.strategy, "init"):
            return self.strategy.init(key, self.params)
        return self.strategy.initialize(key, self.params)

    def ask(self, key: jax.random.KeyArray, state):
        population, next_state = self.strategy.ask(key, state, self.params)
        return jnp.asarray(population, dtype=jnp.float32), next_state

    def tell(self, population: jnp.ndarray, raw_fitness: jnp.ndarray, state):
        return self.strategy.tell(population, fitness_for_evosax(raw_fitness), state, self.params)
