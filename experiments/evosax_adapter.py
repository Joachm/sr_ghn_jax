from __future__ import annotations

from dataclasses import is_dataclass, replace
import importlib
import inspect
import math
import pkgutil
import re

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


def _scalar_from_value(value, default: float) -> float:
    try:
        scalar = float(jax.device_get(value))
    except Exception:
        return default
    if not math.isfinite(scalar):
        return default
    return scalar


def _extract_sigma_like(params, default: float = 1.0) -> float:
    if params is None:
        return default
    for field_name in ("sigma_init", "init_std", "std_init", "sigma"):
        if hasattr(params, field_name):
            return _scalar_from_value(getattr(params, field_name), default)
    return default


def _to_python(value):
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {field: _to_python(getattr(value, field)) for field in value._fields}
    if is_dataclass(value):
        return {field: _to_python(getattr(value, field)) for field in value.__dataclass_fields__}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if hasattr(value, "shape"):
        arr = jax.device_get(value)
        if getattr(arr, "ndim", 0) == 0:
            return _scalar_from_value(arr, 0.0)
        return arr.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _snake_case_name(name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return snake.replace("__", "_")


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
    for module_info in pkgutil.walk_packages(algorithms_mod.__path__, algorithms_mod.__name__ + "."):
        try:
            modules.append(importlib.import_module(module_info.name))
        except Exception:
            continue

    for module in modules:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not inspect.isclass(attr):
                continue
            if getattr(attr, "__module__", None) != module.__name__:
                continue
            has_init = hasattr(attr, "init") or hasattr(attr, "initialize")
            if not (has_init and hasattr(attr, "ask") and hasattr(attr, "tell")):
                continue
            registry[attr_name] = attr
            registry[attr_name.lower()] = attr
            registry[_normalize_name(attr_name)] = attr
            registry[_snake_case_name(attr_name)] = attr
            module_leaf = module.__name__.rsplit(".", 1)[-1]
            registry[module_leaf] = attr
            registry[_normalize_name(module_leaf)] = attr
            if module_leaf == "discovered_es":
                registry["des"] = attr
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
        self.solution = jnp.asarray(solution, dtype=jnp.float32)
        self.pop_size = int(config.pop_size)
        self.algo_name = config.evosax_algo
        self.strategy = _instantiate_strategy(strategy_cls, pop_size=self.pop_size, solution=solution)
        self.params = _override_strategy_params(getattr(self.strategy, "default_params", None), config.evosax_sigma_init)
        self._init_param_names = self._method_param_names("init", "initialize")
        self._ask_param_names = self._method_param_names("ask")
        self._tell_param_names = self._method_param_names("tell")

    def _method_param_names(self, *method_names: str) -> tuple[str, ...]:
        for method_name in method_names:
            if hasattr(self.strategy, method_name):
                return tuple(inspect.signature(getattr(self.strategy, method_name)).parameters.keys())
        return ()

    @property
    def init_signature(self) -> tuple[str, ...]:
        return self._init_param_names

    @property
    def requires_population_init(self) -> bool:
        return self._init_param_names == ("key", "population", "fitness", "params")

    def sample_initial_population(self, key: jax.random.KeyArray) -> jnp.ndarray:
        sigma = _extract_sigma_like(self.params, default=1.0)
        noise = jax.random.normal(key, (self.pop_size, self.solution.shape[0]), dtype=jnp.float32)
        return self.solution[None, :] + jnp.asarray(sigma, dtype=jnp.float32) * noise

    def effective_params_dict(self) -> dict:
        return _to_python(self.params)

    def init(
        self,
        key: jax.random.KeyArray,
        population: jnp.ndarray | None = None,
        raw_fitness: jnp.ndarray | None = None,
    ):
        if hasattr(self.strategy, "init"):
            param_names = self._init_param_names
            if param_names == ("key", "mean", "params"):
                return self.strategy.init(key, self.solution, self.params)
            if param_names == ("key", "params"):
                return self.strategy.init(key, self.params)
            if param_names == ("key", "population", "fitness", "params"):
                if population is None or raw_fitness is None:
                    raise ValueError(
                        f"Strategy '{self.algo_name}' requires initial population and fitness before init."
                    )
                fitness = fitness_for_evosax(raw_fitness)
                return self.strategy.init(key, population, fitness, self.params)
            raise TypeError(f"Unsupported evosax init signature: {param_names}")
        param_names = self._init_param_names
        if param_names == ("key", "params"):
            return self.strategy.initialize(key, self.params)
        raise TypeError(f"Unsupported evosax initialize signature: {param_names}")

    def ask(self, key: jax.random.KeyArray, state):
        param_names = self._ask_param_names
        if len(param_names) == 3:
            population, next_state = self.strategy.ask(key, state, self.params)
        elif len(param_names) == 2:
            population, next_state = self.strategy.ask(key, state)
        else:
            raise TypeError(f"Unsupported evosax ask signature: {param_names}")
        return jnp.asarray(population, dtype=jnp.float32), next_state

    def tell(self, key: jax.random.KeyArray, population: jnp.ndarray, raw_fitness: jnp.ndarray, state):
        fitness = fitness_for_evosax(raw_fitness)
        param_names = self._tell_param_names
        if len(param_names) == 5:
            result = self.strategy.tell(key, population, fitness, state, self.params)
        elif len(param_names) == 4:
            result = self.strategy.tell(population, fitness, state, self.params)
        elif len(param_names) == 3:
            result = self.strategy.tell(population, fitness, state)
        else:
            raise TypeError(f"Unsupported evosax tell signature: {param_names}")
        if isinstance(result, tuple):
            return result[0]
        return result
