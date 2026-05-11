from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from configs import (
    BASELINE_FIXED_LR,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FULL,
    BASELINE_NO_SELF_REFERENCE,
    baseline_overrides,
    make_config_brax_generic,
)
from envs import make_env
from evolution import init_population
from experiments._common import build_graphs_and_specs
from metrics import compute_experiment_metrics
from obs_norm import normalize_obs
from policy import apply_policy
from srghn import SRGHN, make_policy, mutate_with_metadata, mutation_metadata
from experiment_configs import MetaBraxConfig, build_meta_brax_config, print_resolved_config, resolved_config_payload


CARDINAL_HEADINGS = jnp.asarray(
    (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    ),
    dtype=jnp.float32,
)

BASELINE_NAMES = (
    BASELINE_FULL,
    BASELINE_FROZEN_MUTATION,
    BASELINE_FIXED_LR,
    BASELINE_NO_SELF_REFERENCE,
)

VELOCITY_KEY_PAIRS = (
    ("x_velocity", "y_velocity"),
    ("velocity_x", "velocity_y"),
    ("x_vel", "y_vel"),
)

POSITION_KEY_PAIRS = (
    ("x_position", "y_position"),
    ("position_x", "position_y"),
)


@jax.tree_util.register_pytree_node_class
@dataclass
class BraxHeadingTaskBatch:
    headings: jnp.ndarray

    def tree_flatten(self):
        return (self.headings,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (headings,) = children
        return cls(headings=headings)


@jax.tree_util.register_pytree_node_class
@dataclass
class SRGHNMetaState:
    pop: SRGHN
    key: jax.Array
    pop_fitness: jnp.ndarray
    best_fitness: jnp.ndarray
    best_indiv: SRGHN

    def tree_flatten(self):
        return (self.pop, self.key, self.pop_fitness, self.best_fitness, self.best_indiv), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        pop, key, pop_fitness, best_fitness, best_indiv = children
        return cls(pop=pop, key=key, pop_fitness=pop_fitness, best_fitness=best_fitness, best_indiv=best_indiv)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    baseline_name: str
    mutation_exclude_modules: tuple[str, ...] = ()
    fixed_mutation_lr: float | None = None


def _safe_wandb_import():
    try:
        import wandb
    except Exception:
        return None
    return wandb


def wandb_config_payload(
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    *,
    policy_spec=None,
    self_spec=None,
) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["condition"] = asdict(cond)
    if policy_spec is not None:
        payload["policy_spec_shapes"] = [tuple(shape) for shape in policy_spec.shapes]
        payload["policy_num_nodes"] = int(policy_spec.num_nodes)
    if self_spec is not None:
        payload["self_spec_shapes"] = [tuple(shape) for shape in self_spec.shapes]
        payload["self_num_nodes"] = int(self_spec.num_nodes)
    return payload


def _wandb_run_name(cfg: MetaBraxConfig, cond: ConditionSpec) -> str:
    prefix = cfg.wandb_name or "meta_brax_heading"
    return f"{prefix}-{cond.name}"


def _wandb_init_run(
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
    *,
    policy_spec=None,
    self_spec=None,
):
    if cfg.wandb_project is None:
        return None
    wandb = _safe_wandb_import()
    if wandb is None:
        return None
    try:
        if wandb.run is None:
            return wandb.init(
                project=cfg.wandb_project,
                group=cfg.wandb_group,
                name=_wandb_run_name(cfg, cond),
                config=wandb_config_payload(cfg, cond, policy_spec=policy_spec, self_spec=self_spec),
            )
    except Exception:
        return None
    return wandb.run


def _wandb_log_metrics(metrics: dict[str, Any], step: int, *, prefix: str = "train") -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    payload = {f"{prefix}/{key}": float(value) for key, value in metrics.items()}
    payload["gen"] = int(step)
    try:
        wandb.log(payload, step=int(step))
    except Exception:
        pass


def _wandb_log_summary(cfg: MetaBraxConfig, cond: ConditionSpec, payload: dict[str, Any]) -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return

    history = payload["train_history"]
    curve = payload["adaptation_curve_query_return"]
    final_population_fitness = jnp.asarray(payload["final_population_fitness"])
    summary_payload = {
        "final/train_fitness_best": float(history["fitness_best"][-1]),
        "final/train_fitness_mean": float(history["fitness_mean"][-1]),
        "final/train_query_return_best": float(history["query_return_best"][-1]),
        "final/train_query_return_mean": float(history["query_return_mean"][-1]),
        "final/train_diversity": float(history["diversity"][-1]),
        "final/final_population_fitness_best": float(jnp.max(final_population_fitness)),
        "final/final_population_fitness_mean": float(jnp.mean(final_population_fitness)),
        "final/heldout_curve_pre_return_mean": float(curve["mean"][0]),
        "final/heldout_curve_post_return_mean": float(curve["mean"][-1]),
        "final/heldout_curve_stderr_post_return": float(curve["stderr"][-1]),
        "final/heldout_curve_improvement_mean": float(curve["mean"][-1] - curve["mean"][0]),
        "final/heldout_curve_length": int(len(curve["mean"])),
    }
    try:
        wandb.log(summary_payload, step=int(cfg.outer_generations))
    except Exception:
        pass

    if not cfg.wandb_log_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["query_return_best"], label="best")
    axes[0].plot(history["query_return_mean"], label="mean")
    axes[0].set_title("Outer meta-train")
    axes[0].set_xlabel("Outer generation")
    axes[0].set_ylabel("Query return")
    axes[0].legend()

    xs = list(range(len(curve["mean"])))
    axes[1].plot(xs, curve["mean"], label="mean")
    axes[1].fill_between(xs, curve["mean"] - curve["stderr"], curve["mean"] + curve["stderr"], alpha=0.2)
    axes[1].set_title("Held-out adaptation")
    axes[1].set_xlabel("Inner generation")
    axes[1].set_ylabel("Query return")
    axes[1].legend()
    fig.suptitle(cond.name)
    fig.tight_layout()

    try:
        wandb.log({"plots/summary": wandb.Image(fig)}, step=int(cfg.outer_generations))
    except Exception:
        pass
    finally:
        plt.close(fig)


def _wandb_finish_run() -> None:
    wandb = _safe_wandb_import()
    if wandb is None or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def _summary_from_curves(curves: jnp.ndarray) -> dict[str, Any]:
    curves_host = jax.device_get(curves)
    return {
        "all": curves_host,
        "mean": curves_host.mean(axis=0),
        "std": curves_host.std(axis=0),
        "stderr": curves_host.std(axis=0) / math.sqrt(max(curves_host.shape[0], 1)),
    }


def _history_to_host(history: dict[str, jnp.ndarray]) -> dict[str, Any]:
    return {key: jax.device_get(value) for key, value in history.items()}


def parse_condition_spec(text: str, *, fixed_mutation_lr: float = 0.02) -> ConditionSpec:
    if text not in BASELINE_NAMES:
        raise ValueError(f"Unknown meta-Brax condition '{text}'. Expected one of {BASELINE_NAMES}.")
    overrides = baseline_overrides(text, fixed_mutation_lr=fixed_mutation_lr)
    return ConditionSpec(
        name=text,
        baseline_name=text,
        mutation_exclude_modules=tuple(overrides["mutation_exclude_modules"]),
        fixed_mutation_lr=overrides["fixed_mutation_lr"],
    )


def make_runtime_config(cfg: MetaBraxConfig):
    return make_config_brax_generic(
        cfg.env_id,
        seed=cfg.seed,
        pop_size=cfg.outer_pop_size,
        num_generations=cfg.outer_generations,
        episode_horizon=cfg.episode_horizon,
        children_per_parent=cfg.outer_children_per_parent,
        episodes_per_eval=1,
        brax_backend=cfg.brax_backend,
        parameter_block_size=cfg.parameter_block_size,
        mutation_block_ratio=cfg.mutation_block_ratio,
        policy_hidden_dims=cfg.policy_hidden_dims,
    )


def sample_heading_tasks(key: jax.Array, batch_size: int) -> BraxHeadingTaskBatch:
    task_ids = jax.random.randint(key, (batch_size,), 0, CARDINAL_HEADINGS.shape[0], dtype=jnp.int32)
    headings = CARDINAL_HEADINGS[task_ids]
    return BraxHeadingTaskBatch(headings=headings)


def split_support_query_episode_keys(
    key: jax.Array,
    support_episodes: int,
    query_episodes: int,
) -> tuple[jax.Array, jax.Array]:
    total_episodes = support_episodes + query_episodes
    if total_episodes <= 0:
        raise ValueError("support_episodes + query_episodes must be positive.")
    keys = jax.random.split(key, total_episodes)
    return keys[:support_episodes], keys[support_episodes:]


def _metric_mapping(state) -> Any | None:
    metrics = getattr(state, "metrics", None)
    if metrics is None:
        return None
    if hasattr(metrics, "keys"):
        return metrics
    return None


def _extract_planar_velocity_from_metrics(metrics) -> jnp.ndarray | None:
    if metrics is None:
        return None
    for x_key, y_key in VELOCITY_KEY_PAIRS:
        if x_key in metrics and y_key in metrics:
            return jnp.asarray((metrics[x_key], metrics[y_key]), dtype=jnp.float32)
    return None


def _extract_planar_position_from_metrics(metrics) -> jnp.ndarray | None:
    if metrics is None:
        return None
    for x_key, y_key in POSITION_KEY_PAIRS:
        if x_key in metrics and y_key in metrics:
            return jnp.asarray((metrics[x_key], metrics[y_key]), dtype=jnp.float32)
    return None


def _extract_planar_position_from_pipeline_state(pipeline_state) -> jnp.ndarray | None:
    if pipeline_state is None:
        return None
    x = getattr(pipeline_state, "x", None)
    pos = getattr(x, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]
    qp = getattr(pipeline_state, "qp", None)
    pos = getattr(qp, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]
    return None


def _extract_planar_position_from_state(state) -> jnp.ndarray | None:
    pipeline_state = getattr(state, "pipeline_state", None)
    pipeline_position = _extract_planar_position_from_pipeline_state(pipeline_state)
    if pipeline_position is not None:
        return pipeline_position

    qp = getattr(state, "qp", None)
    pos = getattr(qp, "pos", None)
    if pos is not None:
        pos = jnp.asarray(pos, dtype=jnp.float32)
        if pos.ndim >= 2 and pos.shape[-1] >= 2:
            return pos[0, :2]

    metrics = _metric_mapping(state)
    return _extract_planar_position_from_metrics(metrics)


def _state_attr_names(obj) -> tuple[str, ...]:
    mapping = getattr(obj, "__dict__", None)
    if mapping is None:
        return ()
    return tuple(sorted(mapping.keys()))


def _metric_key_names(metrics) -> tuple[str, ...]:
    if metrics is None or not hasattr(metrics, "keys"):
        return ()
    return tuple(sorted(str(key) for key in metrics.keys()))


def _infer_env_dt(env) -> float:
    if hasattr(env, "dt"):
        return float(env.dt)
    sys = getattr(env, "sys", None)
    config = getattr(sys, "config", None)
    if config is not None and hasattr(config, "dt"):
        return float(config.dt)
    raise ValueError("Unable to infer Brax dt from environment.")


def _extract_planar_velocity(next_state, prev_state, dt: float) -> jnp.ndarray:
    metrics = _metric_mapping(next_state)
    metric_velocity = _extract_planar_velocity_from_metrics(metrics)
    if metric_velocity is not None:
        return metric_velocity

    next_position = _extract_planar_position_from_state(next_state)
    prev_position = _extract_planar_position_from_state(prev_state)
    if next_position is not None and prev_position is not None:
        return (next_position - prev_position) / jnp.asarray(dt, dtype=jnp.float32)

    raise ValueError(
        "Unable to extract planar velocity. Expected either one of the metric key pairs "
        f"{VELOCITY_KEY_PAIRS} or root/torso positions from state.pipeline_state.x.pos, "
        "state.pipeline_state.qp.pos, state.qp.pos, or position metrics "
        f"{POSITION_KEY_PAIRS}. Available metric keys: {_metric_key_names(metrics)}. "
        f"Next-state attrs: {_state_attr_names(next_state)}. Prev-state attrs: {_state_attr_names(prev_state)}."
    )


def _projected_heading_reward(prev_state, next_state, heading: jnp.ndarray, dt: float) -> jnp.ndarray:
    planar_velocity = _extract_planar_velocity(next_state, prev_state, dt)
    return jnp.dot(planar_velocity, jnp.asarray(heading, dtype=jnp.float32))


def evaluate_policy_params_on_heading(
    policy_params,
    episode_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> jnp.ndarray:
    runtime_config = make_runtime_config(cfg)
    env, _, _, _, is_discrete, action_shape, action_low, action_high = make_env(runtime_config)
    if is_discrete:
        raise ValueError("meta_brax_heading only supports continuous Brax environments.")
    dt = _infer_env_dt(env)

    def rollout_once(episode_key: jax.Array) -> jnp.ndarray:
        state = env.reset(episode_key)
        obs = state.obs

        def step_fn(carry, _):
            obs_t, state_t, done_t = carry
            obs_in = normalize_obs(obs_t, None, clip=cfg.obs_norm_clip, eps=cfg.obs_norm_eps)
            action = apply_policy(policy_params, obs_in, runtime_config, is_discrete=False)
            action = jnp.asarray(action, dtype=jnp.float32).reshape(action_shape)
            if action_low is not None and action_high is not None:
                action = action_low + (action + 1.0) * 0.5 * (action_high - action_low)

            def do_step(_):
                next_state = env.step(state_t, action)
                reward = _projected_heading_reward(state_t, next_state, heading, dt)
                done = jnp.asarray(next_state.done, dtype=jnp.bool_)
                return next_state.obs, next_state, reward.astype(jnp.float32), done

            def skip_step(_):
                return obs_t, state_t, jnp.zeros((), dtype=jnp.float32), done_t

            next_obs, next_state, reward, done = jax.lax.cond(done_t, skip_step, do_step, operand=None)
            done = jnp.logical_or(done_t, done)
            return (next_obs, next_state, done), reward

        (_, _, _), rewards = jax.lax.scan(
            step_fn,
            (obs, state, jnp.asarray(False)),
            None,
            length=cfg.episode_horizon,
        )
        return jnp.sum(rewards)

    returns = jax.vmap(rollout_once)(episode_keys)
    return jnp.mean(returns)


def evaluate_individual_on_heading(
    indiv: SRGHN,
    episode_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
) -> jnp.ndarray:
    return evaluate_policy_params_on_heading(make_policy(indiv), episode_keys, heading, cfg)


def _stack_parent_with_children(parent: SRGHN, children: SRGHN) -> SRGHN:
    parent_arr, _ = eqx.partition(parent, eqx.is_array)
    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda p, c: jnp.concatenate([p[None, ...], c], axis=0), parent_arr, child_arr)
    return eqx.combine(all_arr, child_static)


def _select_srghn(pop: SRGHN, idx: int | jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    indiv_arr = jax.tree_util.tree_map(lambda value: value[idx], pop_arr)
    return eqx.combine(indiv_arr, pop_static)


def _repeat_srghn(indiv: SRGHN, repeats: int) -> SRGHN:
    arr, static = eqx.partition(indiv, eqx.is_array)
    repeated = jax.tree_util.tree_map(lambda value: jnp.broadcast_to(value, (repeats,) + value.shape), arr)
    return eqx.combine(repeated, static)


def _select_batch_srghn(pop: SRGHN, idx: jnp.ndarray) -> SRGHN:
    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    batch_arr = jax.tree_util.tree_map(lambda value: value[idx], pop_arr)
    return eqx.combine(batch_arr, pop_static)


def _srghn_mutation_kwargs(cond: ConditionSpec) -> dict[str, Any]:
    return {
        "excluded_modules": cond.mutation_exclude_modules,
        "fixed_mutation_lr": cond.fixed_mutation_lr,
    }


def srghn_population_generation(
    pop: SRGHN,
    key: jax.Array,
    pop_size: int,
    children_per_parent: int,
    fitness_fn,
    cond: ConditionSpec,
):
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key)
    probe_keys = jax.random.split(probe_key, pop_size)
    parent_metadata = eqx.filter_vmap(lambda indiv, rng: mutation_metadata(indiv, rng, **mutation_kwargs))(pop, probe_keys)

    if children_per_parent <= 0:
        all_candidates = pop
        all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)
        select_idx = jnp.argsort(all_fitness)[-pop_size:]
        next_pop = _select_batch_srghn(all_candidates, select_idx)
        next_fitness = all_fitness[select_idx]
        elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], parent_metadata)
        metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
        return next_pop, next_fitness, metrics

    pop_arr, pop_static = eqx.partition(pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(pop_size), children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda value: value[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, pop_size * children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutate_with_metadata(indiv, rng, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda parent, child: jnp.concatenate([parent, child], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(fitness_fn)(all_candidates)

    select_idx = jnp.argsort(all_fitness)[-pop_size:]
    next_arr = jax.tree_util.tree_map(lambda value: value[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_value, child_value: jnp.concatenate([parent_value, child_value], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], candidate_metadata)
    metrics = compute_experiment_metrics(pop, all_fitness, parent_metadata, elite_metadata)
    return next_pop, next_fitness, metrics


def init_srghn_local_population(indiv: SRGHN, key: jax.Array, cfg: MetaBraxConfig, cond: ConditionSpec) -> SRGHN:
    if cfg.inner_pop_size <= 1:
        return _repeat_srghn(indiv, 1)
    mutation_kwargs = _srghn_mutation_kwargs(cond)
    child_keys = jax.random.split(key, cfg.inner_pop_size - 1)
    children, _ = eqx.filter_vmap(lambda rng: mutate_with_metadata(indiv, rng, **mutation_kwargs))(child_keys)
    return _stack_parent_with_children(indiv, children)


def srghn_adapt(
    indiv: SRGHN,
    key: jax.Array,
    support_keys: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> SRGHN:
    key_init, key_loop = jax.random.split(key)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return evaluate_individual_on_heading(candidate, support_keys, heading, cfg)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        return (next_pop, rng, next_fitness), best_indiv

    if cfg.inner_generations <= 0:
        return _select_srghn(init_pop, jnp.argmax(init_fitness))

    (_, _, _), best_seq = jax.lax.scan(
        step_fn,
        (init_pop, key_loop, init_fitness),
        None,
        length=cfg.inner_generations,
    )
    return _select_srghn(best_seq, -1)


def srghn_task_curve(
    indiv: SRGHN,
    key: jax.Array,
    heading: jnp.ndarray,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> jnp.ndarray:
    key_adapt, key_episode = jax.random.split(key)
    support_keys, query_keys = split_support_query_episode_keys(key_episode, cfg.support_episodes, cfg.query_episodes)
    key_init, key_loop = jax.random.split(key_adapt)
    init_pop = init_srghn_local_population(indiv, key_init, cfg, cond)

    def support_fitness(candidate: SRGHN) -> jnp.ndarray:
        return evaluate_individual_on_heading(candidate, support_keys, heading, cfg)

    init_fitness = eqx.filter_vmap(support_fitness)(init_pop)
    best0 = _select_srghn(init_pop, jnp.argmax(init_fitness))
    initial_return = evaluate_individual_on_heading(best0, query_keys, heading, cfg)

    def step_fn(carry, _):
        pop, rng, _fitness = carry
        rng, step_key = jax.random.split(rng)
        next_pop, next_fitness, _ = srghn_population_generation(
            pop,
            step_key,
            cfg.inner_pop_size,
            cfg.inner_children_per_parent,
            support_fitness,
            cond,
        )
        best_indiv = _select_srghn(next_pop, jnp.argmax(next_fitness))
        query_return = evaluate_individual_on_heading(best_indiv, query_keys, heading, cfg)
        return (next_pop, rng, next_fitness), query_return

    if cfg.inner_generations <= 0:
        return initial_return[None]

    (_, _, _), query_returns = jax.lax.scan(
        step_fn,
        (init_pop, key_loop, init_fitness),
        None,
        length=cfg.inner_generations,
    )
    return jnp.concatenate([initial_return[None], query_returns], axis=0)


def srghn_meta_fitness(
    indiv: SRGHN,
    key: jax.Array,
    tasks: BraxHeadingTaskBatch,
    cfg: MetaBraxConfig,
    cond: ConditionSpec,
) -> jnp.ndarray:
    task_keys = jax.random.split(key, tasks.headings.shape[0])

    def per_task(heading, task_key):
        key_adapt, key_episode = jax.random.split(task_key)
        support_keys, query_keys = split_support_query_episode_keys(key_episode, cfg.support_episodes, cfg.query_episodes)
        adapted = srghn_adapt(indiv, key_adapt, support_keys, heading, cfg, cond)
        return evaluate_individual_on_heading(adapted, query_keys, heading, cfg)

    return jnp.mean(jax.vmap(per_task)(tasks.headings, task_keys))


def srghn_outer_step(state: SRGHNMetaState, gen: jnp.ndarray, cfg: MetaBraxConfig, cond: ConditionSpec):
    key_next, key_tasks, key_eval, key_evolve = jax.random.split(state.key, 4)
    tasks = sample_heading_tasks(key_tasks, cfg.meta_batch_size)
    eval_keys = jax.random.split(key_eval, cfg.outer_pop_size * (1 + cfg.outer_children_per_parent))

    def batched_fitness(indiv: SRGHN, eval_key: jax.Array) -> jnp.ndarray:
        return srghn_meta_fitness(indiv, eval_key, tasks, cfg, cond)

    mutation_kwargs = _srghn_mutation_kwargs(cond)
    probe_key, child_key = jax.random.split(key_evolve)
    probe_keys = jax.random.split(probe_key, cfg.outer_pop_size)
    parent_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutation_metadata(indiv, rng, **mutation_kwargs)
    )(state.pop, probe_keys)

    pop_arr, pop_static = eqx.partition(state.pop, eqx.is_array)
    parent_idx = jnp.repeat(jnp.arange(cfg.outer_pop_size), cfg.outer_children_per_parent)
    parent_rep_arr = jax.tree_util.tree_map(lambda value: value[parent_idx], pop_arr)
    parents_rep = eqx.combine(parent_rep_arr, pop_static)

    child_keys = jax.random.split(child_key, cfg.outer_pop_size * cfg.outer_children_per_parent)
    children, child_metadata = eqx.filter_vmap(
        lambda indiv, rng: mutate_with_metadata(indiv, rng, **mutation_kwargs)
    )(parents_rep, child_keys)

    child_arr, child_static = eqx.partition(children, eqx.is_array)
    all_arr = jax.tree_util.tree_map(lambda parent, child: jnp.concatenate([parent, child], axis=0), pop_arr, child_arr)
    all_candidates = eqx.combine(all_arr, child_static)
    all_fitness = eqx.filter_vmap(batched_fitness)(all_candidates, eval_keys)

    select_idx = jnp.argsort(all_fitness)[-cfg.outer_pop_size:]
    next_arr = jax.tree_util.tree_map(lambda value: value[select_idx], all_arr)
    next_pop = eqx.combine(next_arr, child_static)
    next_fitness = all_fitness[select_idx]

    candidate_metadata = jax.tree_util.tree_map(
        lambda parent_value, child_value: jnp.concatenate([parent_value, child_value], axis=0),
        parent_metadata,
        child_metadata,
    )
    elite_metadata = jax.tree_util.tree_map(lambda value: value[select_idx], candidate_metadata)

    metrics = compute_experiment_metrics(state.pop, all_fitness, parent_metadata, elite_metadata)
    metrics["query_return_best"] = metrics["fitness_best"]
    metrics["query_return_mean"] = metrics["fitness_mean"]
    jax.debug.callback(partial(_wandb_log_metrics, prefix="train"), metrics, gen)

    gen_best_idx = jnp.argmax(next_fitness)
    gen_best = _select_srghn(next_pop, gen_best_idx)
    gen_best_fit = next_fitness[gen_best_idx]
    improved = gen_best_fit > state.best_fitness
    best_indiv = jax.lax.cond(improved, lambda _: gen_best, lambda _: state.best_indiv, operand=None)
    best_fitness = jnp.maximum(state.best_fitness, gen_best_fit)

    return SRGHNMetaState(
        pop=next_pop,
        key=key_next,
        pop_fitness=next_fitness,
        best_fitness=best_fitness,
        best_indiv=best_indiv,
    ), metrics


@partial(jax.jit, static_argnums=(0, 1))
def run_srghn_compiled(cfg: MetaBraxConfig, cond: ConditionSpec, key: jax.Array):
    runtime_config = make_runtime_config(cfg)
    graphs, specs = build_graphs_and_specs(runtime_config)

    key_pop, key_loop = jax.random.split(key)
    init_pop = init_population(key_pop, runtime_config, graphs, specs)
    init_best = _select_srghn(init_pop, 0)
    init_state = SRGHNMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.outer_pop_size,), dtype=jnp.float32),
        best_fitness=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_indiv=init_best,
    )

    generations = jnp.arange(cfg.outer_generations, dtype=jnp.int32)
    final_state, history = jax.lax.scan(
        lambda carry, gen: srghn_outer_step(carry, gen, cfg, cond),
        init_state,
        generations,
    )

    heldout_tasks = sample_heading_tasks(jax.random.PRNGKey(cfg.seed + 10_000), cfg.heldout_task_batch_size)
    heldout_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 20_000), cfg.heldout_task_batch_size)

    curves = jax.vmap(
        lambda heading, task_key: srghn_task_curve(final_state.best_indiv, task_key, heading, cfg, cond)
    )(heldout_tasks.headings, heldout_keys)
    return final_state, history, curves


def run_condition(cfg: MetaBraxConfig, cond: ConditionSpec) -> dict[str, Any]:
    runtime_config = make_runtime_config(cfg)
    graphs, specs = build_graphs_and_specs(runtime_config)
    del graphs

    _wandb_init_run(cfg, cond, policy_spec=specs.policy_spec, self_spec=specs.self_spec)

    init_key = jax.random.PRNGKey(cfg.seed)
    t0 = time.perf_counter()
    final_state, history, curves = run_srghn_compiled(cfg, cond, init_key)
    jax.block_until_ready(history["fitness_best"])
    jax.block_until_ready(curves)
    train_seconds = time.perf_counter() - t0
    eval_seconds = 0.0
    champion = final_state.best_indiv

    summary_payload = {
        "train_history": _history_to_host(history),
        "adaptation_curve_query_return": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
    }
    _wandb_log_summary(cfg, cond, summary_payload)
    _wandb_finish_run()

    return {
        "kind": "srghn",
        "condition": asdict(cond),
        "config": asdict(cfg),
        "policy_spec_shapes": [tuple(shape) for shape in specs.policy_spec.shapes],
        "self_spec_shapes": [tuple(shape) for shape in specs.self_spec.shapes],
        "train_history": _history_to_host(history),
        "adaptation_curve_query_return": _summary_from_curves(curves),
        "final_population_fitness": jax.device_get(final_state.pop_fitness),
        "champion": champion,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
    }


def maybe_save_plots(output_stem: Path, results: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        history = payload["train_history"]
        ax.plot(history["query_return_best"], label=name)
    ax.set_title("Meta-train best query return")
    ax.set_xlabel("Outer generation")
    ax.set_ylabel("Query return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_train.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, payload in results.items():
        curve = payload["adaptation_curve_query_return"]
        xs = list(range(len(curve["mean"])))
        mean = curve["mean"]
        stderr = curve["stderr"]
        ax.plot(xs, mean, label=name)
        ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_title("Held-out adaptation curve")
    ax.set_xlabel("Inner generation")
    ax.set_ylabel("Query return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_name(output_stem.name + "_adaptation.png"), dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta-RL benchmark for Brax Ant with hidden cardinal headings.")
    parser.add_argument("--conditions", nargs="+", default=BASELINE_NAMES)
    parser.add_argument("--output", default="meta_brax_heading_results.pkl")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-preset", default="default", choices=("default", "fast"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--outer-generations", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--heldout-task-batch-size", type=int, default=None)
    parser.add_argument("--outer-pop-size", type=int, default=None)
    parser.add_argument("--outer-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-pop-size", type=int, default=None)
    parser.add_argument("--inner-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-generations", type=int, default=None)
    parser.add_argument("--support-episodes", type=int, default=None)
    parser.add_argument("--query-episodes", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--gnn-hidden-dim", type=int, default=None)
    parser.add_argument("--gnn-steps-policy", type=int, default=None)
    parser.add_argument("--gnn-steps-self", type=int, default=None)
    parser.add_argument("--stoch-coeff-dim", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=None)
    parser.add_argument("--const-noise-std", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def make_base_cfg(args: argparse.Namespace) -> MetaBraxConfig:
    overrides = {
        key: value
        for key, value in {
            "env_id": args.env_id,
            "brax_backend": args.backend,
            "seed": args.seed,
            "outer_generations": args.outer_generations,
            "meta_batch_size": args.meta_batch_size,
            "heldout_task_batch_size": args.heldout_task_batch_size,
            "outer_pop_size": args.outer_pop_size,
            "outer_children_per_parent": args.outer_children_per_parent,
            "inner_pop_size": args.inner_pop_size,
            "inner_children_per_parent": args.inner_children_per_parent,
            "inner_generations": args.inner_generations,
            "support_episodes": args.support_episodes,
            "query_episodes": args.query_episodes,
            "episode_horizon": args.episode_horizon,
            "policy_hidden_dims": None if args.policy_hidden_dims is None else tuple(args.policy_hidden_dims),
            "embedding_dim": args.embedding_dim,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "gnn_steps_policy": args.gnn_steps_policy,
            "gnn_steps_self": args.gnn_steps_self,
            "stoch_coeff_dim": args.stoch_coeff_dim,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
            "mutation_rate_head_dim": args.mutation_rate_head_dim,
            "const_noise_std": args.const_noise_std,
            "baseline_fixed_mutation_lr": args.fixed_mutation_lr,
            "wandb_project": None if args.no_wandb else args.wandb_project,
            "wandb_group": args.wandb_group,
            "wandb_name": args.wandb_name,
            "wandb_log_plots": args.wandb_log_plots,
        }.items()
        if value is not None
    }
    if args.no_wandb:
        overrides["wandb_project"] = None
    run_preset = "fast" if args.fast else args.run_preset
    return build_meta_brax_config(run_preset=run_preset, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(args)
    if args.print_config:
        print_resolved_config(
            base_cfg,
            family="meta_brax_heading",
            run_preset="fast" if args.fast else args.run_preset,
        )
        return 0
    conditions = [
        parse_condition_spec(text, fixed_mutation_lr=base_cfg.baseline_fixed_mutation_lr)
        for text in args.conditions
    ]

    results: dict[str, Any] = {}
    for cond in conditions:
        print(f"[run] {cond.name} env={base_cfg.env_id} backend={base_cfg.brax_backend}", flush=True)
        payload = run_condition(base_cfg, cond)
        results[cond.name] = payload
        curve = payload["adaptation_curve_query_return"]["mean"]
        print(
            f"[done] {cond.name} train_s={payload['train_seconds']:.2f} eval_s={payload['eval_seconds']:.2f} "
            f"heldout_return={curve.tolist()}",
            flush=True,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading",
            run_preset="fast" if args.fast else args.run_preset,
        ),
        "conditions": [asdict(cond) for cond in conditions],
        "results": results,
    }
    with output_path.open("wb") as handle:
        pickle.dump(save_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}")

    if args.plot:
        maybe_save_plots(output_path.with_suffix(""), results)
        print(f"[saved] plots near {output_path.with_suffix('')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
