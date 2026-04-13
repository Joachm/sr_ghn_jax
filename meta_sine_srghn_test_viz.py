from __future__ import annotations

"""Post-training evaluation and paper-style visualizations for SR-GHN sine meta-learning.

This script is designed to run *after* `meta_sine_srghn.py`.
It reconstructs the trained champion deterministically from the saved config/seed when
that champion is not stored in the training pickle, evaluates the meta-trained policy on
held-out sine tasks, and writes several figures suitable for a paper draft.

Outputs
-------
For each requested baseline, the script saves:
- adaptation_curve_<baseline>.png
- pre_post_scatter_<baseline>.png
- improvement_ecdf_<baseline>.png
- qualitative_steps_<baseline>.png
And for the focal baseline it also saves:
- mutation_dynamics_<baseline>.png    (SR-GHN baselines only)
- candidate_ensemble_<baseline>.png
- fixed_tasks_overlay_<baseline>.png
It also writes:
- evaluation_summary.json
- evaluation_payload.pkl

Example
-------
python meta_sine_srghn_test_viz.py \
  --results meta_sine_srghn_results.pkl \
  --output-dir meta_sine_eval_figures
"""

import argparse
import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from srghn import make_policy, mutate_with_metadata, mutation_metadata
from meta_sine_srghn import (
    MetaSineConfig,
    SRGHNMetaState,
    VectorMetaState,
    _select_srghn,
    build_srghn_graphs_and_specs,
    init_policy_vector_population,
    init_srghn_population,
    regression_forward,
    regression_policy_spec,
    sample_sine_tasks,
    srghn_adapt_one_step,
    srghn_outer_step,
    srghn_query_mse,
    srghn_support_fitness,
    srghn_task_curve,
    unflatten_policy_vector,
    vector_ga_adapt_one_step,
    vector_ga_support_fitness,
    vector_ga_task_curve,
    vector_outer_step,
    vector_query_mse,
)


def _cfg_from_dict(cfg_dict: dict[str, Any]) -> MetaSineConfig:
    data = dict(cfg_dict)
    if isinstance(data.get("policy_hidden_dims"), list):
        data["policy_hidden_dims"] = tuple(int(x) for x in data["policy_hidden_dims"])
    for key in ("clip_params", "clip_std", "clip_update", "mutation_exclude_modules"):
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])
    return MetaSineConfig(**data)


def _load_training_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _maybe_load_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _load_pickle(path)
    except Exception:
        return None


def reconstruct_srghn_champion(cfg: MetaSineConfig):
    graphs, specs = build_srghn_graphs_and_specs(cfg)
    self_graph, policy_graph = graphs
    self_spec, policy_spec = specs

    init_key = jax.random.PRNGKey(cfg.seed)
    key_pop, key_loop = jax.random.split(init_key)
    init_pop = init_srghn_population(key_pop, cfg, (self_graph, policy_graph), (self_spec, policy_spec))
    init_state = SRGHNMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.pop_size,), dtype=jnp.float32),
    )

    @jax.jit
    def run_impl(state: SRGHNMetaState):
        return jax.lax.scan(lambda carry, _: srghn_outer_step(carry, cfg), state, None, length=cfg.outer_generations)

    final_state, history = run_impl(init_state)
    jax.block_until_ready(final_state.pop_fitness)
    best_idx = int(jax.device_get(jnp.argmax(final_state.pop_fitness)))
    champion = _select_srghn(final_state.pop, best_idx)
    return {
        "kind": "srghn",
        "cfg": asdict(cfg),
        "best_index": best_idx,
        "final_population_fitness": np.asarray(jax.device_get(final_state.pop_fitness)),
        "champion": champion,
        "history": {k: np.asarray(jax.device_get(v)) for k, v in history.items()},
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
    }


def reconstruct_vector_champion(cfg: MetaSineConfig):
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)

    init_key = jax.random.PRNGKey(cfg.seed)
    key_pop, key_loop = jax.random.split(init_key)
    init_pop = init_policy_vector_population(
        key_pop,
        cfg.pop_size,
        policy_spec,
        scale=cfg.vector_ga_init_scale,
    )
    init_state = VectorMetaState(
        pop=init_pop,
        key=key_loop,
        pop_fitness=-jnp.inf * jnp.ones((cfg.pop_size,), dtype=jnp.float32),
    )

    @jax.jit
    def run_impl(state: VectorMetaState):
        return jax.lax.scan(lambda carry, _: vector_outer_step(carry, cfg, policy_spec), state, None, length=cfg.outer_generations)

    final_state, history = run_impl(init_state)
    jax.block_until_ready(final_state.pop_fitness)
    best_idx = int(jax.device_get(jnp.argmax(final_state.pop_fitness)))
    champion = final_state.pop[best_idx]
    return {
        "kind": "vector_ga",
        "cfg": asdict(cfg),
        "best_index": best_idx,
        "final_population_fitness": np.asarray(jax.device_get(final_state.pop_fitness)),
        "champion": np.asarray(jax.device_get(champion)),
        "history": {k: np.asarray(jax.device_get(v)) for k, v in history.items()},
        "policy_spec_shapes": [tuple(shape) for shape in policy_spec.shapes],
    }


def get_or_reconstruct_champion(baseline_name: str, cfg: MetaSineConfig, cache_dir: Path):
    cache_path = cache_dir / f"champion_cache_{baseline_name}.pkl"
    cached = _maybe_load_cache(cache_path)
    if cached is not None:
        print(f"[cache] loaded {cache_path}", flush=True)
        return cached

    print(f"[reconstruct] baseline={baseline_name}", flush=True)
    if baseline_name == "vector_ga":
        payload = reconstruct_vector_champion(cfg)
    else:
        payload = reconstruct_srghn_champion(cfg)

    try:
        _save_pickle(cache_path, payload)
        print(f"[cache] saved {cache_path}", flush=True)
    except Exception as exc:
        print(f"[cache] skip save for {baseline_name}: {exc}", flush=True)
    return payload


@jax.tree_util.register_pytree_node_class
class SimpleShowcase:
    def __init__(self, preds, query_mse, support_fitness, mutation_rate_mean, block_fraction, update_rms):
        self.preds = preds
        self.query_mse = query_mse
        self.support_fitness = support_fitness
        self.mutation_rate_mean = mutation_rate_mean
        self.block_fraction = block_fraction
        self.update_rms = update_rms

    def tree_flatten(self):
        return (
            self.preds,
            self.query_mse,
            self.support_fitness,
            self.mutation_rate_mean,
            self.block_fraction,
            self.update_rms,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.jit

def _grid_template(num_points: int, x_min: float, x_max: float):
    return jnp.linspace(x_min, x_max, num_points, dtype=jnp.float32)[:, None]


def evaluate_srghn_baseline(
    champion,
    cfg: MetaSineConfig,
    *,
    eval_tasks: int,
    showcase_tasks: int,
    eval_seed: int,
    plot_points: int,
):
    heldout = sample_sine_tasks(jax.random.PRNGKey(eval_seed), cfg, eval_tasks)
    heldout_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 1), eval_tasks)
    grid_x = _grid_template(plot_points, cfg.x_min, cfg.x_max)

    @jax.jit
    def eval_curves(indiv, tasks, keys):
        return jax.vmap(
            lambda sx, sy, qx, qy, task_key: srghn_task_curve(indiv, task_key, sx, sy, qx, qy, cfg)
        )(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, keys)

    curves = eval_curves(champion, heldout, heldout_keys)
    jax.block_until_ready(curves)
    curves_np = np.asarray(jax.device_get(curves))

    showcase = sample_sine_tasks(jax.random.PRNGKey(eval_seed + 2), cfg, showcase_tasks)
    showcase_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 3), showcase_tasks)

    def single_showcase(indiv, sx, sy, qx, qy, key):
        init_pred = regression_forward(make_policy(indiv), grid_x)
        init_query = srghn_query_mse(indiv, qx, qy)
        init_support = srghn_support_fitness(indiv, sx, sy)

        def step_fn(current, step_key):
            probe_key, adapt_key = jax.random.split(step_key)
            meta = mutation_metadata(
                current,
                probe_key,
                excluded_modules=cfg.mutation_exclude_modules,
                fixed_mutation_lr=cfg.fixed_mutation_lr,
            )
            nxt, best_support = srghn_adapt_one_step(current, adapt_key, sx, sy, cfg)
            pred = regression_forward(make_policy(nxt), grid_x)
            q_mse = srghn_query_mse(nxt, qx, qy)
            return nxt, (
                pred,
                q_mse,
                best_support,
                meta.mutation_rate_mean,
                meta.mutation_block_fraction,
                meta.update_rms,
            )

        step_keys = jax.random.split(key, cfg.inner_steps)
        _, outs = jax.lax.scan(step_fn, indiv, step_keys)
        preds = jnp.concatenate([init_pred[None], outs[0]], axis=0)
        query_mse = jnp.concatenate([init_query[None], outs[1]], axis=0)
        support_fit = jnp.concatenate([init_support[None], outs[2]], axis=0)
        mutation_rate = outs[3]
        block_fraction = outs[4]
        update_rms = outs[5]
        return SimpleShowcase(preds, query_mse, support_fit, mutation_rate, block_fraction, update_rms)

    showcase_out = jax.vmap(lambda sx, sy, qx, qy, key: single_showcase(champion, sx, sy, qx, qy, key))(
        showcase.support_x,
        showcase.support_y,
        showcase.query_x,
        showcase.query_y,
        showcase_keys,
    )
    showcase_out = jax.device_get(showcase_out)

    return {
        "kind": "srghn",
        "curves": curves_np,
        "pre_mse": curves_np[:, 0],
        "post_mse": curves_np[:, -1],
        "improvement": curves_np[:, 0] - curves_np[:, -1],
        "relative_improvement": (curves_np[:, 0] - curves_np[:, -1]) / np.maximum(curves_np[:, 0], 1e-8),
        "grid_x": np.asarray(jax.device_get(grid_x.squeeze(-1))),
        "heldout_tasks": jax.device_get(heldout),
        "showcase_tasks": jax.device_get(showcase),
        "showcase_preds": np.asarray(showcase_out.preds),
        "showcase_query_mse": np.asarray(showcase_out.query_mse),
        "showcase_support_fitness": np.asarray(showcase_out.support_fitness),
        "mutation_rate_mean": np.asarray(showcase_out.mutation_rate_mean),
        "mutation_block_fraction": np.asarray(showcase_out.block_fraction),
        "mutation_update_rms": np.asarray(showcase_out.update_rms),
    }


def evaluate_vector_baseline(
    champion_vec,
    cfg: MetaSineConfig,
    *,
    eval_tasks: int,
    showcase_tasks: int,
    eval_seed: int,
    plot_points: int,
):
    heldout = sample_sine_tasks(jax.random.PRNGKey(eval_seed), cfg, eval_tasks)
    heldout_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 1), eval_tasks)
    grid_x = _grid_template(plot_points, cfg.x_min, cfg.x_max)
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)

    @jax.jit
    def eval_curves(policy_vector, tasks, keys):
        return jax.vmap(
            lambda sx, sy, qx, qy, task_key: vector_ga_task_curve(policy_vector, task_key, sx, sy, qx, qy, cfg, policy_spec)
        )(tasks.support_x, tasks.support_y, tasks.query_x, tasks.query_y, keys)

    curves = eval_curves(jnp.asarray(champion_vec), heldout, heldout_keys)
    jax.block_until_ready(curves)
    curves_np = np.asarray(jax.device_get(curves))

    showcase = sample_sine_tasks(jax.random.PRNGKey(eval_seed + 2), cfg, showcase_tasks)
    showcase_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 3), showcase_tasks)

    def single_showcase(policy_vector, sx, sy, qx, qy, key):
        init_pred = regression_forward(unflatten_policy_vector(policy_vector, policy_spec), grid_x)
        init_query = vector_query_mse(policy_vector, qx, qy, policy_spec)
        init_support = vector_ga_support_fitness(policy_vector, sx, sy, policy_spec)

        def step_fn(current, step_key):
            nxt, best_support = vector_ga_adapt_one_step(current, step_key, sx, sy, cfg, policy_spec)
            pred = regression_forward(unflatten_policy_vector(nxt, policy_spec), grid_x)
            q_mse = vector_query_mse(nxt, qx, qy, policy_spec)
            return nxt, (pred, q_mse, best_support)

        step_keys = jax.random.split(key, cfg.inner_steps)
        _, outs = jax.lax.scan(step_fn, policy_vector, step_keys)
        preds = jnp.concatenate([init_pred[None], outs[0]], axis=0)
        query_mse = jnp.concatenate([init_query[None], outs[1]], axis=0)
        support_fit = jnp.concatenate([init_support[None], outs[2]], axis=0)
        zeros = jnp.zeros((cfg.inner_steps,), dtype=preds.dtype)
        return SimpleShowcase(preds, query_mse, support_fit, zeros, zeros, zeros)

    showcase_out = jax.vmap(lambda sx, sy, qx, qy, key: single_showcase(jnp.asarray(champion_vec), sx, sy, qx, qy, key))(
        showcase.support_x,
        showcase.support_y,
        showcase.query_x,
        showcase.query_y,
        showcase_keys,
    )
    showcase_out = jax.device_get(showcase_out)

    return {
        "kind": "vector_ga",
        "curves": curves_np,
        "pre_mse": curves_np[:, 0],
        "post_mse": curves_np[:, -1],
        "improvement": curves_np[:, 0] - curves_np[:, -1],
        "relative_improvement": (curves_np[:, 0] - curves_np[:, -1]) / np.maximum(curves_np[:, 0], 1e-8),
        "grid_x": np.asarray(jax.device_get(grid_x.squeeze(-1))),
        "heldout_tasks": jax.device_get(heldout),
        "showcase_tasks": jax.device_get(showcase),
        "showcase_preds": np.asarray(showcase_out.preds),
        "showcase_query_mse": np.asarray(showcase_out.query_mse),
        "showcase_support_fitness": np.asarray(showcase_out.support_fitness),
        "mutation_rate_mean": np.asarray(showcase_out.mutation_rate_mean),
        "mutation_block_fraction": np.asarray(showcase_out.block_fraction),
        "mutation_update_rms": np.asarray(showcase_out.update_rms),
    }


def _mean_ci(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    stderr = arr.std(axis=0, ddof=0) / math.sqrt(max(arr.shape[0], 1))
    return mean, 1.96 * stderr


def plot_adaptation_curve(path: Path, baseline_name: str, curves: np.ndarray) -> None:
    mean, ci = _mean_ci(curves)
    xs = np.arange(curves.shape[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(xs, mean, linewidth=2.2)
    ax.fill_between(xs, mean - ci, mean + ci, alpha=0.25)
    ax.set_xlabel("Inner adaptation step")
    ax.set_ylabel("Held-out query MSE")
    ax.set_title(f"Held-out adaptation curve: {baseline_name}")
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pre_post_scatter(path: Path, baseline_name: str, pre_mse: np.ndarray, post_mse: np.ndarray) -> None:
    lim_min = float(min(pre_mse.min(), post_mse.min()))
    lim_max = float(max(pre_mse.max(), post_mse.max()))
    eps = 1e-5
    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    ax.scatter(pre_mse + eps, post_mse + eps, alpha=0.5, s=18)
    diag = np.linspace(lim_min + eps, lim_max + eps, 256)
    ax.plot(diag, diag, linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pre-adaptation query MSE")
    ax.set_ylabel("Post-adaptation query MSE")
    ax.set_title(f"Task-wise adaptation gain: {baseline_name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_improvement_ecdf(path: Path, baseline_name: str, relative_improvement: np.ndarray) -> None:
    vals = np.sort(relative_improvement)
    probs = np.linspace(0.0, 1.0, len(vals), endpoint=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(vals, probs, linewidth=2.2)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Relative improvement = (pre - post) / pre")
    ax.set_ylabel("ECDF")
    ax.set_title(f"How often adaptation helps: {baseline_name}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_qualitative_steps(path: Path, baseline_name: str, eval_payload: dict[str, Any]) -> None:
    tasks = eval_payload["showcase_tasks"]
    preds = eval_payload["showcase_preds"]
    q_mse = eval_payload["showcase_query_mse"]
    grid_x = eval_payload["grid_x"]
    num_tasks = preds.shape[0]
    num_steps = preds.shape[1]

    fig, axes = plt.subplots(num_tasks, num_steps, figsize=(3.6 * num_steps, 2.8 * num_tasks), squeeze=False)
    for i in range(num_tasks):
        sx = np.asarray(tasks.support_x[i]).squeeze(-1)
        sy = np.asarray(tasks.support_y[i])
        qx = np.asarray(tasks.query_x[i]).squeeze(-1)
        qy = np.asarray(tasks.query_y[i])
        order = np.argsort(np.concatenate([sx, qx]))
        merged_x = np.concatenate([sx, qx])[order]
        merged_y = np.concatenate([sy, qy])[order]
        for s in range(num_steps):
            ax = axes[i, s]
            ax.plot(merged_x, merged_y, linewidth=1.8)
            ax.plot(grid_x, preds[i, s], linewidth=2.0)
            ax.scatter(sx, sy, s=18, marker="o")
            ax.scatter(qx, qy, s=10, marker="x", alpha=0.55)
            ax.set_xlim(grid_x.min(), grid_x.max())
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(f"step {s}\nquery MSE={q_mse[i, s]:.4f}")
            else:
                ax.set_title(f"query MSE={q_mse[i, s]:.4f}")
            if s == 0:
                ax.set_ylabel(f"task {i}")
    fig.suptitle(f"Fixed held-out tasks across adaptation steps: {baseline_name}", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_task_overlay(path: Path, baseline_name: str, eval_payload: dict[str, Any]) -> None:
    tasks = eval_payload["showcase_tasks"]
    preds = eval_payload["showcase_preds"]
    grid_x = eval_payload["grid_x"]
    q_mse = eval_payload["showcase_query_mse"]
    num_tasks = preds.shape[0]

    fig, axes = plt.subplots(1, num_tasks, figsize=(4.2 * num_tasks, 3.8), squeeze=False)
    for i in range(num_tasks):
        ax = axes[0, i]
        sx = np.asarray(tasks.support_x[i]).squeeze(-1)
        sy = np.asarray(tasks.support_y[i])
        qx = np.asarray(tasks.query_x[i]).squeeze(-1)
        qy = np.asarray(tasks.query_y[i])
        merged_x = np.concatenate([sx, qx])
        merged_y = np.concatenate([sy, qy])
        order = np.argsort(merged_x)
        ax.plot(merged_x[order], merged_y[order], linewidth=2.0, label="true task")
        for s in range(preds.shape[1]):
            ax.plot(grid_x, preds[i, s], alpha=0.7, linewidth=1.8, label=f"step {s}" if i == 0 else None)
        ax.scatter(sx, sy, s=22, marker="o", label="support" if i == 0 else None)
        ax.scatter(qx, qy, s=12, marker="x", alpha=0.6, label="query" if i == 0 else None)
        ax.set_title(f"task {i}\n{q_mse[i, 0]:.3f}  ->  {q_mse[i, -1]:.3f}")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)
    fig.suptitle(f"Prediction evolution on fixed held-out tasks: {baseline_name}", y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mutation_dynamics(path: Path, baseline_name: str, eval_payload: dict[str, Any]) -> None:
    lr = eval_payload["mutation_rate_mean"]
    block_frac = eval_payload["mutation_block_fraction"]
    upd_rms = eval_payload["mutation_update_rms"]
    steps = np.arange(1, lr.shape[1] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), squeeze=False)
    panels = [
        (lr, "Predicted mutation-rate mean"),
        (block_frac, "Mutated block fraction"),
        (upd_rms, "Proposed update RMS"),
    ]
    for ax, (arr, title) in zip(axes[0], panels):
        mean, ci = _mean_ci(arr)
        ax.plot(steps, mean, linewidth=2.0)
        ax.fill_between(steps, mean - ci, mean + ci, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("Adaptation step")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Held-out task average")
    fig.suptitle(f"Test-time mutation dynamics: {baseline_name}", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_ensemble_srghn(path: Path, baseline_name: str, champion, cfg: MetaSineConfig, eval_payload: dict[str, Any], *, ensemble_children: int, seed: int) -> None:
    task = eval_payload["showcase_tasks"]
    sx = jnp.asarray(task.support_x[0])
    sy = jnp.asarray(task.support_y[0])
    qx = jnp.asarray(task.query_x[0])
    qy = jnp.asarray(task.query_y[0])
    grid_x = jnp.asarray(eval_payload["grid_x"])[:, None]
    child_keys = jax.random.split(jax.random.PRNGKey(seed), ensemble_children)

    children, _ = eqx.filter_vmap(
        lambda k: mutate_with_metadata(
            champion,
            k,
            excluded_modules=cfg.mutation_exclude_modules,
            fixed_mutation_lr=cfg.fixed_mutation_lr,
        )
    )(child_keys)

    child_preds = eqx.filter_vmap(lambda child: regression_forward(make_policy(child), grid_x))(children)
    child_fit = eqx.filter_vmap(lambda child: srghn_support_fitness(child, sx, sy))(children)
    parent_pred = regression_forward(make_policy(champion), grid_x)
    parent_fit = srghn_support_fitness(champion, sx, sy)
    true_x = np.concatenate([np.asarray(sx).squeeze(-1), np.asarray(qx).squeeze(-1)])
    true_y = np.concatenate([np.asarray(sy), np.asarray(qy)])
    order = np.argsort(true_x)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(true_x[order], true_y[order], linewidth=2.2, label="true task")
    norm_fit = np.asarray(jax.device_get(child_fit))
    fit_min = float(norm_fit.min())
    fit_max = float(norm_fit.max())
    denom = max(fit_max - fit_min, 1e-8)
    for i in range(ensemble_children):
        alpha = 0.15 + 0.7 * float((norm_fit[i] - fit_min) / denom)
        ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(child_preds[i])), alpha=alpha, linewidth=1.3)
    ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(parent_pred)), linewidth=2.4, linestyle="--", label=f"parent ({float(parent_fit):.3f})")
    best_idx = int(np.argmax(norm_fit))
    ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(child_preds[best_idx])), linewidth=2.4, label=f"best child ({float(norm_fit[best_idx]):.3f})")
    ax.scatter(np.asarray(sx).squeeze(-1), np.asarray(sy), s=24, marker="o", label="support")
    ax.scatter(np.asarray(qx).squeeze(-1), np.asarray(qy), s=12, marker="x", alpha=0.6, label="query")
    ax.set_title(f"Mutated candidate ensemble on one held-out task: {baseline_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_candidate_ensemble_vector(path: Path, baseline_name: str, champion_vec: np.ndarray, cfg: MetaSineConfig, eval_payload: dict[str, Any], *, ensemble_children: int, seed: int) -> None:
    task = eval_payload["showcase_tasks"]
    sx = jnp.asarray(task.support_x[0])
    sy = jnp.asarray(task.support_y[0])
    qx = jnp.asarray(task.query_x[0])
    qy = jnp.asarray(task.query_y[0])
    grid_x = jnp.asarray(eval_payload["grid_x"])[:, None]
    policy_spec = regression_policy_spec(cfg.policy_hidden_dims)

    parent = jnp.asarray(champion_vec)
    noise = cfg.vector_ga_sigma * jax.random.normal(jax.random.PRNGKey(seed), (ensemble_children, parent.shape[0]), dtype=parent.dtype)
    children = parent[None, :] + noise
    child_fit = jax.vmap(lambda vec: vector_ga_support_fitness(vec, sx, sy, policy_spec))(children)
    child_preds = jax.vmap(lambda vec: regression_forward(unflatten_policy_vector(vec, policy_spec), grid_x))(children)
    parent_pred = regression_forward(unflatten_policy_vector(parent, policy_spec), grid_x)
    parent_fit = vector_ga_support_fitness(parent, sx, sy, policy_spec)
    true_x = np.concatenate([np.asarray(sx).squeeze(-1), np.asarray(qx).squeeze(-1)])
    true_y = np.concatenate([np.asarray(sy), np.asarray(qy)])
    order = np.argsort(true_x)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(true_x[order], true_y[order], linewidth=2.2, label="true task")
    norm_fit = np.asarray(jax.device_get(child_fit))
    fit_min = float(norm_fit.min())
    fit_max = float(norm_fit.max())
    denom = max(fit_max - fit_min, 1e-8)
    for i in range(ensemble_children):
        alpha = 0.15 + 0.7 * float((norm_fit[i] - fit_min) / denom)
        ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(child_preds[i])), alpha=alpha, linewidth=1.3)
    ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(parent_pred)), linewidth=2.4, linestyle="--", label=f"parent ({float(parent_fit):.3f})")
    best_idx = int(np.argmax(norm_fit))
    ax.plot(np.asarray(grid_x).squeeze(-1), np.asarray(jax.device_get(child_preds[best_idx])), linewidth=2.4, label=f"best child ({float(norm_fit[best_idx]):.3f})")
    ax.scatter(np.asarray(sx).squeeze(-1), np.asarray(sy), s=24, marker="o", label="support")
    ax.scatter(np.asarray(qx).squeeze(-1), np.asarray(qy), s=12, marker="x", alpha=0.6, label="query")
    ax.set_title(f"Mutated candidate ensemble on one held-out task: {baseline_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _to_jsonable_summary(eval_payload: dict[str, Any]) -> dict[str, Any]:
    curves = eval_payload["curves"]
    return {
        "num_eval_tasks": int(curves.shape[0]),
        "num_adaptation_points": int(curves.shape[1]),
        "mean_curve": curves.mean(axis=0).tolist(),
        "std_curve": curves.std(axis=0).tolist(),
        "median_pre_mse": float(np.median(eval_payload["pre_mse"])),
        "median_post_mse": float(np.median(eval_payload["post_mse"])),
        "mean_relative_improvement": float(np.mean(eval_payload["relative_improvement"])),
        "frac_tasks_improved": float(np.mean(eval_payload["post_mse"] < eval_payload["pre_mse"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize meta-trained SR-GHN sine-regression models.")
    parser.add_argument("--results", required=True, help="Pickle written by meta_sine_srghn.py")
    parser.add_argument("--output-dir", default="meta_sine_eval_figures")
    parser.add_argument("--baselines", nargs="+", default=None, help="Subset of baselines to evaluate. Defaults to all in results.")
    parser.add_argument("--focal-baseline", default=None, help="Baseline used for the extra focal figures. Defaults to 'full' if present.")
    parser.add_argument("--eval-tasks", type=int, default=512)
    parser.add_argument("--showcase-tasks", type=int, default=4)
    parser.add_argument("--plot-points", type=int, default=256)
    parser.add_argument("--ensemble-children", type=int, default=24)
    parser.add_argument("--eval-seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = _load_training_payload(results_path)
    result_dict = train_payload["results"]
    baseline_names = list(result_dict.keys()) if args.baselines is None else list(args.baselines)
    missing = [name for name in baseline_names if name not in result_dict]
    if missing:
        raise ValueError(f"Baselines not found in results: {missing}")

    if args.focal_baseline is None:
        focal_baseline = "full" if "full" in baseline_names else baseline_names[0]
    else:
        focal_baseline = args.focal_baseline
        if focal_baseline not in baseline_names:
            raise ValueError(f"focal baseline '{focal_baseline}' must be included in --baselines")

    summary: dict[str, Any] = {
        "results_file": str(results_path),
        "focal_baseline": focal_baseline,
        "eval_seed": args.eval_seed,
        "baselines": {},
    }
    eval_payloads: dict[str, Any] = {}

    for baseline_name in baseline_names:
        cfg = _cfg_from_dict(result_dict[baseline_name]["config"])
        champ_payload = get_or_reconstruct_champion(baseline_name, cfg, output_dir)
        champion = champ_payload["champion"]

        print(f"[eval] baseline={baseline_name}", flush=True)
        if baseline_name == "vector_ga":
            eval_payload = evaluate_vector_baseline(
                champion,
                cfg,
                eval_tasks=args.eval_tasks,
                showcase_tasks=args.showcase_tasks,
                eval_seed=args.eval_seed,
                plot_points=args.plot_points,
            )
        else:
            eval_payload = evaluate_srghn_baseline(
                champion,
                cfg,
                eval_tasks=args.eval_tasks,
                showcase_tasks=args.showcase_tasks,
                eval_seed=args.eval_seed,
                plot_points=args.plot_points,
            )

        eval_payloads[baseline_name] = {
            "config": asdict(cfg),
            "champion_payload": champ_payload,
            "evaluation": eval_payload,
        }
        summary["baselines"][baseline_name] = _to_jsonable_summary(eval_payload)

        plot_adaptation_curve(output_dir / f"adaptation_curve_{baseline_name}.png", baseline_name, eval_payload["curves"])
        plot_pre_post_scatter(
            output_dir / f"pre_post_scatter_{baseline_name}.png",
            baseline_name,
            eval_payload["pre_mse"],
            eval_payload["post_mse"],
        )
        plot_improvement_ecdf(
            output_dir / f"improvement_ecdf_{baseline_name}.png",
            baseline_name,
            eval_payload["relative_improvement"],
        )
        plot_qualitative_steps(output_dir / f"qualitative_steps_{baseline_name}.png", baseline_name, eval_payload)

        if baseline_name == focal_baseline:
            plot_fixed_task_overlay(output_dir / f"fixed_tasks_overlay_{baseline_name}.png", baseline_name, eval_payload)
            if baseline_name == "vector_ga":
                plot_candidate_ensemble_vector(
                    output_dir / f"candidate_ensemble_{baseline_name}.png",
                    baseline_name,
                    champion,
                    cfg,
                    eval_payload,
                    ensemble_children=args.ensemble_children,
                    seed=args.eval_seed + 500,
                )
            else:
                plot_mutation_dynamics(output_dir / f"mutation_dynamics_{baseline_name}.png", baseline_name, eval_payload)
                plot_candidate_ensemble_srghn(
                    output_dir / f"candidate_ensemble_{baseline_name}.png",
                    baseline_name,
                    champion,
                    cfg,
                    eval_payload,
                    ensemble_children=args.ensemble_children,
                    seed=args.eval_seed + 500,
                )

        print(
            f"[done] baseline={baseline_name} mean_pre={summary['baselines'][baseline_name]['median_pre_mse']:.4f} "
            f"mean_post={summary['baselines'][baseline_name]['median_post_mse']:.4f}",
            flush=True,
        )

    with (output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _save_pickle(output_dir / "evaluation_payload.pkl", eval_payloads)

    print(f"[saved] {output_dir / 'evaluation_summary.json'}")
    print(f"[saved] {output_dir / 'evaluation_payload.pkl'}")
    print(f"[saved] figures in {output_dir}")


if __name__ == "__main__":
    main()
