from __future__ import annotations

"""Post-training evaluation + paper-friendly visualizations for the paired-loop meta-sine harness."""

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

from srghn import make_policy, mutation_metadata
from meta_sine_srghn import (
    ConditionSpec,
    LocalEvosaxState,
    MetaSineConfig,
    _select_srghn,
    build_srghn_graphs_and_specs,
    init_local_evosax_state,
    init_srghn_local_population,
    init_vector_local_population,
    local_evosax_generation,
    make_evosax_adapter,
    parse_condition_spec,
    regression_forward,
    regression_policy_spec,
    run_condition,
    sample_sine_tasks,
    srghn_population_generation,
    srghn_query_mse,
    srghn_support_fitness,
    srghn_task_curve,
    unflatten_policy_vector,
    validate_condition,
    vector_gaussian_generation,
    vector_query_mse,
    vector_support_fitness,
    vector_task_curve,
)


def _cfg_from_dict(cfg_dict: dict[str, Any]) -> MetaSineConfig:
    data = dict(cfg_dict)
    if isinstance(data.get("policy_hidden_dims"), list):
        data["policy_hidden_dims"] = tuple(int(x) for x in data["policy_hidden_dims"])
    for key in ("clip_params", "clip_std", "clip_update"):
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])
    return MetaSineConfig(**data)



def _cond_from_dict(cond_dict: dict[str, Any]) -> ConditionSpec:
    data = dict(cond_dict)
    if isinstance(data.get("mutation_exclude_modules"), list):
        data["mutation_exclude_modules"] = tuple(data["mutation_exclude_modules"])
    return validate_condition(ConditionSpec(**data))



def _save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)



def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)



def _load_training_payload(path: Path) -> dict[str, Any]:
    return _load_pickle(path)



def _maybe_load_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _load_pickle(path)
    except Exception:
        return None



def get_or_reconstruct_condition(result_name: str, payload: dict[str, Any], cache_dir: Path):
    cache_path = cache_dir / f"reconstructed_{result_name}.pkl"
    cached = _maybe_load_cache(cache_path)
    if cached is not None:
        return cached

    result = payload["results"][result_name]
    if "champion" in result and result["champion"] is not None:
        return result

    cfg = _cfg_from_dict(result["config"])
    cond = _cond_from_dict(result["condition"])
    reconstructed = run_condition(cfg, cond)
    _save_pickle(cache_path, reconstructed)
    return reconstructed



def _grid_template(num_points: int, x_min: float, x_max: float):
    return jnp.linspace(x_min, x_max, num_points, dtype=jnp.float32)[:, None]



def _adapt_trace_srghn(champion, cfg: MetaSineConfig, cond: ConditionSpec, sx, sy, qx, qy, key):
    key_init, key_loop = jax.random.split(key)
    pop = init_srghn_local_population(champion, key_init, cfg, cond)

    def support_fit(indiv):
        return srghn_support_fitness(indiv, sx, sy)

    fit = eqx.filter_vmap(support_fit)(pop)
    best = _select_srghn(pop, jnp.argmax(fit))
    bests = [best]
    fits = [jnp.max(fit)]
    qmses = [srghn_query_mse(best, qx, qy)]

    for _ in range(cfg.inner_generations):
        key_loop, step_key = jax.random.split(key_loop)
        pop, fit, _ = srghn_population_generation(pop, step_key, cfg.inner_pop_size, cfg.inner_children_per_parent, support_fit, cond)
        best = _select_srghn(pop, jnp.argmax(fit))
        bests.append(best)
        fits.append(jnp.max(fit))
        qmses.append(srghn_query_mse(best, qx, qy))
    return bests, jnp.asarray(fits), jnp.asarray(qmses), pop, fit



def _adapt_trace_vector(champion, cfg: MetaSineConfig, cond: ConditionSpec, policy_spec, sx, sy, qx, qy, key):
    support_fit = lambda vec: vector_support_fitness(vec, sx, sy, policy_spec)
    if cond.inner_optimizer == "gaussian":
        key_init, key_loop = jax.random.split(key)
        pop = init_vector_local_population(champion, key_init, cfg)
        fit = eqx.filter_vmap(support_fit)(pop)
        best = pop[jnp.argmax(fit)]
        bests = [best]
        fits = [jnp.max(fit)]
        qmses = [vector_query_mse(best, qx, qy, policy_spec)]
        for _ in range(cfg.inner_generations):
            key_loop, step_key = jax.random.split(key_loop)
            pop, fit, _ = vector_gaussian_generation(pop, step_key, cfg, cfg.inner_pop_size, cfg.inner_children_per_parent, support_fit)
            best = pop[jnp.argmax(fit)]
            bests.append(best)
            fits.append(jnp.max(fit))
            qmses.append(vector_query_mse(best, qx, qy, policy_spec))
        return bests, jnp.asarray(fits), jnp.asarray(qmses), pop, fit

    if cond.inner_optimizer == "evosax":
        adapter = make_evosax_adapter(cfg.inner_pop_size, cond.inner_evosax_algo or "", cfg.inner_evosax_sigma_init, int(champion.shape[0]))
        state = init_local_evosax_state(champion, key, adapter, support_fit)
        bests = [state.best_solution]
        fits = [state.best_fitness]
        qmses = [vector_query_mse(state.best_solution, qx, qy, policy_spec)]
        for _ in range(cfg.inner_generations):
            state = local_evosax_generation(state, adapter, support_fit)
            bests.append(state.best_solution)
            fits.append(state.best_fitness)
            qmses.append(vector_query_mse(state.best_solution, qx, qy, policy_spec))
        return bests, jnp.asarray(fits), jnp.asarray(qmses), state.population, state.fitness

    raise ValueError(f"Unsupported vector inner optimizer: {cond.inner_optimizer}")



def evaluate_condition(condition_name: str, result_payload: dict[str, Any], *, eval_tasks: int, showcase_tasks: int, eval_seed: int, plot_points: int):
    cfg = _cfg_from_dict(result_payload["config"])
    cond = _cond_from_dict(result_payload["condition"])
    grid_x = _grid_template(plot_points, cfg.x_min, cfg.x_max)

    if cond.search_object == "srghn":
        champion = result_payload["champion"]
        heldout = sample_sine_tasks(jax.random.PRNGKey(eval_seed), cfg, eval_tasks)
        heldout_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 1), eval_tasks)
        curves = jax.vmap(
            lambda sx, sy, qx, qy, task_key: srghn_task_curve(champion, task_key, sx, sy, qx, qy, cfg, cond)
        )(heldout.support_x, heldout.support_y, heldout.query_x, heldout.query_y, heldout_keys)
        curves_np = np.asarray(jax.device_get(curves))

        showcase = sample_sine_tasks(jax.random.PRNGKey(eval_seed + 2), cfg, showcase_tasks)
        showcase_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 3), showcase_tasks)
        showcase_payload = []
        for idx in range(showcase_tasks):
            sx = showcase.support_x[idx]
            sy = showcase.support_y[idx]
            qx = showcase.query_x[idx]
            qy = showcase.query_y[idx]
            bests, fits, qmses, final_pop, final_fit = _adapt_trace_srghn(champion, cfg, cond, sx, sy, qx, qy, showcase_keys[idx])
            preds = [np.asarray(jax.device_get(regression_forward(make_policy(b), grid_x))) for b in bests]
            meta = [mutation_metadata(b, jax.random.PRNGKey(eval_seed + 10_000 + idx * 97 + step)) for step, b in enumerate(bests)]
            showcase_payload.append(
                {
                    "support_x": np.asarray(jax.device_get(sx[:, 0])),
                    "support_y": np.asarray(jax.device_get(sy)),
                    "query_x": np.asarray(jax.device_get(qx[:, 0])),
                    "query_y": np.asarray(jax.device_get(qy)),
                    "grid_x": np.asarray(jax.device_get(grid_x[:, 0])),
                    "preds": preds,
                    "support_fitness": np.asarray(jax.device_get(fits)),
                    "query_mse": np.asarray(jax.device_get(qmses)),
                    "mutation_rate_mean": np.asarray(jax.device_get(jnp.asarray([m.mutation_rate_mean for m in meta]))),
                    "mutation_block_fraction": np.asarray(jax.device_get(jnp.asarray([m.mutation_block_fraction for m in meta]))),
                    "update_rms": np.asarray(jax.device_get(jnp.asarray([m.update_rms for m in meta]))),
                    "candidate_grid_preds": np.asarray(
                        jax.device_get(jax.vmap(lambda cand: regression_forward(make_policy(cand), grid_x))(final_pop))
                    ),
                    "candidate_support_fitness": np.asarray(jax.device_get(final_fit)),
                }
            )

    elif cond.search_object == "vector":
        champion = jnp.asarray(result_payload["champion"], dtype=jnp.float32)
        policy_spec = regression_policy_spec(cfg.policy_hidden_dims)
        inner_adapter = None
        if cond.inner_optimizer == "evosax":
            inner_adapter = make_evosax_adapter(cfg.inner_pop_size, cond.inner_evosax_algo or "", cfg.inner_evosax_sigma_init, champion.shape[0])
        heldout = sample_sine_tasks(jax.random.PRNGKey(eval_seed), cfg, eval_tasks)
        heldout_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 1), eval_tasks)
        curves = jax.vmap(
            lambda sx, sy, qx, qy, task_key: vector_task_curve(champion, task_key, sx, sy, qx, qy, cfg, cond, policy_spec, inner_adapter)
        )(heldout.support_x, heldout.support_y, heldout.query_x, heldout.query_y, heldout_keys)
        curves_np = np.asarray(jax.device_get(curves))

        showcase = sample_sine_tasks(jax.random.PRNGKey(eval_seed + 2), cfg, showcase_tasks)
        showcase_keys = jax.random.split(jax.random.PRNGKey(eval_seed + 3), showcase_tasks)
        showcase_payload = []
        for idx in range(showcase_tasks):
            sx = showcase.support_x[idx]
            sy = showcase.support_y[idx]
            qx = showcase.query_x[idx]
            qy = showcase.query_y[idx]
            bests, fits, qmses, final_pop, final_fit = _adapt_trace_vector(champion, cfg, cond, policy_spec, sx, sy, qx, qy, showcase_keys[idx])
            preds = [np.asarray(jax.device_get(regression_forward(unflatten_policy_vector(b, policy_spec), grid_x))) for b in bests]
            showcase_payload.append(
                {
                    "support_x": np.asarray(jax.device_get(sx[:, 0])),
                    "support_y": np.asarray(jax.device_get(sy)),
                    "query_x": np.asarray(jax.device_get(qx[:, 0])),
                    "query_y": np.asarray(jax.device_get(qy)),
                    "grid_x": np.asarray(jax.device_get(grid_x[:, 0])),
                    "preds": preds,
                    "support_fitness": np.asarray(jax.device_get(fits)),
                    "query_mse": np.asarray(jax.device_get(qmses)),
                    "candidate_grid_preds": np.asarray(
                        jax.device_get(jax.vmap(lambda cand: regression_forward(unflatten_policy_vector(cand, policy_spec), grid_x))(final_pop))
                    ),
                    "candidate_support_fitness": np.asarray(jax.device_get(final_fit)),
                }
            )
    else:
        raise ValueError(f"Unsupported condition search object: {cond.search_object}")

    pre = curves_np[:, 0]
    post = curves_np[:, -1]
    improvement = pre - post
    rel_improvement = improvement / np.maximum(pre, 1e-8)

    summary = {
        "condition_name": condition_name,
        "search_object": cond.search_object,
        "outer_optimizer": cond.outer_optimizer,
        "inner_optimizer": cond.inner_optimizer,
        "outer_evosax_algo": cond.outer_evosax_algo,
        "inner_evosax_algo": cond.inner_evosax_algo,
        "pre_mse_mean": float(np.mean(pre)),
        "post_mse_mean": float(np.mean(post)),
        "abs_improvement_mean": float(np.mean(improvement)),
        "rel_improvement_mean": float(np.mean(rel_improvement)),
    }

    return {
        "summary": summary,
        "curves": curves_np,
        "showcase": showcase_payload,
    }



def _plot_adaptation_curve(path: Path, curves: np.ndarray, title: str) -> None:
    xs = np.arange(curves.shape[1])
    mean = curves.mean(axis=0)
    stderr = curves.std(axis=0) / math.sqrt(max(curves.shape[0], 1))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, mean)
    ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)
    ax.set_xlabel("Inner generation")
    ax.set_ylabel("Query MSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def _plot_pre_post_scatter(path: Path, curves: np.ndarray, title: str) -> None:
    pre = curves[:, 0]
    post = curves[:, -1]
    lo = min(pre.min(), post.min())
    hi = max(pre.max(), post.max())
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(pre, post, alpha=0.7, s=18)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("Pre-adaptation query MSE")
    ax.set_ylabel("Post-adaptation query MSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def _plot_improvement_ecdf(path: Path, curves: np.ndarray, title: str) -> None:
    rel = (curves[:, 0] - curves[:, -1]) / np.maximum(curves[:, 0], 1e-8)
    xs = np.sort(rel)
    ys = np.linspace(0.0, 1.0, len(xs), endpoint=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(xs, ys)
    ax.set_xlabel("Relative improvement")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def _plot_qualitative_steps(path: Path, showcase: list[dict[str, Any]], title: str) -> None:
    rows = len(showcase)
    cols = len(showcase[0]["preds"])
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False)
    for r, item in enumerate(showcase):
        grid_x = item["grid_x"]
        order = np.argsort(item["query_x"])
        true_y = np.interp(grid_x, item["query_x"][order], item["query_y"][order])
        for c in range(cols):
            ax = axes[r, c]
            ax.plot(grid_x, true_y, linestyle="--", linewidth=1)
            ax.plot(grid_x, item["preds"][c], linewidth=1.5)
            ax.scatter(item["support_x"], item["support_y"], s=16)
            ax.set_title(f"Task {r+1}, step {c}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def _plot_candidate_ensemble(path: Path, showcase_item: dict[str, Any], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    grid_x = showcase_item["grid_x"]
    ensemble = showcase_item["candidate_grid_preds"]
    fit = showcase_item["candidate_support_fitness"]
    order = np.argsort(fit)
    for idx in order:
        ax.plot(grid_x, ensemble[idx], alpha=0.2, linewidth=1.0)
    ax.scatter(showcase_item["support_x"], showcase_item["support_y"], s=20)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def _plot_dynamics(path: Path, showcase_item: dict[str, Any], title: str) -> None:
    if "mutation_rate_mean" not in showcase_item:
        return
    xs = np.arange(len(showcase_item["mutation_rate_mean"]))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].plot(xs, showcase_item["mutation_rate_mean"])
    axes[0].set_title("Mutation rate")
    axes[1].plot(xs, showcase_item["mutation_block_fraction"])
    axes[1].set_title("Block fraction")
    axes[2].plot(xs, showcase_item["query_mse"])
    axes[2].set_title("Query MSE")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time evaluation + visualizations for paired-loop meta-sine runs.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default="meta_sine_eval_figures")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--eval-tasks", type=int, default=512)
    parser.add_argument("--showcase-tasks", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--plot-points", type=int, default=256)
    parser.add_argument("--focal-condition", default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    training_payload = _load_training_payload(results_path)
    available = list(training_payload["results"].keys())
    selected = args.conditions if args.conditions else available
    focal = args.focal_condition or (selected[0] if selected else None)

    summary: dict[str, Any] = {}
    eval_payload: dict[str, Any] = {}

    for name in selected:
        if name not in training_payload["results"]:
            raise ValueError(f"Condition '{name}' not found in results file. Available: {available}")
        result_payload = get_or_reconstruct_condition(name, training_payload, cache_dir)
        evaluated = evaluate_condition(
            name,
            result_payload,
            eval_tasks=args.eval_tasks,
            showcase_tasks=args.showcase_tasks,
            eval_seed=args.eval_seed,
            plot_points=args.plot_points,
        )
        eval_payload[name] = evaluated
        summary[name] = evaluated["summary"]

        _plot_adaptation_curve(output_dir / f"adaptation_curve_{name}.png", evaluated["curves"], f"Held-out adaptation: {name}")
        _plot_pre_post_scatter(output_dir / f"pre_post_scatter_{name}.png", evaluated["curves"], f"Pre vs post: {name}")
        _plot_improvement_ecdf(output_dir / f"improvement_ecdf_{name}.png", evaluated["curves"], f"Improvement ECDF: {name}")
        _plot_qualitative_steps(output_dir / f"qualitative_steps_{name}.png", evaluated["showcase"], f"Step-by-step fits: {name}")

        if name == focal and evaluated["showcase"]:
            _plot_candidate_ensemble(output_dir / f"candidate_ensemble_{name}.png", evaluated["showcase"][0], f"Candidate ensemble: {name}")
            _plot_dynamics(output_dir / f"mutation_dynamics_{name}.png", evaluated["showcase"][0], f"Inner-loop dynamics: {name}")

    with (output_dir / "evaluation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    _save_pickle(output_dir / "evaluation_payload.pkl", eval_payload)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
