from __future__ import annotations

from dataclasses import asdict
from math import sqrt
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp

from envs import iter_shift_windows


def _to_host(value):
    return jax.device_get(value)


def comparison_label(config) -> str:
    if getattr(config, "optimizer_family", "srghn") == "evosax":
        algo = getattr(config, "evosax_algo", None) or "unknown"
        return f"evosax:{algo}"
    return config.baseline_name


def summarize_shift_window(
    fitness_series: jnp.ndarray,
    shift_start: int,
    shift_end: int | None,
    *,
    thresholds: tuple[float, ...] = (0.5, 0.9, 1.0),
) -> dict[str, Any]:
    n = int(fitness_series.shape[0])
    start = max(0, min(int(shift_start), n - 1))
    pre_idx = max(start - 1, 0)
    end_idx = n - 1 if shift_end is None else max(start, min(int(shift_end), n - 1))
    pre_shift = float(fitness_series[pre_idx])
    post_window = fitness_series[start : end_idx + 1]
    post_min = float(jnp.min(post_window))
    post_final = float(post_window[-1])
    drop = pre_shift - post_min
    area = float(jnp.sum(post_window))

    recovery = {}
    for threshold in thresholds:
        target = pre_shift * float(threshold)
        mask = post_window >= target
        if bool(jnp.any(mask)):
            offset = int(jnp.argmax(mask))
            recovery[f"recover_{int(threshold * 100)}"] = start + offset
        else:
            recovery[f"recover_{int(threshold * 100)}"] = None

    return {
        "shift_start": start,
        "shift_end": shift_end,
        "pre_shift_fitness": pre_shift,
        "post_shift_min_fitness": post_min,
        "post_shift_final_fitness": post_final,
        "immediate_drop": drop,
        "post_shift_area": area,
        **recovery,
    }


def summarize_run(metrics: dict[str, Any], config) -> dict[str, Any]:
    fitness_best = jnp.asarray(metrics["fitness_best"])
    fitness_mean = jnp.asarray(metrics["fitness_mean"])
    shift_summaries = []
    for window in iter_shift_windows(config):
        shift_summaries.append(
            {
                "rule": window.rule,
                "target_value": window.target_value,
                "best": summarize_shift_window(fitness_best, window.start_gen, window.end_gen),
                "mean": summarize_shift_window(fitness_mean, window.start_gen, window.end_gen),
            }
        )
    return {
        "baseline_name": comparison_label(config),
        "env_id": config.env_id,
        "shift_windows": [asdict(window) for window in iter_shift_windows(config)],
        "shift_summaries": shift_summaries,
    }


def build_run_artifact(config, metrics: dict[str, Any]) -> dict[str, Any]:
    host_metrics = _to_host(metrics)
    return {
        "config": config,
        "metrics": host_metrics,
        "summary": summarize_run(host_metrics, config),
    }


def save_pickle(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _aggregate_series(series_list: list[jnp.ndarray]) -> dict[str, Any]:
    stacked = jnp.stack(series_list)
    n = max(stacked.shape[0], 1)
    return {
        "mean": _to_host(jnp.mean(stacked, axis=0)),
        "stderr": _to_host(jnp.std(stacked, axis=0) / sqrt(n)),
    }


def aggregate_comparison_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (comparison_label(result["config"]), result["config"].env_id)
        grouped.setdefault(key, []).append(result)

    aggregated = {}
    for (baseline_name, env_id), runs in grouped.items():
        metric_keys = runs[0]["metrics"].keys()
        metrics = {
            key: _aggregate_series([jnp.asarray(run["metrics"][key]) for run in runs])
            for key in metric_keys
        }
        aggregated[(baseline_name, env_id)] = {
            "num_runs": len(runs),
            "summary": [run["summary"] for run in runs],
            "metrics": metrics,
        }
    return aggregated
