from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptation_analysis import comparison_label
from plot_control_experiments import _set_plot_style


TARGET_METRIC = "fitness_best"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _is_mapping(value) -> bool:
    return hasattr(value, "__contains__") and hasattr(value, "__getitem__")


def _normalize_metric_array(value, *, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Artifact {path} metric '{TARGET_METRIC}' must be a 1D per-update series.")
    return array


def _extract_ppo_run_record(path: Path, env_filter: set[str] | None):
    payload = _load_pickle(path)
    if not _is_mapping(payload) or "config" not in payload or "metrics" not in payload:
        return None

    config = payload["config"]
    metrics = payload["metrics"]
    if not _is_mapping(metrics):
        raise ValueError(f"Artifact {path} has an invalid 'metrics' payload; expected a mapping.")
    if not hasattr(config, "env_id") or not hasattr(config, "seed"):
        raise ValueError(f"Artifact {path} config must expose 'env_id' and 'seed'.")

    if comparison_label(config) != "ppo":
        return None

    env_id = str(config.env_id)
    if env_filter is not None and env_id not in env_filter:
        return None

    if TARGET_METRIC not in metrics:
        raise ValueError(f"Artifact {path} is missing required metric: {TARGET_METRIC}.")

    return {
        "path": path,
        "env_id": env_id,
        "seed": int(config.seed),
        "fitness": _normalize_metric_array(metrics[TARGET_METRIC], path=path),
    }


def discover_ppo_run_artifacts(
    results_root: str | Path,
    *,
    env_ids: Iterable[str] | None = None,
) -> list[dict]:
    root = Path(results_root)
    env_filter = set(env_ids) if env_ids is not None else None

    records = []
    for path in sorted(root.rglob("*.pkl")):
        record = _extract_ppo_run_record(path, env_filter)
        if record is not None:
            records.append(record)

    if not records:
        raise ValueError(
            f"No valid PPO run artifacts were found under {root}. "
            f"Expected pickle payloads with 'config', 'metrics', and metric '{TARGET_METRIC}'."
        )
    return records


def aggregate_ppo_metrics(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["env_id"]].append(record)

    aggregate = {}
    for env_id, runs in sorted(grouped.items()):
        arrays = [run["fitness"] for run in runs]
        expected_length = arrays[0].shape[0]
        for run, array in zip(runs, arrays):
            if array.shape[0] != expected_length:
                raise ValueError(
                    f"Inconsistent update lengths for env='{env_id}'. "
                    f"Expected length {expected_length} but found {array.shape[0]} in {run['path']}."
                )
        stacked = np.stack(arrays, axis=0)
        aggregate[env_id] = {
            "env_id": env_id,
            "num_runs": len(runs),
            "seeds": sorted(run["seed"] for run in runs),
            "fitness": {
                "mean": np.mean(stacked, axis=0),
                "std": np.std(stacked, axis=0),
                "num_runs": len(runs),
            },
        }
    return aggregate


def _plot_fitness_series(ax, xs: np.ndarray, mean: np.ndarray, std: np.ndarray, label: str, color) -> None:
    ax.plot(xs, mean, label=label, color=color, linewidth=2.0)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)


def plot_per_environment(aggregate: dict[str, dict], output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    color = plt.get_cmap("tab10")(0)

    for env_id, entry in aggregate.items():
        series = entry["fitness"]
        xs = np.arange(series["mean"].shape[0])
        fig, ax = plt.subplots(figsize=(10, 5.5))
        _plot_fitness_series(ax, xs, series["mean"], series["std"], f"PPO (n={entry['num_runs']})", color)
        ax.set_title(f"Fitness vs Update | {env_id}")
        ax.set_xlabel("Update")
        ax.set_ylabel("Fitness")
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = output_root / f"{_safe_name(env_id)}_fitness.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written_files.append(out_path)

    return written_files


def plot_combined_overview(aggregate: dict[str, dict], output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    env_ids = list(sorted(aggregate))
    n_envs = len(env_ids)
    ncols = 2 if n_envs > 1 else 1
    nrows = int(np.ceil(n_envs / ncols))
    color = plt.get_cmap("tab10")(0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.5 * ncols, 4.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = list(axes.flat)

    for idx, env_id in enumerate(env_ids):
        ax = axes_flat[idx]
        entry = aggregate[env_id]
        series = entry["fitness"]
        xs = np.arange(series["mean"].shape[0])
        _plot_fitness_series(ax, xs, series["mean"], series["std"], f"PPO (n={entry['num_runs']})", color)
        ax.set_title(env_id)
        ax.set_xlabel("Update")
        ax.set_ylabel("Fitness")

    for ax in axes_flat[n_envs:]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("PPO Fitness", y=1.04)
    fig.tight_layout()
    out_path = output_root / "combined_fitness.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated PPO experiment fitness from saved artifacts.")
    parser.add_argument("--results-root", required=True, help="Root directory to scan recursively for pickle PPO artifacts.")
    parser.add_argument("--output-dir", required=True, help="Directory where plot images will be written.")
    parser.add_argument("--env-id", action="append", default=None, help="Optional environment id filter. Repeat to include multiple.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _set_plot_style()
    try:
        records = discover_ppo_run_artifacts(args.results_root, env_ids=args.env_id)
        aggregate = aggregate_ppo_metrics(records)
        plot_per_environment(aggregate, args.output_dir)
        plot_combined_overview(aggregate, args.output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
