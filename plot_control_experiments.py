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


REQUIRED_METRICS = ("fitness_mean", "fitness_best", "diversity")
METRIC_SPECS = {
    "fitness_mean": {
        "title": "Mean Fitness vs Generation",
        "ylabel": "Fitness",
        "filename": "fitness_mean",
    },
    "fitness_best": {
        "title": "Best Fitness vs Generation",
        "ylabel": "Fitness",
        "filename": "fitness_best",
    },
    "diversity": {
        "title": "Population Diversity vs Generation",
        "ylabel": "Diversity",
        "filename": "diversity",
    },
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f7f7",
            "axes.edgecolor": "#2d2d2d",
            "axes.labelcolor": "#1f1f1f",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d7d7d7",
            "grid.alpha": 0.7,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.color": "#1f1f1f",
            "ytick.color": "#1f1f1f",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _is_mapping(value) -> bool:
    return hasattr(value, "__contains__") and hasattr(value, "__getitem__")


def _normalize_metric_array(value, *, path: Path, metric_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Artifact {path} metric '{metric_name}' must be a 1D per-generation series.")
    return array


def _extract_run_record(path: Path, env_filter: set[str] | None, control_filter: set[str] | None):
    payload = _load_pickle(path)
    if not _is_mapping(payload) or "config" not in payload or "metrics" not in payload:
        return None

    config = payload["config"]
    metrics = payload["metrics"]
    if not _is_mapping(metrics):
        raise ValueError(f"Artifact {path} has an invalid 'metrics' payload; expected a mapping.")
    if not hasattr(config, "env_id") or not hasattr(config, "seed"):
        raise ValueError(f"Artifact {path} config must expose 'env_id' and 'seed'.")

    control_label = comparison_label(config)
    env_id = str(config.env_id)
    if env_filter is not None and env_id not in env_filter:
        return None
    if control_filter is not None and control_label not in control_filter:
        return None

    missing_metrics = [metric_name for metric_name in REQUIRED_METRICS if metric_name not in metrics]
    if missing_metrics:
        missing_list = ", ".join(missing_metrics)
        raise ValueError(f"Artifact {path} is missing required metrics: {missing_list}.")

    series = {
        metric_name: _normalize_metric_array(metrics[metric_name], path=path, metric_name=metric_name)
        for metric_name in REQUIRED_METRICS
    }
    return {
        "path": path,
        "env_id": env_id,
        "seed": int(config.seed),
        "control_label": control_label,
        "metrics": series,
    }


def discover_run_artifacts(
    results_root: str | Path,
    *,
    env_ids: Iterable[str] | None = None,
    controls: Iterable[str] | None = None,
) -> list[dict]:
    root = Path(results_root)
    env_filter = set(env_ids) if env_ids is not None else None
    control_filter = set(controls) if controls is not None else None

    records = []
    for path in sorted(root.rglob("*.pkl")):
        record = _extract_run_record(path, env_filter, control_filter)
        if record is not None:
            records.append(record)

    if not records:
        raise ValueError(
            "No valid run artifacts were found under "
            f"{root}. Expected pickle payloads with 'config', 'metrics', and metrics "
            f"{', '.join(REQUIRED_METRICS)}."
        )
    return records


def aggregate_control_metrics(records: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["env_id"], record["control_label"])].append(record)

    aggregate = {}
    for key, runs in sorted(grouped.items()):
        env_id, control_label = key
        metric_payload = {}
        seeds = sorted(run["seed"] for run in runs)
        for metric_name in REQUIRED_METRICS:
            arrays = [run["metrics"][metric_name] for run in runs]
            expected_length = arrays[0].shape[0]
            for run, array in zip(runs, arrays):
                if array.shape[0] != expected_length:
                    raise ValueError(
                        "Inconsistent generation lengths for "
                        f"env='{env_id}', control='{control_label}', metric='{metric_name}'. "
                        f"Expected length {expected_length} but found {array.shape[0]} in {run['path']}."
                    )
            stacked = np.stack(arrays, axis=0)
            metric_payload[metric_name] = {
                "mean": np.mean(stacked, axis=0),
                "std": np.std(stacked, axis=0),
                "num_runs": len(runs),
            }

        aggregate[key] = {
            "env_id": env_id,
            "control_label": control_label,
            "num_runs": len(runs),
            "seeds": seeds,
            "metrics": metric_payload,
        }
    return aggregate


def _sorted_env_ids(aggregate: dict[tuple[str, str], dict]) -> list[str]:
    return sorted({env_id for env_id, _ in aggregate})


def _sorted_controls(aggregate: dict[tuple[str, str], dict]) -> list[str]:
    return sorted({control for _, control in aggregate})


def build_control_colors(aggregate: dict[tuple[str, str], dict]) -> dict[str, tuple[float, ...]]:
    controls = _sorted_controls(aggregate)
    cmap = plt.get_cmap("tab10" if len(controls) <= 10 else "tab20")
    return {control: cmap(idx % cmap.N) for idx, control in enumerate(controls)}


def _plot_metric_series(ax, xs: np.ndarray, mean: np.ndarray, std: np.ndarray, label: str, color) -> None:
    ax.plot(xs, mean, label=label, color=color, linewidth=2.0)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.18)


def plot_per_environment(
    aggregate: dict[tuple[str, str], dict],
    output_dir: str | Path,
    control_colors: dict[str, tuple[float, ...]],
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for env_id in _sorted_env_ids(aggregate):
        env_entries = {
            control: aggregate[(env_id, control)]
            for control in _sorted_controls(aggregate)
            if (env_id, control) in aggregate
        }
        for metric_name, spec in METRIC_SPECS.items():
            fig, ax = plt.subplots(figsize=(10, 5.5))
            for control_label, entry in env_entries.items():
                series = entry["metrics"][metric_name]
                xs = np.arange(series["mean"].shape[0])
                label = f"{control_label} (n={entry['num_runs']})"
                _plot_metric_series(
                    ax,
                    xs,
                    series["mean"],
                    series["std"],
                    label,
                    control_colors[control_label],
                )
            ax.set_title(f"{spec['title']} | {env_id}")
            ax.set_xlabel("Generation")
            ax.set_ylabel(spec["ylabel"])
            ax.legend(loc="best")
            fig.tight_layout()
            out_path = output_root / f"{_safe_name(env_id)}_{spec['filename']}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written_files.append(out_path)

    return written_files


def plot_combined_overview(
    aggregate: dict[tuple[str, str], dict],
    output_dir: str | Path,
    control_colors: dict[str, tuple[float, ...]],
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    env_ids = _sorted_env_ids(aggregate)
    controls = _sorted_controls(aggregate)

    n_envs = len(env_ids)
    ncols = 2 if n_envs > 1 else 1
    nrows = int(np.ceil(n_envs / ncols))

    for metric_name, spec in METRIC_SPECS.items():
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7.5 * ncols, 4.8 * nrows),
            sharex=True,
            squeeze=False,
        )
        axes_flat = list(axes.flat)

        for env_idx, env_id in enumerate(env_ids):
            ax = axes_flat[env_idx]
            for control_label in controls:
                entry = aggregate.get((env_id, control_label))
                if entry is None:
                    continue
                series = entry["metrics"][metric_name]
                xs = np.arange(series["mean"].shape[0])
                label = f"{control_label} (n={entry['num_runs']})"
                _plot_metric_series(
                    ax,
                    xs,
                    series["mean"],
                    series["std"],
                    label,
                    control_colors[control_label],
                )
            ax.set_title(env_id)
            ax.set_xlabel("Generation")
            ax.set_ylabel(spec["ylabel"])

        for ax in axes_flat[n_envs:]:
            ax.set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(spec["title"], y=1.06)
        fig.tight_layout()
        out_path = output_root / f"combined_{spec['filename']}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written_files.append(out_path)

    return written_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated control experiment metrics from saved artifacts.")
    parser.add_argument("--results-root", required=True, help="Root directory to scan recursively for pickle run artifacts.")
    parser.add_argument("--output-dir", required=True, help="Directory where plot images will be written.")
    parser.add_argument("--env-id", action="append", default=None, help="Optional environment id filter. Repeat to include multiple.")
    parser.add_argument("--control", action="append", default=None, help="Optional control label filter. Repeat to include multiple.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _set_plot_style()
    try:
        records = discover_run_artifacts(args.results_root, env_ids=args.env_id, controls=args.control)
        aggregate = aggregate_control_metrics(records)
        control_colors = build_control_colors(aggregate)
        plot_per_environment(aggregate, args.output_dir, control_colors)
        plot_combined_overview(aggregate, args.output_dir, control_colors)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
