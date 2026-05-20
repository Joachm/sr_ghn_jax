from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_PROJECT = "WalkerWalk"
TARGET_METRIC = "fitness_best"


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _resolve_project_path(entity: str | None, project: str, project_path: str | None) -> str:
    if project_path:
        if "/" not in project_path:
            raise ValueError(
                "--project-path must be in the form 'entity/project' so the W&B API can resolve it."
            )
        return project_path

    if not entity:
        entity = os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError(
            "Missing W&B entity. Provide --entity, set WANDB_ENTITY, or pass --project-path entity/project."
        )
    return f"{entity}/{project}"


def _load_wandb_api():
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("wandb is required to read project histories.") from exc

    return wandb.Api()


def _set_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
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


def _coerce_step(value) -> float | None:
    if value is None:
        return None
    try:
        step = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(step):
        return None
    return step


def _extract_series_from_run(run, metric_name: str = TARGET_METRIC) -> dict | None:
    steps: list[float] = []
    values: list[float] = []

    for row in run.scan_history(keys=["_step", metric_name]):
        if metric_name not in row:
            continue
        step = _coerce_step(row.get("_step", row.get("step")))
        if step is None:
            continue
        try:
            value = float(row[metric_name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        steps.append(step)
        values.append(value)

    if not steps:
        return None

    step_array = np.asarray(steps, dtype=float)
    value_array = np.asarray(values, dtype=float)
    order = np.argsort(step_array, kind="stable")
    step_array = step_array[order]
    value_array = value_array[order]

    unique_steps: list[float] = []
    unique_values: list[float] = []
    for step, value in zip(step_array, value_array):
        if unique_steps and step == unique_steps[-1]:
            unique_values[-1] = value
        else:
            unique_steps.append(step)
            unique_values.append(value)

    return {
        "run_id": getattr(run, "id", ""),
        "run_name": getattr(run, "name", getattr(run, "id", "")),
        "steps": np.asarray(unique_steps, dtype=float),
        "values": np.asarray(unique_values, dtype=float),
    }


def fetch_project_series(project_path: str, *, metric_name: str = TARGET_METRIC) -> list[dict]:
    api = _load_wandb_api()
    runs = list(api.runs(project_path, order="+created_at"))
    if not runs:
        raise ValueError(f"No runs were found for W&B project '{project_path}'.")

    records = []
    for run in runs:
        series = _extract_series_from_run(run, metric_name=metric_name)
        if series is not None:
            records.append(series)

    if not records:
        raise ValueError(
            f"None of the runs in '{project_path}' contained metric '{metric_name}' in their history."
        )
    return records


def _build_common_step_grid(series_list: Sequence[dict]) -> np.ndarray:
    steps = [np.asarray(series["steps"], dtype=float) for series in series_list]
    return np.unique(np.concatenate(steps))


def _interpolate_series(steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if steps.size == 1:
        output = np.full(grid.shape, np.nan, dtype=float)
        output[np.isclose(grid, steps[0])] = values[0]
        return output

    output = np.full(grid.shape, np.nan, dtype=float)
    mask = (grid >= steps[0]) & (grid <= steps[-1])
    if np.any(mask):
        output[mask] = np.interp(grid[mask], steps, values)
    return output


def aggregate_project_series(series_list: Sequence[dict]) -> dict:
    if not series_list:
        raise ValueError("At least one run series is required for aggregation.")

    grid = _build_common_step_grid(series_list)
    interpolated = []
    for series in series_list:
        interpolated.append(
            _interpolate_series(
                np.asarray(series["steps"], dtype=float),
                np.asarray(series["values"], dtype=float),
                grid,
            )
        )

    stacked = np.stack(interpolated, axis=0)
    counts = np.sum(np.isfinite(stacked), axis=0)
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("The runs do not overlap on any step values after alignment.")

    mean = np.nanmean(stacked, axis=0)[valid]
    std = np.nanstd(stacked, axis=0)[valid]
    return {
        "steps": grid[valid],
        "mean": mean,
        "std": std,
        "num_runs": len(series_list),
    }


def plot_project_series(
    aggregate: dict,
    *,
    project_path: str,
    metric_name: str = TARGET_METRIC,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(aggregate["steps"], dtype=float)
    mean = np.asarray(aggregate["mean"], dtype=float)
    std = np.asarray(aggregate["std"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    color = plt.get_cmap("tab10")(0)
    label = f"{project_path} (n={aggregate['num_runs']})"
    ax.plot(steps, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_title(f"{metric_name} over training")
    ax.set_xlabel("Step")
    ax.set_ylabel(metric_name)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fitness_best from a W&B project and plot mean/std across runs."
    )
    parser.add_argument(
        "--project",
        "--project-name",
        dest="project",
        default=DEFAULT_PROJECT,
        help="W&B project name. Defaults to WalkerWalk.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (username or team). Defaults to WANDB_ENTITY if set.",
    )
    parser.add_argument(
        "--project-path",
        default=None,
        help="Explicit W&B project path in the form 'entity/project'. Overrides --entity and --project.",
    )
    parser.add_argument(
        "--metric",
        "--metric-name",
        dest="metric",
        default=TARGET_METRIC,
        help="Metric to read from run history. Defaults to fitness_best.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to walkerwalk_fitness_best.png in the current directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _set_plot_style()

    project_path = _resolve_project_path(args.entity, args.project, args.project_path)
    output_path = args.output or f"{_safe_name(args.project)}_{_safe_name(args.metric)}.png"

    series_list = fetch_project_series(project_path, metric_name=args.metric)
    aggregate = aggregate_project_series(series_list)
    plot_project_series(aggregate, project_path=project_path, metric_name=args.metric, output_path=output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
