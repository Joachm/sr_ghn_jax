"""Export five-seed classic-control W&B histories into plot-ready CSV files."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "joachimwp"
METRIC = "fitness_best"
ENVIRONMENTS = (
    "Acrobot-v1",
    "Pendulum-v1",
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
)
PROJECTS = {
    "OpenES": "sr-ghn_control_open_es",
    "CMA-ES": "sr-ghn_control_cma_es",
    "SAMR-GA": "sr-ghn_control_samr_ga",
    "GESMR": "sr-ghn_control_gesmr_ga",
    "sr-ghn": "srghn-gymnax12",
}
OUTPUT_DIR = Path("data/classic_control_wandb")
RUN_NAME = re.compile(r"^(?P<environment>.+)-seed(?P<seed>\d+)$")


def selected_runs(api: wandb.Api, method: str, project: str) -> list[tuple[object, str, int]]:
    """Return only seed0 through seed4, validating all requested environments."""
    selected = []
    for run in api.runs(f"{ENTITY}/{project}", order="+created_at"):
        match = RUN_NAME.match(run.name)
        if match is None:
            raise RuntimeError(f"Unexpected run name in {project}: {run.name}")

        environment = match.group("environment")
        seed = int(match.group("seed"))
        if environment not in ENVIRONMENTS:
            raise RuntimeError(f"Unexpected environment in {project}: {environment}")
        if seed < 5:
            selected.append((run, environment, seed))

    expected = {(environment, seed) for environment in ENVIRONMENTS for seed in range(5)}
    actual = {(environment, seed) for _, environment, seed in selected}
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise RuntimeError(f"{method} run selection mismatch; missing={missing}, extra={extra}")
    return selected


def export() -> None:
    """Download raw histories and their per-environment five-seed summaries."""
    api = wandb.Api(timeout=60)
    points = []
    manifest_runs = []

    for method, project in PROJECTS.items():
        for run, environment, seed in selected_runs(api, method, project):
            run_points = []
            for row in run.scan_history(keys=[METRIC], page_size=1_000):
                value = row.get(METRIC)
                step = row.get("_step")
                if value is not None and step is not None:
                    run_points.append(
                        {
                            "environment": environment,
                            "method": method,
                            "project": project,
                            "run_id": run.id,
                            "run_name": run.name,
                            "seed": seed,
                            "step": int(step),
                            METRIC: float(value),
                        }
                    )
            if not run_points:
                raise RuntimeError(f"{project}/{run.id} has no {METRIC} history.")

            points.extend(run_points)
            manifest_runs.append(
                {
                    "environment": environment,
                    "method": method,
                    "project": project,
                    "run_id": run.id,
                    "run_name": run.name,
                    "seed": seed,
                    "url": run.url,
                    "points": len(run_points),
                    "first_step": run_points[0]["step"],
                    "last_step": run_points[-1]["step"],
                }
            )

    raw = pd.DataFrame(points).sort_values(["environment", "method", "seed", "step"])
    if raw.duplicated(["environment", "method", "seed", "step"]).any():
        raise RuntimeError("At least one run logged fitness_best more than once per step.")

    summary = (
        raw.groupby(["environment", "method", "project", "step"])[METRIC]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": f"{METRIC}_mean",
                "std": f"{METRIC}_std",
            }
        )
        .sort_values(["environment", "method", "step"])
    )
    if not summary["n_runs"].eq(5).all():
        raise RuntimeError("Runs do not share identical logged steps within a method/environment.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_DIR / "fitness_best_by_run.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fitness_best_mean_std.csv", index=False)
    manifest = {
        "entity": ENTITY,
        "metric": METRIC,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "environments": ENVIRONMENTS,
        "methods": PROJECTS,
        "seed_selection": "seed0 through seed4 inclusive for every method",
        "standard_deviation": "sample (ddof=1)",
        "runs": manifest_runs,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# Classic-control W&B export\n\n"
        "This directory contains `fitness_best` histories for five seeds of each\n"
        "method/environment combination. The SR-GHN source project has ten seeds,\n"
        "but only `seed0` through `seed4` are included to match the baselines.\n\n"
        "- `fitness_best_by_run.csv`: source metric points, one row per run step.\n"
        "- `fitness_best_mean_std.csv`: five-seed mean and sample standard deviation\n"
        "  per environment, method, and step.\n"
        "- `manifest.json`: project mapping, retained run IDs, and source URLs.\n",
        encoding="utf-8",
    )

    print(f"Exported {len(raw):,} raw points across {len(manifest_runs)} runs.")
    print(f"Raw histories: {OUTPUT_DIR / 'fitness_best_by_run.csv'}")
    print(f"Mean and std:  {OUTPUT_DIR / 'fitness_best_mean_std.csv'}")


if __name__ == "__main__":
    export()
