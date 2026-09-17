"""Export SpaceInvaders W&B histories into plot-ready CSV files.

The export preserves all individual repetitions and also produces the
per-generation mean and sample standard deviation used for comparison plots.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "joachimwp"
METRIC = "fitness_best"
PROJECTS = {
    "sr-ghn": "BiggerSpace",
    "CMA-ES": "SpaceCMA",
    "SAMR-GA": "SpaceSAMRGA",
    "OpenES": "SpaceOpen",
    "GESMR": "SPACE_GESMR",
}
OUTPUT_DIR = Path("data/spaceinvaders_wandb")


def export() -> None:
    """Download each requested run and write raw and aggregated histories."""
    api = wandb.Api(timeout=60)
    series = []
    manifest_runs = []

    for method, project in PROJECTS.items():
        runs = list(api.runs(f"{ENTITY}/{project}", order="+created_at"))
        if len(runs) != 5:
            raise RuntimeError(
                f"Expected five runs in {ENTITY}/{project}; found {len(runs)}."
            )

        for run in runs:
            rows = []
            for row in run.scan_history(page_size=1_000):
                value = row.get(METRIC)
                step = row.get("_step")
                if value is not None and step is not None:
                    rows.append(
                        {
                            "method": method,
                            "project": project,
                            "run_id": run.id,
                            "run_name": run.name,
                            "step": int(step),
                            METRIC: float(value),
                        }
                    )

            if not rows:
                raise RuntimeError(f"{ENTITY}/{project}/{run.id} has no {METRIC} history.")

            series.extend(rows)
            manifest_runs.append(
                {
                    "method": method,
                    "project": project,
                    "run_id": run.id,
                    "run_name": run.name,
                    "url": run.url,
                    "points": len(rows),
                    "first_step": rows[0]["step"],
                    "last_step": rows[-1]["step"],
                }
            )

    raw = pd.DataFrame(series).sort_values(["method", "run_name", "step"])
    # One metric value per W&B step is expected. Fail loudly if that invariant
    # changes, rather than silently averaging duplicate logs within a run.
    duplicate_steps = raw.duplicated(["method", "run_id", "step"])
    if duplicate_steps.any():
        raise RuntimeError("At least one run logged fitness_best more than once per step.")

    summary = (
        raw.groupby(["method", "project", "step"])[METRIC]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": f"{METRIC}_mean",
                "std": f"{METRIC}_std",
            }
        )
        .sort_values(["method", "step"])
    )
    if not (summary["n_runs"] == 5).all():
        raise RuntimeError("Runs do not share an identical set of logged steps.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_DIR / "fitness_best_by_run.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fitness_best_mean_std.csv", index=False)

    manifest = {
        "entity": ENTITY,
        "metric": METRIC,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "methods": PROJECTS,
        "runs": manifest_runs,
        "standard_deviation": "sample (ddof=1)",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Exported {len(raw):,} raw points across {len(manifest_runs)} runs.")
    print(f"Raw histories: {OUTPUT_DIR / 'fitness_best_by_run.csv'}")
    print(f"Mean and std:  {OUTPUT_DIR / 'fitness_best_mean_std.csv'}")


if __name__ == "__main__":
    export()
