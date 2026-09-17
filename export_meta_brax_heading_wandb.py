"""Export comparable Meta Brax Heading runs into plot-ready CSV tables."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "joachimwp"
PROJECT = "meta_brax_heading"
TRAIN_METRIC = "train/fitness_best"
HELDOUT_METRIC = "final/heldout_curve_post_return_mean"
OUTPUT_DIR = Path("data/meta_brax_heading_wandb")
COMPARABLE_CONFIG = {
    "outer_generations": 1_500,
    "inner_pop_size": 2,
    "outer_pop_size": 42,
    "meta_batch_size": 12,
}
METHOD_LABELS = {
    "srghn_full": "sr-ghn",
    "simple_ga_simple_ga": "SimpleGA",
    "samr_ga_samr_ga": "SAMR-GA",
    "gesmr_ga_gesmr_ga": "GESMR-GA",
    "sep_cma_es_sep_cma_es": "Sep-CMA-ES",
}
EXPECTED_SEEDS = set(range(5))


def is_comparable(run: object) -> bool:
    """Return whether a run uses the final shared experiment budget."""
    return all(run.config.get(key) == value for key, value in COMPARABLE_CONFIG.items())


def select_runs(api: wandb.Api) -> tuple[list[object], list[dict[str, object]]]:
    """Select one latest successful run for every method/seed combination."""
    candidates: dict[str, dict[int, list[object]]] = {}
    excluded = []

    for run in api.runs(f"{ENTITY}/{PROJECT}", order="+created_at"):
        condition = run.config.get("condition", {})
        method_id = condition.get("name")
        seed = run.config.get("seed")
        if not is_comparable(run):
            continue
        if run.state != "finished" or TRAIN_METRIC not in run.summary or HELDOUT_METRIC not in run.summary:
            excluded.append(
                {
                    "run_id": run.id,
                    "method_id": method_id,
                    "seed": seed,
                    "state": run.state,
                    "reason": "not finished with both requested metrics",
                }
            )
            continue
        if method_id not in METHOD_LABELS:
            excluded.append(
                {
                    "run_id": run.id,
                    "method_id": method_id,
                    "seed": seed,
                    "state": run.state,
                    "reason": "method lacks five completed comparable seeds",
                }
            )
            continue
        candidates.setdefault(method_id, {}).setdefault(seed, []).append(run)

    selected = []
    for method_id, by_seed in candidates.items():
        if set(by_seed) != EXPECTED_SEEDS:
            raise RuntimeError(
                f"{method_id} does not provide exactly seeds 0 through 4: {sorted(by_seed)}"
            )
        for seed, runs in by_seed.items():
            # Duplicated seed-zero runs have identical configurations; use the
            # latest completed run, matching the final repeated experiment set.
            run = max(runs, key=lambda item: item.created_at)
            selected.append(run)
            for duplicate in runs:
                if duplicate.id != run.id:
                    excluded.append(
                        {
                            "run_id": duplicate.id,
                            "method_id": method_id,
                            "seed": seed,
                            "state": duplicate.state,
                            "reason": "superseded duplicate of selected seed",
                        }
                    )

    if len(selected) != len(METHOD_LABELS) * len(EXPECTED_SEEDS):
        raise RuntimeError(f"Expected 25 selected runs; found {len(selected)}.")
    return selected, excluded


def export() -> None:
    """Write raw training histories, aggregates, and held-out evaluations."""
    api = wandb.Api(timeout=60)
    runs, excluded = select_runs(api)
    training_rows = []
    heldout_rows = []
    manifest_runs = []

    for run in runs:
        method_id = run.config["condition"]["name"]
        method = METHOD_LABELS[method_id]
        seed = int(run.config["seed"])
        points = []
        for row in run.scan_history(keys=[TRAIN_METRIC], page_size=1_000):
            value = row.get(TRAIN_METRIC)
            step = row.get("_step")
            if value is not None and step is not None:
                points.append(
                    {
                        "method": method,
                        "method_id": method_id,
                        "run_id": run.id,
                        "run_name": run.name,
                        "seed": seed,
                        "step": int(step),
                        TRAIN_METRIC: float(value),
                    }
                )
        if not points:
            raise RuntimeError(f"{run.id} has no {TRAIN_METRIC} history.")

        training_rows.extend(points)
        heldout_rows.append(
            {
                "method": method,
                "method_id": method_id,
                "run_id": run.id,
                "run_name": run.name,
                "seed": seed,
                HELDOUT_METRIC: float(run.summary[HELDOUT_METRIC]),
            }
        )
        manifest_runs.append(
            {
                "method": method,
                "method_id": method_id,
                "run_id": run.id,
                "run_name": run.name,
                "seed": seed,
                "url": run.url,
                "training_points": len(points),
                "first_step": points[0]["step"],
                "last_step": points[-1]["step"],
            }
        )

    training = pd.DataFrame(training_rows).sort_values(["method", "seed", "step"])
    if training.duplicated(["method", "seed", "step"]).any():
        raise RuntimeError("At least one selected run logged train/fitness_best twice per step.")
    training_summary = (
        training.groupby(["method", "method_id", "step"])[TRAIN_METRIC]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": "train_fitness_best_mean",
                "std": "train_fitness_best_std",
            }
        )
        .sort_values(["method", "step"])
    )
    if not training_summary["n_runs"].eq(5).all():
        raise RuntimeError("Selected training runs do not share identical logged steps.")

    heldout = pd.DataFrame(heldout_rows).sort_values(["method", "seed"])
    heldout_summary = (
        heldout.groupby(["method", "method_id"])[HELDOUT_METRIC]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": "heldout_mean",
                "std": "heldout_std",
                "median": "heldout_median",
                "min": "heldout_min",
                "max": "heldout_max",
            }
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training.to_csv(OUTPUT_DIR / "train_fitness_best_by_run.csv", index=False)
    training_summary.to_csv(OUTPUT_DIR / "train_fitness_best_mean_std.csv", index=False)
    heldout.to_csv(OUTPUT_DIR / "heldout_scores_by_run.csv", index=False)
    heldout_summary.to_csv(OUTPUT_DIR / "heldout_scores_summary.csv", index=False)

    manifest = {
        "entity": ENTITY,
        "project": PROJECT,
        "training_metric": TRAIN_METRIC,
        "heldout_metric": HELDOUT_METRIC,
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "selection_config": COMPARABLE_CONFIG,
        "selection_policy": "latest finished run per method and seed 0 through 4",
        "standard_deviation": "sample (ddof=1)",
        "method_labels": METHOD_LABELS,
        "selected_runs": manifest_runs,
        "excluded_comparable_runs": excluded,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(
        "# Meta Brax Heading W&B export\n\n"
        "The export contains the comparable final-budget runs in `meta_brax_heading`: five\n"
        "seeds (`0` through `4`) for each of SR-GHN, SimpleGA, SAMR-GA, GESMR-GA, and\n"
        "Sep-CMA-ES. `srghn_full` is labelled `sr-ghn`.\n\n"
        "- `train_fitness_best_by_run.csv`: training source points per selected run.\n"
        "- `train_fitness_best_mean_std.csv`: per-method training means and sample standard deviations.\n"
        "- `heldout_scores_by_run.csv`: one final held-out score per selected run; direct box-plot input.\n"
        "- `heldout_scores_summary.csv`: descriptive statistics for those box plots.\n"
        "- `manifest.json`: run provenance and excluded/incomplete-run decisions.\n",
        encoding="utf-8",
    )
    print(f"Exported {len(training):,} training points across {len(runs)} runs.")
    print(f"Exported {len(heldout):,} held-out evaluation scores.")


if __name__ == "__main__":
    export()
