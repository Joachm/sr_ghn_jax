# export_wandb.py

from pathlib import Path
import argparse
import json

import pandas as pd
import wandb


def flatten_dict(d, prefix=""):
    """Flatten nested dictionaries for easy DataFrame storage."""
    out = {}

    for key, value in d.items():
        key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_dict(value, key))
        else:
            out[key] = value

    return out


def export_wandb(
    entity,
    project,
    metrics=None,
    filters=None,
    output_dir="wandb_export",
):
    """
    Export run metadata and full metric histories from a W&B project.

    Returns
    -------
    runs_df : pd.DataFrame
        One row per W&B run. Contains run information, config, and summary.

    history_df : pd.DataFrame
        Long-ish history table. Each row corresponds to one logged W&B step.
    """

    api = wandb.Api(timeout=60)

    path = f"{entity}/{project}"

    runs = api.runs(
        path=path,
        filters=filters,
        order="+created_at",
    )

    metadata_rows = []
    history_frames = []

    for i, run in enumerate(runs):
        print(f"[{i + 1}] {run.name} ({run.id})")

        # ------------------------------------------------------------
        # Run-level information
        # ------------------------------------------------------------
        run_metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            "tags": ",".join(run.tags),
        }

        # W&B config = hyperparameters / experiment settings
        config = {
            f"config.{k}": v
            for k, v in flatten_dict(dict(run.config)).items()
            if not k.startswith("_")
        }

        # W&B summary = final / aggregated metrics
        summary = {
            f"summary.{k}": v
            for k, v in flatten_dict(dict(run.summary)).items()
            if not k.startswith("_")
        }

        metadata_rows.append(
            {
                **run_metadata,
                **config,
                **summary,
            }
        )

        # ------------------------------------------------------------
        # Full time-series history
        # ------------------------------------------------------------
        #
        # Calling scan_history() without metric keys means we don't
        # accidentally lose rows when metrics were logged at different
        # times.
        history = pd.DataFrame(
            run.scan_history(page_size=1000)
        )

        if history.empty:
            continue

        # Keep requested metrics plus useful W&B axes.
        if metrics is not None:
            standard_columns = [
                "_step",
                "_timestamp",
                "_runtime",
            ]

            wanted = standard_columns + list(metrics)
            available = [c for c in wanted if c in history.columns]

            history = history[available]

        # Attach run-level identifiers to every history row.
        history.insert(0, "run_id", run.id)
        history.insert(1, "run_name", run.name)

        # It is often useful to have important hyperparameters directly
        # available when grouping plots.
        for key, value in run.config.items():
            if key.startswith("_"):
                continue

            # Only copy scalar-ish values into every row.
            if isinstance(value, (str, int, float, bool)) or value is None:
                history[f"config.{key}"] = value

        history_frames.append(history)

    # ------------------------------------------------------------------
    # Combine everything
    # ------------------------------------------------------------------

    runs_df = pd.DataFrame(metadata_rows)

    if history_frames:
        history_df = pd.concat(
            history_frames,
            ignore_index=True,
            sort=False,
        )
    else:
        history_df = pd.DataFrame()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "runs.csv"
    history_path = output_dir / "history.csv"

    runs_df.to_csv(runs_path, index=False)
    history_df.to_csv(history_path, index=False)

    print()
    print(f"Saved run metadata -> {runs_path}")
    print(f"Saved history      -> {history_path}")

    return runs_df, history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)

    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to export, e.g. train/loss val/loss val/accuracy",
    )

    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help='JSON W&B run filter, e.g. \'{"state": "finished"}\'',
    )

    parser.add_argument(
        "--output",
        default="wandb_export",
    )

    args = parser.parse_args()

    filters = json.loads(args.filters) if args.filters else None

    export_wandb(
        entity=args.entity,
        project=args.project,
        metrics=args.metrics,
        filters=filters,
        output_dir=args.output,
    )
