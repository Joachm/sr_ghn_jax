from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np


def _load(path: str):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _plot_series(ax, xs, mean, stderr, label):
    ax.plot(xs, mean, label=label)
    ax.fill_between(xs, mean - stderr, mean + stderr, alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated adaptation experiment results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="adaptation_plots")
    args = parser.parse_args()

    payload = _load(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = payload["aggregate"]
    first_entry = next(iter(aggregate.values()))
    fitness_len = len(first_entry["metrics"]["fitness_best"]["mean"])
    xs = np.arange(fitness_len)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (baseline_name, _env_id), entry in aggregate.items():
        mean = np.asarray(entry["metrics"]["fitness_best"]["mean"])
        stderr = np.asarray(entry["metrics"]["fitness_best"]["stderr"])
        _plot_series(ax, xs, mean, stderr, baseline_name)
    ax.set_title("Best Fitness vs Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fitness_vs_generation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (baseline_name, _env_id), entry in aggregate.items():
        mean = np.asarray(entry["metrics"]["population_mutation_rate_mean"]["mean"])
        stderr = np.asarray(entry["metrics"]["population_mutation_rate_mean"]["stderr"])
        _plot_series(ax, xs, mean, stderr, baseline_name)
    ax.set_title("Population Mutation Rate vs Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mutation_rate_vs_generation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (baseline_name, _env_id), entry in aggregate.items():
        mean = np.asarray(entry["metrics"]["population_update_rms_mean"]["mean"])
        stderr = np.asarray(entry["metrics"]["population_update_rms_mean"]["stderr"])
        _plot_series(ax, xs, mean, stderr, baseline_name)
    ax.set_title("Population Update RMS vs Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Update RMS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "update_rms_vs_generation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    baselines = []
    values = []
    for (baseline_name, _env_id), entry in aggregate.items():
        summaries = entry["summary"]
        recovery = []
        for run in summaries:
            for shift_summary in run["shift_summaries"]:
                best_summary = shift_summary["best"]
                if best_summary["recover_90"] is not None:
                    recovery.append(best_summary["recover_90"] - best_summary["shift_start"])
        baselines.append(baseline_name)
        values.append(np.mean(recovery) if recovery else np.nan)
    ax.bar(baselines, values)
    ax.set_title("Mean Recovery Time to 90% Pre-shift Fitness")
    ax.set_ylabel("Generations")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_time_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (baseline_name, _env_id), entry in aggregate.items():
        mean = np.asarray(entry["metrics"]["diversity"]["mean"])
        stderr = np.asarray(entry["metrics"]["diversity"]["stderr"])
        _plot_series(ax, xs, mean, stderr, baseline_name)
    ax.set_title("Population Diversity vs Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "diversity_vs_generation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
