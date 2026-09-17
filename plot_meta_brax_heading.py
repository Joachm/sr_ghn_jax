"""Create a publication-quality Meta Brax Heading training/evaluation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DATA_DIR = Path("data/meta_brax_heading_wandb")
DEFAULT_TRAINING_INPUT = DATA_DIR / "train_fitness_best_mean_std.csv"
DEFAULT_HELDOUT_INPUT = DATA_DIR / "heldout_scores_by_run.csv"
DEFAULT_OUTPUT_DIR = Path("figures/meta_brax_heading")
HELDOUT_METRIC = "final/heldout_curve_post_return_mean"
METHODS = (
    ("SimpleGA", "SimpleGA", "#CC79A7"),
    ("GESMR-GA", "GESMR-GA", "#E69F00"),
    ("SAMR-GA", "SAMR-GA", "#009E73"),
    ("Sep-CMA-ES", "Sep-CMA-ES", "#D55E00"),
    ("sr-ghn", "SR-GHN", "#0072B2"),
)
Y_LIMITS = (-200, 3_700)


def configure_style() -> None:
    """Set consistent vector-friendly styling for a paper figure."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.labelcolor": "#202020",
            "xtick.color": "#353535",
            "ytick.color": "#353535",
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_data(training_path: Path, heldout_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source tables and enforce their five-method/five-run structure."""
    training = pd.read_csv(training_path)
    heldout = pd.read_csv(heldout_path)
    expected = {method for method, _, _ in METHODS}
    training_columns = {"method", "step", "n_runs", "train_fitness_best_mean", "train_fitness_best_std"}
    heldout_columns = {"method", "seed", HELDOUT_METRIC}

    if missing := training_columns.difference(training.columns):
        raise ValueError(f"{training_path} is missing columns: {sorted(missing)}")
    if missing := heldout_columns.difference(heldout.columns):
        raise ValueError(f"{heldout_path} is missing columns: {sorted(missing)}")
    if set(training["method"].unique()) != expected or set(heldout["method"].unique()) != expected:
        raise ValueError("Input tables do not contain the expected comparable method set.")
    if not training["n_runs"].eq(5).all() or not heldout.groupby("method").size().eq(5).all():
        raise ValueError("Every method must contain exactly five repetitions.")
    return training, heldout


def style_axis(axis: plt.Axes) -> None:
    """Apply clean, low-ink axes suitable for publication."""
    axis.set_ylim(*Y_LIMITS)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.78)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(length=3.2, width=0.8)


def make_figure(training: pd.DataFrame, heldout: pd.DataFrame) -> plt.Figure:
    """Render mean training curves beside all held-out evaluation samples."""
    configure_style()
    figure, (training_axis, heldout_axis) = plt.subplots(
        1,
        2,
        figsize=(8.25, 3.9),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    figure.set_constrained_layout_pads(h_pad=0.08, hspace=0.04, w_pad=0.06, wspace=0.06)

    # All uncertainty bands are below every mean curve for clear comparisons.
    for method_index, (method, _, color) in enumerate(METHODS):
        series = training.loc[training["method"] == method]
        training_axis.fill_between(
            series["step"],
            series["train_fitness_best_mean"] - series["train_fitness_best_std"],
            series["train_fitness_best_mean"] + series["train_fitness_best_std"],
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=method_index + 1,
        )
    for method_index, (method, label, color) in enumerate(METHODS):
        series = training.loc[training["method"] == method]
        training_axis.plot(
            series["step"],
            series["train_fitness_best_mean"],
            color=color,
            linewidth=2.35 if method == "sr-ghn" else 1.75,
            label=label,
            solid_capstyle="round",
            zorder=10 + method_index,
        )

    training_axis.set_title("(a) Training", loc="left", pad=5)
    training_axis.set_xlabel("Evolution step")
    training_axis.set_ylabel("Return")
    training_axis.set_xlim(left=0)
    training_axis.margins(x=0)
    training_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    style_axis(training_axis)
    training_axis.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.55,
    )

    positions = np.arange(1, len(METHODS) + 1)
    values = [heldout.loc[heldout["method"] == method, HELDOUT_METRIC].to_numpy() for method, _, _ in METHODS]
    boxplot = heldout_axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.35},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"edgecolor": "#555555", "linewidth": 1.0},
    )
    for patch, (_, _, color) in zip(boxplot["boxes"], METHODS, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)

    # Fixed offsets make all five observations visible and deterministic.
    offsets = np.array([-0.14, -0.07, 0.0, 0.07, 0.14])
    for position, method_values, (_, _, color) in zip(positions, values, METHODS, strict=True):
        heldout_axis.scatter(
            position + offsets,
            method_values,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )

    heldout_axis.set_title("(b) Evaluation", loc="left", pad=5)
    heldout_axis.set_xticks(positions, [label for _, label, _ in METHODS], rotation=22, ha="right")
    heldout_axis.set_xlim(0.45, len(METHODS) + 0.55)
    style_axis(heldout_axis)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-input", type=Path, default=DEFAULT_TRAINING_INPUT)
    parser.add_argument("--heldout-input", type=Path, default=DEFAULT_HELDOUT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training, heldout = load_data(args.training_input, args.heldout_input)
    figure = make_figure(training, heldout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "meta_brax_heading_comparison"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
