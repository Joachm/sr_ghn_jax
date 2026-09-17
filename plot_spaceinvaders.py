"""Create the SpaceInvaders fitness comparison figure for publication use.

The figure shows the mean best fitness over five seeds for each method, with
translucent bands representing one sample standard deviation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd


DEFAULT_INPUT = Path("data/spaceinvaders_wandb/fitness_best_mean_std.csv")
DEFAULT_OUTPUT_DIR = Path("figures/spaceinvaders")
METHODS = (
    ("sr-ghn", "SR-GHN", "#0072B2"),
    ("CMA-ES", "CMA-ES", "#D55E00"),
    ("SAMR-GA", "SAMR-GA", "#009E73"),
    ("OpenES", "OpenES", "#CC79A7"),
    ("GESMR", "GESMR", "#E69F00"),
)


def thousands(value: float, _: int) -> str:
    """Format evolution steps compactly without sacrificing legibility."""
    if value == 0:
        return "0"
    return f"{value / 1_000:g}k"


def configure_style() -> None:
    """Use a restrained, vector-friendly style suitable for paper figures."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#202020",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#353535",
            "ytick.color": "#353535",
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def load_summary(path: Path) -> pd.DataFrame:
    """Read and validate the exported per-step five-seed summary."""
    required = {
        "method",
        "step",
        "n_runs",
        "fitness_best_mean",
        "fitness_best_std",
    }
    data = pd.read_csv(path)
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    expected = {method for method, _, _ in METHODS}
    unexpected = set(data["method"].unique()).difference(expected)
    if unexpected:
        raise ValueError(f"Unexpected methods in {path}: {sorted(unexpected)}")
    if set(data["method"].unique()) != expected:
        raise ValueError("The summary must contain all five comparison methods.")
    if not data["n_runs"].eq(5).all():
        raise ValueError("Every summary row must contain exactly five runs.")

    return data.sort_values(["method", "step"])


def make_figure(data: pd.DataFrame) -> plt.Figure:
    """Render mean curves and plus/minus one-standard-deviation bands."""
    configure_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.45), layout="constrained")
    figure.set_constrained_layout_pads(h_pad=0.06, hspace=0.04, w_pad=0.06, wspace=0.04)

    for method, label, color in METHODS:
        series = data.loc[data["method"] == method]
        x = series["step"]
        mean = series["fitness_best_mean"]
        std = series["fitness_best_std"]
        line_width = 2.5 if method == "sr-ghn" else 1.9

        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=line_width,
            label=label,
            solid_capstyle="round",
            zorder=2,
        )

    axis.set_xlabel("Evolution step")
    axis.set_ylabel("Best fitness")
    axis.set_xlim(left=0)
    axis.margins(x=0, y=0.04)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(thousands))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(length=3.5, width=0.8)
    axis.text(
        0.01,
        0.98,
        "SpaceInvaders (MinAtar)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#202020",
    )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.56, 1.17),
        ncol=5,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_summary(args.input)
    figure = make_figure(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_dir / "spaceinvaders_fitness_comparison"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)

    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
