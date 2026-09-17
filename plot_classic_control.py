"""Create a publication-quality classic-control fitness comparison figure.

The five panels show mean best fitness over five seeds for each method. Shaded
bands represent plus/minus one sample standard deviation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd


DEFAULT_INPUT = Path("data/classic_control_wandb/fitness_best_mean_std.csv")
DEFAULT_OUTPUT_DIR = Path("figures/classic_control")
METHODS = (
    ("OpenES", "OpenES", "#CC79A7"),
    ("sr-ghn", "SR-GHN", "#0072B2"),
    ("CMA-ES", "CMA-ES", "#D55E00"),
    ("SAMR-GA", "SAMR-GA", "#009E73"),
    ("GESMR", "GESMR", "#E69F00"),
)
PANELS = (
    ("Acrobot-v1", "(a) Acrobot"),
    ("Pendulum-v1", "(b) Pendulum"),
    ("CartPole-v1", "(c) CartPole"),
    ("MountainCar-v0", "(d) MountainCar"),
    ("MountainCarContinuous-v0", "(e) MountainCarContinuous"),
)
Y_LIMITS = {
    "Acrobot-v1": (-525, 0),
    "Pendulum-v1": (-1300, 25),
    "CartPole-v1": (0, 525),
    "MountainCar-v0": (-205, -75),
    "MountainCarContinuous-v0": (-500, 105),
}


def compact_steps(value: float, _: int) -> str:
    """Format evolution steps compactly on the shared horizontal axis."""
    if value == 0:
        return "0"
    return f"{value / 1_000:g}k"


def configure_style() -> None:
    """Set a restrained, paper-friendly visual style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.75,
            "axes.labelcolor": "#202020",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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
    """Read the plot-ready data and enforce the five-seed invariant."""
    required = {
        "environment",
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

    expected_methods = {method for method, _, _ in METHODS}
    expected_environments = {environment for environment, _ in PANELS}
    if set(data["method"].unique()) != expected_methods:
        raise ValueError("The summary must contain exactly the five comparison methods.")
    if set(data["environment"].unique()) != expected_environments:
        raise ValueError("The summary must contain exactly the five classic-control tasks.")
    if not data["n_runs"].eq(5).all():
        raise ValueError("Every summary row must contain exactly five runs.")
    return data.sort_values(["environment", "method", "step"])


def style_axis(axis: plt.Axes) -> None:
    """Apply consistent panel styling while preserving task-specific y limits."""
    axis.set_xlim(left=0)
    axis.margins(x=0, y=0.05)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    axis.xaxis.set_major_formatter(FuncFormatter(compact_steps))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.78)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(length=3.0, width=0.75)


def make_figure(data: pd.DataFrame) -> plt.Figure:
    """Render five task panels and a separate legend/information cell."""
    configure_style()
    figure, axes = plt.subplots(3, 2, figsize=(8.25, 6.75), layout="constrained")
    figure.set_constrained_layout_pads(h_pad=0.08, hspace=0.08, w_pad=0.035, wspace=0.02)
    flat_axes = axes.ravel()
    legend_handles = None

    for axis, (environment, title) in zip(flat_axes, PANELS):
        panel = data.loc[data["environment"] == environment]
        # Keep all uncertainty bands behind every mean curve.
        for method_index, (method, _, color) in enumerate(METHODS):
            series = panel.loc[panel["method"] == method]
            axis.fill_between(
                series["step"],
                series["fitness_best_mean"] - series["fitness_best_std"],
                series["fitness_best_mean"] + series["fitness_best_std"],
                color=color,
                alpha=0.09,
                linewidth=0,
                zorder=method_index + 1,
            )

        # OpenES is drawn first so all other method curves remain above it.
        for method_index, (method, label, color) in enumerate(METHODS):
            series = panel.loc[panel["method"] == method]
            line_width = 2.35 if method == "sr-ghn" else 1.75
            axis.plot(
                series["step"],
                series["fitness_best_mean"],
                color=color,
                linewidth=line_width,
                label=label,
                solid_capstyle="round",
                zorder=10 + method_index,
            )

        style_axis(axis)
        axis.set_ylim(*Y_LIMITS[environment])
        axis.set_title(title, loc="left", pad=5, color="#202020")
        legend_handles = axis.get_legend_handles_labels()

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    handles, labels = legend_handles
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncol=1,
        frameon=False,
        handlelength=2.25,
        handletextpad=0.7,
        labelspacing=0.7,
    )
    figure.supxlabel("Evolution step", fontsize=11, color="#202020")
    figure.supylabel("Best fitness", fontsize=11, color="#202020")
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure = make_figure(load_summary(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "classic_control_fitness_comparison"
    figure.savefig(stem.with_suffix(".pdf"))
    figure.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)
    print(f"Saved {stem.with_suffix('.pdf')}")
    print(f"Saved {stem.with_suffix('.png')} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
