from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from math import ceil, prod
from pathlib import Path
import pickle
from typing import Iterable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hypernets import DeterministicHead


@dataclass(frozen=True)
class CellSpec:
    hidden_dim: int
    block_size: int
    relative_block_size: float
    num_blocks: int
    num_blocks_per_hidden: float
    output_head_params: int
    stochastic_block_scaled_params: int


@dataclass(frozen=True)
class GridConfig:
    rows: int
    cols: int
    hidden_dims: tuple[int, ...]
    block_sizes: tuple[int, ...]
    steps: int
    learning_rate: float
    seeds: tuple[int, ...]
    target_seeds: tuple[int, ...]
    target_kind: str
    coeff_dim: int


@dataclass(frozen=True)
class GridCellResult:
    spec: CellSpec
    mean_rmse: float
    std_rmse: float
    mean_relative_rmse: float
    std_relative_rmse: float
    mean_cosine_similarity: float
    std_cosine_similarity: float
    num_runs: int
    run_metrics: tuple[dict, ...]


class SyntheticDeterministicPolicy(eqx.Module):
    context: jnp.ndarray
    det: DeterministicHead
    out_shape: tuple[int, int] = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        block_size: int,
        out_shape: tuple[int, int],
        *,
        key,
    ):
        k_ctx, k_det = jax.random.split(key, 2)
        self.context = 0.1 * jax.random.normal(k_ctx, (hidden_dim,), dtype=jnp.float32)
        self.det = DeterministicHead(hidden_dim, hidden_dim, block_size, key=k_det)
        self.out_shape = tuple(int(dim) for dim in out_shape)
        self.out_dim = int(prod(self.out_shape))

    def __call__(self) -> jnp.ndarray:
        return self.det(self.context, self.out_dim).reshape(self.out_shape)


PAPER_DPI = 220


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
            "axes.grid": False,
            "legend.frameon": False,
            "xtick.color": "#1f1f1f",
            "ytick.color": "#1f1f1f",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _make_key(seed: int):
    key_fn = getattr(jax.random, "key", None)
    if key_fn is not None:
        return key_fn(seed)
    return jax.random.PRNGKey(seed)


class _Adam:
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def init(self, params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {
            "count": jnp.asarray(0, dtype=jnp.int32),
            "m": zeros,
            "v": zeros,
        }

    def update(self, grads, state):
        count = state["count"] + jnp.asarray(1, dtype=state["count"].dtype)
        m = jax.tree_util.tree_map(
            lambda m_prev, g: self.beta1 * m_prev + (1.0 - self.beta1) * g,
            state["m"],
            grads,
        )
        v = jax.tree_util.tree_map(
            lambda v_prev, g: self.beta2 * v_prev + (1.0 - self.beta2) * jnp.square(g),
            state["v"],
            grads,
        )

        count_f = count.astype(jnp.float32)
        lr = jnp.asarray(self.learning_rate, dtype=jnp.float32)
        beta1 = jnp.asarray(self.beta1, dtype=jnp.float32)
        beta2 = jnp.asarray(self.beta2, dtype=jnp.float32)
        eps = jnp.asarray(self.eps, dtype=jnp.float32)

        m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**count_f), m)
        v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**count_f), v)
        updates = jax.tree_util.tree_map(
            lambda m_item, v_item: -lr * m_item / (jnp.sqrt(v_item) + eps),
            m_hat,
            v_hat,
        )
        new_state = {"count": count, "m": m, "v": v}
        return updates, new_state


def _normalize_sizes(values: Iterable[int], upper_bound: int) -> tuple[int, ...]:
    cleaned = sorted({max(1, min(int(value), upper_bound)) for value in values}, reverse=True)
    if not cleaned:
        raise ValueError("Expected at least one valid positive value.")
    return tuple(cleaned)


def _block_sizes_from_ratios(ratios: Sequence[float], out_dim: int) -> tuple[int, ...]:
    sizes = []
    for ratio in ratios:
        if ratio <= 0.0:
            raise ValueError("All block-size ratios must be positive.")
        sizes.append(max(1, min(out_dim, int(ceil(out_dim * float(ratio))))))
    return _normalize_sizes(sizes, out_dim)


def _default_block_size_schedule(out_dim: int) -> tuple[int, ...]:
    return _block_sizes_from_ratios((1.0, 0.125, 0.03125, 0.0078125), out_dim)


def _default_hidden_dim_schedule() -> tuple[int, ...]:
    return (8, 16, 32, 64)


def _make_structured_target(shape: tuple[int, int]) -> jnp.ndarray:
    rows, cols = shape
    ys = jnp.linspace(-1.0, 1.0, rows, dtype=jnp.float32)
    xs = jnp.linspace(-1.0, 1.0, cols, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")

    smooth = jnp.sin(jnp.pi * (2.25 * xx + 1.35 * yy))
    interference = 0.65 * jnp.cos(jnp.pi * (4.75 * xx - 3.10 * yy))
    checker = 0.25 * jnp.sin(8.0 * jnp.pi * xx) * jnp.cos(6.0 * jnp.pi * yy)
    diagonal = 0.8 * jnp.exp(-8.0 * (xx - yy) ** 2)
    hotspot_a = 0.9 * jnp.exp(-9.0 * ((xx - 0.35) ** 2 + (yy + 0.25) ** 2))
    hotspot_b = -0.7 * jnp.exp(-11.0 * ((xx + 0.45) ** 2 + (yy - 0.40) ** 2))
    stripe = 0.25 * jnp.where((jnp.arange(rows)[:, None] % 5) == 0, 1.0, 0.0)

    target = smooth + interference + checker + diagonal + hotspot_a + hotspot_b + stripe
    target = target - jnp.mean(target)
    target = target / jnp.maximum(jnp.std(target), 1e-6)
    return target.astype(jnp.float32)


def _make_random_target(shape: tuple[int, int], seed: int) -> jnp.ndarray:
    key = _make_key(seed)
    target = jax.random.normal(key, shape, dtype=jnp.float32)
    target = target - jnp.mean(target)
    return target / jnp.maximum(jnp.std(target), 1e-6)


def make_target(shape: tuple[int, int], *, kind: str, seed: int = 0) -> jnp.ndarray:
    if kind == "structured":
        return _make_structured_target(shape)
    if kind == "random":
        return _make_random_target(shape, seed)
    raise ValueError(f"Unknown target kind: {kind}")


def _cell_spec(block_size: int, out_dim: int, hidden_dim: int, coeff_dim: int) -> CellSpec:
    num_blocks = int(ceil(out_dim / block_size))
    return CellSpec(
        hidden_dim=int(hidden_dim),
        block_size=int(block_size),
        relative_block_size=float(block_size / out_dim),
        num_blocks=num_blocks,
        num_blocks_per_hidden=float(num_blocks / hidden_dim),
        output_head_params=int(hidden_dim * block_size + block_size),
        stochastic_block_scaled_params=int(coeff_dim * block_size),
    )


def _loss_and_metrics(model: SyntheticDeterministicPolicy, target: jnp.ndarray):
    pred = model()
    residual = pred - target
    mse = jnp.mean(jnp.square(residual))
    rmse = jnp.sqrt(mse)
    target_rms = jnp.sqrt(jnp.mean(jnp.square(target)))
    rel_rmse = rmse / jnp.maximum(target_rms, 1e-8)
    cosine = jnp.vdot(pred.reshape(-1), target.reshape(-1)) / jnp.maximum(
        jnp.linalg.norm(pred.reshape(-1)) * jnp.linalg.norm(target.reshape(-1)), 1e-8
    )
    metrics = {
        "loss": mse,
        "rmse": rmse,
        "relative_rmse": rel_rmse,
        "cosine_similarity": cosine,
    }
    return mse, metrics


def _make_train_step(optimizer: _Adam):
    @eqx.filter_jit
    def train_step(model: SyntheticDeterministicPolicy, opt_state, target: jnp.ndarray):
        (loss, metrics), grads = eqx.filter_value_and_grad(_loss_and_metrics, has_aux=True)(model, target)
        grads = eqx.filter(grads, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, metrics

    return train_step


@eqx.filter_jit
def _eval_metrics(model: SyntheticDeterministicPolicy, target: jnp.ndarray):
    _, metrics = _loss_and_metrics(model, target)
    return metrics


def run_single_seed(
    *,
    target: jnp.ndarray,
    block_size: int,
    hidden_dim: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> dict:
    model_key = _make_key(seed)
    model = SyntheticDeterministicPolicy(
        hidden_dim=hidden_dim,
        block_size=block_size,
        out_shape=tuple(int(dim) for dim in target.shape),
        key=model_key,
    )
    optimizer = _Adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = _make_train_step(optimizer)

    for _ in range(steps):
        model, opt_state, _, _ = train_step(model, opt_state, target)

    metrics = _eval_metrics(model, target)
    return {
        "seed": int(seed),
        "rmse": float(metrics["rmse"]),
        "relative_rmse": float(metrics["relative_rmse"]),
        "cosine_similarity": float(metrics["cosine_similarity"]),
    }


def run_hidden_block_grid(config: GridConfig) -> dict:
    out_dim = config.rows * config.cols
    target_payload = [
        {
            "target_seed": int(target_seed),
            "target": np.asarray(
                jax.device_get(make_target((config.rows, config.cols), kind=config.target_kind, seed=target_seed)),
                dtype=float,
            ),
        }
        for target_seed in config.target_seeds
    ]

    cell_results: list[GridCellResult] = []
    for hidden_dim in config.hidden_dims:
        for block_size in config.block_sizes:
            run_metrics = []
            for target_item in target_payload:
                target = jnp.asarray(target_item["target"], dtype=jnp.float32)
                for seed in config.seeds:
                    metrics = run_single_seed(
                        target=target,
                        block_size=block_size,
                        hidden_dim=hidden_dim,
                        steps=config.steps,
                        learning_rate=config.learning_rate,
                        seed=seed,
                    )
                    metrics["target_seed"] = int(target_item["target_seed"])
                    run_metrics.append(metrics)

            rel_rmse = np.asarray([item["relative_rmse"] for item in run_metrics], dtype=float)
            rmse = np.asarray([item["rmse"] for item in run_metrics], dtype=float)
            cosine = np.asarray([item["cosine_similarity"] for item in run_metrics], dtype=float)
            cell_results.append(
                GridCellResult(
                    spec=_cell_spec(block_size, out_dim, hidden_dim, config.coeff_dim),
                    mean_rmse=float(np.mean(rmse)),
                    std_rmse=float(np.std(rmse)),
                    mean_relative_rmse=float(np.mean(rel_rmse)),
                    std_relative_rmse=float(np.std(rel_rmse)),
                    mean_cosine_similarity=float(np.mean(cosine)),
                    std_cosine_similarity=float(np.std(cosine)),
                    num_runs=len(run_metrics),
                    run_metrics=tuple(run_metrics),
                )
            )

    mean_relative_rmse = np.zeros((len(config.hidden_dims), len(config.block_sizes)), dtype=float)
    std_relative_rmse = np.zeros_like(mean_relative_rmse)
    mean_cosine = np.zeros_like(mean_relative_rmse)

    by_key = {(cell.spec.hidden_dim, cell.spec.block_size): cell for cell in cell_results}
    for row, hidden_dim in enumerate(config.hidden_dims):
        for col, block_size in enumerate(config.block_sizes):
            cell = by_key[(hidden_dim, block_size)]
            mean_relative_rmse[row, col] = cell.mean_relative_rmse
            std_relative_rmse[row, col] = cell.std_relative_rmse
            mean_cosine[row, col] = cell.mean_cosine_similarity

    return {
        "config": asdict(config),
        "cells": cell_results,
        "mean_relative_rmse_matrix": mean_relative_rmse,
        "std_relative_rmse_matrix": std_relative_rmse,
        "mean_cosine_matrix": mean_cosine,
        "targets": target_payload,
    }


def _column_label(cell: GridCellResult) -> str:
    return (
        f"b={cell.spec.block_size}\n"
        f"{100.0 * cell.spec.relative_block_size:.2f}% vec\n"
        f"{cell.spec.num_blocks} blocks"
    )


def _text_color_for_value(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return "black"
    norm = (value - vmin) / (vmax - vmin)
    return "white" if norm > 0.55 else "#111111"


def plot_rmse_heatmap(results: dict, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    mean_rmse = np.asarray(results["mean_relative_rmse_matrix"], dtype=float)
    std_rmse = np.asarray(results["std_relative_rmse_matrix"], dtype=float)
    config = results["config"]
    hidden_dims = [int(value) for value in config["hidden_dims"]]
    block_sizes = [int(value) for value in config["block_sizes"]]
    cells = {(cell.spec.hidden_dim, cell.spec.block_size): cell for cell in results["cells"]}

    fig, ax = plt.subplots(figsize=(2.25 * len(block_sizes) + 1.8, 1.3 * len(hidden_dims) + 1.8), constrained_layout=True)
    im = ax.imshow(mean_rmse, aspect="auto", cmap="magma_r")

    ax.set_title("SR-GHN deterministic-head reconstruction error")
    ax.set_xlabel("Block size setting")
    ax.set_ylabel("Deterministic hidden dimension")
    ax.set_xticks(np.arange(len(block_sizes)))
    ax.set_yticks(np.arange(len(hidden_dims)))
    ax.set_xticklabels([_column_label(cells[(hidden_dims[0], block_size)]) for block_size in block_sizes])
    ax.set_yticklabels([f"h={hidden_dim}" for hidden_dim in hidden_dims])

    vmin = float(np.min(mean_rmse))
    vmax = float(np.max(mean_rmse))
    for row, hidden_dim in enumerate(hidden_dims):
        for col, block_size in enumerate(block_sizes):
            cell_value = mean_rmse[row, col]
            cell_std = std_rmse[row, col]
            text_color = _text_color_for_value(cell_value, vmin, vmax)
            ax.text(
                col,
                row,
                f"{cell_value:.3f}\n±{cell_std:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="semibold",
            )

    ax.set_xticks(np.arange(-0.5, len(block_sizes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hidden_dims), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Mean relative RMSE")

    png_path = output_root / "block_hidden_rmse_heatmap.png"
    pdf_path = output_root / "block_hidden_rmse_heatmap.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_tradeoff_curves(results: dict, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = results["config"]
    hidden_dims = [int(value) for value in config["hidden_dims"]]
    block_sizes = [int(value) for value in config["block_sizes"]]
    cells = {(cell.spec.hidden_dim, cell.spec.block_size): cell for cell in results["cells"]}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    for hidden_dim in hidden_dims:
        xs = []
        ys = []
        yerr = []
        for block_size in block_sizes:
            cell = cells[(hidden_dim, block_size)]
            xs.append(cell.spec.num_blocks)
            ys.append(cell.mean_relative_rmse)
            yerr.append(cell.std_relative_rmse)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        ax_left.plot(xs, ys, marker="o", linewidth=2.2, markersize=6, label=f"h={hidden_dim}")
        ax_left.fill_between(xs, ys - yerr, ys + yerr, alpha=0.14)

    ax_left.set_xscale("log", base=2)
    ax_left.set_xlabel("Number of output blocks")
    ax_left.set_ylabel("Mean relative RMSE")
    ax_left.set_title("Error vs. block count")
    ax_left.legend(loc="upper left", ncol=2, fontsize=9)

    for hidden_dim in hidden_dims:
        ratios = []
        ys = []
        for block_size in block_sizes:
            cell = cells[(hidden_dim, block_size)]
            ratios.append(cell.spec.num_blocks_per_hidden)
            ys.append(cell.mean_relative_rmse)
        ax_right.plot(ratios, ys, marker="o", linewidth=2.2, markersize=6, label=f"h={hidden_dim}")

    ax_right.set_xscale("log", base=2)
    ax_right.set_xlabel("Block-count / hidden-dim ratio")
    ax_right.set_ylabel("Mean relative RMSE")
    ax_right.set_title("Error vs. shared-capacity ratio")

    png_path = output_root / "block_hidden_tradeoff_curves.png"
    pdf_path = output_root / "block_hidden_tradeoff_curves.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def save_results_pickle(results: dict, output_dir: str | Path) -> Path:
    serializable = {
        "config": results["config"],
        "mean_relative_rmse_matrix": np.asarray(results["mean_relative_rmse_matrix"], dtype=float),
        "std_relative_rmse_matrix": np.asarray(results["std_relative_rmse_matrix"], dtype=float),
        "mean_cosine_matrix": np.asarray(results["mean_cosine_matrix"], dtype=float),
        "targets": results["targets"],
        "cells": [
            {
                "spec": asdict(cell.spec),
                "mean_rmse": cell.mean_rmse,
                "std_rmse": cell.std_rmse,
                "mean_relative_rmse": cell.mean_relative_rmse,
                "std_relative_rmse": cell.std_relative_rmse,
                "mean_cosine_similarity": cell.mean_cosine_similarity,
                "std_cosine_similarity": cell.std_cosine_similarity,
                "num_runs": cell.num_runs,
                "run_metrics": list(cell.run_metrics),
            }
            for cell in results["cells"]
        ],
    }
    out_path = Path(output_dir) / "block_hidden_tradeoff_results.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def save_results_csv(results: dict, output_dir: str | Path) -> Path:
    out_path = Path(output_dir) / "block_hidden_tradeoff_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hidden_dim",
                "block_size",
                "relative_block_size",
                "num_blocks",
                "num_blocks_per_hidden",
                "output_head_params",
                "stochastic_block_scaled_params",
                "num_runs",
                "mean_rmse",
                "std_rmse",
                "mean_relative_rmse",
                "std_relative_rmse",
                "mean_cosine_similarity",
                "std_cosine_similarity",
            ]
        )
        for cell in results["cells"]:
            writer.writerow(
                [
                    cell.spec.hidden_dim,
                    cell.spec.block_size,
                    cell.spec.relative_block_size,
                    cell.spec.num_blocks,
                    cell.spec.num_blocks_per_hidden,
                    cell.spec.output_head_params,
                    cell.spec.stochastic_block_scaled_params,
                    cell.num_runs,
                    cell.mean_rmse,
                    cell.std_rmse,
                    cell.mean_relative_rmse,
                    cell.std_relative_rmse,
                    cell.mean_cosine_similarity,
                    cell.std_cosine_similarity,
                ]
            )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep both deterministic-head block_size and hidden_dim, and write a paper-ready "
            "RMSE matrix over their combinations."
        )
    )
    parser.add_argument("--rows", type=int, default=32, help="Target matrix height.")
    parser.add_argument("--cols", type=int, default=32, help="Target matrix width.")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=None, help="Hidden dimensions to sweep.")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit block sizes to sweep. Overrides --block-ratios when provided.",
    )
    parser.add_argument(
        "--block-ratios",
        type=float,
        nargs="*",
        default=None,
        help="Relative block sizes to sweep, expressed as fractions of the flattened target size.",
    )
    parser.add_argument("--steps", type=int, default=2500, help="Optimization steps per cell and seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="Initialization seeds.")
    parser.add_argument(
        "--target-seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="Random target seeds to average over. For structured targets, multiple seeds are redundant.",
    )
    parser.add_argument(
        "--target-kind",
        choices=("structured", "random"),
        default="random",
        help="Type of target matrix to reconstruct.",
    )
    parser.add_argument(
        "--coeff-dim",
        type=int,
        default=32,
        help="Reference stochastic coefficient width used only for reporting block-scaled stochastic params.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for plots and tables.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _set_plot_style()

    rows = int(args.rows)
    cols = int(args.cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must both be positive.")
    out_dim = rows * cols

    if args.block_sizes:
        block_sizes = _normalize_sizes(args.block_sizes, out_dim)
    elif args.block_ratios:
        block_sizes = _block_sizes_from_ratios(args.block_ratios, out_dim)
    else:
        block_sizes = _default_block_size_schedule(out_dim)

    hidden_dims = _normalize_sizes(args.hidden_dims or _default_hidden_dim_schedule(), out_dim)

    config = GridConfig(
        rows=rows,
        cols=cols,
        hidden_dims=tuple(sorted(hidden_dims)),
        block_sizes=block_sizes,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        seeds=tuple(int(seed) for seed in args.seeds),
        target_seeds=tuple(int(seed) for seed in args.target_seeds),
        target_kind=str(args.target_kind),
        coeff_dim=int(args.coeff_dim),
    )

    default_out_dir = Path("paper_figures") / f"block_hidden_tradeoff_{rows}x{cols}"
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_out_dir

    results = run_hidden_block_grid(config)
    plot_rmse_heatmap(results, output_dir)
    plot_tradeoff_curves(results, output_dir)
    save_results_pickle(results, output_dir)
    save_results_csv(results, output_dir)
    print(f"Saved block-size/hidden-dim tradeoff artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
