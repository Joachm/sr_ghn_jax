from __future__ import annotations

import argparse
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
class CaseSpec:
    block_size: int
    relative_block_size: float
    num_blocks: int
    output_head_params: int
    stochastic_block_scaled_params: int


class SyntheticDeterministicPolicy(eqx.Module):
    """Minimal wrapper around the repo's DeterministicHead.

    This isolates the block-size effect in the deterministic hypernetwork without
    introducing environment or rollout confounds. The model still uses the exact
    block-wise head implementation used by SR-GHNs.
    """

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


@dataclass(frozen=True)
class TrainingConfig:
    rows: int
    cols: int
    hidden_dim: int
    steps: int
    learning_rate: float
    seeds: tuple[int, ...]
    block_sizes: tuple[int, ...]
    target_kind: str
    coeff_dim: int


@dataclass(frozen=True)
class CaseResult:
    spec: CaseSpec
    seed_metrics: tuple[dict, ...]
    mean_rmse: float
    std_rmse: float
    mean_relative_rmse: float
    std_relative_rmse: float
    mean_cosine_similarity: float
    representative_seed: int
    representative_prediction: np.ndarray
    representative_residual: np.ndarray
    representative_history: np.ndarray


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
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _normalize_block_sizes(block_sizes: Iterable[int], out_dim: int) -> tuple[int, ...]:
    cleaned = sorted({max(1, min(int(size), out_dim)) for size in block_sizes}, reverse=True)
    if not cleaned:
        raise ValueError("At least one valid block size is required.")
    return tuple(cleaned)


def _block_sizes_from_ratios(ratios: Sequence[float], out_dim: int) -> tuple[int, ...]:
    sizes = []
    for ratio in ratios:
        if ratio <= 0.0:
            raise ValueError("All block-size ratios must be positive.")
        sizes.append(max(1, min(out_dim, int(ceil(out_dim * float(ratio))))))
    return _normalize_block_sizes(sizes, out_dim)


def _default_block_size_schedule(out_dim: int) -> tuple[int, ...]:
    ratios = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    return _block_sizes_from_ratios(ratios, out_dim)


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
    return target / jnp.maximum(jnp.std(target), 1e-6)


def make_target(shape: tuple[int, int], *, kind: str, seed: int = 0) -> jnp.ndarray:
    if kind == "structured":
        return _make_structured_target(shape)
    if kind == "random":
        return _make_random_target(shape, seed)
    raise ValueError(f"Unknown target kind: {kind}")


def _count_array_params(module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def _case_spec(block_size: int, out_dim: int, hidden_dim: int, coeff_dim: int) -> CaseSpec:
    num_blocks = int(ceil(out_dim / block_size))
    output_head_params = hidden_dim * block_size + block_size
    stochastic_block_scaled_params = coeff_dim * block_size
    return CaseSpec(
        block_size=int(block_size),
        relative_block_size=float(block_size / out_dim),
        num_blocks=num_blocks,
        output_head_params=output_head_params,
        stochastic_block_scaled_params=stochastic_block_scaled_params,
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
    pred = model()
    residual = pred - target
    return metrics, pred, residual


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

    history = []
    eval_every = max(1, steps // 100)
    for step in range(steps):
        model, opt_state, _, metrics = train_step(model, opt_state, target)
        if step == 0 or (step + 1) % eval_every == 0 or step + 1 == steps:
            history.append(
                (
                    step + 1,
                    float(metrics["rmse"]),
                    float(metrics["relative_rmse"]),
                    float(metrics["cosine_similarity"]),
                )
            )

    metrics, pred, residual = _eval_metrics(model, target)
    return {
        "seed": int(seed),
        "rmse": float(metrics["rmse"]),
        "relative_rmse": float(metrics["relative_rmse"]),
        "cosine_similarity": float(metrics["cosine_similarity"]),
        "prediction": np.asarray(jax.device_get(pred)),
        "residual": np.asarray(jax.device_get(residual)),
        "history": np.asarray(history, dtype=float),
        "det_total_params": _count_array_params(model.det),
        "det_out_proj_params": _count_array_params(model.det.out_proj),
    }


def run_block_size_sweep(config: TrainingConfig) -> dict:
    target = np.asarray(jax.device_get(make_target((config.rows, config.cols), kind=config.target_kind, seed=config.seeds[0])))
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    case_results = []

    for block_size in config.block_sizes:
        seed_results = [
            run_single_seed(
                target=jnp.asarray(target),
                block_size=block_size,
                hidden_dim=config.hidden_dim,
                steps=config.steps,
                learning_rate=config.learning_rate,
                seed=seed,
            )
            for seed in config.seeds
        ]
        spec = _case_spec(block_size, config.rows * config.cols, config.hidden_dim, config.coeff_dim)
        rel_rmses = np.asarray([seed_result["relative_rmse"] for seed_result in seed_results], dtype=float)
        representative_index = int(np.argmin(rel_rmses))
        representative = seed_results[representative_index]
        case_results.append(
            CaseResult(
                spec=spec,
                seed_metrics=tuple(
                    {
                        "seed": seed_result["seed"],
                        "rmse": seed_result["rmse"],
                        "relative_rmse": seed_result["relative_rmse"],
                        "cosine_similarity": seed_result["cosine_similarity"],
                        "det_total_params": seed_result["det_total_params"],
                        "det_out_proj_params": seed_result["det_out_proj_params"],
                    }
                    for seed_result in seed_results
                ),
                mean_rmse=float(np.mean([seed_result["rmse"] for seed_result in seed_results])),
                std_rmse=float(np.std([seed_result["rmse"] for seed_result in seed_results])),
                mean_relative_rmse=float(np.mean(rel_rmses)),
                std_relative_rmse=float(np.std(rel_rmses)),
                mean_cosine_similarity=float(np.mean([seed_result["cosine_similarity"] for seed_result in seed_results])),
                representative_seed=int(representative["seed"]),
                representative_prediction=np.asarray(representative["prediction"], dtype=float),
                representative_residual=np.asarray(representative["residual"], dtype=float),
                representative_history=np.asarray(representative["history"], dtype=float),
            )
        )

    return {
        "config": asdict(config),
        "target": target,
        "target_rms": target_rms,
        "cases": case_results,
    }


def _block_label(spec: CaseSpec) -> str:
    return f"b={spec.block_size}\n{spec.relative_block_size * 100:.1f}% of vector\n{spec.num_blocks} blocks"


def plot_reconstruction_grid(results: dict, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    target = np.asarray(results["target"], dtype=float)
    cases: list[CaseResult] = results["cases"]

    vmin = float(np.min(target))
    vmax = float(np.max(target))
    residual_bound = float(max(np.max(np.abs(case.representative_residual)) for case in cases))
    width = max(2.7 * (len(cases) + 1), 10.0)
    fig, axes = plt.subplots(
        2,
        len(cases) + 1,
        figsize=(width, 5.9),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )

    target_im = axes[0, 0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0, 0].set_title("Target matrix")
    axes[0, 0].set_ylabel("Policy reconstruction")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    zero_resid = np.zeros_like(target)
    resid_im = axes[1, 0].imshow(
        zero_resid,
        cmap="coolwarm",
        vmin=-residual_bound,
        vmax=residual_bound,
        aspect="equal",
    )
    axes[1, 0].set_title("Zero residual")
    axes[1, 0].set_ylabel("Residual")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    for column, case in enumerate(cases, start=1):
        pred_ax = axes[0, column]
        resid_ax = axes[1, column]

        pred_ax.imshow(case.representative_prediction, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        pred_ax.set_title(
            _block_label(case.spec) + f"\nrel. RMSE={case.mean_relative_rmse:.3f}",
            fontsize=10.5,
        )
        pred_ax.set_xticks([])
        pred_ax.set_yticks([])

        resid_ax.imshow(
            case.representative_residual,
            cmap="coolwarm",
            vmin=-residual_bound,
            vmax=residual_bound,
            aspect="equal",
        )
        resid_ax.set_title(f"seed {case.representative_seed}", fontsize=10.5)
        resid_ax.set_xticks([])
        resid_ax.set_yticks([])

    cbar_top = fig.colorbar(target_im, ax=list(axes[0, :]), shrink=0.82, location="right", pad=0.02)
    cbar_top.set_label("Weight value")
    cbar_bottom = fig.colorbar(resid_im, ax=list(axes[1, :]), shrink=0.82, location="right", pad=0.02)
    cbar_bottom.set_label("Prediction - target")

    png_path = output_root / "block_size_reconstructions.png"
    pdf_path = output_root / "block_size_reconstructions.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_tradeoff_summary(results: dict, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[CaseResult] = sorted(results["cases"], key=lambda case: case.spec.relative_block_size, reverse=True)

    rel_block = np.asarray([case.spec.relative_block_size for case in cases], dtype=float)
    rel_rmse = np.asarray([case.mean_relative_rmse for case in cases], dtype=float)
    rel_rmse_std = np.asarray([case.std_relative_rmse for case in cases], dtype=float)
    out_head_params = np.asarray([case.spec.output_head_params for case in cases], dtype=float)
    num_blocks = np.asarray([case.spec.num_blocks for case in cases], dtype=int)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    ax_left.plot(rel_block, rel_rmse, marker="o", linewidth=2.3, markersize=6)
    ax_left.fill_between(rel_block, rel_rmse - rel_rmse_std, rel_rmse + rel_rmse_std, alpha=0.18)
    ax_left.set_xscale("log", base=2)
    ax_left.invert_xaxis()
    ax_left.set_xlabel("Relative block size (block_size / #policy parameters)")
    ax_left.set_ylabel("Relative RMSE")
    ax_left.set_title("Reconstruction quality vs. block size")
    ax_left.set_xticks(rel_block, labels=[f"{value * 100:.1f}%" for value in rel_block], rotation=25, ha="right")
    for x, y, n_blocks in zip(rel_block, rel_rmse, num_blocks):
        ax_left.annotate(f"{n_blocks} blocks", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax_right.plot(out_head_params, rel_rmse, marker="o", linewidth=2.3, markersize=6)
    ax_right.fill_between(out_head_params, rel_rmse - rel_rmse_std, rel_rmse + rel_rmse_std, alpha=0.18)
    ax_right.set_xscale("log", base=2)
    ax_right.set_xlabel("Deterministic output-head parameters")
    ax_right.set_ylabel("Relative RMSE")
    ax_right.set_title("Compression / fidelity tradeoff")
    for x, y, case in zip(out_head_params, rel_rmse, cases):
        ax_right.annotate(f"b={case.spec.block_size}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    png_path = output_root / "block_size_tradeoff_summary.png"
    pdf_path = output_root / "block_size_tradeoff_summary.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_training_curves(results: dict, output_dir: str | Path) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[CaseResult] = sorted(results["cases"], key=lambda case: case.spec.relative_block_size, reverse=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for case in cases:
        history = np.asarray(case.representative_history, dtype=float)
        ax.plot(history[:, 0], history[:, 2], linewidth=2.0, label=_block_label(case.spec).replace("\n", " | "))
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Relative RMSE")
    ax.set_title("Optimization trajectories for representative runs")
    ax.legend(loc="upper right", fontsize=8)

    png_path = output_root / "block_size_training_curves.png"
    pdf_path = output_root / "block_size_training_curves.pdf"
    fig.savefig(png_path, dpi=PAPER_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def save_results_pickle(results: dict, output_dir: str | Path) -> Path:
    serializable = {
        "config": results["config"],
        "target": np.asarray(results["target"], dtype=float),
        "target_rms": float(results["target_rms"]),
        "cases": [
            {
                "spec": asdict(case.spec),
                "seed_metrics": list(case.seed_metrics),
                "mean_rmse": case.mean_rmse,
                "std_rmse": case.std_rmse,
                "mean_relative_rmse": case.mean_relative_rmse,
                "std_relative_rmse": case.std_relative_rmse,
                "mean_cosine_similarity": case.mean_cosine_similarity,
                "representative_seed": case.representative_seed,
                "representative_prediction": np.asarray(case.representative_prediction, dtype=float),
                "representative_residual": np.asarray(case.representative_residual, dtype=float),
                "representative_history": np.asarray(case.representative_history, dtype=float),
            }
            for case in results["cases"]
        ],
    }
    out_path = Path(output_dir) / "block_size_tradeoff_results.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how shrinking the SR-GHN deterministic head block_size trades off "
            "output-head parameter count against the ability to reconstruct a target matrix."
        )
    )
    parser.add_argument("--rows", type=int, default=32, help="Target matrix height.")
    parser.add_argument("--cols", type=int, default=32, help="Target matrix width.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for the deterministic head.")
    parser.add_argument("--steps", type=int, default=2500, help="Optimization steps per block-size / seed.")
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Adam learning rate.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="Initialization seeds to average over.")
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
        help=(
            "Relative block sizes to sweep, expressed as fractions of the flattened target size. "
            "Example: --block-ratios 1.0 0.5 0.25 0.125"
        ),
    )
    parser.add_argument(
        "--target-kind",
        choices=("structured", "random"),
        default="structured",
        help="Type of target matrix to reconstruct.",
    )
    parser.add_argument(
        "--coeff-dim",
        type=int,
        default=32,
        help="Reference stochastic coefficient width used only for reporting block-scaled stochastic params.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots and the serialized results payload.",
    )
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
        block_sizes = _normalize_block_sizes(args.block_sizes, out_dim)
    elif args.block_ratios:
        block_sizes = _block_sizes_from_ratios(args.block_ratios, out_dim)
    else:
        block_sizes = _default_block_size_schedule(out_dim)

    config = TrainingConfig(
        rows=rows,
        cols=cols,
        hidden_dim=int(args.hidden_dim),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        seeds=tuple(int(seed) for seed in args.seeds),
        block_sizes=block_sizes,
        target_kind=str(args.target_kind),
        coeff_dim=int(args.coeff_dim),
    )

    default_out_dir = Path("paper_figures") / f"block_size_tradeoff_{rows}x{cols}"
    output_dir = Path(args.output_dir) if args.output_dir is not None else default_out_dir

    results = run_block_size_sweep(config)
    plot_reconstruction_grid(results, output_dir)
    plot_tradeoff_summary(results, output_dir)
    plot_training_curves(results, output_dir)
    save_results_pickle(results, output_dir)
    print(f"Saved block-size tradeoff artifacts to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
