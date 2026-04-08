from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from adaptation_analysis import build_run_artifact, save_pickle
from configs import make_config_nonstationary_gymnax
from experiments._adaptation import GYMNAX_MINATAR_SUITE_ENVIRONMENTS, gymnax_minatar_suite_shift_windows
from experiments._common import run_experiment


POLICY_CONV_CHANNELS = (8,)
POLICY_CONV_KERNEL_SIZES = ((3, 3),)
POLICY_CONV_STRIDES = ((1, 1),)
POLICY_HIDDEN_DIMS = (128,)


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _label_for_config(config) -> str:
    if config.optimizer_family == "evosax":
        return f"evosax_{_safe_name(config.evosax_algo or 'unknown')}"
    return _safe_name(config.baseline_name)


def _namespace_for_config(config) -> str:
    return (
        f"{_label_for_config(config)}"
        f"_g{config.num_generations}"
        f"_p{config.pop_size}"
        f"_cpp{config.children_per_parent}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the full nonstationary Gymnax MinAtar adaptation suite.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list; defaults to 0..4.")
    parser.add_argument("--output-dir", default="results/gymnax_minatar_suite")
    parser.add_argument("--project", default=None)
    parser.add_argument("--optimizer-family", default="srghn", choices=("srghn", "evosax"))
    parser.add_argument("--baseline", default="srghn_full")
    parser.add_argument("--evosax-algo", default=None)
    parser.add_argument("--evosax-sigma-init", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-generations", type=int, default=12000)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--episode-horizon", type=int, default=2500)
    parser.add_argument("--parameter-block-size", type=int, default=1024*4)
    parser.add_argument("--mutation-block-ratio", type=float, default=1.0)
    args = parser.parse_args()

    seeds = args.seeds or list(range(5))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for env_id in GYMNAX_MINATAR_SUITE_ENVIRONMENTS:
        if "Space" in env_id:
            for seed in seeds:
                config = make_config_nonstationary_gymnax(
                    env_id,
                    seed=seed,
                    pop_size=args.pop_size,
                    num_generations=args.num_generations,
                    episode_horizon=args.episode_horizon,
                    children_per_parent=args.children_per_parent,
                    episodes_per_eval=args.episodes_per_eval,
                    parameter_block_size=args.parameter_block_size,
                    mutation_block_ratio=args.mutation_block_ratio,
                    shift_windows=gymnax_minatar_suite_shift_windows(env_id),
                    optimizer_family=args.optimizer_family,
                    baseline_name=args.baseline,
                    evosax_algo=args.evosax_algo,
                    evosax_sigma_init=args.evosax_sigma_init,
                    policy_architecture="cnn_mlp",
                    policy_conv_channels=POLICY_CONV_CHANNELS,
                    policy_conv_kernel_sizes=POLICY_CONV_KERNEL_SIZES,
                    policy_conv_strides=POLICY_CONV_STRIDES,
                    policy_hidden_dims=POLICY_HIDDEN_DIMS,
                    wandb_project=args.project or ("srghn-minatar_3" if args.optimizer_family == "srghn" else f"sr-ghn_control_{_safe_name(args.evosax_algo or 'unknown')}_minatar"),
                    wandb_group=_safe_name(env_id),
                    wandb_name=f"{_safe_name(env_id)}-seed{seed}",
                    fixed_mutation_lr=args.fixed_mutation_lr,
                )
                suite_dir = output_root / _namespace_for_config(config)
                suite_dir.mkdir(parents=True, exist_ok=True)
                failure_log = suite_dir / "failures.log"
                env_dir = suite_dir / _safe_name(env_id)
                env_dir.mkdir(parents=True, exist_ok=True)
                out_path = env_dir / f"seed_{seed}.pkl"
                if args.skip_existing and out_path.exists():
                    print(f"[skip] {env_id} seed={seed} -> {out_path}")
                    continue
                print(f"[run] env={env_id} seed={seed}")
                try:
                    _, metrics = run_experiment(config)
                    artifact = build_run_artifact(config, metrics)
                    save_pickle(out_path, artifact)
                    print(f"[saved] {out_path}")
                except Exception:
                    tb = traceback.format_exc()
                    message = f"[failed] env={env_id} seed={seed}\n{tb}\n"
                    print(message, flush=True)
                    with failure_log.open("a", encoding="utf-8") as f:
                        f.write(message)


if __name__ == "__main__":
    main()
