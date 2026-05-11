from __future__ import annotations

import argparse

from adaptation_analysis import build_run_artifact, save_pickle
from experiment_configs import (
    BASELINE_PRESET_NAMES,
    NONSTATIONARY_GYMNAX_TASK_PRESET_NAMES,
    print_resolved_config,
    resolve_nonstationary_gymnax_config,
)
from experiments._common import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a nonstationary Gymnax adaptation experiment.")
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--variant", "--task-preset", dest="task_preset", default="cartpole_flip", choices=NONSTATIONARY_GYMNAX_TASK_PRESET_NAMES[:3])
    parser.add_argument("--optimizer-family", default="srghn", choices=("srghn", "evosax"))
    parser.add_argument("--baseline", default="srghn_full", choices=BASELINE_PRESET_NAMES)
    parser.add_argument("--run-preset", default="default", choices=("default", "debug"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--evosax-algo", default=None)
    parser.add_argument("--evosax-sigma-init", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=None)
    parser.add_argument("--episodes-per-eval", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--output", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "seed": args.seed,
            "pop_size": args.pop_size,
            "num_generations": args.num_generations,
            "episode_horizon": args.episode_horizon,
            "children_per_parent": args.children_per_parent,
            "episodes_per_eval": args.episodes_per_eval,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
            "optimizer_family": args.optimizer_family,
            "baseline_name": args.baseline,
            "evosax_algo": args.evosax_algo,
            "evosax_sigma_init": args.evosax_sigma_init,
            "wandb_project": args.project,
            "fixed_mutation_lr": args.fixed_mutation_lr,
        }.items()
        if value is not None
    }
    config = resolve_nonstationary_gymnax_config(
        args.env_id,
        task_preset=args.task_preset,
        run_preset=args.run_preset,
        overrides=overrides,
    )
    if args.print_config:
        print_resolved_config(
            config,
            family="control.nonstationary_gymnax",
            task_preset=args.task_preset,
            baseline=args.baseline,
            run_preset=args.run_preset,
        )
        return 0
    _, metrics = run_experiment(config)
    artifact = build_run_artifact(config, metrics)
    safe_env_id = args.env_id.replace("/", "_").replace(":", "_")
    label = args.baseline if args.optimizer_family == "srghn" else f"evosax_{args.evosax_algo or 'unknown'}"
    out_name = args.output or f"nonstationary_gymnax_{safe_env_id}_{args.task_preset}_{label}_g{config.num_generations}_p{config.pop_size}_seed{config.seed}.pkl"
    save_pickle(out_name, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
