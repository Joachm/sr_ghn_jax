from __future__ import annotations

import argparse

from adaptation_analysis import build_run_artifact, save_pickle
from experiment_configs import (
    BASELINE_PRESET_NAMES,
    NONSTATIONARY_BRAX_TASK_PRESET_NAMES,
    print_resolved_config,
    resolve_nonstationary_brax_config,
)
from experiments._common import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a nonstationary Brax SR-GHN experiment.")
    parser.add_argument("--env-id", default="ant")
    parser.add_argument("--variant", "--task-preset", dest="task_preset", default="brax_direction_switch", choices=NONSTATIONARY_BRAX_TASK_PRESET_NAMES)
    parser.add_argument("--baseline", default="srghn_full", choices=BASELINE_PRESET_NAMES)
    parser.add_argument("--run-preset", default="default", choices=("default", "debug"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=None)
    parser.add_argument("--episodes-per-eval", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
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
            "brax_backend": args.backend,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
            "baseline_name": args.baseline,
            "fixed_mutation_lr": args.fixed_mutation_lr,
        }.items()
        if value is not None
    }
    config = resolve_nonstationary_brax_config(
        args.env_id,
        task_preset=args.task_preset,
        run_preset=args.run_preset,
        target_speed=args.target_speed,
        overrides=overrides,
    )
    if args.print_config:
        print_resolved_config(
            config,
            family="control.nonstationary_brax",
            task_preset=args.task_preset,
            baseline=args.baseline,
            run_preset=args.run_preset,
        )
        return 0
    _, metrics = run_experiment(config)
    artifact = build_run_artifact(config, metrics)
    safe_env_id = args.env_id.replace("/", "_").replace(":", "_")
    out_name = args.output or f"nonstationary_brax_{safe_env_id}_{args.task_preset}_{args.baseline}_seed{config.seed}.pkl"
    save_pickle(out_name, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
