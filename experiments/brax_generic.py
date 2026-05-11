from __future__ import annotations

import argparse
import pickle

from experiment_configs import print_resolved_config, resolve_brax_generic_config
from experiments._common import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic Brax SRGHN experiment.")
    parser.add_argument("--env-id", required=True, help="Brax environment id, e.g., ant")
    parser.add_argument(
        "--backend",
        default=None,
        help="Brax backend, e.g., spring, generalized, mjx (defaults to Brax default).",
    )
    parser.add_argument("--run-preset", default="default", choices=("default", "debug"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=None)
    parser.add_argument("--episodes-per-eval", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        key: value
        for key, value in {
            "brax_backend": args.backend,
            "seed": args.seed,
            "pop_size": args.pop_size,
            "num_generations": args.num_generations,
            "episode_horizon": args.episode_horizon,
            "children_per_parent": args.children_per_parent,
            "episodes_per_eval": args.episodes_per_eval,
            "parameter_block_size": args.parameter_block_size,
            "mutation_block_ratio": args.mutation_block_ratio,
        }.items()
        if value is not None
    }
    config = resolve_brax_generic_config(args.env_id, run_preset=args.run_preset, overrides=overrides)
    if args.print_config:
        print_resolved_config(config, family="control.brax_generic", run_preset=args.run_preset)
        return 0
    _, metrics = run_experiment(config)
    out_name = f"brax_{args.env_id}_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
