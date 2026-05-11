from __future__ import annotations

import argparse
import pickle
from time import perf_counter

from experiment_configs import print_resolved_config, resolve_mujoco_playground_config
from experiments._common import run_experiment
from solution_artifacts import (
    build_solution_artifact,
    materialize_host_tree,
    save_solution_artifact,
    select_best_individual_from_fitness,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic MuJoCo Playground SRGHN experiment.")
    parser.add_argument(
        "--env-id",
        required=True,
        help="MuJoCo Playground id, e.g., humanoid:run or humanoid/run",
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
    config = resolve_mujoco_playground_config(args.env_id, run_preset=args.run_preset, overrides=overrides)
    if args.print_config:
        print_resolved_config(config, family="control.mujoco_playground_generic", run_preset=args.run_preset)
        return 0
    t0 = perf_counter()
    final_state, metrics = run_experiment(config)
    print("[post-run] selecting best individual from cached final fitness...")
    best_individual, best_index, best_fitness, population_fitness = select_best_individual_from_fitness(
        final_state.pop,
        final_state.pop_fitness,
    )
    print("[post-run] materializing winner, normalizer, fitness, and metrics to host memory...")
    best_individual, population_fitness, obs_norm_state, metrics = materialize_host_tree(
        (best_individual, population_fitness, final_state.obs_norm, metrics)
    )
    print(f"[post-run] host materialization finished in {perf_counter() - t0:.1f}s")
    safe_env_id = args.env_id.replace("/", "_").replace(":", "_")
    out_name = f"mujoco_playground_{safe_env_id}_metrics.pkl"
    print(f"[post-run] writing metrics to {out_name}...")
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    solution_artifact = build_solution_artifact(
        config=config,
        individual=best_individual,
        obs_norm_state=obs_norm_state,
        best_index=best_index,
        best_fitness=best_fitness,
        population_fitness=population_fitness,
        metrics=metrics,
    )
    solution_name = f"mujoco_playground_{safe_env_id}_solution.pkl"
    print(f"[post-run] writing solution artifact to {solution_name}...")
    save_solution_artifact(solution_name, solution_artifact)
    print(f"[post-run] completed in {perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
