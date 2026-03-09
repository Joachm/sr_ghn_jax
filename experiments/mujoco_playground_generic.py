from __future__ import annotations

import argparse
import pickle
from time import perf_counter

from configs import make_config_mujoco_playground_generic
from experiments._common import run_experiment
from solution_artifacts import (
    build_solution_artifact,
    materialize_host_tree,
    save_solution_artifact,
    select_best_individual_from_fitness,
)


def main():
    parser = argparse.ArgumentParser(description="Run a generic MuJoCo Playground SRGHN experiment.")
    parser.add_argument(
        "--env-id",
        required=True,
        help="MuJoCo Playground id, e.g., humanoid:run or humanoid/run",
    )
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=None)
    parser.add_argument("--episodes-per-eval", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    args = parser.parse_args()

    config_kwargs = {}
    if args.seed is not None:
        config_kwargs["seed"] = args.seed
    if args.pop_size is not None:
        config_kwargs["pop_size"] = args.pop_size
    if args.num_generations is not None:
        config_kwargs["num_generations"] = args.num_generations
    if args.episode_horizon is not None:
        config_kwargs["episode_horizon"] = args.episode_horizon
    if args.children_per_parent is not None:
        config_kwargs["children_per_parent"] = args.children_per_parent
    if args.episodes_per_eval is not None:
        config_kwargs["episodes_per_eval"] = args.episodes_per_eval
    if args.parameter_block_size is not None:
        config_kwargs["parameter_block_size"] = args.parameter_block_size
    if args.mutation_block_ratio is not None:
        config_kwargs["mutation_block_ratio"] = args.mutation_block_ratio

    config = make_config_mujoco_playground_generic(args.env_id, **config_kwargs)
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


if __name__ == "__main__":
    main()
