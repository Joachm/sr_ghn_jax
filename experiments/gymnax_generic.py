from __future__ import annotations

import argparse
import pickle

from configs import make_config_gymnax_generic
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a generic Gymnax SRGHN experiment.")
    parser.add_argument("--env-id", required=True, help="Gymnax environment id, e.g., CartPole-v1")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=300)
    parser.add_argument("--episode-horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--parameter-block-size", type=int, default=64)
    parser.add_argument("--mutation-block-ratio", type=float, default=0.125)
    args = parser.parse_args()

    config = make_config_gymnax_generic(
        args.env_id,
        seed=args.seed,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        episode_horizon=args.episode_horizon,
        children_per_parent=args.children_per_parent,
        episodes_per_eval=args.episodes_per_eval,
        parameter_block_size=args.parameter_block_size,
        mutation_block_ratio=args.mutation_block_ratio,
    )
    final_state, metrics = run_experiment(config)
    out_name = f"gymnax_{args.env_id}_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
