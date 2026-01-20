from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    config = make_config_gymnax_generic(
        args.env_id,
        seed=args.seed,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        episode_horizon=args.episode_horizon,
        children_per_parent=args.children_per_parent,
        episodes_per_eval=args.episodes_per_eval,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
