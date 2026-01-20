from __future__ import annotations

import argparse
from configs import make_config_mujoco_playground_generic
from experiments._common import run_experiment


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

    config = make_config_mujoco_playground_generic(args.env_id, **config_kwargs)
    run_experiment(config)


if __name__ == "__main__":
    main()
