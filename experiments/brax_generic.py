from __future__ import annotations

import argparse
import pickle

from configs import make_config_brax_generic
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a generic Brax SRGHN experiment.")
    parser.add_argument("--env-id", required=True, help="Brax environment id, e.g., ant")
    parser.add_argument(
        "--backend",
        default=None,
        help="Brax backend, e.g., spring, generalized, mjx (defaults to Brax default).",
    )
    parser.add_argument("--pop-size", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--children-per-parent", type=int, default=None)
    parser.add_argument("--episodes-per-eval", type=int, default=None)
    args = parser.parse_args()

    config_kwargs = {
        "brax_backend": args.backend,
    }
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

    config = make_config_brax_generic(args.env_id, **config_kwargs)
    final_state, metrics = run_experiment(config)
    out_name = f"brax_{args.env_id}_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
