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
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=1000)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    args = parser.parse_args()

    config = make_config_brax_generic(
        args.env_id,
        seed=args.seed,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        episode_horizon=args.episode_horizon,
        children_per_parent=args.children_per_parent,
        episodes_per_eval=args.episodes_per_eval,
        brax_backend=args.backend,
    )
    final_state, metrics = run_experiment(config)
    out_name = f"brax_{args.env_id}_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
