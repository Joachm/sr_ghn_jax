from __future__ import annotations

import argparse
import pickle

from configs import make_config_ant_brax
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the Ant Brax SRGHN experiment.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Brax backend, e.g., spring, generalized, mjx (defaults to Brax default).",
    )
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    args = parser.parse_args()

    config = make_config_ant_brax()
    config_updates = {}
    if args.backend is not None:
        config_updates["brax_backend"] = args.backend
    if args.parameter_block_size is not None:
        config_updates["parameter_block_size"] = args.parameter_block_size
    if args.mutation_block_ratio is not None:
        config_updates["mutation_block_ratio"] = args.mutation_block_ratio
    if config_updates:
        config = config.__class__(**{**config.__dict__, **config_updates})
    final_state, metrics = run_experiment(config)
    out_name = f"ant_brax_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
