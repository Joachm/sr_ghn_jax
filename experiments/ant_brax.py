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
    args = parser.parse_args()

    config = make_config_ant_brax()
    if args.backend is not None:
        config = config.__class__(**{**config.__dict__, "brax_backend": args.backend})
    final_state, metrics = run_experiment(config)
    out_name = f"ant_brax_metrics.pkl"
    with open(out_name, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
