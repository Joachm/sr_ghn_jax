from __future__ import annotations

import pickle

from ..configs import make_config_ant_brax
from ._common import run_experiment


def main():
    config = make_config_ant_brax()
    final_state, metrics = run_experiment(config)
    with open("ant_brax_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
