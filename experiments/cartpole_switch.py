from __future__ import annotations

import pickle

from configs import make_config_cartpole_switch
from experiments._common import run_experiment


def main():
    config = make_config_cartpole_switch()
    final_state, metrics = run_experiment(config)
    with open("cartpole_switch_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    main()
