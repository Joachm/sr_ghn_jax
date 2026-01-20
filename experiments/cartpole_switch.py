from __future__ import annotations

from configs import make_config_cartpole_switch
from experiments._common import run_experiment


def main():
    config = make_config_cartpole_switch()
    run_experiment(config)


if __name__ == "__main__":
    main()
