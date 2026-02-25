from __future__ import annotations

import argparse

from configs import make_config_cartpole_switch
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the CartPole switch SRGHN experiment.")
    parser.add_argument(
        "--self-reg-mode",
        choices=("weight_norm", "weight_decay", "none"),
        default=None,
        help="Self-regularization mode for SRGHN mutation.",
    )
    parser.add_argument(
        "--self-weight-decay",
        type=float,
        default=None,
        help="Weight decay factor used when --self-reg-mode=weight_decay.",
    )
    parser.add_argument(
        "--stoch-cov-rank",
        type=int,
        default=None,
        help="Low-rank covariance rank for correlated mutation exploration (0 disables).",
    )
    parser.add_argument(
        "--stoch-cov-scale",
        type=float,
        default=None,
        help="Scale applied to low-rank correlated mutation component.",
    )
    args = parser.parse_args()

    config = make_config_cartpole_switch()
    overrides = {}
    if args.self_reg_mode is not None:
        overrides["self_reg_mode"] = args.self_reg_mode
    if args.self_weight_decay is not None:
        overrides["self_weight_decay"] = args.self_weight_decay
    if args.stoch_cov_rank is not None:
        overrides["stoch_cov_rank"] = args.stoch_cov_rank
    if args.stoch_cov_scale is not None:
        overrides["stoch_cov_scale"] = args.stoch_cov_scale
    if overrides:
        config = config.__class__(**{**config.__dict__, **overrides})
    run_experiment(config)


if __name__ == "__main__":
    main()
