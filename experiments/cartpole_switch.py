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
    parser.add_argument(
        "--policy-head-max-out",
        type=int,
        default=None,
        help="Optional deterministic policy-head output width cap/floor (must be >= max policy leaf).",
    )
    parser.add_argument(
        "--self-shard-size",
        type=int,
        default=None,
        help="Shard size used for self-mutation stochastic head output.",
    )
    parser.add_argument(
        "--shard-graph-mode",
        choices=("dense", "sibling_chain", "hub"),
        default=None,
        help="Shard graph topology: dense clique+bipartite, sibling_chain, or hub.",
    )
    parser.add_argument(
        "--shard-residual-scale",
        type=float,
        default=None,
        help="Residual blend scale in group+residual updates (delta = group + scale * residual).",
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
    if args.policy_head_max_out is not None:
        overrides["policy_head_max_out"] = args.policy_head_max_out
    if args.self_shard_size is not None:
        overrides["self_shard_size"] = args.self_shard_size
    if args.shard_graph_mode is not None:
        overrides["shard_graph_mode"] = args.shard_graph_mode
    if args.shard_residual_scale is not None:
        overrides["shard_residual_scale"] = args.shard_residual_scale
    if overrides:
        config = config.__class__(**{**config.__dict__, **overrides})
    run_experiment(config)


if __name__ == "__main__":
    main()
