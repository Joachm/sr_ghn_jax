from __future__ import annotations

import argparse
from configs import make_config_gymnax_generic
from experiments._common import run_experiment


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("policy hidden dims must be a comma-separated list of positive ints.")
    return dims


def main():
    parser = argparse.ArgumentParser(description="Run a generic Gymnax SRGHN experiment.")
    parser.add_argument("--env-id", required=True, help="Gymnax environment id, e.g., CartPole-v1")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=300)
    parser.add_argument("--episode-horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument(
        "--policy-hidden-dims",
        type=_parse_hidden_dims,
        default=(32,),
        help="Comma-separated hidden sizes, e.g., 64,64,32",
    )
    parser.add_argument(
        "--freeze-stoch-output-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze/unfreeze stochastic output basis (use --freeze-stoch-output-head or --no-freeze-stoch-output-head).",
    )
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
    args = parser.parse_args()

    config = make_config_gymnax_generic(
        args.env_id,
        seed=args.seed,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        episode_horizon=args.episode_horizon,
        children_per_parent=args.children_per_parent,
        episodes_per_eval=args.episodes_per_eval,
        policy_hidden_dims=args.policy_hidden_dims,
        freeze_stoch_output_head=(
            False if args.freeze_stoch_output_head is None else args.freeze_stoch_output_head
        ),
    )
    overrides = {}
    if args.self_reg_mode is not None:
        overrides["self_reg_mode"] = args.self_reg_mode
    if args.self_weight_decay is not None:
        overrides["self_weight_decay"] = args.self_weight_decay
    if overrides:
        config = config.__class__(**{**config.__dict__, **overrides})
    run_experiment(config)


if __name__ == "__main__":
    main()
