from __future__ import annotations

import argparse
from configs import make_config_brax_generic
from experiments._common import run_experiment


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("policy hidden dims must be a comma-separated list of positive ints.")
    return dims


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
    parser.add_argument(
        "--policy-hidden-dims",
        type=_parse_hidden_dims,
        default=None,
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
    if args.policy_hidden_dims is not None:
        config_kwargs["policy_hidden_dims"] = args.policy_hidden_dims
    if args.freeze_stoch_output_head is not None:
        config_kwargs["freeze_stoch_output_head"] = args.freeze_stoch_output_head
    if args.self_reg_mode is not None:
        config_kwargs["self_reg_mode"] = args.self_reg_mode
    if args.self_weight_decay is not None:
        config_kwargs["self_weight_decay"] = args.self_weight_decay

    config = make_config_brax_generic(args.env_id, **config_kwargs)
    run_experiment(config)


if __name__ == "__main__":
    main()
