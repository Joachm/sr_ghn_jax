from __future__ import annotations

import argparse
from configs import make_config_ant_brax
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run the Ant Brax SRGHN experiment.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Brax backend, e.g., spring, generalized, mjx (defaults to Brax default).",
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

    config = make_config_ant_brax()
    overrides = {}
    if args.backend is not None:
        overrides["brax_backend"] = args.backend
    if args.self_reg_mode is not None:
        overrides["self_reg_mode"] = args.self_reg_mode
    if args.self_weight_decay is not None:
        overrides["self_weight_decay"] = args.self_weight_decay
    if overrides:
        config = config.__class__(**{**config.__dict__, **overrides})
    run_experiment(config)


if __name__ == "__main__":
    main()
