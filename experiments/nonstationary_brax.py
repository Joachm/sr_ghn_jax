from __future__ import annotations

import argparse

from adaptation_analysis import build_run_artifact, save_pickle
from configs import make_config_nonstationary_brax
from experiments._adaptation import brax_shift_windows
from experiments._common import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run a nonstationary Brax SR-GHN experiment.")
    parser.add_argument("--env-id", default="ant")
    parser.add_argument("--variant", default="brax_direction_switch", choices=("brax_direction_switch", "brax_speed_target_switch"))
    parser.add_argument("--baseline", default="srghn_full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=1000)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--parameter-block-size", type=int, default=64)
    parser.add_argument("--mutation-block-ratio", type=float, default=0.125)
    parser.add_argument("--fixed-mutation-lr", type=float, default=0.02)
    parser.add_argument("--target-speed", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = make_config_nonstationary_brax(
        args.env_id,
        seed=args.seed,
        pop_size=args.pop_size,
        num_generations=args.num_generations,
        episode_horizon=args.episode_horizon,
        children_per_parent=args.children_per_parent,
        episodes_per_eval=args.episodes_per_eval,
        brax_backend=args.backend,
        parameter_block_size=args.parameter_block_size,
        mutation_block_ratio=args.mutation_block_ratio,
        shift_windows=brax_shift_windows(args.variant, target_speed=args.target_speed),
        baseline_name=args.baseline,
        fixed_mutation_lr=args.fixed_mutation_lr,
    )
    _, metrics = run_experiment(config)
    artifact = build_run_artifact(config, metrics)
    safe_env_id = args.env_id.replace("/", "_").replace(":", "_")
    out_name = args.output or f"nonstationary_brax_{safe_env_id}_{args.variant}_{args.baseline}_seed{args.seed}.pkl"
    save_pickle(out_name, artifact)


if __name__ == "__main__":
    main()
