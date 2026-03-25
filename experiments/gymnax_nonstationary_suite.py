from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from adaptation_analysis import build_run_artifact, save_pickle
from configs import make_config_nonstationary_gymnax
from experiments._adaptation import gymnax_suite_shift_windows
from experiments._common import run_experiment


ENVIRONMENTS = (
    "CartPole-v1",
    "Acrobot-v1",
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
)


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def main():
    parser = argparse.ArgumentParser(description="Run the full nonstationary Gymnax SR-GHN suite.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list; defaults to 0..9.")
    parser.add_argument("--output-dir", default="results/gymnax_nonstationary_suite")
    parser.add_argument("--project", default="srghn-gymnax12")
    parser.add_argument("--fixed-mutation-lr", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-generations", type=int, default=1500)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--children-per-parent", type=int, default=4)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--episode-horizon", type=int, default=500)
    parser.add_argument("--parameter-block-size", type=int, default=1024)
    parser.add_argument("--mutation-block-ratio", type=float, default=1.)
    args = parser.parse_args()

    seeds = args.seeds or list(range(10))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "failures.log"

    for env_id in ENVIRONMENTS:
        env_dir = output_dir / _safe_name(env_id)
        env_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            out_path = env_dir / f"seed_{seed}.pkl"
            if args.skip_existing and out_path.exists():
                print(f"[skip] {env_id} seed={seed} -> {out_path}")
                continue

            config = make_config_nonstationary_gymnax(
                env_id,
                seed=seed,
                pop_size=args.pop_size,
                num_generations=args.num_generations,
                episode_horizon=args.episode_horizon,
                children_per_parent=args.children_per_parent,
                episodes_per_eval=args.episodes_per_eval,
                parameter_block_size=args.parameter_block_size,
                mutation_block_ratio=args.mutation_block_ratio,
                shift_windows=gymnax_suite_shift_windows(env_id),
                wandb_project=args.project,
                wandb_group=_safe_name(env_id),
                wandb_name=f"{_safe_name(env_id)}-seed{seed}",
                fixed_mutation_lr=args.fixed_mutation_lr,
            )
            print(f"[run] env={env_id} seed={seed}")
            try:
                _, metrics = run_experiment(config)
                artifact = build_run_artifact(config, metrics)
                save_pickle(out_path, artifact)
                print(f"[saved] {out_path}")
            except Exception:
                tb = traceback.format_exc()
                message = f"[failed] env={env_id} seed={seed}\n{tb}\n"
                print(message, flush=True)
                with failure_log.open("a", encoding="utf-8") as f:
                    f.write(message)


if __name__ == "__main__":
    main()
