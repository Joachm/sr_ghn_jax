from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from adaptation_analysis import build_run_artifact, save_pickle
from baselines.ppo.config import PPO_GYMNAX_SUITE_ENVIRONMENTS, make_ppo_config_nonstationary_gymnax
from baselines.ppo.runner import run_ppo
from experiment_configs import print_resolved_config, resolve_ppo_nonstationary_gymnax_config


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _namespace_for_config(config) -> str:
    return (
        f"ppo_u{config.num_generations}"
        f"_nenv{config.num_envs}"
        f"_roll{config.rollout_length}"
        f"_mb{config.num_minibatches}"
        f"_ep{config.update_epochs}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PPO baseline for the nonstationary Gymnax suite.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list; defaults to 0..4.")
    parser.add_argument("--output-dir", default="results/ppo_gymnax_nonstationary_suite")
    parser.add_argument("--project", default=None)
    parser.add_argument("--task-preset", default="suite_default", choices=("suite_default",))
    parser.add_argument("--run-preset", default="default", choices=("default", "debug"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-generations", type=int, default=1500)
    parser.add_argument("--episode-horizon", type=int, default=500)
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    seeds = args.seeds or list(range(5))
    template = resolve_ppo_nonstationary_gymnax_config(
        PPO_GYMNAX_SUITE_ENVIRONMENTS[0],
        task_preset=args.task_preset,
        run_preset=args.run_preset,
        overrides={
            "num_generations": args.num_generations,
            "episode_horizon": args.episode_horizon,
            "episodes_per_eval": args.episodes_per_eval,
            "wandb_project": args.project,
            "num_envs": args.num_envs,
            "rollout_length": args.rollout_length,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "clip_eps": args.clip_eps,
            "gae_lambda": args.gae_lambda,
            "gamma": args.gamma,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "max_grad_norm": args.max_grad_norm,
        },
    )
    if args.print_config:
        print_resolved_config(
            template,
            family="ppo.gymnax_nonstationary_suite",
            task_preset=args.task_preset,
            run_preset=args.run_preset,
        )
        return 0
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for env_id in PPO_GYMNAX_SUITE_ENVIRONMENTS:
        for seed in seeds:
            config = resolve_ppo_nonstationary_gymnax_config(
                env_id,
                task_preset=args.task_preset,
                run_preset=args.run_preset,
                overrides={
                    "seed": seed,
                    "num_generations": args.num_generations,
                    "episode_horizon": args.episode_horizon,
                    "episodes_per_eval": args.episodes_per_eval,
                    "wandb_project": args.project,
                    "wandb_group": _safe_name(env_id),
                    "wandb_name": f"{_safe_name(env_id)}-seed{seed}",
                    "num_envs": args.num_envs,
                    "rollout_length": args.rollout_length,
                    "num_minibatches": args.num_minibatches,
                    "update_epochs": args.update_epochs,
                    "learning_rate": args.learning_rate,
                    "clip_eps": args.clip_eps,
                    "gae_lambda": args.gae_lambda,
                    "gamma": args.gamma,
                    "entropy_coef": args.entropy_coef,
                    "value_coef": args.value_coef,
                    "max_grad_norm": args.max_grad_norm,
                },
            )
            suite_dir = output_root / _namespace_for_config(config)
            suite_dir.mkdir(parents=True, exist_ok=True)
            failure_log = suite_dir / "failures.log"
            env_dir = suite_dir / _safe_name(env_id)
            env_dir.mkdir(parents=True, exist_ok=True)
            out_path = env_dir / f"seed_{seed}.pkl"
            if args.skip_existing and out_path.exists():
                print(f"[skip] {env_id} seed={seed} -> {out_path}")
                continue
            print(f"[run] env={env_id} seed={seed}")
            try:
                _, metrics = run_ppo(config)
                artifact = build_run_artifact(config, metrics)
                save_pickle(out_path, artifact)
                print(f"[saved] {out_path}")
            except Exception:
                tb = traceback.format_exc()
                message = f"[failed] env={env_id} seed={seed}\n{tb}\n"
                print(message, flush=True)
                with failure_log.open("a", encoding="utf-8") as handle:
                    handle.write(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
