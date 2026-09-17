from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from adaptation_analysis import build_run_artifact, save_pickle
from experiment_configs import print_resolved_config, resolve_nonstationary_gymnax_config
from experiments._adaptation import GYMNAX_MINATAR_SUITE_ENVIRONMENTS, gymnax_minatar_suite_shift_windows
from experiments._common import run_experiment


POLICY_CONV_CHANNELS = (8,)
POLICY_CONV_KERNEL_SIZES = ((3, 3),)
POLICY_CONV_STRIDES = ((1, 1),)
POLICY_HIDDEN_DIMS = (128,)


def population_for_eval_budget(
    optimizer_family: str,
    eval_budget: int,
    children_per_parent: int,
    replacement_mode: str = "elitist_union",
) -> int:
    """Resolve the family-specific population size for one candidate budget."""
    if eval_budget <= 0:
        raise ValueError("eval_budget must be positive.")
    if children_per_parent < 0:
        raise ValueError("children_per_parent must be non-negative.")
    if optimizer_family == "evosax":
        return eval_budget
    if optimizer_family != "srghn":
        raise ValueError(f"Unknown optimizer family: {optimizer_family}")
    if replacement_mode == "generational":
        if eval_budget % 1:
            raise ValueError("Evaluation budget must be a whole population size.")
        return eval_budget
    if replacement_mode != "elitist_union":
        raise ValueError(f"Unknown SR-GHN replacement mode: {replacement_mode}")
    candidates_per_parent = 1 + children_per_parent
    if eval_budget % candidates_per_parent:
        raise ValueError(
            f"Evaluation budget {eval_budget} is not divisible by "
            f"1 + children_per_parent = {candidates_per_parent}."
        )
    return eval_budget // candidates_per_parent


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _label_for_config(config) -> str:
    if config.optimizer_family == "evosax":
        return f"evosax_{_safe_name(config.evosax_algo or 'unknown')}"
    return _safe_name(config.baseline_name)


def _namespace_for_config(config) -> str:
    return (
        f"{_label_for_config(config)}"
        f"_g{config.num_generations}"
        f"_p{config.pop_size}"
        f"_cpp{config.children_per_parent}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full nonstationary Gymnax MinAtar adaptation suite.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Explicit seed list; defaults to 0..4.")
    parser.add_argument("--output-dir", default="results/gymnax_minatar_suite")
    parser.add_argument("--project", default=None)
    parser.add_argument("--optimizer-family", default="srghn", choices=("srghn", "evosax"))
    parser.add_argument("--baseline", default="srghn_full")
    parser.add_argument("--task-preset", default="minatar_suite", choices=("minatar_suite",))
    parser.add_argument("--run-preset", default="default", choices=("default", "debug"))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--evosax-algo", default=None)
    parser.add_argument("--evosax-sigma-init", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-generations", type=int, default=24000)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument(
        "--eval-budget-per-generation",
        type=int,
        default=None,
        help="Candidate/environment evaluations per generation; resolves --pop-size per optimizer family.",
    )
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--srghn-replacement-mode", choices=("elitist_union", "generational"), default="generational")
    parser.add_argument("--episodes-per-eval", type=int, default=1)
    parser.add_argument("--episode-horizon", type=int, default=2500)
    parser.add_argument("--parameter-block-size", type=int, default=1024*4)
    parser.add_argument("--mutation-block-ratio", type=float, default=1.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = args.seeds or list(range(5))
    resolved_pop_size = (
        population_for_eval_budget(
            args.optimizer_family,
            args.eval_budget_per_generation,
            args.children_per_parent,
            args.srghn_replacement_mode,
        )
        if args.eval_budget_per_generation is not None
        else args.pop_size
    )
    if args.print_config:
        template = resolve_nonstationary_gymnax_config(
            GYMNAX_MINATAR_SUITE_ENVIRONMENTS[0],
            task_preset=args.task_preset,
            run_preset=args.run_preset,
            overrides={
                "optimizer_family": args.optimizer_family,
                "baseline_name": args.baseline,
                "evosax_algo": args.evosax_algo,
                "evosax_sigma_init": args.evosax_sigma_init,
                "fixed_mutation_lr": args.fixed_mutation_lr,
                "num_generations": args.num_generations,
                "pop_size": resolved_pop_size,
                "children_per_parent": args.children_per_parent,
                        "srghn_replacement_mode": args.srghn_replacement_mode,
                "episodes_per_eval": args.episodes_per_eval,
                "episode_horizon": args.episode_horizon,
                "parameter_block_size": args.parameter_block_size,
                "mutation_block_ratio": args.mutation_block_ratio,
            },
        )
        print(
            f"resident_population_size={template.pop_size} "
            f"num_reproducers={template.pop_size // template.children_per_parent if template.srghn_replacement_mode == 'generational' else template.pop_size} "
            f"evaluated_candidates_per_generation={template.pop_size if template.srghn_replacement_mode == 'generational' else template.pop_size * (1 + template.children_per_parent)}"
        )
        print_resolved_config(
            template,
            family="control.gymnax_minatar_suite",
            task_preset=args.task_preset,
            baseline=args.baseline,
            run_preset=args.run_preset,
        )
        return 0
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for env_id in GYMNAX_MINATAR_SUITE_ENVIRONMENTS:
        if "Space" in env_id:
            for seed in seeds:
                config = resolve_nonstationary_gymnax_config(
                    env_id,
                    task_preset=args.task_preset,
                    run_preset=args.run_preset,
                    overrides={
                        "seed": seed,
                        "pop_size": resolved_pop_size,
                        "num_generations": args.num_generations,
                        "episode_horizon": args.episode_horizon,
                        "children_per_parent": args.children_per_parent,
                        "srghn_replacement_mode": args.srghn_replacement_mode,
                        "episodes_per_eval": args.episodes_per_eval,
                        "parameter_block_size": args.parameter_block_size,
                        "mutation_block_ratio": args.mutation_block_ratio,
                        "optimizer_family": args.optimizer_family,
                        "baseline_name": args.baseline,
                        "evosax_algo": args.evosax_algo,
                        "evosax_sigma_init": args.evosax_sigma_init,
                        "policy_conv_channels": POLICY_CONV_CHANNELS,
                        "policy_conv_kernel_sizes": POLICY_CONV_KERNEL_SIZES,
                        "policy_conv_strides": POLICY_CONV_STRIDES,
                        "policy_hidden_dims": POLICY_HIDDEN_DIMS,
                        "wandb_project": args.project or ("srghn-minatar_3" if args.optimizer_family == "srghn" else f"sr-ghn_control_{_safe_name(args.evosax_algo or 'unknown')}_minatar"),
                        "wandb_group": _safe_name(env_id),
                        "wandb_name": f"{_safe_name(env_id)}-seed{seed}",
                        "fixed_mutation_lr": args.fixed_mutation_lr,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
