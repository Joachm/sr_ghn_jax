from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

from experiment_configs import print_resolved_config, resolved_config_payload
from meta_brax_heading import BASELINE_NAMES, MetaBraxConfig, make_base_cfg, parse_condition_spec, run_condition


DEFAULT_SWEEP_CONDITIONS = ("srghn_full",)
DEFAULT_INNER_POP_SIZES = (2, 4, 8)
DEFAULT_INNER_GENERATIONS = (0, 1, 2, 4)
DEFAULT_OUTER_POP_SIZES = (8, 16, 32)
DEFAULT_WANDB_PROJECT = "meta_brax_heading_srghn_inner_sweep"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep SR-GHN inner-loop adaptation budgets for the meta-Brax heading task."
    )
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_SWEEP_CONDITIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--inner-pop-sizes", nargs="+", type=int, default=DEFAULT_INNER_POP_SIZES)
    parser.add_argument("--inner-generations-values", nargs="+", type=int, default=DEFAULT_INNER_GENERATIONS)
    parser.add_argument("--outer-pop-sizes", nargs="+", type=int, default=DEFAULT_OUTER_POP_SIZES)
    parser.add_argument("--output", default="meta_brax_heading_srghn_inner_sweep.pkl")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-preset", default="default", choices=("default", "fast"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", default="ant")
    parser.add_argument("--backend", default="spring")
    parser.add_argument("--outer-generations", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--heldout-task-batch-size", type=int, default=None)
    parser.add_argument("--outer-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-children-per-parent", type=int, default=None)
    parser.add_argument("--support-episodes", type=int, default=None)
    parser.add_argument("--query-episodes", type=int, default=None)
    parser.add_argument("--episode-horizon", type=int, default=None)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--gnn-hidden-dim", type=int, default=None)
    parser.add_argument("--gnn-steps-policy", type=int, default=None)
    parser.add_argument("--gnn-steps-self", type=int, default=None)
    parser.add_argument("--stoch-coeff-dim", type=int, default=None)
    parser.add_argument("--parameter-block-size", type=int, default=None)
    parser.add_argument("--mutation-block-ratio", type=float, default=None)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=None)
    parser.add_argument("--const-noise-std", type=float, default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="meta_brax_heading_srghn_inner_sweep")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _make_base_cfg_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.setdefault("outer_pop_size", None)
    values.setdefault("inner_pop_size", None)
    values.setdefault("inner_generations", None)
    values.setdefault("outer_evosax_sigma_init", None)
    values.setdefault("inner_evosax_sigma_init", None)
    values.setdefault("outer_evosax_std_decay", None)
    values.setdefault("inner_evosax_std_decay", None)
    return argparse.Namespace(**values)


def _make_run_name(condition_name: str, inner_pop: int, inner_generations: int, outer_pop: int, seed: int) -> str:
    return (
        f"sweep_{condition_name}"
        f"_innerpop{inner_pop}"
        f"_innergen{inner_generations}"
        f"_outerpop{outer_pop}"
        f"_seed{seed}"
    )


def _inner_candidate_evals(inner_pop_size: int, inner_children_per_parent: int, inner_generations: int) -> int:
    if inner_generations <= 0:
        return int(inner_pop_size)
    return int(inner_pop_size * (1 + inner_children_per_parent) * inner_generations)


def _outer_candidate_evals(outer_pop_size: int, outer_children_per_parent: int) -> int:
    return int(outer_pop_size * (1 + outer_children_per_parent))


def _summary_row(result: dict, *, seed: int, inner_pop: int, inner_generations: int, outer_pop: int) -> dict:
    cfg = result["config"]
    curve = result["adaptation_curve_query_return"]
    history = result["train_history"]
    inner_budget = _inner_candidate_evals(
        inner_pop,
        int(cfg["inner_children_per_parent"]),
        inner_generations,
    )
    outer_budget = _outer_candidate_evals(
        outer_pop,
        int(cfg["outer_children_per_parent"]),
    )
    return {
        "condition": result["condition"]["name"],
        "seed": int(seed),
        "inner_pop_size": int(inner_pop),
        "inner_generations": int(inner_generations),
        "outer_pop_size": int(outer_pop),
        "inner_candidate_evals_per_task": inner_budget,
        "outer_candidate_evals_per_generation": outer_budget,
        "champion_meta_fitness": float(result["champion_meta_fitness"]),
        "heldout_pre_return_mean": float(curve["mean"][0]),
        "heldout_post_return_mean": float(curve["mean"][-1]),
        "heldout_improvement_mean": float(curve["mean"][-1] - curve["mean"][0]),
        "final_train_query_return_best": float(history["query_return_best"][-1]),
        "final_train_query_return_mean": float(history["query_return_mean"][-1]),
        "train_seconds": float(result["train_seconds"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(_make_base_cfg_args(args))
    if args.print_config:
        print_resolved_config(
            base_cfg,
            family="meta_brax_heading_srghn_inner_sweep",
            run_preset="fast" if args.fast else args.run_preset,
        )
        return 0

    seeds = args.seeds or [args.seed]
    inner_pop_sizes = tuple(int(value) for value in args.inner_pop_sizes)
    inner_generations_values = tuple(int(value) for value in args.inner_generations_values)
    outer_pop_sizes = tuple(int(value) for value in args.outer_pop_sizes)

    results: list[dict] = []
    summary_rows: list[dict] = []

    for condition_name in args.conditions:
        if condition_name not in BASELINE_NAMES:
            raise ValueError("This sweep runner is intended for SR-GHN conditions only.")
        for inner_pop in inner_pop_sizes:
            for inner_generations in inner_generations_values:
                for outer_pop in outer_pop_sizes:
                    for seed in seeds:
                        cfg_updates = {
                            **asdict(base_cfg),
                            "seed": seed,
                            "inner_pop_size": inner_pop,
                            "inner_generations": inner_generations,
                            "outer_pop_size": outer_pop,
                        }
                        if not args.no_wandb:
                            cfg_updates["wandb_project"] = args.wandb_project
                            cfg_updates["wandb_group"] = args.wandb_group
                            cfg_updates["wandb_name"] = args.wandb_name or _make_run_name(
                                condition_name,
                                inner_pop,
                                inner_generations,
                                outer_pop,
                                seed,
                            )
                        else:
                            cfg_updates["wandb_project"] = None
                        cfg = MetaBraxConfig(**cfg_updates)
                        cond = parse_condition_spec(condition_name, fixed_mutation_lr=cfg.baseline_fixed_mutation_lr)
                        print(
                            "[run] "
                            f"condition={condition_name} seed={seed} "
                            f"inner_pop={inner_pop} inner_gen={inner_generations} outer_pop={outer_pop}",
                            flush=True,
                        )
                        result = run_condition(cfg, cond)
                        results.append(
                            {
                                "condition_name": condition_name,
                                "seed": seed,
                                "inner_pop_size": inner_pop,
                                "inner_generations": inner_generations,
                                "outer_pop_size": outer_pop,
                                "result": result,
                            }
                        )
                        summary = _summary_row(
                            result,
                            seed=seed,
                            inner_pop=inner_pop,
                            inner_generations=inner_generations,
                            outer_pop=outer_pop,
                        )
                        summary_rows.append(summary)
                        print(
                            "[done] "
                            f"condition={condition_name} seed={seed} "
                            f"inner_pop={inner_pop} inner_gen={inner_generations} outer_pop={outer_pop} "
                            f"heldout_post={summary['heldout_post_return_mean']:.3f} "
                            f"improvement={summary['heldout_improvement_mean']:.3f} "
                            f"train_s={summary['train_seconds']:.1f}",
                            flush=True,
                        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading_srghn_inner_sweep",
            run_preset="fast" if args.fast else args.run_preset,
        ),
        "conditions": list(args.conditions),
        "seeds": list(seeds),
        "inner_pop_sizes": list(inner_pop_sizes),
        "inner_generations_values": list(inner_generations_values),
        "outer_pop_sizes": list(outer_pop_sizes),
        "summary": summary_rows,
        "results": results,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
