from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from math import sqrt

import jax
import jax.numpy as jnp

from experiment_configs import print_resolved_config, resolved_config_payload
from meta_brax_heading import (
    ALL_CONDITION_NAMES,
    BASELINE_NAMES,
    MetaBraxConfig,
    make_base_cfg,
    parse_condition_spec,
    run_condition,
)


def _aggregate_series(series_list: list[jnp.ndarray]) -> dict[str, object]:
    stacked = jnp.stack(series_list)
    n = max(stacked.shape[0], 1)
    return {
        "mean": jax.device_get(stacked.mean(axis=0)),
        "stderr": jax.device_get(stacked.std(axis=0) / sqrt(n)),
    }


def aggregate_results(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        name = str(result["condition"]["name"])
        grouped.setdefault(name, []).append(result)

    aggregate: dict[str, object] = {}
    for name, runs in grouped.items():
        history_keys = runs[0]["train_history"].keys()
        aggregate[name] = {
            "num_runs": len(runs),
            "condition": runs[0]["condition"],
            "config": runs[0]["config"],
            "train_history": {
                key: _aggregate_series([jnp.asarray(run["train_history"][key]) for run in runs])
                for key in history_keys
            },
            "adaptation_curve_query_return": _aggregate_series(
                [jnp.asarray(run["adaptation_curve_query_return"]["mean"]) for run in runs]
            ),
            "final_population_fitness": _aggregate_series(
                [jnp.asarray(run["final_population_fitness"]) for run in runs]
            ),
        }
    return aggregate


def vector_outer_candidate_evals(cfg: MetaBraxConfig) -> int:
    return cfg.outer_pop_size


def srghn_outer_candidate_evals(cfg: MetaBraxConfig) -> int:
    return cfg.outer_pop_size * (1 + cfg.outer_children_per_parent)


def vector_inner_support_candidate_evals(cfg: MetaBraxConfig) -> int:
    return cfg.inner_pop_size * (1 + cfg.inner_generations)


def srghn_inner_support_candidate_evals(cfg: MetaBraxConfig) -> int:
    # Parent support fitness is cached within an adaptation task; only children
    # require new support rollouts after the initial population evaluation.
    return cfg.inner_pop_size * (1 + cfg.inner_generations * cfg.inner_children_per_parent)


def environment_episodes_per_task(cfg: MetaBraxConfig, *, candidate_evals: int) -> int:
    """Count support and query environment episodes for one adaptation task."""
    return candidate_evals * cfg.support_episodes + cfg.query_episodes


def environment_episodes_per_outer_generation(
    cfg: MetaBraxConfig, *, outer_candidate_evals: int, inner_candidate_evals: int
) -> int:
    return outer_candidate_evals * cfg.meta_batch_size * environment_episodes_per_task(
        cfg, candidate_evals=inner_candidate_evals
    )


def budget_matched_srghn_config(vector_cfg: MetaBraxConfig) -> MetaBraxConfig:
    """Map a vector baseline budget to SR-GHN without changing its outer operator."""
    candidates_per_outer_parent = 3  # one parent plus two children
    if vector_cfg.outer_pop_size % candidates_per_outer_parent:
        raise ValueError(
            "Budget-matched SR-GHN requires a vector outer population divisible by three."
        )
    return replace(
        vector_cfg,
        outer_pop_size=vector_cfg.outer_pop_size // candidates_per_outer_parent,
        outer_children_per_parent=2,
        inner_pop_size=vector_cfg.inner_pop_size,
        inner_children_per_parent=1,
        inner_generations=vector_cfg.inner_generations,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-seed comparison for the meta-Brax heading task.")
    parser.add_argument("--conditions", nargs="+", default=BASELINE_NAMES)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="meta_brax_heading_compare.pkl")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-preset", default="default", choices=("default", "fast"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--backend", "--brax-backend", dest="backend", default=None)
    parser.add_argument("--outer-generations", type=int, default=None)
    parser.add_argument("--meta-batch-size", type=int, default=None)
    parser.add_argument("--heldout-task-batch-size", type=int, default=None)
    parser.add_argument("--outer-pop-size", type=int, default=None)
    parser.add_argument("--outer-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-pop-size", type=int, default=None)
    parser.add_argument("--inner-children-per-parent", type=int, default=None)
    parser.add_argument("--inner-generations", type=int, default=None)
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
    parser.add_argument("--outer-evosax-sigma-init", type=float, default=None)
    parser.add_argument("--inner-evosax-sigma-init", type=float, default=None)
    parser.add_argument("--outer-evosax-std-decay", type=float, default=None)
    parser.add_argument("--inner-evosax-std-decay", type=float, default=None)
    parser.add_argument(
        "--budget-match-srghn",
        action="store_true",
        help="Use the cached-inner-fitness SR-GHN configuration that matches vector candidate budgets.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(args)
    if args.print_config:
        print_resolved_config(
            base_cfg,
            family="meta_brax_heading_compare",
            run_preset="fast" if args.fast else "default",
        )
        return 0
    seeds = args.seeds or list(range(5))

    results: list[dict[str, object]] = []
    for baseline_name in args.conditions:
        for seed in seeds:
            cfg = MetaBraxConfig(**{**asdict(base_cfg), "seed": seed})
            cond = parse_condition_spec(baseline_name, fixed_mutation_lr=cfg.baseline_fixed_mutation_lr)
            if args.budget_match_srghn and cond.search_object == "srghn":
                cfg = budget_matched_srghn_config(cfg)
            print(f"[run] baseline={baseline_name} seed={seed}", flush=True)
            result = run_condition(cfg, cond)
            result["candidate_evaluation_budget"] = {
                "outer_per_generation": (
                    srghn_outer_candidate_evals(cfg)
                    if cond.search_object == "srghn"
                    else vector_outer_candidate_evals(cfg)
                ),
                "inner_support_per_task": (
                    srghn_inner_support_candidate_evals(cfg)
                    if cond.search_object == "srghn"
                    else vector_inner_support_candidate_evals(cfg)
                ),
            }
            results.append(result)

    payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading_compare",
            run_preset="fast" if args.fast else "default",
        ),
        "seeds": seeds,
        "budget_match_srghn": args.budget_match_srghn,
        "results": results,
        "aggregate": aggregate_results(results),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
