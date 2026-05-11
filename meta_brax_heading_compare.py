from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path
from math import sqrt

import jax
import jax.numpy as jnp

from experiment_configs import print_resolved_config, resolved_config_payload
from meta_brax_heading import (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-seed SR-GHN comparison for the meta-Brax heading task.")
    parser.add_argument("--conditions", nargs="+", default=BASELINE_NAMES)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="meta_brax_heading_compare.pkl")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", default="ant")
    parser.add_argument("--backend", default="spring")
    parser.add_argument("--outer-generations", type=int, default=200)
    parser.add_argument("--meta-batch-size", type=int, default=4)
    parser.add_argument("--heldout-task-batch-size", type=int, default=16)
    parser.add_argument("--outer-pop-size", type=int, default=16)
    parser.add_argument("--outer-children-per-parent", type=int, default=1)
    parser.add_argument("--inner-pop-size", type=int, default=4)
    parser.add_argument("--inner-children-per-parent", type=int, default=1)
    parser.add_argument("--inner-generations", type=int, default=2)
    parser.add_argument("--support-episodes", type=int, default=2)
    parser.add_argument("--query-episodes", type=int, default=2)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--policy-hidden-dims", nargs="+", type=int, default=(32, 32, 32))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--gnn-hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-steps-policy", type=int, default=10)
    parser.add_argument("--gnn-steps-self", type=int, default=10)
    parser.add_argument("--stoch-coeff-dim", type=int, default=32)
    parser.add_argument("--parameter-block-size", type=int, default=64)
    parser.add_argument("--mutation-block-ratio", type=float, default=0.125)
    parser.add_argument("--mutation-rate-head-dim", type=int, default=5)
    parser.add_argument("--const-noise-std", type=float, default=1e-3)
    parser.add_argument("--fixed-mutation-lr", type=float, default=0.02)
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
            print(f"[run] baseline={baseline_name} seed={seed}", flush=True)
            results.append(run_condition(cfg, cond))

    payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading_compare",
            run_preset="fast" if args.fast else "default",
        ),
        "seeds": seeds,
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
