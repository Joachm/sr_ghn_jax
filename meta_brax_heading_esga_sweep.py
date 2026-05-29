from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict
from pathlib import Path

from experiment_configs import print_resolved_config, resolved_config_payload
from meta_brax_heading import MetaBraxConfig, make_base_cfg, parse_condition_spec, run_condition


DEFAULT_SWEEP_CONDITIONS = ("open_es_open_es", "simple_ga_simple_ga")
DEFAULT_SIGMA_VALUES = (0.05, 0.1, 0.2, 0.5)
DEFAULT_DECAY_VALUES = (1.0, 0.9995, 0.999)
DEFAULT_WANDB_PROJECT = "meta_brax_heading_prelim_sweeps"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep sigma and std-decay settings for Open_ES and SimpleGA on the meta-Brax heading task."
    )
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_SWEEP_CONDITIONS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sigma-values", nargs="+", type=float, default=DEFAULT_SIGMA_VALUES)
    parser.add_argument("--decay-values", nargs="+", type=float, default=DEFAULT_DECAY_VALUES)
    parser.add_argument("--output", default="meta_brax_heading_esga_sweep.pkl")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-id", default="ant")
    parser.add_argument("--backend", default="spring")
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
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default="meta_brax_heading_esga_sweep")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-log-plots", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _make_run_name(condition_name: str, sigma: float, decay: float, seed: int) -> str:
    sigma_token = str(sigma).replace(".", "p")
    decay_token = str(decay).replace(".", "p")
    return f"sweep_{condition_name}_sigma{sigma_token}_decay{decay_token}_seed{seed}"


def _summary_row(result: dict, *, sigma: float, decay: float, seed: int) -> dict:
    curve = result["adaptation_curve_query_return"]
    history = result["train_history"]
    return {
        "condition": result["condition"]["name"],
        "seed": int(seed),
        "sigma_init": float(sigma),
        "std_decay": float(decay),
        "champion_meta_fitness": float(result["champion_meta_fitness"]),
        "heldout_pre_return_mean": float(curve["mean"][0]),
        "heldout_post_return_mean": float(curve["mean"][-1]),
        "heldout_improvement_mean": float(curve["mean"][-1] - curve["mean"][0]),
        "final_train_query_return_best": float(history["query_return_best"][-1]),
        "final_train_query_return_mean": float(history["query_return_mean"][-1]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = make_base_cfg(args)
    if args.print_config:
        print_resolved_config(
            base_cfg,
            family="meta_brax_heading_esga_sweep",
            run_preset="fast" if args.fast else "default",
        )
        return 0

    seeds = args.seeds or [args.seed]
    sigma_values = tuple(float(value) for value in args.sigma_values)
    decay_values = tuple(float(value) for value in args.decay_values)

    results: list[dict] = []
    summary_rows: list[dict] = []

    for condition_name in args.conditions:
        if condition_name not in DEFAULT_SWEEP_CONDITIONS:
            raise ValueError(
                f"{condition_name!r} is not supported by this sweep runner. "
                f"Use one of {DEFAULT_SWEEP_CONDITIONS}."
            )
        for sigma in sigma_values:
            for decay in decay_values:
                for seed in seeds:
                    cfg_updates = {
                        **asdict(base_cfg),
                        "seed": seed,
                        "outer_evosax_sigma_init": sigma,
                        "inner_evosax_sigma_init": sigma,
                        "outer_evosax_std_decay": decay,
                        "inner_evosax_std_decay": decay,
                    }
                    if not args.no_wandb:
                        cfg_updates["wandb_project"] = args.wandb_project
                        cfg_updates["wandb_group"] = args.wandb_group
                        cfg_updates["wandb_name"] = args.wandb_name or _make_run_name(condition_name, sigma, decay, seed)
                    else:
                        cfg_updates["wandb_project"] = None
                    cfg = MetaBraxConfig(**cfg_updates)
                    cond = parse_condition_spec(condition_name, fixed_mutation_lr=cfg.baseline_fixed_mutation_lr)
                    print(
                        f"[run] condition={condition_name} seed={seed} sigma={sigma:.6g} decay={decay:.6g}",
                        flush=True,
                    )
                    result = run_condition(cfg, cond)
                    results.append(
                        {
                            "condition_name": condition_name,
                            "sigma_init": sigma,
                            "std_decay": decay,
                            "seed": seed,
                            "result": result,
                        }
                    )
                    summary = _summary_row(result, sigma=sigma, decay=decay, seed=seed)
                    summary_rows.append(summary)
                    print(
                        "[done] "
                        f"condition={condition_name} seed={seed} sigma={sigma:.6g} decay={decay:.6g} "
                        f"heldout_post={summary['heldout_post_return_mean']:.3f} "
                        f"improvement={summary['heldout_improvement_mean']:.3f}",
                        flush=True,
                    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_config": asdict(base_cfg),
        "resolved_config": resolved_config_payload(
            base_cfg,
            family="meta_brax_heading_esga_sweep",
            run_preset="fast" if args.fast else "default",
        ),
        "conditions": list(args.conditions),
        "seeds": list(seeds),
        "sigma_values": list(sigma_values),
        "decay_values": list(decay_values),
        "summary": summary_rows,
        "results": results,
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[saved] {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
