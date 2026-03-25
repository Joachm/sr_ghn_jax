from __future__ import annotations

import argparse

from adaptation_analysis import aggregate_comparison_results, build_run_artifact, save_pickle
from configs import make_config_nonstationary_brax, make_config_nonstationary_gymnax
from experiments._adaptation import DEFAULT_BASELINES, brax_shift_windows, gymnax_shift_windows
from experiments._common import run_experiment


def _run_cartpole(seed: int, baseline: str, fixed_mutation_lr: float):
    config = make_config_nonstationary_gymnax(
        "CartPole-v1",
        seed=seed,
        num_generations=1500,
        shift_windows=gymnax_shift_windows("cartpole_flip"),
        baseline_name=baseline,
        fixed_mutation_lr=fixed_mutation_lr,
    )
    _, metrics = run_experiment(config)
    return build_run_artifact(config, metrics)


def _run_cartpole_evosax(seed: int, algo: str, sigma_init: float | None):
    config = make_config_nonstationary_gymnax(
        "CartPole-v1",
        seed=seed,
        num_generations=1500,
        shift_windows=gymnax_shift_windows("cartpole_flip"),
        optimizer_family="evosax",
        baseline_name="evosax",
        evosax_algo=algo,
        evosax_sigma_init=sigma_init,
    )
    _, metrics = run_experiment(config)
    return build_run_artifact(config, metrics)


def _run_repeated(seed: int, baseline: str, fixed_mutation_lr: float):
    config = make_config_nonstationary_gymnax(
        "CartPole-v1",
        seed=seed,
        num_generations=1500,
        shift_windows=gymnax_shift_windows("cartpole_repeated"),
        baseline_name=baseline,
        fixed_mutation_lr=fixed_mutation_lr,
    )
    _, metrics = run_experiment(config)
    return build_run_artifact(config, metrics)


def _run_repeated_evosax(seed: int, algo: str, sigma_init: float | None):
    config = make_config_nonstationary_gymnax(
        "CartPole-v1",
        seed=seed,
        num_generations=1500,
        shift_windows=gymnax_shift_windows("cartpole_repeated"),
        optimizer_family="evosax",
        baseline_name="evosax",
        evosax_algo=algo,
        evosax_sigma_init=sigma_init,
    )
    _, metrics = run_experiment(config)
    return build_run_artifact(config, metrics)


def _run_brax(seed: int, baseline: str, fixed_mutation_lr: float, backend: str | None):
    config = make_config_nonstationary_brax(
        "ant",
        seed=seed,
        brax_backend=backend,
        shift_windows=brax_shift_windows("brax_direction_switch"),
        baseline_name=baseline,
        fixed_mutation_lr=fixed_mutation_lr,
    )
    _, metrics = run_experiment(config)
    return build_run_artifact(config, metrics)


def main():
    parser = argparse.ArgumentParser(description="Run SR-GHN adaptation baseline comparisons.")
    parser.add_argument("--suite", default="cartpole", choices=("cartpole", "brax", "repeated"))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--fixed-mutation-lr", type=float, default=None)
    parser.add_argument("--evosax-algos", nargs="*", default=())
    parser.add_argument("--evosax-sigma-init", type=float, default=None)
    parser.add_argument("--skip-srghn", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.suite == "cartpole":
        seeds = args.seeds or list(range(16))
        fixed_mutation_lr = 0.05 if args.fixed_mutation_lr is None else args.fixed_mutation_lr
        runner = lambda seed, baseline: _run_cartpole(seed, baseline, fixed_mutation_lr)
    elif args.suite == "repeated":
        seeds = args.seeds or list(range(16))
        fixed_mutation_lr = 0.05 if args.fixed_mutation_lr is None else args.fixed_mutation_lr
        runner = lambda seed, baseline: _run_repeated(seed, baseline, fixed_mutation_lr)
    else:
        seeds = args.seeds or list(range(8))
        fixed_mutation_lr = 0.02 if args.fixed_mutation_lr is None else args.fixed_mutation_lr
        runner = lambda seed, baseline: _run_brax(seed, baseline, fixed_mutation_lr, args.backend)

    results = []
    if not args.skip_srghn:
        for baseline in DEFAULT_BASELINES:
            for seed in seeds:
                print(f"[run] suite={args.suite} baseline={baseline} seed={seed}")
                results.append(runner(seed, baseline))

    if args.evosax_algos:
        if args.suite == "brax":
            raise ValueError("evosax controls are only implemented for Gymnax suites.")
        evosax_runner = _run_cartpole_evosax if args.suite == "cartpole" else _run_repeated_evosax
        for algo in args.evosax_algos:
            for seed in seeds:
                print(f"[run] suite={args.suite} baseline=evosax:{algo} seed={seed}")
                results.append(evosax_runner(seed, algo, args.evosax_sigma_init))

    payload = {
        "suite": args.suite,
        "seeds": seeds,
        "results": results,
        "aggregate": aggregate_comparison_results(results),
    }
    out_name = args.output or f"adaptation_compare_{args.suite}.pkl"
    save_pickle(out_name, payload)


if __name__ == "__main__":
    main()
