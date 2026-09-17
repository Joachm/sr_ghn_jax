#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m experiments.gymnax_minatar_suite \
    --optimizer-family srghn \
    --baseline srghn_full \
    --seeds 0 1 2 3 4 \
    --pop-size 200 \
    --children-per-parent 2 \
    --srghn-replacement-mode generational \
    --eval-budget-per-generation 200 \
    --num-generations 24000 \
    --episodes-per-eval 1 \
    --episode-horizon 2500 \
    --parameter-block-size 4096 \
    --mutation-block-ratio 1.0 \
    --project srghn-spaceinvaders-population-eval-matched \
    --output-dir results/gymnax_minatar_population_eval_matched

python meta_brax_heading_compare.py \
    --conditions srghn_full \
    --seeds 0 1 2 3 4 \
    --outer-pop-size 42 \
    --outer-children-per-parent 2 \
    --inner-pop-size 2 \
    --inner-children-per-parent 1 \
    --inner-generations 4 \
    --outer-generations 1500 \
    --meta-batch-size 12 \
    --heldout-task-batch-size 16 \
    --support-episodes 2 \
    --query-episodes 2 \
    --episode-horizon 1000 \
    --parameter-block-size 1024 \
    --mutation-block-ratio 1.0 \
    --env-id ant \
    --brax-backend spring \
    --outer-replacement-mode generational \
    --budget-match-srghn \
    --wandb-project meta_brax_heading_population_eval_matched \
    --wandb-group srghn_population_eval_matched \
    --output results/meta_brax_heading_population_eval_matched.pkl
