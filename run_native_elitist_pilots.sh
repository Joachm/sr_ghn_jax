#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# SpaceInvaders: native SR-GHN, 25 parents + 175 children = 200 evaluations.
python -m experiments.gymnax_minatar_suite \
    --optimizer-family srghn \
    --baseline srghn_full \
    --seeds 0 \
    --num-generations 6000 \
    --pop-size 25 \
    --children-per-parent 7 \
    --srghn-replacement-mode elitist_union \
    --eval-budget-per-generation 200 \
    --episodes-per-eval 1 \
    --episode-horizon 2500 \
    --parameter-block-size 4096 \
    --mutation-block-ratio 1.0 \
    --project srghn-spaceinvaders-native-elitist-pilot \
    --output-dir results/gymnax_minatar_native_elitist_pilot

# Meta-Brax A: 14 parents + 28 children = 42 outer evaluations.
python meta_brax_heading_compare.py \
    --conditions srghn_full \
    --seeds 0 \
    --outer-generations 500 \
    --outer-pop-size 14 \
    --outer-children-per-parent 2 \
    --outer-replacement-mode elitist_union \
    --inner-pop-size 2 \
    --inner-children-per-parent 1 \
    --inner-generations 4 \
    --meta-batch-size 12 \
    --heldout-task-batch-size 16 \
    --support-episodes 2 \
    --query-episodes 2 \
    --episode-horizon 1000 \
    --parameter-block-size 1024 \
    --mutation-block-ratio 1.0 \
    --env-id ant \
    --brax-backend spring \
    --wandb-project meta-brax-native-elitist-p14-c2-pilot \
    --wandb-group meta_brax_native_elitist_pilots \
    --output results/meta_brax_native_elitist_p14_c2_pilot.pkl

# Meta-Brax B: 21 parents + 21 children = 42 outer evaluations.
python meta_brax_heading_compare.py \
    --conditions srghn_full \
    --seeds 0 \
    --outer-generations 500 \
    --outer-pop-size 21 \
    --outer-children-per-parent 1 \
    --outer-replacement-mode elitist_union \
    --inner-pop-size 2 \
    --inner-children-per-parent 1 \
    --inner-generations 4 \
    --meta-batch-size 12 \
    --heldout-task-batch-size 16 \
    --support-episodes 2 \
    --query-episodes 2 \
    --episode-horizon 1000 \
    --parameter-block-size 1024 \
    --mutation-block-ratio 1.0 \
    --env-id ant \
    --brax-backend spring \
    --wandb-project meta-brax-native-elitist-p21-c1-pilot \
    --wandb-group meta_brax_native_elitist_pilots \
    --output results/meta_brax_native_elitist_p21_c1_pilot.pkl
