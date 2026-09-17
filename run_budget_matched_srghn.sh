#!/usr/bin/env bash
# Rerun only SR-GHN at the candidate-evaluation budget of existing controls.
set -euo pipefail

cd "$(dirname "$0")"

python -m experiments.gymnax_minatar_suite \
    --optimizer-family srghn \
    --baseline srghn_full \
    --seeds 0 1 2 3 4 \
    --children-per-parent 1 \
    --eval-budget-per-generation 200 \
    --output-dir results/gymnax_minatar_budget_matched

# Existing vector comparison: outer P=42, inner P=2, four inner generations,
# and a meta-batch of 12 headings. The flag resolves SR-GHN to outer P=14 with
# two children per parent; inner adaptation caches parents and evaluates only
# one new child per parent and generation.
python meta_brax_heading_compare.py \
    --conditions srghn_full \
    --seeds 0 1 2 3 4 \
    --outer-pop-size 42 \
    --inner-pop-size 2 \
    --inner-generations 4 \
    --meta-batch-size 12 \
    --support-episodes 2 \
    --query-episodes 2 \
    --budget-match-srghn \
    --output results/meta_brax_heading_srghn_budget_matched.pkl
