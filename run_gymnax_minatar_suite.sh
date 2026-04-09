#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=3 python -m experiments.gymnax_minatar_suite "$@"
