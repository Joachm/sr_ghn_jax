#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=2 python -m experiments.mujoco_playground_suite "$@"
