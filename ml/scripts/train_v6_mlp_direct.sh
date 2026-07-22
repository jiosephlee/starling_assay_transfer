#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data1/joseph/miniconda3/envs/openrlhf/bin/python}"
exec "${PYTHON_BIN}" -m starling_ml.train_v6_mlp \
  --variant direct \
  --cache ml/artifacts/v6_mlp/cache \
  --output ml/artifacts/v6_mlp/runs/direct_soft_plus_listnet \
  --listnet-weight 0.1 \
  "$@"
