#!/usr/bin/env bash
# Precompute frozen MolFormer + MiniLM embeddings for the 82K base molecules.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-ml/configs/default.yaml}"

exec "${PYTHON}" -m starling_ml.precompute_embeddings --config "${CONFIG}" "$@"
