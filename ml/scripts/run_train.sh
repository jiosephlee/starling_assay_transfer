#!/usr/bin/env bash
# Train the transfer-classification model. Multi-GPU via torchrun by default.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-ml/configs/default.yaml}"
NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"

if [[ "${NPROC}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC}" -m starling_ml.train --config "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m starling_ml.train --config "${CONFIG}" "$@"
fi
