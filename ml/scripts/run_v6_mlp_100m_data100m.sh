#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data1/joseph/miniconda3/envs/openrlhf/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/ml${PYTHONPATH:+:$PYTHONPATH}"
ROOT="ml/artifacts/v6_5_molformer_pubmedbert_100m_data100m"
PROJECT="${WANDB_PROJECT:-assay-transfer-soft}"
GROUP="${WANDB_GROUP:-assay-transfer-raw-pair-v6-5-soft}"
MODE="difference_product"
STEPS=19531

"$PYTHON_BIN" - <<'PY'
import wandb
if wandb.Api(timeout=10).viewer is None:
    raise RuntimeError("W&B authentication preflight failed")
PY

CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" "$PYTHON_BIN" -m starling_ml.train_v6_mlp_100m \
  --fusion-mode "$MODE" \
  --embedding-cache "$ROOT/embedding_cache" \
  --pair-cache "$ROOT/pair_cache" \
  --schedule "$ROOT/group_schedule.npy" \
  --schedule-steps "$STEPS" \
  --stop-after "$STEPS" \
  --output "$ROOT/runs/$MODE" \
  --wandb-project "$PROJECT" \
  --wandb-group "$GROUP" \
  --wandb-run-name "mlp-100m-difference_product_assay-transfer-raw-pair-v6-5-soft-100m-data-a100"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" "$PYTHON_BIN" ml/scripts/evaluate_v6_mlp_100m.py \
  --checkpoint "$ROOT/runs/$MODE/best.pt" \
  --selection "ml/artifacts/v6_5_molformer_pubmedbert_100m/fusion_selection.json" \
  --embedding-cache "$ROOT/embedding_cache" \
  --pair-cache "$ROOT/pair_cache" \
  --output "$ROOT/final_evaluation.json"
