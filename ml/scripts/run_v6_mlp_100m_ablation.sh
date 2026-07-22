#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data1/joseph/miniconda3/envs/openrlhf/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/ml${PYTHONPATH:+:$PYTHONPATH}"
ROOT="ml/artifacts/v6_5_molformer_pubmedbert_100m"
PROJECT="${WANDB_PROJECT:-assay-transfer-soft}"
GROUP="${WANDB_GROUP:-assay-transfer-raw-pair-v6-5-soft}"

"$PYTHON_BIN" - <<'PY'
import wandb
viewer = wandb.Api(timeout=10).viewer
if viewer is None:
    raise RuntimeError("W&B authentication preflight failed")
print(f"W&B authenticated as {viewer}")
PY

GPU_INDEX=0
PILOT_PIDS=""
for MODE in concat difference difference_product; do
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON_BIN" -m starling_ml.train_v6_mlp_100m \
    --fusion-mode "$MODE" \
    --embedding-cache "$ROOT/embedding_cache" \
    --schedule "$ROOT/group_schedule.npy" \
    --pair-cache "$ROOT/pair_cache" \
    --output "$ROOT/runs/$MODE" \
    --wandb-project "$PROJECT" \
    --wandb-group "$GROUP" \
    --stop-after 1000 &
  PILOT_PIDS="$PILOT_PIDS $!"
  GPU_INDEX=$((GPU_INDEX + 1))
done
for PILOT_PID in $PILOT_PIDS; do
  wait "$PILOT_PID"
done

"$PYTHON_BIN" ml/scripts/select_v6_mlp_100m_fusion.py \
  --runs "$ROOT/runs" \
  --output "$ROOT/fusion_selection.json"

WINNER=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$ROOT/fusion_selection.json")
CUDA_VISIBLE_DEVICES="${WINNER_GPU:-0}" "$PYTHON_BIN" -m starling_ml.train_v6_mlp_100m \
  --fusion-mode "$WINNER" \
  --embedding-cache "$ROOT/embedding_cache" \
  --pair-cache "$ROOT/pair_cache" \
  --schedule "$ROOT/group_schedule.npy" \
  --output "$ROOT/runs/$WINNER" \
  --resume "$ROOT/runs/$WINNER/last.pt" \
  --wandb-project "$PROJECT" \
  --wandb-group "$GROUP" \
  --stop-after 5000

CUDA_VISIBLE_DEVICES="${WINNER_GPU:-0}" "$PYTHON_BIN" ml/scripts/evaluate_v6_mlp_100m.py \
  --checkpoint "$ROOT/runs/$WINNER/best.pt" \
  --selection "$ROOT/fusion_selection.json" \
  --embedding-cache "$ROOT/embedding_cache" \
  --pair-cache "$ROOT/pair_cache" \
  --output "$ROOT/final_evaluation.json"
