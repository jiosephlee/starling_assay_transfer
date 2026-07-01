#!/usr/bin/env bash
# Quick 3x3 LR x per-device-batch sweep, 200 steps each, eval every 25 steps.
# Runs at production scale (8 GPUs), logs each run to wandb (grouped), and reports the
# best eval AUROC / accuracy / macro-F1 per run plus the per-metric winner.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH=ml
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lr_bs_sweep}"
PY="${PYTHON:-/data1/joseph/miniconda3/envs/openrlhf/bin/python}"
NPROC="${NPROC:-8}"
LOGDIR="${LOGDIR:-ml/artifacts/sweep_logs}"
mkdir -p "$LOGDIR"
RESULTS="$LOGDIR/results.tsv"
printf "lr\tper_dev_bs\tglobal_bs\tbest_auroc\tbest_acc\tbest_macro_f1\n" > "$RESULTS"

# Overridable via env. Example (source-value sweep, eval-section only):
#   CONFIG=ml/configs/default.yaml LRS="1e-4 3e-4 1e-3" BSS="8192 16384 32768" STEPS=300 \
#   LOGDIR=ml/artifacts/srcval_sweep_logs WANDB_RUN_GROUP=ssv2_srcval_sweep \
#   EXTRA="model.use_source_value=true train.wandb_val_mirror=false" bash sweep_lr_bs.sh
LRS="${LRS:-1e-4 3e-4 1e-3}"
BSS="${BSS:-8192 16384 32768}"
EVAL_STEPS="${EVAL_STEPS:-25}"
EXTRA="${EXTRA:-}"
CONFIG="${CONFIG:-}"
if [ -n "${EPOCHS:-}" ]; then DUR="train.num_train_epochs=${EPOCHS}"; else DUR="train.max_steps=${STEPS:-200}"; fi

for lr in $LRS; do
  for bs in $BSS; do
    name="lr${lr}_bs${bs}"
    log="$LOGDIR/${name}.log"
    gbs=$(( bs * NPROC ))
    echo "=== running ${name} (global_bs=${gbs}) ==="
    "$PY" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29561}" \
      -m starling_ml.train ${CONFIG:+--config $CONFIG} \
      --set train.torch_compile=false ${DUR} train.eval_steps="${EVAL_STEPS}" \
      train.logging_steps="${EVAL_STEPS}" train.report_to=wandb train.run_name="${name}" \
      train.warmup_ratio=0.05 train.learning_rate="${lr}" train.per_device_batch_size="${bs}" \
      ${EXTRA} \
      paths.output_dir="$LOGDIR/$name" > "$log" 2>&1
    best=$("$PY" - "$log" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
def best(metric):
    vs = [float(m) for m in re.findall(rf"'eval_val_{metric}': '([0-9.eE+-]+)'", txt)]
    return f"{max(vs):.4f}" if vs else "NA"
print("\t".join(best(m) for m in ("auroc", "accuracy", "macro_f1")))
PYEOF
)
    printf "%s\t%s\t%s\t%s\n" "$lr" "$bs" "$gbs" "$best" >> "$RESULTS"
    rm -rf "$LOGDIR/$name"   # drop the per-run saved model
  done
done

echo ""
echo "=== FULL RESULTS ==="
column -t "$RESULTS"
echo ""
echo "=== WINNER PER METRIC ==="
"$PY" - "$RESULTS" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
def num(x):
    try: return float(x)
    except: return float("-inf")
for col in ("best_auroc", "best_acc", "best_macro_f1"):
    w = max(rows, key=lambda r: num(r[col]))
    print(f"{col:14s} winner: lr={w['lr']:5s} per_dev_bs={w['per_dev_bs']:6s} "
          f"(global_bs={w['global_bs']:7s})  ->  {w[col]}")
PYEOF
