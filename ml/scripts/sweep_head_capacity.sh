#!/usr/bin/env bash
# Head-capacity scan: vary the SwiGLU head size on the frozen embeddings, everything else
# fixed (lr 3e-4, batch 65536, 7 fields). Tests whether val AUROC rises with head params
# (capacity-limited / double descent) or is flat (frozen-feature ceiling).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH=ml
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-head_capacity_scan}"
PY="${PYTHON:-/data1/joseph/miniconda3/envs/openrlhf/bin/python}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-300}"
LOGDIR=ml/artifacts/headscan_logs
mkdir -p "$LOGDIR"
RESULTS="$LOGDIR/results.tsv"
printf "d_model\td_ff\tn_blocks\tparams_M\tbest_val_auroc\tbest_macro_f1\tfinal_train_auroc\n" > "$RESULTS"

# (d_model d_ff n_blocks)
CONFIGS=("256 512 2" "512 1024 4" "768 2048 8" "1024 3072 12" "1536 4096 16")
for c in "${CONFIGS[@]}"; do
  read -r dm dff nb <<< "$c"
  name="dm${dm}_ff${dff}_nb${nb}"
  log="$LOGDIR/${name}.log"
  params=$("$PY" - "$dm" "$dff" "$nb" <<'PYEOF'
import sys, json, numpy as np
from starling_ml.config import Config
from starling_ml.model import TransferPairModel
dm, dff, nb = map(int, sys.argv[1:4])
cfg = Config.from_yaml("ml/configs/default.yaml")
cfg.model.d_model, cfg.model.d_ff, cfg.model.n_blocks = dm, dff, nb
F = len(json.load(open(cfg.paths.embeddings_dir + "/manifest.json"))["metadata_fields"])
mol = np.zeros((8, 768), np.float16); meta = np.zeros((8, F, 384), np.float16); pres = np.ones((8, F), np.uint8)
m = TransferPairModel(cfg.model, cfg.loss, mol, meta, pres)
print(round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 1))
PYEOF
)
  echo "=== ${name} (${params}M params) ==="
  "$PY" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port=29562 -m starling_ml.train \
    --set train.torch_compile=false train.max_steps="${STEPS}" train.eval_steps=25 \
    train.logging_steps=25 train.report_to=wandb train.run_name="${name}" train.warmup_ratio=0.05 \
    model.d_model="${dm}" model.d_ff="${dff}" model.n_blocks="${nb}" \
    paths.output_dir="$LOGDIR/$name" > "$log" 2>&1
  best=$("$PY" - "$log" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
def best(m):
    vs = [float(x) for x in re.findall(rf"'eval_val_{m}': '([0-9.eE+-]+)'", txt)]
    return f"{max(vs):.4f}" if vs else "NA"
ta = [float(x) for x in re.findall(r"'eval_train_sample_auroc': '([0-9.eE+-]+)'", txt)]
print(f"{best('auroc')}\t{best('macro_f1')}\t{ta[-1]:.4f}" if ta else f"{best('auroc')}\t{best('macro_f1')}\tNA")
PYEOF
)
  printf "%s\t%s\t%s\t%s\t%s\n" "$dm" "$dff" "$nb" "$params" "$best" >> "$RESULTS"
  rm -rf "$LOGDIR/$name"
done

echo ""
echo "=== HEAD CAPACITY SCAN (val AUROC vs params) ==="
column -t "$RESULTS"
