#!/usr/bin/env bash
# Train the transfer-classification model. This is where the record-level KNN
# eval callback runs (it embeds molecules with the live model during eval and
# reads datasets/starling_eval/condition_key_v3_record_splits_hf, which is in git).
#
# Multi-GPU via torchrun (one process per GPU). Requires the pair/memmap training
# data (regenerate with pipeline_cpu.sbatch) and precomputed embeddings
# (precompute_embeddings_gpu.sbatch) to exist first.
#
# CLUSTER-SPECIFIC: change --partition, --account, --gres, and paths for your site.
#
#SBATCH --job-name=ob-train
#SBATCH --partition=gpu             # CHANGE: your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2               # CHANGE: number/type of GPUs, e.g. gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ob_train_%j.out
#SBATCH --error=logs/ob_train_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate your environment (CHANGE to match your cluster):
# module load cuda/12.1 python/3.11
# source .venv/bin/activate

export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-ml/configs/shared_eval_same_species_v2.yaml}"

# One process per allocated GPU.
NPROC="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
if [[ "${NPROC}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC}" -m starling_ml.train --config "${CONFIG}" "$@"
else
  exec "${PYTHON}" -m starling_ml.train --config "${CONFIG}" "$@"
fi
