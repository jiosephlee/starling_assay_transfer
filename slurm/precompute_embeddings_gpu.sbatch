#!/usr/bin/env bash
# Precompute frozen MolFormer + MiniLM embeddings for the ~82K base molecules.
# GPU job (single GPU is enough). Run once before training.
#
# CLUSTER-SPECIFIC: change --partition, --account, --gres, and paths for your site.
#
#SBATCH --job-name=ob-precompute
#SBATCH --partition=gpu             # CHANGE: your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # CHANGE: e.g. gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ob_precompute_%j.out
#SBATCH --error=logs/ob_precompute_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate your environment (CHANGE to match your cluster):
# module load cuda/12.1 python/3.11
# source .venv/bin/activate

export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-ml/configs/shared_eval_same_species_v2.yaml}"

exec "${PYTHON}" -m starling_ml.precompute_embeddings --config "${CONFIG}" "$@"
