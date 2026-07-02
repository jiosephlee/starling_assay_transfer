#!/usr/bin/env bash
# Regenerate the oral-bioavailability datasets (base -> pairs -> record splits)
# from the tracked TDC seed. CPU-heavy: pair enumeration parallelizes over cores.
#
# CLUSTER-SPECIFIC: change --partition, --account, and paths for your site.
# Reference specs (from the original Genoa/EPYC runs): 64 CPUs, 128-256G RAM.
#
#SBATCH --job-name=ob-pipeline
#SBATCH --partition=cpu              # CHANGE: your CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ob_pipeline_%j.out
#SBATCH --error=logs/ob_pipeline_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Activate your environment (CHANGE to match your cluster):
# module load python/3.11
# source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export ARROW_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/oral_bioavailability_v3.yaml}"

exec "${PYTHON}" scripts/run_oral_bioavailability_pipeline.py --config "${CONFIG}" "$@"
