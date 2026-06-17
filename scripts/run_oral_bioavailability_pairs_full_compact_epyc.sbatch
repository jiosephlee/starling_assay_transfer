#!/bin/bash -l
#SBATCH --job-name=ob-pairs-full
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/datasets/logs/ob_pairs_full_%j.out
#SBATCH --error=/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/datasets/logs/ob_pairs_full_%j.err

set -euo pipefail

cd /vast/projects/myatskar/design-documents/joseph/starling_assay_transfer
mkdir -p datasets/logs

export PYTHONPATH="/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/create_transfer_pairs_compact_parquet.py \
  --input datasets/base/Oral_bioavailability_cleaned \
  --enumerate-all \
  --workers "${SLURM_CPUS_PER_TASK:-64}" \
  --tasks-per-worker 4 \
  --output-dir datasets/pairs_compact/oral_bioavailability_pairs_full \
  --row-group-size 250000 \
  --parquet-compression zstd \
  --progress-every 0 \
  --overwrite \
  "$@"
