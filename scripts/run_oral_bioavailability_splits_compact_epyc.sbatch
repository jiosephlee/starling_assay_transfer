#!/bin/bash -l
#SBATCH --job-name=ob-splits-compact
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/datasets/logs/ob_splits_compact_%j.out
#SBATCH --error=/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/datasets/logs/ob_splits_compact_%j.err

set -euo pipefail

cd /vast/projects/myatskar/design-documents/joseph/starling_assay_transfer
mkdir -p datasets/logs

export PYTHONPATH="/vast/projects/myatskar/design-documents/joseph/starling_assay_transfer/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ARROW_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/create_splits_from_compact_pairs.py \
  --input-dir datasets/pairs_compact/oral_bioavailability_pairs_full \
  --output-dir datasets/pairs_split_compact/oral_bioavailability_pair_splits \
  --eval-pairs-per-split 30000 \
  --similarity-buckets 6 \
  --similarity-thresholds 0.10 0.20 0.40 0.60 0.80 \
  --batch-size 250000 \
  --row-group-size 250000 \
  --bucket-file-row-limit 10000000 \
  --parquet-compression zstd \
  --progress-every-seconds 300 \
  --keep-bucketed-input \
  --overwrite \
  "$@"
