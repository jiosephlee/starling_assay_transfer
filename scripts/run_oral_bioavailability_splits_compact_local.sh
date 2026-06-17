#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-48}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-datasets/pairs_compact/oral_bioavailability_pairs_full}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/pairs_split_compact/oral_bioavailability_pair_splits}"
EVAL_PAIRS_PER_SPLIT="${EVAL_PAIRS_PER_SPLIT:-30000}"
SIMILARITY_BUCKETS="${SIMILARITY_BUCKETS:-6}"
SIMILARITY_THRESHOLDS="${SIMILARITY_THRESHOLDS:-0.10 0.20 0.40 0.60 0.80}"
BATCH_SIZE="${BATCH_SIZE:-250000}"
ROW_GROUP_SIZE="${ROW_GROUP_SIZE:-250000}"
BUCKET_FILE_ROW_LIMIT="${BUCKET_FILE_ROW_LIMIT:-10000000}"
PARQUET_COMPRESSION="${PARQUET_COMPRESSION:-zstd}"
PROGRESS_EVERY_SECONDS="${PROGRESS_EVERY_SECONDS:-300}"
EXTRA_ARGS=()
if [[ "${REUSE_BUCKETED_INPUT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-bucketed-input)
fi
if [[ "${REUSE_CANDIDATE_INDEX:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-candidate-index)
fi
if [[ "${RESUME_CHECKPOINTS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--resume-checkpoints)
fi

"${PYTHON}" scripts/create_splits_from_compact_pairs.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-pairs-per-split "${EVAL_PAIRS_PER_SPLIT}" \
  --similarity-buckets "${SIMILARITY_BUCKETS}" \
  --similarity-thresholds ${SIMILARITY_THRESHOLDS} \
  --batch-size "${BATCH_SIZE}" \
  --row-group-size "${ROW_GROUP_SIZE}" \
  --bucket-file-row-limit "${BUCKET_FILE_ROW_LIMIT}" \
  --parquet-compression "${PARQUET_COMPRESSION}" \
  --progress-every-seconds "${PROGRESS_EVERY_SECONDS}" \
  --keep-bucketed-input \
  --overwrite \
  "${EXTRA_ARGS[@]}" \
  "$@"
