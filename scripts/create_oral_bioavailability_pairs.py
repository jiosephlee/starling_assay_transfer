#!/usr/bin/env python3
"""Canonical Oral Bioavailability compact pair generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INTERNAL_DIR = SCRIPT_DIR / "internal"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(INTERNAL_DIR))

import create_transfer_pairs_compact_parquet as compact_pairs  # noqa: E402


DEFAULT_INPUT = "datasets/base/Oral_bioavailability_cleaned_v3"
DEFAULT_OUTPUT_ROOT = Path("datasets/pairs_compact")
MODE_OUTPUTS = {
    "no_constraints": "oral_bioavailability_pairs_no_constraints_v3",
    "same_species_v2": "oral_bioavailability_pairs_same_species_v2_v3",
    "condition_key": "oral_bioavailability_pairs_condition_key_v3",
}
MODE_SAME_COLUMNS = {
    "no_constraints": [],
    "same_species_v2": ["species_or_population_normalized"],
    "condition_key": ["condition_key_repro"],
}


def build(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir or (args.output_root / MODE_OUTPUTS[args.mode])
    pair_args = argparse.Namespace(
        input=args.input,
        output_dir=output_dir,
        smiles_column=args.smiles_column,
        value_column=args.value_column,
        metadata_columns=args.metadata_columns,
        same_metadata_columns=MODE_SAME_COLUMNS[args.mode],
        transfer_threshold=args.transfer_threshold,
        not_transfer_threshold=args.not_transfer_threshold,
        pair_sample_fraction=args.pair_sample_fraction,
        enumerate_all=args.enumerate_all,
        max_candidate_pairs=args.max_candidate_pairs,
        similarity_buckets=args.similarity_buckets,
        similarity_thresholds=args.similarity_thresholds,
        seed=args.seed,
        attempt_factor=args.attempt_factor,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_nbits=args.fingerprint_nbits,
        read_batch_size=args.read_batch_size,
        row_group_size=args.row_group_size,
        parquet_compression=args.parquet_compression,
        progress_every=args.progress_every,
        workers=args.workers,
        tasks_per_worker=args.tasks_per_worker,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    return compact_pairs.build(pair_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--mode", choices=tuple(MODE_OUTPUTS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--metadata-columns", nargs="+", default=compact_pairs.DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--pair-sample-fraction", type=float, default=0.01)
    parser.add_argument("--enumerate-all", action="store_true")
    parser.add_argument("--max-candidate-pairs", type=int, default=None)
    parser.add_argument("--similarity-buckets", type=int, default=5)
    parser.add_argument("--similarity-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--attempt-factor", type=int, default=10)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--fingerprint-nbits", type=int, default=2048)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--row-group-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-worker", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mode != "no_constraints" and not args.enumerate_all:
        parser.error("constrained modes require --enumerate-all")
    if args.max_candidate_pairs is not None and args.enumerate_all:
        parser.error("--max-candidate-pairs is not used with --enumerate-all")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(json.dumps({"output_dir": metadata["output_dir"], "pairs_written": metadata["pairs_written"]}, indent=2))


if __name__ == "__main__":
    main()
