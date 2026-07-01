#!/usr/bin/env python3
"""Canonical Oral Bioavailability shared split/full materialization."""

from __future__ import annotations

import argparse
import shutil
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

SCRIPT_DIR = Path(__file__).resolve().parent
INTERNAL_DIR = SCRIPT_DIR / "internal"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(INTERNAL_DIR))

import create_oral_bioavailability_shared_eval_splits as shared_splits  # noqa: E402


def build(args: argparse.Namespace) -> dict[str, object]:
    if args.train_direction_mode == "both":
        split_args = argparse.Namespace(**vars(args))
        split_args.train_direction_mode = "bidirectional"
        delattr(split_args, "all_train_direction_modes")
        bidirectional = shared_splits.build(split_args)
        unidirectional = derive_unidirectional_from_bidirectional(args, bidirectional)
        return {"bidirectional": bidirectional, "unidirectional": unidirectional}
    split_args = argparse.Namespace(**vars(args))
    delattr(split_args, "all_train_direction_modes")
    return {args.train_direction_mode: shared_splits.build(split_args)}


def universe_output_dirs(args: argparse.Namespace, mode: str) -> dict[str, Path]:
    universes = set(args.universes)
    return {
        key: value
        for key, value in shared_splits.split_output_dirs(args.output_root, mode, args.output_name_suffix).items()
        if key in universes
    }


def filter_train_a_to_b(source_train: Path, dest_train: Path, args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    dataset = ds.dataset(source_train, format="parquet")
    writer = shared_splits.RollingParquetWriter(
        dest_train,
        dataset.schema,
        file_row_limit=args.parquet_file_row_limit,
        compression=args.parquet_compression,
    )
    rows = 0
    labels: Counter[str] = Counter()
    try:
        for batch in dataset.to_batches(batch_size=args.batch_size):
            table = pa.Table.from_batches([batch], schema=dataset.schema)
            mask = pc.equal(table["direction"], pa.scalar("a_to_b"))
            filtered = table.filter(mask)
            if filtered.num_rows:
                writer.write_table(filtered)
                rows += filtered.num_rows
                labels.update(filtered["transfer_label"].to_pylist())
    finally:
        writer.close()
    return rows, dict(sorted(labels.items()))


def derive_unidirectional_metadata(
    source_meta: dict[str, object],
    *,
    source_dir: Path,
    output_dir: Path,
    train_rows: int,
    train_label_counts: dict[str, int],
) -> dict[str, object]:
    meta = dict(source_meta)
    rows_by_split = dict(meta.get("rows_by_split", {}))
    rows_by_split["train"] = train_rows
    eval_subset_counts = {
        split: dict(counts)
        for split, counts in dict(meta.get("eval_subset_counts", {})).items()
    }
    eval_subset_counts["train"] = {"none": train_rows}
    direction_counts = {
        split: dict(counts)
        for split, counts in dict(meta.get("direction_counts", {})).items()
    }
    direction_counts["train"] = {"a_to_b": train_rows}
    label_counts = {
        split: dict(counts)
        for split, counts in dict(meta.get("transfer_label_counts", {})).items()
    }
    label_counts["train"] = train_label_counts
    meta.update(
        {
            "created_at_utc": shared_splits.utc_now(),
            "output_dir": str(output_dir),
            "train_direction_mode": "unidirectional",
            "train_directions": ["a_to_b"],
            "rows_by_split": rows_by_split,
            "eval_subset_counts": eval_subset_counts,
            "direction_counts": direction_counts,
            "transfer_label_counts": label_counts,
            "derived_from_bidirectional_output": str(source_dir),
            "derivation_policy": (
                "validation/test copied unchanged from bidirectional output; train filtered to direction == a_to_b"
            ),
        }
    )
    return meta


def derive_unidirectional_from_bidirectional(args: argparse.Namespace, bidirectional_meta: dict[str, object]) -> dict[str, object]:
    if args.arrow_cpu_count:
        pa.set_cpu_count(args.arrow_cpu_count)
        pa.set_io_thread_count(args.arrow_cpu_count)
    source_dirs = universe_output_dirs(args, "bidirectional")
    output_dirs = universe_output_dirs(args, "unidirectional")
    shared_splits.prepare_outputs(output_dirs, args.overwrite, args.splits)
    staged_dirs = shared_splits.make_staged_output_dirs(output_dirs)
    universe_metadata: dict[str, object] = {}
    installed = False
    try:
        for universe, source_dir in source_dirs.items():
            staged_dir = staged_dirs[universe]
            for split in ("validation", "test"):
                if split in args.splits and (source_dir / split).exists():
                    shutil.copytree(source_dir / split, staged_dir / split)
            train_rows = 0
            train_labels: dict[str, int] = {}
            if "train" in args.splits:
                train_rows, train_labels = filter_train_a_to_b(source_dir / "train", staged_dir / "train", args)
            source_meta = json.loads((source_dir / "metadata.json").read_text())
            universe_metadata[universe] = derive_unidirectional_metadata(
                source_meta,
                source_dir=source_dir,
                output_dir=output_dirs[universe],
                train_rows=train_rows,
                train_label_counts=train_labels,
            )
            shared_splits.write_json(staged_dir / "metadata.json", universe_metadata[universe])
        shared_splits.install_staged_output_dirs(staged_dirs, output_dirs, args.overwrite)
        installed = True
    finally:
        if not installed:
            shared_splits.cleanup_staged_output_dirs(staged_dirs.values())

    source_selection = args.output_root / "oral_bioavailability_shared_eval_selection.json"
    dest_selection = args.output_root / "oral_bioavailability_shared_eval_unidirectional_selection.json"
    if source_selection.exists():
        shutil.copy2(source_selection, dest_selection)
    metadata = dict(bidirectional_meta)
    metadata.update(
        {
            "created_at_utc": shared_splits.utc_now(),
            "train_direction_mode": "unidirectional",
            "outputs": {key: str(value) for key, value in output_dirs.items()},
            "universe_rows_by_split": {
                key: value["rows_by_split"] for key, value in universe_metadata.items()
            },
            "universe_validation": {
                key: value["validation"] for key, value in universe_metadata.items()
            },
            "derived_from_bidirectional_outputs": {key: str(value) for key, value in source_dirs.items()},
            "derivation_policy": (
                "validation/test copied unchanged from bidirectional output; train filtered to direction == a_to_b"
            ),
        }
    )
    shared_splits.write_json(args.output_root / "oral_bioavailability_shared_eval_unidirectional_metadata.json", metadata)
    return metadata


def summarize_result(result: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for mode, mode_result in result.items():
        if not isinstance(mode_result, dict):
            summary[mode] = mode_result
            continue
        summary[mode] = {
            "outputs": mode_result.get("outputs", {}),
            "rows_by_split": mode_result.get("universe_rows_by_split", {}),
            "validation": mode_result.get("universe_validation", {}),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default="datasets/base/Oral_bioavailability_cleaned_v3")
    parser.add_argument("--condition-key-pairs", type=Path, default=Path("datasets/pairs_compact/oral_bioavailability_pairs_condition_key_v3"))
    parser.add_argument("--same-species-pairs", type=Path, default=Path("datasets/pairs_compact/oral_bioavailability_pairs_same_species_v2_v3"))
    parser.add_argument("--no-constraints-pairs", type=Path, default=Path("datasets/pairs_compact/oral_bioavailability_pairs_no_constraints_v3"))
    parser.add_argument(
        "--universes",
        nargs="+",
        choices=("condition_key", "same_species_v2", "no_constraints"),
        default=["condition_key", "same_species_v2", "no_constraints"],
        help="Pair universes to materialize. Selection is still based on condition_key.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets/pairs_split_full"))
    parser.add_argument("--output-name-suffix", default="_v3")
    parser.add_argument("--splits", nargs="+", choices=shared_splits.SPLITS, default=list(shared_splits.SPLITS))
    parser.add_argument("--train-direction-mode", choices=("bidirectional", "unidirectional", "both"), default="both")
    parser.add_argument("--all-train-direction-modes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-columns", nargs="+", default=shared_splits.DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--shared-eval-compatibility-column", default=shared_splits.DEFAULT_SHARED_COMPATIBILITY_COLUMN)
    parser.add_argument("--pair-metadata-columns", nargs="+", default=[
        "bioavailability_report_type",
        "species_or_population",
        "dose",
        "oral_exposure_mode",
        "qualifying_conditions",
        "comparator",
        "extra_details",
    ])
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--eval-directions-per-subset", type=int, default=10_000)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=25)
    parser.add_argument("--preferred-no-train-molecules", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--parquet-file-row-limit", type=int, default=10_000_000)
    parser.add_argument("--progress-every-seconds", type=float, default=300.0)
    parser.add_argument("--arrow-cpu-count", type=int, default=None)
    parser.add_argument("--disjoint-eval-molecules", action="store_true")
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all_train_direction_modes:
        args.train_direction_mode = "both"
    return args


def main() -> None:
    print(json.dumps(summarize_result(build(parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
