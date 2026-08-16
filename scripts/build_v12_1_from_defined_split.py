#!/usr/bin/env python3
"""Rebuild V12.1 targets over exactly frozen V12 pair and ranking membership."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v12_1_targets import (
    CALIBRATION_FIELD, TARGET_CONTRACT, is_decisive, load_distance_calibration,
    parsed_calibrations,
)
from pipeline.v12_contract import DEFAULT_REGISTRY, load_registry, validate_registry
from pipeline.v3_policy import file_sha256
from scripts import build_v11_categorical_ablation as v11_release
from scripts import build_v11_intern_raw_pair as v11
from scripts import build_v12_1_intern_raw_pair as v12_1
from scripts import build_v12_intern_raw_pair as v12
from scripts import build_v12_release as v12_release
from scripts import verify_v11_intern_raw_pair as v11_verify


DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v12_1")
VERSION = "v12.1-train-percentile-distance-cdf-with-categorical"
MEMBERSHIP_CONTRACT = "frozen_v12_ordered_pair_and_ranking_membership.v1"
WRITE_BATCH_ROWS = 8192
RANKING_FIELDS = (
    "ranking_query_id", "ranking_member_index", "ranking_family",
    "ranking_query_category_id",
)
TARGET_FIELDS = {
    "completion", "is_decisive", "metadata", "target_a", "target_b", "target_z",
    "target_distribution", "percentile_distance", "percentile_distance_cdf",
}


def _reference_manifest(
    root: Path, task_id: str, source: Path, registry_path: Path,
) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != v12_release.VERSION or manifest.get("task_id") != task_id:
        raise ValueError("reference release is not the requested V12 task")
    if manifest.get("eligible_source_sha256") != file_sha256(source):
        raise ValueError("reference release and eligible source disagree")
    if manifest.get("prompt_projection", {}).get("sha256") != file_sha256(registry_path):
        raise ValueError("reference release and current V12 registry disagree")
    for split in v12_release.SPLITS:
        path = root / split / "data.parquet"
        if file_sha256(path) != manifest["parquet_sha256"][split]:
            raise ValueError(f"reference {split} Parquet hash mismatch")
    return manifest


def _source_index(source: Path, calibration_path: Path) -> dict[str, dict[str, Any]]:
    calibrations = parsed_calibrations(load_distance_calibration(calibration_path))
    output = {}
    for row in pq.read_table(source).to_pylist():
        if row["measurement_kind"] != "binary":
            row[CALIBRATION_FIELD] = calibrations[str(row["pair_bucket_key"])]
        output[str(row["child_id"])] = row
    return output


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    return v11_verify._same(left, right)


def _rebuilt(
    reference: Mapping[str, Any], source: Mapping[str, dict[str, Any]], split: str,
) -> dict[str, Any]:
    query = source[str(reference["query_record_id"])]
    retrieval = source[str(reference["retrieval_record_id"])]
    row = v12.row_for(query, retrieval, split)
    row["is_decisive"] = is_decisive(row, query)
    v11._fix_metadata_v11(row, target_contract=TARGET_CONTRACT)
    for field in RANKING_FIELDS:
        row[field] = reference.get(field)
    invariant_fields = (set(reference) | set(row)) - TARGET_FIELDS
    for field in invariant_fields:
        if not _same(reference.get(field), row.get(field)):
            raise ValueError(f"frozen V12 membership changed field {field!r}")
    return row


def _references(path: Path) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(WRITE_BATCH_ROWS):
        yield from batch.to_pylist()


def _examples(
    reference_root: Path, source: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    examples, seen = [], set()
    for split in v12_release.SPLITS:
        for reference in _references(reference_root / split / "data.parquet"):
            key = (str(reference["measurement_kind"]), split.endswith("ranking"))
            if key not in seen:
                examples.append(_rebuilt(reference, source, split))
                seen.add(key)
            if len(seen) >= 6:
                return examples
    return examples


def _schema(examples: list[dict[str, Any]], reference_schema: pa.Schema) -> pa.Schema:
    inferred = pa.Table.from_pylist(examples).schema
    fields = [
        reference_schema.field(field.name)
        if pa.types.is_null(field.type) and field.name in reference_schema.names else field
        for field in inferred
    ]
    return pa.schema(fields)


def _write_split(
    reference_path: Path, output_path: Path, source: Mapping[str, dict[str, Any]],
    split: str, schema: pa.Schema,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", compression_level=3)
    count = 0
    try:
        for batch in pq.ParquetFile(reference_path).iter_batches(WRITE_BATCH_ROWS):
            rows = [_rebuilt(row, source, split) for row in batch.to_pylist()]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            count += len(rows)
    finally:
        writer.close()
    temporary.replace(output_path)
    return count


def _membership(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    splits = {}
    for split in v12_release.SPLITS:
        path = root / split / "data.parquet"
        splits[split] = {
            "rows": int(reference["rows"][split]),
            "ordered_pair_id_sha256": v12_release._pair_id_sha256(path),
        }
        if split.endswith("ranking"):
            splits[split]["ranking_identity_sha256"] = v12_release._ranking_identity_sha256(path)
    return {
        "schema_version": MEMBERSHIP_CONTRACT, "reference_version": reference["version"],
        "reference_root": str(root),
        "reference_manifest_sha256": file_sha256(root / "manifest.json"), "splits": splits,
    }


def _manifest(
    task_id: str, staged: Path, source: Path, reference_root: Path,
    reference: Mapping[str, Any], counts: Mapping[str, int], schema: pa.Schema,
) -> dict[str, Any]:
    membership = _membership(reference_root, reference)
    for split in v12_release.SPLITS:
        path = staged / split / "data.parquet"
        if v12_release._pair_id_sha256(path) != membership["splits"][split]["ordered_pair_id_sha256"]:
            raise RuntimeError(f"V12.1 {split} ordered pair membership changed")
        if split.endswith("ranking") and (
            v12_release._ranking_identity_sha256(path)
            != membership["splits"][split]["ranking_identity_sha256"]
        ):
            raise RuntimeError(f"V12.1 {split} ranking identity changed")
    components = {}
    for name, details in reference["train_components"].items():
        copied = dict(details)
        copied["reference_v12_component_parquet_sha256"] = copied.pop(
            "component_parquet_sha256"
        )
        copied["target_rebuild_scope"] = "same_ordered_pair_range_new_target_fields"
        components[name] = copied
    calibration_files = {
        path.name: {"path": f"calibration/{path.name}", "sha256": file_sha256(path)}
        for path in sorted((staged / "calibration").iterdir()) if path.is_file()
    }
    return {
        "version": VERSION, "task_id": task_id, "dataset_variant": v12_release.VARIANT,
        "source": str(source), "eligible_source_sha256": file_sha256(source),
        "target_contract": TARGET_CONTRACT, "membership_contract": membership,
        "gold_lineage": reference["gold_lineage"], "split_contract": reference["split_contract"],
        "train_components": components, "ranking": reference["ranking"],
        "calibration": {
            "artifacts": calibration_files, "fit_split": "train",
            "pair_scope": "all distinct-record pairs including same-parent pairs",
        },
        "rows": dict(counts), "release_schema": schema.names,
        "parquet_sha256": {
            split: file_sha256(staged / split / "data.parquet")
            for split in v12_release.SPLITS
        },
        "paths": {split: f"{split}/data.parquet" for split in v12_release.SPLITS},
    }


def _card(task_id: str, manifest: Mapping[str, Any]) -> str:
    rows = manifest["rows"]
    return f"""---
pretty_name: Assay Transfer Raw Pair V12.1 Intern - {v12_release.TASK_TITLES[task_id]}
task_categories:
  - text-classification
language:
  - en
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/data.parquet
      - split: validation_ranking
        path: validation_ranking/data.parquet
      - split: test_ranking
        path: test_ranking/data.parquet
---

# Assay Transfer Raw Pair V12.1 Intern - {v12_release.TASK_TITLES[task_id]}

V12.1 preserves the exact ordered V12 membership across {rows['train']:,} train,
{rows['validation_ranking']:,} validation-ranking, and {rows['test_ranking']:,} test-ranking
rows. Only target-derived fields change. Its second CDF is fit on every distinct-record unordered
pair in train, including pairs whose records share a normalized parent.
"""


def build_task_release(
    task_id: str, reference_root: Path, source: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT, registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    reference = _reference_manifest(reference_root, task_id, source, registry_path)
    calibration = source.parent / "train_percentile_distance_calibration.json.gz"
    source_rows = _source_index(source, calibration)
    v12_1.configure_engine(registry_path)
    reset_prompt_cache()
    examples = _examples(reference_root, source_rows)
    schema = _schema(examples, pq.read_schema(reference_root / "train/data.parquet"))
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v12_1.stage.", dir=task_output))
    staged = stage_parent / v12_release.VARIANT
    staged.mkdir()
    try:
        counts = {}
        for split in v12_release.SPLITS:
            counts[split] = _write_split(
                reference_root / split / "data.parquet", staged / split / "data.parquet",
                source_rows, split, schema,
            )
        calibration_dir = staged / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "train_value_calibration.json.gz",
            "train_percentile_distance_calibration.json.gz", "split_manifest.json",
        ):
            shutil.copyfile(source.parent / name, calibration_dir / name)
        manifest = _manifest(
            task_id, staged, source, reference_root, reference, counts, schema
        )
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staged / "README.md").write_text(_card(task_id, manifest), encoding="utf-8")
        v11_release._promote(staged, task_output / v12_release.VARIANT)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(registry["tasks"]))
    parser.add_argument("--reference-release", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    args.source = args.source or v12.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    args = _parse_args()
    manifest = build_task_release(
        args.task, args.reference_release, args.source, args.output_root, args.registry
    )
    print(json.dumps(manifest["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
