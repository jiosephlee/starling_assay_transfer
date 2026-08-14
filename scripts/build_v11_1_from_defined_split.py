#!/usr/bin/env python3
"""Rebuild V11.1 targets over an exactly frozen V11 train/evaluation membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_1_targets import (
    REFERENCE_PAIR_LIMIT, TARGET_CONTRACT, attach_distance_calibrations,
    build_distance_calibration, parsed_calibrations, write_distance_calibration,
)
from pipeline.v11_contract import (
    DEFAULT_REGISTRY, categorical_ablation_config, heldout_reservation_manifest,
    load_registry, validate_registry,
)
from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v11_targets import TARGET_EPSILON
from pipeline.v3_policy import file_sha256
from scripts import build_v11_1_intern_raw_pair as v11_1
from scripts import build_v11_categorical_ablation as v11_release
from scripts import build_v11_intern_raw_pair as v11
from scripts import verify_v11_intern_raw_pair as v11_verify
from scripts.build_v11_1_categorical_release import (
    CALIBRATION_FILE, HUB_DATASET_IDS, VARIANT, VERSION,
)


REFERENCE_VERSION = "v11-txagent-v7-value-cdf-with-categorical"
MEMBERSHIP_CONTRACT = "frozen_v11_ordered_pair_membership.v1"
WRITE_BATCH_ROWS = 8_192
RANKING_FIELDS = (
    "ranking_query_id", "ranking_member_index", "ranking_family",
    "ranking_query_category_id",
)
TARGET_DERIVED_FIELDS = {
    "completion", "is_decisive", "metadata", "target_a", "target_b", "target_z",
    "target_distribution", "percentile_distance", "percentile_distance_cdf",
}


def _reference_manifest(
    root: Path, task_id: str, source: Path, registry_path: Path,
) -> dict[str, Any]:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != REFERENCE_VERSION or manifest.get("task_id") != task_id:
        raise ValueError("reference release is not the requested V11 task")
    if manifest.get("dataset_variant") != VARIANT:
        raise ValueError("reference release is not the with-categorical variant")
    if manifest.get("eligible_source_sha256") != file_sha256(source):
        raise ValueError("reference release and eligible source hashes disagree")
    if manifest.get("prompt_projection", {}).get("sha256") != file_sha256(registry_path):
        raise ValueError("reference release and prompt registry hashes disagree")
    for split in v11_release.SPLITS:
        parquet = root / split / "data.parquet"
        if file_sha256(parquet) != manifest["parquet_sha256"][split]:
            raise ValueError(f"reference {split} Parquet hash mismatch")
    return manifest


def _prepared_source(
    task_id: str, source: Path,
) -> tuple[list[dict], dict[str, dict], dict[str, Any], dict[str, int]]:
    full_rows = pq.read_table(source).to_pylist()
    calibration = build_distance_calibration(full_rows, task_id)
    attached, rejected = attach_distance_calibrations(
        full_rows, parsed_calibrations(calibration)
    )
    return full_rows, {row["child_id"]: row for row in attached}, calibration, rejected


def _same_invariant(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left == right
    return v11_verify._same(left, right)


def _assert_invariants(reference: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    fields = (set(reference) | set(rebuilt)) - TARGET_DERIVED_FIELDS
    for field in fields:
        if not _same_invariant(reference.get(field), rebuilt.get(field)):
            raise ValueError(f"frozen pair changed invariant field {field!r}")


def _rebuilt_row(
    reference: Mapping[str, Any], source: Mapping[str, dict], split: str,
) -> dict[str, Any]:
    try:
        query = source[str(reference["query_record_id"])]
        retrieval = source[str(reference["retrieval_record_id"])]
    except KeyError as exc:
        raise ValueError(f"reference pair uses an uncalibrated or missing record: {exc}") from exc
    row = v11._row_for_v11(query, retrieval, split)
    row["is_decisive"] = v11_1.is_decisive(row, query)
    v11._fix_metadata_v11(row, target_contract=TARGET_CONTRACT)
    for field in RANKING_FIELDS:
        row[field] = reference.get(field)
    _assert_invariants(reference, row)
    return row


def _reference_rows(path: Path, start: int, count: int) -> Iterator[dict[str, Any]]:
    position, end = 0, start + count
    for batch in pq.ParquetFile(path).iter_batches(WRITE_BATCH_ROWS):
        batch_start, position = position, position + batch.num_rows
        if position <= start:
            continue
        if batch_start >= end:
            break
        offset = max(0, start - batch_start)
        length = min(position, end) - batch_start - offset
        yield from batch.slice(offset, length).to_pylist()


def _component_ranges(manifest: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    order = manifest["measurement_kind_policy"]["train_components"]
    if order != ["continuous", "categorical"]:
        raise ValueError("reference train component order changed")
    ranges, start = [], 0
    for name in order:
        count = int(manifest["train_components"][name]["achieved_pairs"])
        ranges.append((name, start, count))
        start += count
    if start != int(manifest["rows"]["train"]):
        raise ValueError("reference component ranges do not cover train")
    return ranges


def _schema_examples(
    reference_path: Path, manifest: Mapping[str, Any], source: Mapping[str, dict],
) -> list[dict[str, Any]]:
    wanted = {
        kind for component in manifest["train_components"].values()
        for kind, count in component["achieved_by_measurement_kind"].items() if count
    }
    examples, seen = [], set()
    for batch in pq.ParquetFile(reference_path).iter_batches(WRITE_BATCH_ROWS):
        kinds = batch.column(batch.schema.get_field_index("measurement_kind")).to_pylist()
        for index, kind in enumerate(kinds):
            if kind in seen:
                continue
            reference = batch.slice(index, 1).to_pylist()[0]
            examples.append(_rebuilt_row(reference, source, "train"))
            seen.add(kind)
        if seen == wanted:
            break
    if seen != wanted:
        raise ValueError(f"could not sample reference kinds: {sorted(wanted - seen)}")
    return examples


def _release_schema(examples: list[dict], reference_schema: pa.Schema) -> pa.Schema:
    inferred = pa.Table.from_pylist(examples).schema
    fields = []
    for field in inferred:
        if pa.types.is_null(field.type) and field.name in reference_schema.names:
            fields.append(reference_schema.field(field.name))
        else:
            fields.append(field)
    return pa.schema(fields)


def _write_rows(
    path: Path, references: Iterator[dict], source: Mapping[str, dict],
    split: str, schema: pa.Schema,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary, schema, compression="zstd", compression_level=v11_release.TRAIN_ZSTD_LEVEL
    )
    count, rows = 0, []
    try:
        for reference in references:
            rows.append(_rebuilt_row(reference, source, split))
            if len(rows) >= WRITE_BATCH_ROWS:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count, rows = count + len(rows), []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            count += len(rows)
    finally:
        writer.close()
    temporary.replace(path)
    return count


def _ranking_identity_sha256(path: Path) -> str:
    columns = ["pair_id", *RANKING_FIELDS]
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(WRITE_BATCH_ROWS, columns=columns):
        for row in batch.to_pylist():
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(payload.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _reference_membership(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    splits = {}
    for split in v11_release.SPLITS:
        path = root / split / "data.parquet"
        splits[split] = {
            "rows": int(manifest["rows"][split]),
            "parquet_sha256": file_sha256(path),
            "ordered_pair_id_sha256": v11_release._pair_id_sha256(path),
        }
    splits["validation_ranking"]["ranking_identity_sha256"] = _ranking_identity_sha256(
        root / "validation_ranking/data.parquet"
    )
    return {
        "schema_version": MEMBERSHIP_CONTRACT,
        "reference_version": manifest["version"],
        "reference_root": str(root),
        "reference_manifest_sha256": file_sha256(root / "manifest.json"),
        "splits": splits,
    }


def _dataset_card(task_id: str, manifest: Mapping[str, Any]) -> str:
    rows = manifest["rows"]
    title = v11_release.TASK_TITLES[task_id]
    return f"""---
pretty_name: Assay Transfer Raw Pair V11.1 Intern — {title} — With Categorical
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
---

# Assay Transfer Raw Pair V11.1 Intern — {title} — With Categorical

This release freezes the exact ordered V11 train and ranking pair memberships: {rows['train']:,}
train rows and {rows['validation_ranking']:,} ranking rows. It recomputes all target-derived
fields under the V11.1 percentile-distance-CDF contract. Continuous and ordinal calibrations use
up to {REFERENCE_PAIR_LIMIT:,} cross-molecule reference pairs per bucket; binary targets retain
the 0.95/0.05 agreement rule. The release does not emit `target_z`.
"""


def _manifest(
    task_id: str, staged: Path, source: Path, registry_path: Path,
    reference_root: Path, reference: Mapping[str, Any], components: Mapping[str, Any],
    schema: pa.Schema, calibration_path: Path, rejected: Mapping[str, int],
) -> dict[str, Any]:
    hashes, counts = v11_release._split_artifacts(staged)
    return {
        "version": VERSION, "dataset_variant": VARIANT,
        "hub_dataset_id": HUB_DATASET_IDS[task_id], "task_id": task_id,
        "source": str(source), "eligible_source_sha256": file_sha256(source),
        "eligible_source_manifest_sha256": file_sha256(source.parent / "manifest.json"),
        "prompt_projection": {
            "schema_version": load_registry(registry_path)["schema_version"],
            "path": str(registry_path), "sha256": file_sha256(registry_path),
        },
        "target_contract": {"name": TARGET_CONTRACT, "probability_epsilon": TARGET_EPSILON},
        "reference_membership": _reference_membership(reference_root, reference),
        "percentile_distance_calibration": {
            "path": CALIBRATION_FILE, "sha256": file_sha256(calibration_path),
            "reference_pair_limit": REFERENCE_PAIR_LIMIT,
            "non_pairable_source_records_by_measurement_kind": dict(rejected),
        },
        "construction": reference["construction"],
        "measurement_kind_policy": reference["measurement_kind_policy"],
        "train_components": dict(components),
        "eligible_record_inventory": reference["eligible_record_inventory"],
        "source_record_sampling": reference["source_record_sampling"],
        "heldout_reservation": reference["heldout_reservation"],
        "ranking_validation": reference["ranking_validation"],
        "target_derived_fields_recomputed": sorted(TARGET_DERIVED_FIELDS - {"target_z"}),
        "removed_fields": ["target_z"], "release_schema": schema.names,
        "rows": counts, "parquet_sha256": hashes,
        "paths": {split: f"{split}/data.parquet" for split in v11_release.SPLITS},
    }


def _build_components(
    reference_root: Path, component_root: Path, reference: Mapping[str, Any],
    source: Mapping[str, dict], schema: pa.Schema, registry: Mapping[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    paths, manifests = [], {}
    release = categorical_ablation_config(registry)
    reference_train = reference_root / "train/data.parquet"
    for name, start, count in _component_ranges(reference):
        path = component_root / name / "train/data.parquet"
        actual = _write_rows(
            path, _reference_rows(reference_train, start, count), source, "train", schema
        )
        if actual != count:
            raise RuntimeError(f"{name}: rebuilt row count changed")
        config = release["train_components"][name]
        paths.append(path)
        manifests[name] = v11_release._component_manifest(name, config, path, actual)
        expected_pair_hash = reference["train_components"][name]["pair_id_sha256"]
        if manifests[name]["pair_id_sha256"] != expected_pair_hash:
            raise RuntimeError(f"{name}: ordered pair membership changed")
    return paths, manifests


def _verify_live_reservation(
    task_id: str, full_rows: list[dict], registry: Mapping[str, Any], reference: Mapping[str, Any],
) -> set[str]:
    live = heldout_reservation_manifest(task_id, registry)
    if live != reference["heldout_reservation"]:
        raise ValueError("reference heldout reservation is no longer live")
    return heldout_union_molecules(task_id, full_rows, registry)


def _build_staged(
    task_id: str, staged: Path, component_root: Path, source_path: Path,
    reference_root: Path, registry_path: Path, registry: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    full_rows, source, calibration, rejected = _prepared_source(task_id, source_path)
    _verify_live_reservation(task_id, full_rows, registry, reference)
    reference_train = reference_root / "train/data.parquet"
    examples = _schema_examples(reference_train, reference, source)
    ranking_reference = reference_root / "validation_ranking/data.parquet"
    ranking_rows = [
        _rebuilt_row(row, source, "validation_ranking")
        for row in _reference_rows(ranking_reference, 0, int(reference["rows"]["validation_ranking"]))
    ]
    schema = _release_schema(examples + ranking_rows[:1], pq.read_schema(reference_train))
    component_paths, components = _build_components(
        reference_root, component_root, reference, source, schema, registry
    )
    v11_release._write_combined_train(component_paths, staged / "train/data.parquet", schema)
    v11_release._write_ranking(ranking_rows, staged / "validation_ranking/data.parquet", schema)
    calibration_path = staged / CALIBRATION_FILE
    write_distance_calibration(calibration, calibration_path)
    manifest = _manifest(
        task_id, staged, source_path, registry_path, reference_root, reference,
        components, schema, calibration_path, rejected,
    )
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staged / "README.md").write_text(_dataset_card(task_id, manifest), encoding="utf-8")
    return manifest


def build_task_release(
    task_id: str, reference_root: Path, source: Path,
    output_root: Path = v11_1.DEFAULT_OUTPUT_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    v11_1.configure_engine(task_id, registry, registry_path)
    source = v11._prepare_source(task_id, source, None, registry)
    reset_prompt_cache()
    reference = _reference_manifest(reference_root, task_id, source, registry_path)
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v11_1.fixed.", dir=task_output))
    components = Path(tempfile.mkdtemp(prefix=f".{task_id}.fixed.", dir=stage_parent))
    staged = stage_parent / VARIANT
    staged.mkdir()
    try:
        manifest = _build_staged(
            task_id, staged, components, source, reference_root,
            registry_path, registry, reference,
        )
        v11_release._promote(staged, task_output / VARIANT)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(HUB_DATASET_IDS))
    parser.add_argument("--reference-release", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=v11_1.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    args.source = args.source or v11.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    args = _parse_args()
    manifest = build_task_release(
        args.task, args.reference_release, args.source, args.output_root, args.registry
    )
    print(json.dumps(manifest["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
