#!/usr/bin/env python3
"""Build the single V11 with-categorical Intern release for one task."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import scripts.build_v7_intern_raw_pair as v7
from pipeline.pair_core import stable_hash
from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_contract import (
    DEFAULT_REGISTRY, categorical_ablation_config, heldout_reservation_manifest,
    load_registry, task_config, validate_registry,
)
from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v11_ranking import build_ranking_anchors, materialize_ranking_rows
from pipeline.v11_targets import TARGET_CONTRACT, TARGET_EPSILON
from pipeline.v3_policy import file_sha256
from scripts import build_v11_intern_raw_pair as v11


DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v11_categorical_ablation")
DEFAULT_COMPONENT_ROOT = Path("tmp/v11_value_cdf_ranking_release/components")
VARIANT = "with_categorical"
SPLITS = ("train", "validation_ranking")
CONCAT_BATCH_ROWS = 32_768
TRAIN_ZSTD_LEVEL = 3
TASK_TITLES = {
    "bioavailability_ma": "Bioavailability",
    "bbb_martins": "Blood-Brain Barrier",
    "skin_reaction": "Skin Reaction",
}
HUB_DATASET_IDS = {
    "bioavailability_ma": "jiosephlee/assay-transfer-raw-pair-v11-bioavailability-ma-with-categorical-intern",
    "bbb_martins": "jiosephlee/assay-transfer-raw-pair-v11-bbb-martins-with-categorical-intern",
    "skin_reaction": "jiosephlee/assay-transfer-raw-pair-v11-skin-reaction-with-categorical-intern",
}


def _kinds(rows: Iterable[dict], allowed: Iterable[str]) -> list[dict]:
    accepted = set(allowed)
    return [row for row in rows if str(row["measurement_kind"]) in accepted]


def _unreserved(rows: Iterable[dict], reserved: set[str]) -> list[dict]:
    return [row for row in rows if str(row["canonical_smiles"]) not in reserved]


def _sample_configured_sources(
    rows: list[dict], task: Mapping[str, Any], task_id: str,
) -> tuple[list[dict], dict[str, Any]]:
    rules = task["construction"].get("source_record_sampling", {})
    retained_ids, report = set(), {}
    for source, rule in rules.items():
        source_rows = [row for row in rows if str(row["source_id"]) == source]
        ordered = sorted(
            source_rows,
            key=lambda row: stable_hash(f"v11-source-sample:{task_id}:{source}:{row['child_id']}"),
        )
        wanted = len(ordered) * int(rule["numerator"]) // int(rule["denominator"])
        retained_ids.update(str(row["child_id"]) for row in ordered[:wanted])
        report[source] = {
            **dict(rule), "input_records": len(ordered), "retained_records": wanted,
            "seed_namespace": f"v11-source-sample:{task_id}:{source}",
        }
    sampled = [
        row for row in rows
        if str(row["source_id"]) not in rules or str(row["child_id"]) in retained_ids
    ]
    return sampled, report


def _pair_id_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["pair_id"]):
        for pair_id in batch.column(0).to_pylist():
            digest.update(str(pair_id).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _parquet_kind_counts(path: Path) -> dict[str, int]:
    counts: Counter = Counter()
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["measurement_kind"]):
        counts.update(str(value) for value in batch.column(0).to_pylist())
    return dict(sorted(counts.items()))


def _degree_maxima(path: Path) -> dict[str, int]:
    query, retrieval = Counter(), Counter()
    columns = ["query_record_id", "retrieval_record_id"]
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        query.update(str(value) for value in batch.column(0).to_pylist())
        retrieval.update(str(value) for value in batch.column(1).to_pylist())
    return {
        "observed_max_query_degree": max(query.values(), default=0),
        "observed_max_retrieval_degree": max(retrieval.values(), default=0),
    }


def _component_manifest(
    name: str, config: Mapping[str, Any], path: Path, achieved: int,
) -> dict[str, Any]:
    degrees = _degree_maxima(path)
    query_cap = int(config["query_degree_cap"])
    retrieval_cap = int(config["retrieval_degree_cap"])
    if degrees["observed_max_query_degree"] > query_cap:
        raise RuntimeError(f"{name}: emitted rows exceed query degree cap")
    if degrees["observed_max_retrieval_degree"] > retrieval_cap:
        raise RuntimeError(f"{name}: emitted rows exceed retrieval degree cap")
    return {
        "component_name": name, "measurement_kinds": list(config["measurement_kinds"]),
        "max_pairs": int(config["max_pairs"]), "achieved_pairs": achieved,
        "achieved_by_measurement_kind": _parquet_kind_counts(path),
        "query_degree_cap": query_cap, "retrieval_degree_cap": retrieval_cap,
        "pair_id_sha256": _pair_id_sha256(path),
        "component_parquet_sha256": file_sha256(path), **degrees,
    }


def _record_inventory(rows: list[dict], reserved: set[str]) -> dict[str, Any]:
    heldout = [row for row in rows if str(row["canonical_smiles"]) in reserved]
    train = [row for row in rows if str(row["canonical_smiles"]) not in reserved]
    count = lambda values: dict(sorted(Counter(  # noqa: E731
        str(row["measurement_kind"]) for row in values
    ).items()))
    return {
        "all_eligible_by_measurement_kind": count(rows),
        "scaffold_reserved_by_measurement_kind": count(heldout),
        "train_candidate_by_measurement_kind": count(train),
    }


def _write_component(root: Path, rows: list[dict], cap: int) -> tuple[Path, int]:
    text, count = v7._write_train_flat_variable(root, rows, cap)
    path = Path(text)
    if pq.ParquetFile(path).metadata.num_rows != count:
        raise RuntimeError("train component row-count mismatch")
    return path, count


def _release_schema(component_paths: list[Path], ranking_rows: list[dict]) -> pa.Schema:
    if not ranking_rows:
        raise RuntimeError("ranking validation has no complete anchor groups")
    schemas = [pq.read_schema(path) for path in component_paths]
    schemas.append(pa.Table.from_pylist(ranking_rows).schema)
    return pa.unify_schemas(schemas, promote_options="permissive")


def _align_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if field.name not in table.column_names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        column = table[field.name]
        arrays.append(column if column.type == field.type else pc.cast(column, field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_combined_train(inputs: list[Path], output: Path, schema: pa.Schema) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary, schema, compression="zstd", compression_level=TRAIN_ZSTD_LEVEL
    )
    count = 0
    try:
        for path in inputs:
            for batch in pq.ParquetFile(path).iter_batches(CONCAT_BATCH_ROWS):
                writer.write_table(_align_table(pa.Table.from_batches([batch]), schema))
                count += batch.num_rows
    finally:
        writer.close()
    temporary.replace(output)
    return count


def _write_ranking(rows: list[dict], output: Path, schema: pa.Schema) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        _align_table(table, schema), temporary, compression="zstd",
        compression_level=TRAIN_ZSTD_LEVEL,
    )
    temporary.replace(output)
    return len(rows)


def _split_artifacts(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    paths = {split: root / split / "data.parquet" for split in SPLITS}
    hashes = {split: file_sha256(path) for split, path in paths.items()}
    counts = {split: pq.ParquetFile(path).metadata.num_rows for split, path in paths.items()}
    return hashes, counts


def _dataset_card(task_id: str, manifest: Mapping[str, Any]) -> str:
    rows, components = manifest["rows"], manifest["train_components"]
    component_text = ", ".join(
        f"{name}={details['achieved_pairs']:,}" for name, details in components.items()
    )
    return f"""---
pretty_name: Assay Transfer Raw Pair V11 Intern — {TASK_TITLES[task_id]} — With Categorical
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

# Assay Transfer Raw Pair V11 Intern — {TASK_TITLES[task_id]} — With Categorical

This V11 release contains continuous, binary, and ordinal assay-transfer training pairs.
Train has {rows['train']:,} rows ({component_text}). Scaffold gold `valid` and `test` identities
are merged into one {rows['validation_ranking']:,}-row ranking-validation split and excluded in
full from train.

Continuous and ordinal soft targets use within-bucket empirical value-CDF separation. Binary
targets use 0.95/0.05 same/different-category supervision. Ranking groups contain 20 candidates;
categorical groups contain five query-category and fifteen non-query-category candidates. Metric
calculation is intentionally downstream in therapeutic-tuning.
"""


def _release_manifest(
    task_id: str, root: Path, source: Path, registry_path: Path,
    registry: Mapping[str, Any], components: Mapping[str, dict], inventory: Mapping[str, Any],
    ranking: Mapping[str, Any], sampling: Mapping[str, Any], schema: pa.Schema,
) -> dict[str, Any]:
    hashes, counts = _split_artifacts(root)
    release = categorical_ablation_config(registry)
    return {
        "version": "v11-txagent-v7-value-cdf-with-categorical",
        "dataset_variant": VARIANT, "hub_dataset_id": HUB_DATASET_IDS[task_id],
        "task_id": task_id, "source": str(source),
        "eligible_source_sha256": file_sha256(source),
        "eligible_source_manifest_sha256": file_sha256(source.parent / "manifest.json"),
        "prompt_projection": {
            "schema_version": registry["schema_version"], "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "target_contract": {"name": TARGET_CONTRACT, "probability_epsilon": TARGET_EPSILON},
        "construction": v11.construction_config(task_id, registry),
        "measurement_kind_policy": {
            "reservation": "all_eligible_measurement_kinds",
            "evaluation": list(release["evaluation_measurement_kinds"]),
            "train_components": ["continuous", "categorical"],
        },
        "train_components": dict(components), "eligible_record_inventory": dict(inventory),
        "source_record_sampling": dict(sampling),
        "heldout_reservation": heldout_reservation_manifest(task_id, registry),
        "ranking_validation": dict(ranking), "release_schema": schema.names,
        "rows": counts, "parquet_sha256": hashes,
        "paths": {split: f"{split}/data.parquet" for split in SPLITS},
    }


def _write_release_files(root: Path, task_id: str, manifest: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(_dataset_card(task_id, manifest), encoding="utf-8")


def _promote(staged: Path, final: Path) -> None:
    backup = final.with_name(f".{final.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved_old = False
    try:
        if final.exists():
            final.replace(backup)
            moved_old = True
        staged.replace(final)
    except Exception:
        if moved_old and not final.exists():
            backup.replace(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _prepare_rows(
    task_id: str, source: Path, registry_path: Path, rebuild_source: bool,
) -> tuple[Path, list[dict], dict, set[str], dict[str, Any]]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    v11.configure_engine(task_id, registry, registry_path)
    if rebuild_source:
        v11.write_eligible_records(task_id, source.parent, registry=registry)
    source = v11._prepare_source(task_id, source, None, registry)
    reset_prompt_cache()
    rows = pq.read_table(source).to_pylist()
    rows, sampling = _sample_configured_sources(rows, task_config(task_id, registry), task_id)
    reserved = heldout_union_molecules(task_id, rows, registry)
    return source, rows, registry, reserved, sampling


def _build_staged_release(
    task_id: str, staged: Path, components_root: Path, source: Path,
    rows: list[dict], registry_path: Path, registry: Mapping[str, Any], reserved: set[str],
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    release = categorical_ablation_config(registry)
    train_rows = _unreserved(rows, reserved)
    component_paths, components = [], {}
    for name in ("continuous", "categorical"):
        config = release["train_components"][name]
        selected = _kinds(train_rows, config["measurement_kinds"])
        path, count = _write_component(components_root / name, selected, int(config["max_pairs"]))
        component_paths.append(path)
        components[name] = _component_manifest(name, config, path, count)
    heldout = [row for row in rows if str(row["canonical_smiles"]) in reserved]
    anchors, ranking_diagnostics = build_ranking_anchors(
        heldout, task_config(task_id, registry), release
    )
    ranking_rows = materialize_ranking_rows(
        anchors, v11._row_for_v11, v11._fix_metadata_v11, v11.is_decisive
    )
    schema = _release_schema(component_paths, ranking_rows)
    train_count = _write_combined_train(component_paths, staged / "train/data.parquet", schema)
    expected_train = sum(item["achieved_pairs"] for item in components.values())
    if train_count != expected_train:
        raise RuntimeError("combined train does not match its component row counts")
    _write_ranking(ranking_rows, staged / "validation_ranking/data.parquet", schema)
    inventory = _record_inventory(rows, reserved)
    manifest = _release_manifest(
        task_id, staged, source, registry_path, registry, components,
        inventory, ranking_diagnostics, sampling, schema,
    )
    _write_release_files(staged, task_id, manifest)
    return manifest


def build_task_release(
    task_id: str, source: Path, output_root: Path, component_root: Path,
    registry_path: Path = DEFAULT_REGISTRY, rebuild_source: bool = False,
) -> dict[str, Any]:
    source, rows, registry, reserved, sampling = _prepare_rows(
        task_id, source, registry_path, rebuild_source
    )
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)
    component_root.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".with_categorical.stage.", dir=task_output))
    component_parent = Path(tempfile.mkdtemp(prefix=f".{task_id}.", dir=component_root))
    staged = stage_parent / VARIANT
    staged.mkdir()
    try:
        manifest = _build_staged_release(
            task_id, staged, component_parent, source, rows,
            registry_path, registry, reserved, sampling,
        )
        _promote(staged, task_output / VARIANT)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)
        shutil.rmtree(component_parent, ignore_errors=True)


def build_task_pair(*args, **kwargs) -> dict[str, dict]:
    """Compatibility wrapper returning the sole remaining release by variant name."""
    return {VARIANT: build_task_release(*args, **kwargs)}


def _parse_args() -> argparse.Namespace:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(registry["tasks"]))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--rebuild-source", action="store_true")
    args = parser.parse_args()
    args.source = args.source or v11.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    args = _parse_args()
    manifest = build_task_release(
        args.task, args.source, args.output_root, args.component_root,
        args.registry, args.rebuild_source,
    )
    print(json.dumps(manifest["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
