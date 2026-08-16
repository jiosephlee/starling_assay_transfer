#!/usr/bin/env python3
"""Build one leakage-safe assay-transfer V12 task release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.build_v7_intern_raw_pair as v7
from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v11_targets import TARGET_EPSILON
from pipeline.v12_contract import (
    DEFAULT_REGISTRY, load_registry, release_config, task_config, validate_registry,
)
from pipeline.v12_ranking import build_ranking_anchors, materialize_ranking_rows
from pipeline.v12_source import write_eligible_records
from pipeline.v12_targets import TARGET_CONTRACT
from pipeline.v3_policy import file_sha256
from scripts import build_v11_categorical_ablation as v11_release
from scripts import build_v12_intern_raw_pair as v12


DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v12")
DEFAULT_COMPONENT_ROOT = Path("tmp/v12_release/components")
SPLITS = ("train", "validation_ranking", "test_ranking")
VERSION = "v12-txagent-lineage-train-calibrated-with-categorical"
VARIANT = "with_categorical"
TASK_TITLES = v11_release.TASK_TITLES


def _pair_id_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["pair_id"]):
        for value in batch.column(0).to_pylist():
            digest.update(str(value).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _ranking_identity_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    columns = ["pair_id", "ranking_query_id", "ranking_member_index", "ranking_family"]
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        for row in batch.to_pylist():
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
            digest.update(payload.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _write_component(
    root: Path, rows: list[dict[str, Any]], maximum: int,
) -> tuple[Path, int]:
    path_text, count = v7._write_train_flat_variable(root, rows, maximum)
    path = Path(path_text)
    if pq.ParquetFile(path).metadata.num_rows != count:
        raise RuntimeError("component row-count mismatch")
    return path, count


def _component_manifest(
    name: str, config: Mapping[str, Any], path: Path, count: int,
) -> dict[str, Any]:
    manifest = v11_release._component_manifest(name, config, path, count)
    relationships = Counter()
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["pair_relationship"]):
        relationships.update(str(value) for value in batch.column(0).to_pylist())
    manifest["pair_relationship_counts"] = dict(sorted(relationships.items()))
    return manifest


def _ranking_rows(
    rows: list[dict[str, Any]], task: Mapping[str, Any], split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train = [row for row in rows if row["dataset_split"] == "train"]
    queries = [row for row in rows if row["dataset_split"] == split]
    anchors, diagnostics = build_ranking_anchors(queries, train, task, split)
    materialized = materialize_ranking_rows(
        anchors, split, v12.row_for, v11_release.v11._fix_metadata_v11,
        v11_release.v11.is_decisive,
    )
    return materialized, diagnostics


def _release_schema(component_paths: list[Path], rankings: list[list[dict]]) -> pa.Schema:
    schemas = [pq.read_schema(path) for path in component_paths]
    for rows in rankings:
        if rows:
            schemas.append(pa.Table.from_pylist(rows).schema)
    if len(schemas) == len(component_paths):
        raise RuntimeError("both V12 ranking splits are empty")
    return pa.unify_schemas(schemas, promote_options="permissive")


def _write_outputs(
    staged: Path, components: list[Path], rankings: Mapping[str, list[dict]], schema: pa.Schema,
) -> dict[str, int]:
    counts = {
        "train": v11_release._write_combined_train(
            components, staged / "train/data.parquet", schema
        )
    }
    for split, rows in rankings.items():
        counts[f"{split}_ranking"] = v11_release._write_ranking(
            rows, staged / f"{split}_ranking/data.parquet", schema
        )
    return counts


def _copy_calibrations(source_dir: Path, staged: Path) -> dict[str, Any]:
    output = {}
    for name in (
        "train_value_calibration.json.gz", "train_percentile_distance_calibration.json.gz",
        "split_manifest.json",
    ):
        destination = staged / "calibration" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / name, destination)
        output[name] = {"path": f"calibration/{name}", "sha256": file_sha256(destination)}
    return output


def _manifest(
    task_id: str, staged: Path, source: Path, registry_path: Path,
    components: Mapping[str, Any], rankings: Mapping[str, Any], counts: Mapping[str, int],
    schema: pa.Schema,
) -> dict[str, Any]:
    split_hashes = {split: file_sha256(staged / split / "data.parquet") for split in SPLITS}
    pair_hashes = {split: _pair_id_sha256(staged / split / "data.parquet") for split in SPLITS}
    ranking_hashes = {
        split: _ranking_identity_sha256(staged / split / "data.parquet")
        for split in ("validation_ranking", "test_ranking")
    }
    source_manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    return {
        "version": VERSION, "dataset_variant": VARIANT, "task_id": task_id,
        "source": str(source), "eligible_source_sha256": file_sha256(source),
        "eligible_source_manifest_sha256": file_sha256(source.parent / "manifest.json"),
        "gold_lineage": source_manifest["lineage"],
        "split_contract": source_manifest["split_contract"],
        "prompt_projection": {"path": str(registry_path), "sha256": file_sha256(registry_path)},
        "target_contract": {"name": TARGET_CONTRACT, "probability_epsilon": TARGET_EPSILON},
        "train_components": dict(components), "ranking": dict(rankings),
        "calibration": _copy_calibrations(source.parent, staged),
        "rows": dict(counts), "release_schema": schema.names,
        "parquet_sha256": split_hashes, "ordered_pair_id_sha256": pair_hashes,
        "ranking_identity_sha256": ranking_hashes,
        "paths": {split: f"{split}/data.parquet" for split in SPLITS},
        "freeze_policy": "split assignments and ordered ranking membership frozen before training",
        "model_selection_policy": "checkpoint selection on validation_ranking only; test_ranking after freeze",
    }


def _dataset_card(task_id: str, manifest: Mapping[str, Any]) -> str:
    rows = manifest["rows"]
    return f"""---
pretty_name: Assay Transfer Raw Pair V12 Intern - {TASK_TITLES[task_id]}
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

# Assay Transfer Raw Pair V12 Intern - {TASK_TITLES[task_id]}

V12 contains {rows['train']:,} train pairs, {rows['validation_ranking']:,} validation-ranking
rows, and {rows['test_ranking']:,} test-ranking rows. Splits are molecule-disjoint and grouped by
TxAgent normalized-parent identity. Test is restricted to the eligible intersection with the
gold valid+test molecule universe; validation is a stable held-out sample from gold train.

Source sampling and the split happen before the >=25-train-record bucket gate. Target CDFs use
train records only. TxAgent's global sample SD is retained only as measurement metadata. Every
ranking query is held out and its 24 candidates come from train.
"""


def _prepare_source(
    task_id: str, source: Path, registry_path: Path, rebuild: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    if rebuild or not source.is_file():
        write_eligible_records(task_id, source.parent, registry_path)
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "starling_txagent_eligible_v12":
        raise ValueError("unexpected V12 eligible source stage")
    stale_registry = manifest.get("prompt_projection", {}).get("sha256") != file_sha256(
        registry_path
    )
    stale_inputs = any(
        not Path(entry["path"]).is_file()
        or file_sha256(Path(entry["path"])) != entry["sha256"]
        for entry in manifest.get("inputs", {}).values()
    )
    if stale_registry or stale_inputs:
        write_eligible_records(task_id, source.parent, registry_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["artifacts"]["records.parquet"] != file_sha256(source):
        raise ValueError("V12 eligible source hash mismatch")
    return pq.read_table(source).to_pylist(), registry


def _build_staged(
    task_id: str, staged: Path, component_root: Path, source: Path,
    rows: list[dict[str, Any]], registry_path: Path, registry: Mapping[str, Any],
) -> dict[str, Any]:
    release, task = release_config(registry), task_config(task_id, registry)
    train = [row for row in rows if row["dataset_split"] == "train"]
    component_paths, component_manifests = [], {}
    for name in ("continuous", "categorical"):
        config = release["train_components"][name]
        kinds = set(config["measurement_kinds"])
        selected = [row for row in train if row["measurement_kind"] in kinds]
        path, count = _write_component(component_root / name, selected, int(config["max_pairs"]))
        component_paths.append(path)
        component_manifests[name] = _component_manifest(name, config, path, count)
    ranking_rows, ranking_diagnostics = {}, {}
    for split in ("validation", "test"):
        ranking_rows[split], ranking_diagnostics[split] = _ranking_rows(rows, task, split)
    schema = _release_schema(component_paths, list(ranking_rows.values()))
    counts = _write_outputs(staged, component_paths, ranking_rows, schema)
    manifest = _manifest(
        task_id, staged, source, registry_path, component_manifests,
        ranking_diagnostics, counts, schema,
    )
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staged / "README.md").write_text(_dataset_card(task_id, manifest), encoding="utf-8")
    return manifest


def build_task_release(
    task_id: str, source: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
    component_root: Path = DEFAULT_COMPONENT_ROOT, registry_path: Path = DEFAULT_REGISTRY,
    rebuild_source: bool = False,
) -> dict[str, Any]:
    rows, registry = _prepare_source(task_id, source, registry_path, rebuild_source)
    v12.configure_engine(registry_path)
    reset_prompt_cache()
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)
    component_root.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v12.stage.", dir=task_output))
    components = Path(tempfile.mkdtemp(prefix=f".{task_id}.", dir=component_root))
    staged = stage_parent / VARIANT
    staged.mkdir()
    try:
        manifest = _build_staged(
            task_id, staged, components, source, rows, registry_path, registry
        )
        v11_release._promote(staged, task_output / VARIANT)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)
        shutil.rmtree(components, ignore_errors=True)


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
    args.source = args.source or v12.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
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
