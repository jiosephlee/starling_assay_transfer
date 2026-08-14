#!/usr/bin/env python3
"""Legacy selection-based V11.1 builder; not the canonical release entry point."""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_1_targets import (
    TARGET_CONTRACT, attach_distance_calibrations, build_distance_calibration,
    parsed_calibrations, target_for, write_distance_calibration,
)
from pipeline.v11_contract import (
    DEFAULT_REGISTRY, categorical_ablation_config, heldout_reservation_manifest,
    load_registry, task_config, validate_registry,
)
from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v11_ranking import build_ranking_anchors, materialize_ranking_rows
from pipeline.v11_targets import TARGET_EPSILON
from pipeline.v3_policy import file_sha256
from scripts import build_v11_1_intern_raw_pair as v11_1
from scripts import build_v11_categorical_ablation as v11_release
from scripts import build_v11_intern_raw_pair as v11


CALIBRATION_FILE = "calibration/percentile_distance_cdf.json.gz"
VARIANT = "with_categorical"
VERSION = "v11.1-txagent-v7-percentile-distance-cdf-with-categorical"
HUB_DATASET_IDS = {
    task: f"jiosephlee/assay-transfer-raw-pair-v11.1-{task.replace('_', '-')}-with-categorical-intern"
    for task in ("bioavailability_ma", "bbb_martins", "skin_reaction")
}


def _dataset_card(task_id: str, manifest: Mapping[str, Any]) -> str:
    rows, components = manifest["rows"], manifest["train_components"]
    component_text = ", ".join(
        f"{name}={details['achieved_pairs']:,}" for name, details in components.items()
    )
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

This V11.1 release contains {rows['train']:,} training rows ({component_text}) and
{rows['validation_ranking']:,} scaffold-held-out ranking rows. Continuous and ordinal targets
apply an endpoint-anchored empirical CDF to within-bucket value-percentile distances. Binary
targets use 0.95/0.05 same/different-category supervision. `target_a` is the canonical graded
equivalence target; this release does not emit `target_z`.
"""


def _manifest(
    task_id: str, source: Path, registry_path: Path, registry: Mapping[str, Any],
    components: Mapping[str, Any], inventory: Mapping[str, Any], ranking: Mapping[str, Any],
    sampling: Mapping[str, Any], schema, calibration_path: Path, rejected: Mapping[str, int],
    root: Path,
) -> dict[str, Any]:
    hashes, counts = v11_release._split_artifacts(root)
    release = categorical_ablation_config(registry)
    return {
        "version": VERSION, "dataset_variant": VARIANT,
        "hub_dataset_id": HUB_DATASET_IDS[task_id], "task_id": task_id,
        "source": str(source), "eligible_source_sha256": file_sha256(source),
        "eligible_source_manifest_sha256": file_sha256(source.parent / "manifest.json"),
        "prompt_projection": {
            "schema_version": registry["schema_version"], "path": str(registry_path),
            "sha256": file_sha256(registry_path),
        },
        "target_contract": {"name": TARGET_CONTRACT, "probability_epsilon": TARGET_EPSILON},
        "percentile_distance_calibration": {
            "path": CALIBRATION_FILE, "sha256": file_sha256(calibration_path),
            "excluded_records_by_measurement_kind": dict(rejected),
        },
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
        "paths": {split: f"{split}/data.parquet" for split in v11_release.SPLITS},
    }


def _prepare_rows(
    task_id: str, source: Path, registry_path: Path, rebuild_source: bool,
) -> tuple[Path, list[dict], dict, set[str], dict, dict, dict]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    v11_1.configure_engine(task_id, registry, registry_path)
    if rebuild_source:
        v11.write_eligible_records(task_id, source.parent, registry=registry)
    source = v11._prepare_source(task_id, source, None, registry)
    reset_prompt_cache()
    full_rows = pq.read_table(source).to_pylist()
    calibration = build_distance_calibration(full_rows, task_id)
    rows, rejected = attach_distance_calibrations(full_rows, parsed_calibrations(calibration))
    rows, sampling = v11_release._sample_configured_sources(
        rows, task_config(task_id, registry), task_id
    )
    reserved = heldout_union_molecules(task_id, rows, registry)
    return source, rows, registry, reserved, sampling, calibration, rejected


def _components(
    rows: list[dict], reserved: set[str], release: Mapping[str, Any], root: Path,
) -> tuple[list[Path], dict[str, Any]]:
    train_rows = v11_release._unreserved(rows, reserved)
    paths, manifests = [], {}
    for name in ("continuous", "categorical"):
        config = release["train_components"][name]
        selected = v11_release._kinds(train_rows, config["measurement_kinds"])
        path, count = v11_release._write_component(root / name, selected, int(config["max_pairs"]))
        paths.append(path)
        manifests[name] = v11_release._component_manifest(name, config, path, count)
    return paths, manifests


def _build_staged(
    task_id: str, staged: Path, component_root: Path, source: Path, rows: list[dict],
    registry_path: Path, registry: Mapping[str, Any], reserved: set[str], sampling: Mapping,
    calibration: Mapping[str, Any], rejected: Mapping[str, int],
) -> dict[str, Any]:
    release = categorical_ablation_config(registry)
    component_paths, components = _components(rows, reserved, release, component_root)
    heldout = [row for row in rows if str(row["canonical_smiles"]) in reserved]
    anchors, diagnostics = build_ranking_anchors(
        heldout, task_config(task_id, registry), release, target_builder=target_for
    )
    metadata = functools.partial(v11._fix_metadata_v11, target_contract=TARGET_CONTRACT)
    ranking_rows = materialize_ranking_rows(
        anchors, v11._row_for_v11, metadata, v11_1.is_decisive
    )
    schema = v11_release._release_schema(component_paths, ranking_rows)
    train_count = v11_release._write_combined_train(
        component_paths, staged / "train/data.parquet", schema
    )
    if train_count != sum(item["achieved_pairs"] for item in components.values()):
        raise RuntimeError("combined V11.1 train count does not match components")
    v11_release._write_ranking(ranking_rows, staged / "validation_ranking/data.parquet", schema)
    calibration_path = staged / CALIBRATION_FILE
    write_distance_calibration(calibration, calibration_path)
    inventory = v11_release._record_inventory(rows, reserved)
    manifest = _manifest(
        task_id, source, registry_path, registry, components, inventory, diagnostics,
        sampling, schema, calibration_path, rejected, staged,
    )
    (staged / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staged / "README.md").write_text(_dataset_card(task_id, manifest), encoding="utf-8")
    return manifest


def build_task_release(
    task_id: str, source: Path, output_root: Path = v11_1.DEFAULT_OUTPUT_ROOT,
    component_root: Path = v11_1.DEFAULT_COMPONENT_ROOT,
    registry_path: Path = DEFAULT_REGISTRY, rebuild_source: bool = False,
) -> dict[str, Any]:
    prepared = _prepare_rows(task_id, source, registry_path, rebuild_source)
    source, rows, registry, reserved, sampling, calibration, rejected = prepared
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)
    component_root.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v11_1.stage.", dir=task_output))
    component_parent = Path(tempfile.mkdtemp(prefix=f".{task_id}.", dir=component_root))
    staged = stage_parent / VARIANT
    staged.mkdir()
    try:
        manifest = _build_staged(
            task_id, staged, component_parent, source, rows, registry_path, registry,
            reserved, sampling, calibration, rejected,
        )
        v11_release._promote(staged, task_output / VARIANT)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)
        shutil.rmtree(component_parent, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(HUB_DATASET_IDS))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=v11_1.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=v11_1.DEFAULT_COMPONENT_ROOT)
    parser.add_argument("--rebuild-source", action="store_true")
    args = parser.parse_args()
    args.source = args.source or v11.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    raise SystemExit(
        "selection-based V11.1 builds are noncanonical; "
        "use scripts/build_v11_1_from_defined_split.py with --reference-release"
    )


if __name__ == "__main__":
    main()
