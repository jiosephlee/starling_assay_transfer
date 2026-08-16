#!/usr/bin/env python3
"""Compose three task-separated V12 or V12.1 releases without changing row order."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.v3_policy import file_sha256
from scripts import build_v11_categorical_ablation as v11_release
from scripts import build_v12_release as v12_release


TASKS = ("bbb_martins", "bioavailability_ma", "skin_reaction")


def _load_inputs(roots: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    if set(roots) != set(TASKS):
        raise ValueError("composed V12 release requires exactly the three canonical tasks")
    manifests = {
        task: json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for task, root in roots.items()
    }
    versions = {manifest["version"] for manifest in manifests.values()}
    if len(versions) != 1:
        raise ValueError("cannot compose mixed V12/V12.1 versions")
    for task, manifest in manifests.items():
        if manifest.get("task_id") != task:
            raise ValueError(f"release root for {task} belongs to another task")
    return manifests


def _schema(roots: Mapping[str, Path]) -> pa.Schema:
    schemas = [
        pq.read_schema(roots[task] / split / "data.parquet")
        for task in TASKS for split in v12_release.SPLITS
    ]
    return pa.unify_schemas(schemas, promote_options="permissive")


def _write_split(
    roots: Mapping[str, Path], split: str, output: Path, schema: pa.Schema,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd", compression_level=3)
    count = 0
    try:
        for task in TASKS:
            path = roots[task] / split / "data.parquet"
            for batch in pq.ParquetFile(path).iter_batches(32768):
                writer.write_table(v11_release._align_table(pa.Table.from_batches([batch]), schema))
                count += batch.num_rows
    finally:
        writer.close()
    temporary.replace(output)
    return count


def _copy_calibrations(roots: Mapping[str, Path], staged: Path) -> dict[str, Any]:
    output = {}
    for task in TASKS:
        source = roots[task] / "calibration"
        if not source.is_dir():
            continue
        destination = staged / "calibration" / task
        shutil.copytree(source, destination)
        output[task] = {
            path.name: file_sha256(path)
            for path in sorted(destination.iterdir()) if path.is_file()
        }
    return output


def _card(version: str, rows: Mapping[str, int]) -> str:
    return f"""---
pretty_name: Assay Transfer {version} Intern - Composed Tasks
task_categories:
  - text-classification
language:
  - en
---

# Assay Transfer {version} Intern - Composed Tasks

This is an ordered concatenation of the BBB, Bioavailability, and Skin Reaction task releases.
It contains {rows['train']:,} train, {rows['validation_ranking']:,} validation-ranking, and
{rows['test_ranking']:,} test-ranking rows. Task-local lineage, split, and calibration manifests
remain authoritative and are content-addressed below the composed manifest.
"""


def build_composed_release(roots: Mapping[str, Path], output: Path) -> dict[str, Any]:
    manifests = _load_inputs(roots)
    schema = _schema(roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix=".v12.composed.", dir=output.parent))
    staged = stage_parent / output.name
    staged.mkdir()
    try:
        rows = {
            split: _write_split(roots, split, staged / split / "data.parquet", schema)
            for split in v12_release.SPLITS
        }
        manifest = {
            "version": next(iter(manifests.values()))["version"],
            "release_type": "composed_task_concatenation", "task_order": list(TASKS),
            "task_inputs": {
                task: {
                    "root": str(roots[task]),
                    "manifest_sha256": file_sha256(roots[task] / "manifest.json"),
                    "rows": manifests[task]["rows"], "gold_lineage": manifests[task]["gold_lineage"],
                }
                for task in TASKS
            },
            "rows": rows, "release_schema": schema.names,
            "parquet_sha256": {
                split: file_sha256(staged / split / "data.parquet")
                for split in v12_release.SPLITS
            },
            "calibration": _copy_calibrations(roots, staged),
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staged / "README.md").write_text(
            _card(manifest["version"], rows), encoding="utf-8"
        )
        v11_release._promote(staged, output)
        return manifest
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def _task_roots(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        task, separator, path = value.partition("=")
        if not separator:
            raise ValueError("--task-release must be TASK=PATH")
        output[task] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-release", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_composed_release(_task_roots(args.task_release), args.output)
    print(json.dumps(manifest["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
