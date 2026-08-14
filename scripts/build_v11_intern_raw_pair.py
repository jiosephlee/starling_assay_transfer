#!/usr/bin/env python3
"""V11 row/source utilities and compatibility entry point for the sole release."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Any, Mapping

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
from pipeline.source_normalization.starling_txagent_eligible_v7 import write_eligible_records
from pipeline.v11_contract import (
    DEFAULT_REGISTRY, artifact_paths, eligible_projection_manifest, load_registry,
    task_artifact_root, task_config, validate_registry,
)
from pipeline.v11_prompt_rendering import render_prompt, reset_prompt_cache
from pipeline.v11_targets import TARGET_CONTRACT, is_decisive, target_for
from pipeline.v3_policy import file_sha256


DEFAULT_ELIGIBLE_ROOT = Path("datasets/eligible/assay_transfer_starling_txagent_v11")
DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v11_categorical_ablation")
TRAIN_RETRIEVAL_DEGREE_CAP = 6
TRAIN_QUERY_DEGREE_CAP = 6
_ORIGINAL_ROW_FOR = pair_core.row_for


def construction_config(task_id: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    return dict(task_config(task_id, registry)["construction"])


def _side_values(prefix: str, record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "geometry_value", "finite_scalar_value", "value_percentile",
        "canonical_category_id", "canonical_category_rank",
        "canonical_measurement_scale_id", "canonical_unit_text",
        "raw_measurement_text", "raw_unit_text",
    )
    return {f"{prefix}_{field}": record.get(field) for field in fields}


def _row_for_v11(query: dict, retrieval: dict, split: str) -> dict:
    row = _ORIGINAL_ROW_FOR(query, retrieval, split)
    row.update({
        "task_id": query["task_id"], "source_id": query["source_id"],
        "measurement_kind": query["measurement_kind"],
        "pair_bucket_key": query["pair_bucket_key"],
        "canonical_measurement_scale_id": query.get("canonical_measurement_scale_id"),
        "calibration_standard_deviation_ddof": query["calibration_standard_deviation_ddof"],
        "calibration_standard_deviation_value_field": query[
            "calibration_standard_deviation_value_field"
        ],
        **_side_values("query", query), **_side_values("retrieval", retrieval),
    })
    return row


def _fix_metadata_v11(row: dict, target_contract: str = TARGET_CONTRACT) -> None:
    distribution = row["target_distribution"]
    row["metadata"] = {
        "soft_targets": {
            "(A)": distribution["transfer"], "(B)": distribution["nontransfer"],
        },
        "groups": {
            key: row[key] for key in (
                "assay_concept", "task_id", "source_id", "measurement_kind"
            )
        },
        "absolute_geometry_difference": row["absolute_geometry_difference"],
        "calibration_sample_standard_deviation": row[
            "calibration_sample_standard_deviation"
        ],
        "target_contract": target_contract,
    }


def configure_engine(
    task_id: str, registry: Mapping[str, Any], registry_path: Path = DEFAULT_REGISTRY,
) -> None:
    pair_core.target_for = target_for
    pair_driver.target_for = target_for
    v7.target_for = target_for
    v7.row_for = _row_for_v11
    v7._fix_metadata = _fix_metadata_v11
    v7._is_decisive = is_decisive
    pair_core.render_prompt = functools.partial(render_prompt, registry_path=registry_path)
    pair_core.eligible = v7._eligible_with_pair_bucket
    v7.TRAIN_RETRIEVAL_DEGREE_CAP = TRAIN_RETRIEVAL_DEGREE_CAP
    v7.TRAIN_QUERY_DEGREE_CAP = TRAIN_QUERY_DEGREE_CAP


def _stale_upstream_inputs(
    manifest: Mapping[str, Any], task_id: str, artifact_root: Path | None,
    registry: Mapping[str, Any] | None = None,
) -> list[str]:
    recorded = manifest.get("inputs") or {}
    if not recorded:
        return []
    contract = dict(registry or load_registry())
    root = artifact_root or task_artifact_root(task_id, contract)
    stale = []
    for name, path in artifact_paths(root).items():
        entry = recorded.get(name) or {}
        if not path.is_file() or entry.get("sha256") != file_sha256(path):
            stale.append(name)
    return sorted(stale)


def _read_source_manifest(source: Path, task_id: str) -> tuple[Path, dict]:
    manifest_path = source.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"v11 eligible source lacks manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "starling_txagent_eligible_v7":
        raise ValueError("unexpected v11 eligible-source stage")
    if manifest.get("task_id") != task_id:
        raise ValueError("v11 eligible source belongs to a different task")
    if manifest.get("eligible_records_sha256") != file_sha256(source):
        raise ValueError("v11 eligible-source hash mismatch")
    return manifest_path, manifest


def _prepare_source(
    task_id: str, source: Path, artifact_root: Path | None,
    registry: Mapping[str, Any] | None = None,
) -> Path:
    contract = dict(registry or load_registry())
    if not source.is_file():
        write_eligible_records(task_id, source.parent, artifact_root=artifact_root, registry=contract)
    _, manifest = _read_source_manifest(source, task_id)
    stale = _stale_upstream_inputs(manifest, task_id, artifact_root, contract)
    projection = eligible_projection_manifest(task_id, contract)
    if stale or manifest.get("eligible_projection") != projection:
        write_eligible_records(task_id, source.parent, artifact_root=artifact_root, registry=contract)
        reset_prompt_cache()
        _read_source_manifest(source, task_id)
    return source


def build_task(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.build_v11_categorical_ablation import build_task_release

    return build_task_release(
        args.task, args.source, args.output_root, args.component_root,
        args.registry, args.rebuild_source,
    )


def _parse_args() -> argparse.Namespace:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(registry["tasks"]))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--component-root", type=Path,
        default=Path("tmp/v11_value_cdf_ranking_release/components"),
    )
    parser.add_argument("--rebuild-source", action="store_true")
    args = parser.parse_args()
    args.source = args.source or DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    print(json.dumps(build_task(_parse_args())["rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
