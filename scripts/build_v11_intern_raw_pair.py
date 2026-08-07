#!/usr/bin/env python3
"""Build one task-separated v11 dataset from a complete TxAgent v7 artifact.

The pair engine retains v10's construction policy while task/source inventory,
priority phases, quotas, and prompts come from the executable v11 registry.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
import scripts.build_v8_intern_raw_pair as v8
import scripts.build_v10_intern_raw_pair as v10
from pipeline.pair_core import record_key
from pipeline.source_normalization.starling_txagent_eligible_v7 import (
    heldout_union_molecules,
    write_eligible_records,
)
from pipeline.v11_contract import (
    DEFAULT_REGISTRY,
    artifact_paths,
    eligible_projection_manifest,
    heldout_reservation_manifest,
    load_registry,
    task_artifact_root,
    task_config,
    validate_registry,
)
from pipeline.v11_prompt_rendering import render_prompt, reset_prompt_cache
from pipeline.v11_targets import candidate_label, is_decisive, target_for
from pipeline.v3_policy import file_sha256
from scripts.build_raw_pair import build


DEFAULT_ELIGIBLE_ROOT = Path("datasets/eligible/assay_transfer_starling_txagent_v11")
DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v11_intern")
HUB_DATASET_ID = "jiosephlee/assay-transfer-raw-pair-v11-intern"
TRAIN_RETRIEVAL_DEGREE_CAP = 6
TRAIN_QUERY_DEGREE_CAP = 6
_ORIGINAL_ROW_FOR = pair_core.row_for


def construction_config(task_id: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    return dict(task_config(task_id, registry)["construction"])


def _row_for_v11(query: dict, retrieval: dict, split: str) -> dict:
    row = _ORIGINAL_ROW_FOR(query, retrieval, split)
    row.update(
        {
            "task_id": query["task_id"],
            "source_id": query["source_id"],
            "measurement_kind": query["measurement_kind"],
        }
    )
    return row


def _fix_metadata_v11(row: dict) -> None:
    distribution = row["target_distribution"]
    row["metadata"] = {
        "soft_targets": {
            "(A)": distribution["transfer"],
            "(B)": distribution["nontransfer"],
        },
        "groups": {
            "assay_concept": row["assay_concept"],
            "task_id": row["task_id"],
            "source_id": row["source_id"],
            "measurement_kind": row["measurement_kind"],
        },
        "target_z": row["target_z"],
        "absolute_geometry_difference": row["absolute_geometry_difference"],
        "calibration_distance_p05": row["calibration_distance_p05"],
        "calibration_distance_p95": row["calibration_distance_p95"],
    }


def configure_engine(
    task_id: str, registry: Mapping[str, Any], registry_path: Path = DEFAULT_REGISTRY
) -> None:
    config = construction_config(task_id, registry)
    pair_core.target_for = target_for
    pair_driver.target_for = target_for
    v7.target_for = target_for
    v8.target_for = target_for
    v7.row_for = _row_for_v11
    v8.row_for = _row_for_v11
    v7._fix_metadata = _fix_metadata_v11
    v8._fix_metadata = _fix_metadata_v11
    v7._is_decisive = is_decisive
    v8._is_decisive = is_decisive
    v8._candidate_label = candidate_label
    v8.RANKING_ANCHOR_WIDTH = int(config["ranking_anchor_width"])
    v8.RECORD_DEGREE_CAP = int(config["record_degree_cap"])
    v8.ORDINARY_DEGREE_CAP = int(config["ordinary_degree_cap"])
    v8.ORDINARY_PER_CONCEPT_LABEL = dict(config["ordinary_pairs_per_label"])
    v10.RANKING_ANCHORS_BY_SPLIT = dict(config["ranking_anchors"])
    pair_core.render_prompt = functools.partial(render_prompt, registry_path=registry_path)
    v7.TRAIN_RETRIEVAL_DEGREE_CAP = TRAIN_RETRIEVAL_DEGREE_CAP
    v7.TRAIN_QUERY_DEGREE_CAP = TRAIN_QUERY_DEGREE_CAP


def _shared_state(reserved: set[str]) -> tuple:
    return (
        reserved,
        set(),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )


def _construct_phases(
    rows: list[dict], task_id: str, registry: Mapping[str, Any], reserved: set[str]
) -> dict[str, dict]:
    restricted = [row for row in rows if str(row["canonical_smiles"]) in reserved]
    state = _shared_state(reserved)
    pieces: dict[str, list[dict]] = {"test": [], "validation": []}
    phases = task_config(task_id, registry)["priority_phases"]
    for phase in phases:
        concepts = set(phase)
        pieces["test"].append(
            v10.run_construction(restricted, "test", concepts, *state)
        )
        pieces["validation"].append(
            v10.run_construction(restricted, "validation", concepts, *state)
        )
    return {split: v10._merge_results(*values) for split, values in pieces.items()}


def _partition_rows(
    rows: list[dict], results: Mapping[str, dict], reserved: set[str]
) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    validation = results["validation"]["molecules"]
    test = results["test"]["molecules"]
    for row in rows:
        molecule = str(row["canonical_smiles"])
        if molecule in validation:
            output["validation"].append(row)
        if molecule in test:
            output["test"].append(row)
        if molecule not in validation and molecule not in test and molecule not in reserved:
            output["train"].append(row)
    return {name: sorted(values, key=record_key) for name, values in output.items()}


def make_split_eval(task_id: str, registry: Mapping[str, Any]):
    cache: dict[str, Any] = {}

    def split_records(rows: list[dict]) -> dict[str, list[dict]]:
        if not cache:
            reserved = heldout_union_molecules(task_id, rows, registry)
            results = _construct_phases(rows, task_id, registry, reserved)
            cache.update(
                results=results,
                reserved=reserved,
                split_rows=_partition_rows(rows, results, reserved),
            )
        return cache["split_rows"]

    return split_records, cache


def _configure_driver(cache: dict, split_records, train_iterator) -> None:
    ordinary, ranking = v8._make_eval_readers(cache["results"])
    pair_core.eligible = v7._eligible_with_pair_bucket
    pair_core.iter_list_groups = train_iterator
    pair_driver._ordinary = ordinary
    pair_driver._ranking = ranking
    pair_driver.iter_list_groups = train_iterator
    pair_driver._write_train_flat = v7._write_train_flat_variable
    pair_driver._split_records = split_records


def _diagnostics(cache: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    results = cache["results"]
    achieved = {}
    for split in ("validation", "test"):
        counts = Counter(
            (row["source_id"], "transfer" if row["binary_label"] else "nontransfer")
            for row in results[split]["ordinary"]
        )
        achieved[split] = {
            source: {label: counts[source, label] for label in ("transfer", "nontransfer")}
            for source in config["ordinary_pairs_per_label"]
        }
    return {
        "reserved_pool_molecules": len(cache["reserved"]),
        "ranking_anchors_achieved": {
            split: dict(results[split]["ranking_anchor_concepts"])
            for split in ("validation", "test")
        },
        "ordinary_pairs_achieved": achieved,
    }


def _stale_upstream_inputs(
    manifest: Mapping[str, Any], task_id: str, artifact_root: Path | None,
    registry: Mapping[str, Any] | None = None,
) -> list[str]:
    """Names of TxAgent artifacts whose current hash differs from the one the manifest recorded.

    An eligible manifest without recorded inputs predates provenance tracking; there is nothing to
    compare it against, so it is left alone rather than forced into a rebuild.
    """
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


def _prepare_source(
    task_id: str, source: Path, artifact_root: Path | None,
    registry: Mapping[str, Any] | None = None,
) -> Path:
    contract = dict(registry or load_registry())
    if not source.is_file():
        write_eligible_records(
            task_id, source.parent, artifact_root=artifact_root, registry=contract
        )
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
    # The manifest hash only proves the artifact matches itself. Skipping regeneration is only
    # safe while the upstream TxAgent artifacts it was derived from are also unchanged -- those
    # do get rebuilt, so check them rather than silently building on a stale source.
    stale = _stale_upstream_inputs(manifest, task_id, artifact_root, contract)
    projection = eligible_projection_manifest(task_id, contract)
    if stale or manifest.get("eligible_projection") != projection:
        write_eligible_records(
            task_id, source.parent, artifact_root=artifact_root, registry=contract
        )
        reset_prompt_cache()
    return source


def build_task(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    validate_registry(registry, tasks=(args.task,))
    configure_engine(args.task, registry, args.registry)
    source = _prepare_source(args.task, args.source, args.artifact_root, registry)
    # Prompt blocks are memoized by child_id; a regenerated source can reuse an id with new
    # contents, so drop anything cached before this build's records are read.
    reset_prompt_cache()
    rows = pq.read_table(source).to_pylist()
    split_records, cache = make_split_eval(args.task, registry)
    split_rows = split_records(rows)
    train_iterator = (
        functools.partial(v7._iter_list_groups_capped, min_group_size=4)
        if args.listnet
        else v7._iter_list_groups_capped
    )
    _configure_driver(cache, split_records, train_iterator)
    volume = args.train_groups if args.listnet else args.train_pairs
    expected = {name: len(values) for name, values in split_rows.items()}
    manifest = build(
        source, args.output, volume, listnet=args.listnet,
        expected_records=len(rows), expected_split=expected, records=rows,
    )
    config = construction_config(args.task, registry)
    manifest.update(
        version="v11-txagent-v7-task-separated",
        hub_dataset_id=HUB_DATASET_ID,
        task_id=args.task,
        eligible_source_sha256=file_sha256(source),
        eligible_source_manifest_sha256=file_sha256(source.parent / "manifest.json"),
        prompt_projection={
            "schema_version": registry["schema_version"],
            "path": str(args.registry),
            "sha256": file_sha256(args.registry),
        },
        construction=dict(config),
        heldout_reservation=heldout_reservation_manifest(args.task, registry),
        diagnostics=_diagnostics(cache, config),
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(registry["tasks"]))
    parser.add_argument("--registry", type=Path, default=Path("configs/assay_transfer/v11/prompt_projection.json"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--listnet", action="store_true")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    args = parser.parse_args()
    args.source = args.source or DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    args.output = args.output or DEFAULT_OUTPUT_ROOT / args.task
    return args


def main() -> None:
    print(json.dumps(build_task(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
