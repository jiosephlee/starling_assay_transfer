#!/usr/bin/env python3
"""V12 pair engine: train-only calibration and same-parent record comparisons."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v12_contract import DEFAULT_REGISTRY
from pipeline.v12_targets import is_decisive, target_for
from scripts import build_v11_intern_raw_pair as v11


DEFAULT_ELIGIBLE_ROOT = Path("datasets/eligible/assay_transfer_starling_txagent_v12")
DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v12")
TRAIN_QUERY_DEGREE_CAP = 6
TRAIN_RETRIEVAL_DEGREE_CAP = 6


def eligible(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> bool:
    if str(query["child_id"]) == str(retrieval["child_id"]):
        return False
    fields = (
        "task_id", "source_id", "measurement_kind", "canonical_endpoint_key",
        "canonical_measurement_scale_id", "pair_bucket_key",
    )
    return all(query.get(field) == retrieval.get(field) for field in fields)


def pair_relationship(query: Mapping[str, Any], retrieval: Mapping[str, Any]) -> str:
    if query["normalized_parent_identity_key"] == retrieval["normalized_parent_identity_key"]:
        if query["canonical_smiles"] == retrieval["canonical_smiles"]:
            return "same_parent_same_canonical_smiles_distinct_records"
        return "same_parent_different_canonical_smiles"
    return "different_parent"


def row_for(query: dict[str, Any], retrieval: dict[str, Any], split: str) -> dict[str, Any]:
    row = v11._row_for_v11(query, retrieval, split)
    row.update({
        "query_parent_identity_key": query["normalized_parent_identity_key"],
        "retrieval_parent_identity_key": retrieval["normalized_parent_identity_key"],
        "pair_relationship": pair_relationship(query, retrieval),
    })
    return row


def configure_engine(registry_path: Path = DEFAULT_REGISTRY) -> None:
    pair_core.target_for = target_for
    pair_driver.target_for = target_for
    v7.target_for = target_for
    v7.row_for = row_for
    v7._fix_metadata = v11._fix_metadata_v11
    v7._is_decisive = is_decisive
    pair_core.render_prompt = functools.partial(render_prompt, registry_path=registry_path)
    pair_core.eligible = eligible
    v7.TRAIN_RETRIEVAL_DEGREE_CAP = TRAIN_RETRIEVAL_DEGREE_CAP
    v7.TRAIN_QUERY_DEGREE_CAP = TRAIN_QUERY_DEGREE_CAP


__all__ = ["configure_engine", "eligible", "pair_relationship", "row_for"]
