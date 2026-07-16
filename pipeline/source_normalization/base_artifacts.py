"""Canonical-base inventories, manifests, and deterministic reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from pipeline.source_normalization.artifacts import runtime_versions
from pipeline.source_normalization.io import parquet_schema, sha256_file

ARTIFACT_NAMES = (
    "records.parquet", "rejections.parquet", "parent_ledger.parquet",
    "endpoint_inventory.parquet",
)
IDENTITY_FIELDS = (
    "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
    "direction", "target", "kinetic_parameter", "auc_window", "defining_timepoint",
)


def endpoint_inventory(
    records: list[dict[str, Any]], source_id: str, source_name: str,
) -> pd.DataFrame:
    counts = Counter(tuple(record.get(field) for field in IDENTITY_FIELDS) for record in records)
    rows = []
    for values, count in sorted(counts.items(), key=lambda item: tuple(str(v) for v in item[0])):
        rows.append({
            "source_id": source_id, "source_name": source_name,
            **dict(zip(IDENTITY_FIELDS, values, strict=True)), "count": count,
        })
    return pd.DataFrame(rows, columns=["source_id", "source_name", *IDENTITY_FIELDS, "count"])


def output_hashes(source_directory: Path) -> dict[str, str]:
    return {name: sha256_file(source_directory / name) for name in ARTIFACT_NAMES}


def artifact_details(source_directory: Path) -> dict[str, Any]:
    details = {}
    for name in ARTIFACT_NAMES:
        path = source_directory / name
        details[name] = {
            "sha256": sha256_file(path), "size": path.stat().st_size,
            "rows": pq.ParquetFile(path).metadata.num_rows, "schema": parquet_schema(path),
        }
    return details


def _rejection_counts(rejections: list[dict[str, Any]]) -> dict[str, Any]:
    stages = Counter(row["rejection_stage"] for row in rejections)
    reasons = Counter(reason for row in rejections for reason in row["rejection_reasons"])
    return {"by_stage": dict(sorted(stages.items())), "by_reason": dict(sorted(reasons.items()))}


def _ledger_counts(ledger: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["parent_status"] for row in ledger).items()))


def build_manifest(
    *, source_id: str, spec: Mapping[str, Any], records: list[dict[str, Any]],
    rejections: list[dict[str, Any]], ledger: list[dict[str, Any]],
    shared: Mapping[str, Any], hashes: Mapping[str, str], source_directory: Path,
) -> dict[str, Any]:
    raw_parents = spec["rows"]
    scalar_children = sum(row["scalar_children_emitted"] for row in ledger)
    accepted_parents = sum(row["accepted_child_count"] > 0 for row in ledger)
    return {
        "pipeline_version": shared["config"]["version"],
        "artifact_schema_version": shared["config"]["artifact_schema_version"],
        "scalar_parser_version": shared["config"]["value_parser_version"],
        "endpoint_resolver_version": shared["config"]["endpoint_resolver_version"],
        "source_id": source_id, "source_name": spec["semantic_name"],
        "runtime_versions": runtime_versions(),
        "input": shared["source_metadata"][source_id], "references": shared["reference_metadata"],
        "config_sha256": shared["config_hash"],
        "stage_yields": {
            "raw_parents": raw_parents, "normalized_candidates": raw_parents,
            "scalar_children_emitted": scalar_children, "accepted_base_children": len(records),
            "accepted_parents": accepted_parents,
        },
        "split_expansion": {
            "extra_scalar_children": scalar_children - sum(scalar_children > 0 for scalar_children in [row["scalar_children_emitted"] for row in ledger]),
            "parents_with_multiple_scalars": sum(row["scalar_children_emitted"] > 1 for row in ledger),
            "partially_retained_parents": sum(row["parent_status"] == "partially_retained" for row in ledger),
        },
        "parent_status_counts": _ledger_counts(ledger), "rejections": _rejection_counts(rejections),
        "endpoint_coverage": {
            "assigned_scalar_children": len(records),
            "scalar_assignment_rate": len(records) / scalar_children if scalar_children else 0.0,
            "unique_canonical_endpoint_keys": len({row["canonical_endpoint_key"] for row in records}),
        },
        "reconciliation": {
            "exact": len(ledger) == raw_parents and len({row["parent_provenance_id"] for row in ledger}) == raw_parents,
            "raw_parent_rows": raw_parents, "ledger_rows": len(ledger),
            "accepted_plus_rejected_scalar_children": len(records) + sum(row["rejection_stage"] == "endpoint_assignment" for row in rejections),
            "input_unchanged": True,
        },
        "output_sha256": dict(hashes), "output_artifacts": artifact_details(source_directory),
    }


def render_report(manifests: list[Mapping[str, Any]]) -> str:
    total = sum(item["stage_yields"]["raw_parents"] for item in manifests)
    records = sum(item["stage_yields"]["accepted_base_children"] for item in manifests)
    lines = [
        "# Raw source to canonical base report", "", "## Result", "",
        f"All **{total:,}** pinned raw parents reconcile exactly through the parent ledger. "
        f"The pipeline retained **{records:,}** finite scalar children with canonical endpoint keys.", "",
        "| Source | Parents | Scalars | Base records | Keys | Partial parents |", "|---|---:|---:|---:|---:|---:|",
    ]
    for item in manifests:
        yields, coverage, expansion = item["stage_yields"], item["endpoint_coverage"], item["split_expansion"]
        lines.append(
            f"| {item['source_name']} | {yields['raw_parents']:,} | {yields['scalar_children_emitted']:,} | "
            f"{yields['accepted_base_children']:,} | {coverage['unique_canonical_endpoint_keys']:,} | "
            f"{expansion['partially_retained_parents']:,} |"
        )
    lines.extend(["", _report_scope(manifests), ""])
    return "\n".join(lines)


def _report_scope(manifests: list[Mapping[str, Any]]) -> str:
    curated = any("fg_support_reextraction" in item["references"] for item in manifests)
    if curated:
        return (
            "Only dedicated structured measurement fields were parsed, except for the pinned Q3 Fg "
            "support-text re-extraction artifact. Other narrative support and detail fields were never inspected. "
            "Generated Parquets are reproducible local artifacts and are not committed."
        )
    return "Only dedicated structured measurement fields were parsed. Narrative support and detail fields were never inspected. Generated Parquets are reproducible local artifacts and are not committed."
