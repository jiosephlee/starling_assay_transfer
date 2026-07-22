#!/usr/bin/env python3
"""Validate canonical bases and compose the v3 eligible-record artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.v3_policy import V3Policies, file_sha256, resolve_path

REQUIRED = (
    "child_id", "parent_provenance_id", "record_id", "source_id", "input_sha256",
    "source_smiles", "canonical_smiles", "canonical_endpoint_key", "endpoint_family", "endpoint_subtype",
    "unit_basis", "scalar_value", "scalar_is_approximate",
)
CONTEXT_FIELDS = (
    "molecule_name", "species_or_population", "report_or_statistic_type", "dose",
    "study_or_assay_system", "measured_process", "biological_context", "medium",
    "formulation_or_solid_form", "transporter_or_enzyme", "substrate_status",
    "intestinal_site", "molecular_form", "enzyme_or_pathway", "qualifying_conditions",
    "comparator", "extra_details",
)


def _first(row: dict[str, Any], *names: str) -> Any:
    return next((row.get(name) for name in names if row.get(name) not in (None, "")), None)


def _context(row: dict[str, Any]) -> dict[str, Any]:
    values = {
        "molecule_name": row.get("molecule_name"),
        "species_or_population": _first(row, "species_or_population", "species", "species_exact"),
        "report_or_statistic_type": _first(row, "bioavailability_report_type", "statistic_type"),
        "dose": _first(row, "dose", "oral_dose"),
        "study_or_assay_system": _first(row, "study_context", "oral_exposure_mode", "assay_system"),
        "measured_process": _first(row, "exposure_measure", "endpoint_category", "gut_wall_process", "metric_type"),
        "biological_context": row.get("biological_context"),
        "medium": row.get("condition_medium"),
        "formulation_or_solid_form": row.get("formulation_or_solid_form"),
        "transporter_or_enzyme": row.get("transporter_or_enzyme"),
        "substrate_status": row.get("substrate_status"),
        "intestinal_site": row.get("intestinal_site"),
        "molecular_form": row.get("molecular_form"),
        "enzyme_or_pathway": row.get("enzyme_or_pathway"),
        "qualifying_conditions": row.get("qualifying_conditions"),
        "comparator": _first(row, "comparator", "comparator_exposure"),
        "extra_details": row.get("extra_details"),
    }
    return {name: None if values[name] is None else str(values[name]) for name in CONTEXT_FIELDS}


def _validate_base(base: Path, expected_schema: str) -> tuple[Path, dict[str, Any]]:
    manifest_path, records_path = base / "manifest.json", base / "records.parquet"
    if not manifest_path.exists() or not records_path.exists():
        raise FileNotFoundError(f"canonical base is incomplete: {base}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_schema_version") != expected_schema:
        raise ValueError(f"unexpected base schema for {base}")
    expected = manifest["output_artifacts"]["records.parquet"]["sha256"]
    if file_sha256(records_path) != expected:
        raise ValueError(f"records hash mismatch for {base}")
    missing = set(REQUIRED) - set(pq.read_schema(records_path).names)
    if missing:
        raise ValueError(f"base {base} missing columns: {sorted(missing)}")
    return records_path, manifest


def _eligible_row(row: dict[str, Any], policies: V3Policies) -> tuple[dict[str, Any] | None, str | None]:
    concept = policies.concept_for(row)
    metric = policies.metric_for(row)
    if concept is None:
        return None, "unsupported_concept"
    if metric is None:
        return None, "unsupported_metric"
    if not row.get("canonical_smiles") or not row.get("canonical_endpoint_key"):
        return None, "missing_identity"
    value = float(row["scalar_value"])
    transformed = metric.transform_value(value, row.get("unit_basis"))
    if transformed is None:
        return None, "invalid_metric_domain"
    out = {key: row.get(key) for key in REQUIRED}
    out.update({"assay_concept": concept, "metric_type": metric.name,
                "comparison_value": transformed, "transfer_max": metric.transfer_max,
                "not_transfer_min": metric.not_transfer_min, "threshold_display": metric.display})
    out["measurement_label"] = row.get("measurement_label")
    out.update({f"context_{k}": v for k, v in _context(row).items()})
    return out, None


def _load_rows(paths: list[Path], policies: V3Policies) -> tuple[list[dict], list[dict], list[dict]]:
    kept, rejected, inputs = [], [], []
    seen = set()
    expected = policies.release["base_schema_version"]
    for base in paths:
        records_path, manifest = _validate_base(base, expected)
        inputs.append({"path": str(base), "records_sha256": file_sha256(records_path),
                       "rows": manifest["output_artifacts"]["records.parquet"]["rows"]})
        for row in pq.read_table(records_path).to_pylist():
            if row["child_id"] in seen:
                raise ValueError(f"duplicate child_id across bases: {row['child_id']}")
            seen.add(row["child_id"])
            out, reason = _eligible_row(row, policies)
            if reason:
                rejected.append({"child_id": row["child_id"], "source_id": row["source_id"],
                                 "canonical_endpoint_key": row["canonical_endpoint_key"],
                                 "endpoint_family": row["endpoint_family"],
                                 "endpoint_subtype": row["endpoint_subtype"],
                                 "unit_basis": row["unit_basis"], "scalar_value": row["scalar_value"],
                                 "reason": reason})
            else:
                kept.append(out)
    return kept, rejected, inputs


def build(args: argparse.Namespace) -> dict[str, Any]:
    policies = V3Policies(resolve_path(args.release))
    bases = [resolve_path(path) for path in args.base]
    rows, rejected, inputs = _load_rows(bases, policies)
    if not rows:
        raise RuntimeError("no v3-eligible canonical records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output_dir / "records.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(rejected), args.output_dir / "rejections.parquet", compression="zstd")
    concepts = Counter(row["assay_concept"] for row in rows)
    reasons = Counter(row["reason"] for row in rejected)
    manifest = {"stage": "compose_v3", "schema_version": policies.release["artifact_schema_version"],
                "policy_versions": policies.version_bundle, "base_inputs": inputs,
                "eligible_records": len(rows), "rejected_records": len(rejected),
                "records_by_concept": dict(concepts), "rejections_by_reason": dict(reasons),
                "records_sha256": file_sha256(args.output_dir / "records.parquet")}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
