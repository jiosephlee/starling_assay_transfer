"""Row-level normalization and acceptance gates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from pipeline.normalize.assay_species_normalization import resolve_species_record
from pipeline.source_normalization.measurement import (
    approved_category,
    extract_embedded_unit,
    parse_measurement,
)
from pipeline.source_normalization.structures import (
    canonicalize_smiles,
    compare_exported_smiles,
    tdc_match,
)
from pipeline.source_normalization.text import as_nonempty_text, normalize_lexical, normalize_unit

PRIMARY_REASON_ORDER = (
    "missing_global_identifier",
    "unresolved_smiles_mapping",
    "conflicting_smiles_mapping",
    "invalid_structure",
    "wildcard_structure",
    "tdc_molecule_match",
    "missing_endpoint_alias",
    "missing_measurement",
    "unparseable_measurement",
)


def stable_hash(*values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_record_id(source_id: str, input_hash: str, row_number: int) -> str:
    return stable_hash("record_v1", source_id, input_hash, row_number)


def _native(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _native_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _native(value) for key, value in row.items()}


def _mapped_structure(
    row: Mapping[str, Any], mapping: Mapping[str, str]
) -> tuple[dict[str, Any], str | None]:
    identifier = as_nonempty_text(row.get("global_identifier"))
    source_smiles = as_nonempty_text(row.get("smiles"))
    if identifier is None:
        return _empty_structure(source_smiles), "missing_global_identifier"
    mapped = mapping.get(identifier)
    if mapped is None:
        return _empty_structure(source_smiles), "unresolved_smiles_mapping"
    canonical, error = canonicalize_smiles(mapped)
    fields = {
        "source_smiles": source_smiles,
        "authoritative_smiles": mapped,
        "smiles_comparison_status": compare_exported_smiles(source_smiles, mapped),
        "canonical_smiles": canonical,
    }
    return fields, error


def _empty_structure(source_smiles: str | None) -> dict[str, Any]:
    return {
        "source_smiles": source_smiles,
        "authoritative_smiles": None,
        "smiles_comparison_status": None,
        "canonical_smiles": None,
    }


def _direct_structure(row: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    source_smiles = as_nonempty_text(row.get("smiles"))
    canonical, error = canonicalize_smiles(source_smiles)
    fields = {
        "source_smiles": source_smiles,
        "authoritative_smiles": source_smiles,
        "smiles_comparison_status": "exact_match" if source_smiles else "missing",
        "canonical_smiles": canonical,
    }
    return fields, error


def structure_fields(
    row: Mapping[str, Any], spec: Mapping[str, Any], mapping: Mapping[str, str]
) -> tuple[dict[str, Any], str | None]:
    if spec["structure_mode"] == "mapped":
        return _mapped_structure(row, mapping)
    return _direct_structure(row)


def _categorical_value(
    row: Mapping[str, Any], raw_value: Any, spec: Mapping[str, Any], allowlists: Mapping[str, Any]
) -> tuple[str | None, bool]:
    categorical_column = spec.get("categorical_column")
    categorical_raw = row.get(categorical_column) if categorical_column else None
    candidates = []
    if categorical_column:
        candidates.append((categorical_raw, spec.get("categorical_allowlist")))
    candidates.append((raw_value, spec.get("categorical_measurement_allowlist")))
    for value, allowlist_name in candidates:
        if allowlist_name:
            category = approved_category(value, allowlists[allowlist_name])
            if category is not None:
                return category, True
    present = as_nonempty_text(categorical_raw) is not None
    return None, present


def measurement_fields(
    row: Mapping[str, Any], spec: Mapping[str, Any], allowlists: Mapping[str, Any]
) -> tuple[dict[str, Any], str | None]:
    raw_value = row.get(spec["measurement_column"])
    categorical, categorical_present = _categorical_value(row, raw_value, spec, allowlists)
    parsed = parse_measurement(raw_value, categorical)
    raw_present = as_nonempty_text(raw_value) is not None
    if not parsed.successful:
        reason = "unparseable_measurement" if raw_present or categorical_present else "missing_measurement"
    else:
        reason = None
    fields = {"measurement_raw": as_nonempty_text(raw_value), **parsed.columns()}
    return fields, reason


def unit_fields(row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    if "unit_constant" in spec:
        raw = spec["unit_constant"]
    elif "unit_column" in spec:
        raw = row.get(spec["unit_column"])
    elif spec.get("embedded_unit"):
        raw = extract_embedded_unit(row.get(spec["measurement_column"]))
    else:
        raw = None
    return {"unit_raw": as_nonempty_text(raw), "unit_normalized": normalize_unit(raw)}


def endpoint_fields(row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = spec.get("endpoint_constant")
    if raw is None:
        raw = row.get(spec["endpoint_column"])
    return {"endpoint_alias_raw": as_nonempty_text(raw), "endpoint_alias_normalized": normalize_lexical(raw)}


def normalized_column_name(spec: Mapping[str, Any], column: str) -> str:
    overrides = spec.get("normalized_field_names", {})
    return overrides.get(column, f"{column}_normalized")


def normalized_fields(row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for column in spec["normalized_fields"]:
        normalizer = normalize_unit if column == spec.get("unit_column") else normalize_lexical
        fields[normalized_column_name(spec, column)] = normalizer(row.get(column))
    return fields


def _reasons(
    structure_error: str | None,
    is_tdc: bool,
    endpoint: Mapping[str, Any],
    measurement_error: str | None,
) -> list[str]:
    reasons = []
    if structure_error:
        reasons.append(structure_error)
    if is_tdc:
        reasons.append("tdc_molecule_match")
    if endpoint["endpoint_alias_normalized"] is None:
        reasons.append("missing_endpoint_alias")
    if measurement_error:
        reasons.append(measurement_error)
    return reasons


def normalize_row(
    row: Mapping[str, Any], context: Mapping[str, Any], row_number: int
) -> tuple[dict[str, Any], list[str]]:
    source_id, spec = context["source_id"], context["spec"]
    original = _native_row(row)
    structure, structure_error = structure_fields(original, spec, context["mapping"])
    endpoint = endpoint_fields(original, spec)
    measurement, measurement_error = measurement_fields(original, spec, context["allowlists"])
    is_tdc = tdc_match(structure["authoritative_smiles"], structure["canonical_smiles"], context["tdc"])
    reasons = _reasons(structure_error, is_tdc, endpoint, measurement_error)
    metadata = {
        "record_id": stable_record_id(source_id, context["input_hash"], row_number),
        "source_id": source_id,
        "source_row_number": row_number,
        "input_sha256": context["input_hash"],
    }
    derived = {**metadata, **structure, **endpoint, **measurement, **unit_fields(original, spec)}
    derived["species_exact"] = resolve_species_record(original, spec["species_collection"])
    return {**original, **derived, **normalized_fields(original, spec)}, reasons


def primary_reason(reasons: list[str]) -> str:
    return min(reasons, key=PRIMARY_REASON_ORDER.index)


def rejection_record(record: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    keep = (
        "record_id", "source_id", "source_row_number", "input_sha256", "pmid",
        "extraction_id", "global_identifier", "source_smiles", "authoritative_smiles",
        "canonical_smiles", "smiles_comparison_status", "endpoint_alias_raw", "measurement_raw",
    )
    result = {key: record.get(key) for key in keep}
    return {**result, "rejection_reasons": reasons, "primary_rejection_reason": primary_reason(reasons)}


def duplicate_key(record: Mapping[str, Any], combination_fields: list[str]) -> str:
    measurement = [
        record.get(field) for field in (
            "scalar_value", "scalar_is_approximate", "lower_bound", "upper_bound",
            "interval_lower", "interval_upper", "categorical_value",
        )
    ]
    structured = [record.get(field) for field in combination_fields]
    return stable_hash("duplicate_v1", record.get("canonical_smiles"), measurement, structured)


def annotate_duplicates(records: list[dict[str, Any]], combination_fields: list[str]) -> None:
    keys = [duplicate_key(record, combination_fields) for record in records]
    sizes = Counter(keys)
    for record, key in zip(records, keys, strict=True):
        record["duplicate_group_id"] = key
        record["duplicate_group_size"] = sizes[key]
