"""Direct raw-source to finite scalar canonical-base orchestration."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

from pipeline.source_normalization.base_artifacts import (
    build_manifest, endpoint_inventory, output_hashes, render_report,
)
from pipeline.source_normalization.endpoint_keys import assign_canonical_endpoint
from pipeline.source_normalization.io import (
    load_config, load_mapping, load_tdc_exclusions, read_source, required_identifiers,
    resolve_path, sha256_file, verify_artifact, write_json, write_parquet,
)
from pipeline.source_normalization.normalize import normalize_row, stable_hash
from pipeline.source_normalization.scalar import (
    PARSER_VERSION, FragmentRejection, ScalarEmission, split_scalar_measurements,
)

_STRUCTURAL_REASONS = {
    "missing_global_identifier", "unresolved_smiles_mapping", "conflicting_smiles_mapping",
    "invalid_structure", "wildcard_structure", "tdc_molecule_match", "missing_endpoint_alias",
}


def _preflight(config: Mapping[str, Any], data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = {key: verify_artifact(spec, data_root) for key, spec in config["sources"].items()}
    references = {key: verify_artifact(pin, data_root) for key, pin in config["references"].items()}
    return sources, references


def _shared_context(
    config: Mapping[str, Any], config_path: Path, data_root: Path,
    source_metadata: Mapping[str, Any], reference_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = required_identifiers(config, data_root)
    mapping_path = resolve_path(data_root, config["references"]["smiles_mapping"]["path"])
    mapping, mapping_stats = load_mapping(mapping_path, identifiers)
    tdc_path = resolve_path(data_root, config["references"]["tdc_exclusion"]["path"])
    tdc, tdc_stats = load_tdc_exclusions(tdc_path)
    references = dict(reference_metadata)
    references["smiles_mapping"] = {**references["smiles_mapping"], **mapping_stats}
    references["tdc_exclusion"] = {**references["tdc_exclusion"], **tdc_stats}
    return {
        "config": config, "config_hash": sha256_file(config_path),
        "source_metadata": source_metadata, "reference_metadata": references,
        "mapping": mapping, "tdc": tdc,
    }


def _row_context(source_id: str, spec: Mapping[str, Any], shared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source_id, "spec": spec, "mapping": shared["mapping"],
        "tdc": shared["tdc"], "allowlists": shared["config"]["categorical_allowlists"],
        "input_hash": spec["sha256"],
    }


def _child_id(parent_id: str, label: str | None, start: int, end: int, index: int) -> str:
    return stable_hash("child_v1", parent_id, PARSER_VERSION, label, [start, end], index)


def _base_rejection(record: Mapping[str, Any], child_id: str | None, stage: str) -> dict[str, Any]:
    keep = (
        "record_id", "source_id", "source_name", "source_row_number", "input_sha256", "pmid", "extraction_id",
        "global_identifier", "source_smiles", "authoritative_smiles", "canonical_smiles",
        "endpoint_alias_raw", "measurement_raw",
    )
    payload = {key: record.get(key) for key in keep}
    payload["parent_provenance_id"] = record["record_id"]
    payload["child_id"] = child_id
    payload["rejection_stage"] = stage
    return payload


def _structural_rejection(record: Mapping[str, Any], reasons: list[str]) -> dict[str, Any]:
    payload = _base_rejection(record, None, "structural")
    ordered = [reason for reason in reasons if reason in _STRUCTURAL_REASONS]
    return {**payload, "rejection_reasons": ordered, "primary_rejection_reason": ordered[0]}


def _fragment_rejection(
    record: Mapping[str, Any], fragment: FragmentRejection, index: int,
) -> tuple[dict[str, Any], str]:
    child = _child_id(record["record_id"], fragment.measurement_label, fragment.span_start, fragment.span_end, index)
    payload = _base_rejection(record, child, "scalar_fragment")
    reason = fragment.rejection_reason
    payload.update(fragment.columns())
    payload.update({"rejection_reasons": [reason], "primary_rejection_reason": reason})
    return payload, child


def _endpoint_rejection(
    record: Mapping[str, Any], emission: ScalarEmission, child: str, reason: str,
) -> dict[str, Any]:
    payload = _base_rejection(record, child, "endpoint_assignment")
    payload.update(emission.columns())
    payload.update({"rejection_reasons": [reason], "primary_rejection_reason": reason})
    return payload


def _accepted_record(
    record: Mapping[str, Any], emission: ScalarEmission, child: str, assignment: Any,
) -> dict[str, Any]:
    excluded = {
        "scalar_value", "scalar_is_approximate", "lower_bound", "upper_bound",
        "interval_lower", "interval_upper", "categorical_value", "measurement_kind",
    }
    candidate = {key: value for key, value in record.items() if key not in excluded}
    candidate["parent_provenance_id"] = record["record_id"]
    candidate["child_id"] = child
    return {**candidate, **emission.columns(), **assignment.columns()}


def _parent_status(emitted: int, accepted: int, rejected: int, structural: bool = False) -> str:
    if structural:
        return "structurally_rejected"
    if accepted and rejected:
        return "partially_retained"
    if accepted > 1:
        return "split"
    if accepted == 1:
        return "retained_once"
    return "endpoint_rejected" if emitted else "scalar_rejected"


def _ledger_row(
    record: Mapping[str, Any], emitted: int, accepted: int,
    rejection_count: int, child_ids: list[str], structural: bool = False,
) -> dict[str, Any]:
    return {
        "parent_provenance_id": record["record_id"], "source_id": record["source_id"],
        "source_name": record["source_name"],
        "source_row_number": record["source_row_number"], "input_sha256": record["input_sha256"],
        "parent_status": _parent_status(emitted, accepted, rejection_count, structural),
        "scalar_children_emitted": emitted, "accepted_child_count": accepted,
        "rejected_child_or_fragment_count": rejection_count, "child_ids": child_ids,
    }


def _process_candidate(
    record: dict[str, Any], source_id: str, spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    emissions, fragments = split_scalar_measurements(
        record.get(spec["measurement_column"]), record.get("endpoint_alias_raw"), record.get("unit_raw"),
    )
    accepted, rejected, child_ids = [], [], []
    for index, fragment in enumerate(fragments):
        rejection, child = _fragment_rejection(record, fragment, index)
        rejected.append(rejection)
        child_ids.append(child)
    for emission in emissions:
        child = _child_id(record["record_id"], emission.measurement_label, emission.span_start, emission.span_end, emission.emission_index)
        child_ids.append(child)
        assignment, reason = assign_canonical_endpoint(source_id, record, emission)
        if assignment is None:
            rejected.append(_endpoint_rejection(record, emission, child, reason or "endpoint_assignment_failed"))
        else:
            accepted.append(_accepted_record(record, emission, child, assignment))
    ledger = _ledger_row(record, len(emissions), len(accepted), len(rejected), child_ids)
    return accepted, rejected, ledger


def _normalize_source(
    frame: pd.DataFrame, source_id: str, spec: Mapping[str, Any], shared: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    context = _row_context(source_id, spec, shared)
    records, rejections, ledger = [], [], []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
        candidate, reasons = normalize_row(row, context, row_number)
        candidate["source_name"] = spec["semantic_name"]
        structural = [reason for reason in reasons if reason in _STRUCTURAL_REASONS]
        if structural:
            rejections.append(_structural_rejection(candidate, structural))
            ledger.append(_ledger_row(candidate, 0, 0, 1, [], structural=True))
            continue
        accepted, rejected, parent = _process_candidate(candidate, source_id, spec)
        records.extend(accepted)
        rejections.extend(rejected)
        ledger.append(parent)
    return records, rejections, ledger


def _assert_invariants(
    records: list[dict[str, Any]], ledger: list[dict[str, Any]], spec: Mapping[str, Any], shared: Mapping[str, Any],
) -> None:
    if len(ledger) != spec["rows"] or len({row["parent_provenance_id"] for row in ledger}) != spec["rows"]:
        raise RuntimeError("parent ledger does not reconcile exactly")
    for record in records:
        if not math.isfinite(record["scalar_value"]) or not record["canonical_endpoint_key"]:
            raise RuntimeError("accepted base record lacks a finite scalar or endpoint key")
        if record["canonical_smiles"] in shared["tdc"] or record["authoritative_smiles"] in shared["tdc"]:
            raise RuntimeError("accepted base structure overlaps TDC")


def _write_source(
    source_id: str, spec: Mapping[str, Any], data_root: Path,
    output_root: Path, shared: Mapping[str, Any],
) -> dict[str, Any]:
    frame = read_source(spec, data_root)
    records, rejections, ledger = _normalize_source(frame, source_id, spec, shared)
    _assert_invariants(records, ledger, spec, shared)
    source_name = spec["semantic_name"]
    source_directory = output_root / source_name
    write_parquet(pd.DataFrame(records), source_directory / "records.parquet")
    write_parquet(pd.DataFrame(rejections), source_directory / "rejections.parquet")
    write_parquet(pd.DataFrame(ledger), source_directory / "parent_ledger.parquet")
    inventory = endpoint_inventory(records, source_id, source_name)
    write_parquet(inventory, source_directory / "endpoint_inventory.parquet")
    hashes = output_hashes(source_directory)
    manifest = build_manifest(
        source_id=source_id, spec=spec, records=records, rejections=rejections,
        ledger=ledger, shared=shared, hashes=hashes, source_directory=source_directory,
    )
    write_json(manifest, source_directory / "manifest.json")
    return manifest


def _postflight(config: Mapping[str, Any], data_root: Path) -> None:
    for pin in [*config["sources"].values(), *config["references"].values()]:
        path = resolve_path(data_root, pin["path"])
        if sha256_file(path) != pin["sha256"]:
            raise RuntimeError(f"input changed during base construction: {path}")


def _refresh_source_manifest(
    source_id: str, spec: Mapping[str, Any], output_root: Path, shared: Mapping[str, Any],
) -> dict[str, Any]:
    source_directory = output_root / spec["semantic_name"]
    required = ("records.parquet", "rejections.parquet", "parent_ledger.parquet", "endpoint_inventory.parquet")
    missing = [name for name in required if not (source_directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing base artifacts for {source_id}: {missing}")
    records = pq.read_table(source_directory / "records.parquet").to_pylist()
    rejections = pq.read_table(source_directory / "rejections.parquet").to_pylist()
    ledger = pq.read_table(source_directory / "parent_ledger.parquet").to_pylist()
    manifest = build_manifest(
        source_id=source_id, spec=spec, records=records, rejections=rejections,
        ledger=ledger, shared=shared, hashes=output_hashes(source_directory),
        source_directory=source_directory,
    )
    write_json(manifest, source_directory / "manifest.json")
    return manifest


def run_normalization(
    config_path: Path, data_root: Path, output_root: Path | None = None,
    report_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build final canonical base artifacts directly from pinned raw sources."""
    config = load_config(config_path)
    source_metadata, reference_metadata = _preflight(config, data_root)
    shared = _shared_context(config, config_path, data_root, source_metadata, reference_metadata)
    outputs = output_root or data_root / config["output_directory"]
    manifests = [_write_source(key, spec, data_root, outputs, shared) for key, spec in config["sources"].items()]
    _postflight(config, data_root)
    report = report_path or config_path.parents[1] / config["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(manifests), encoding="utf-8")
    return manifests


def refresh_manifests(
    config_path: Path, data_root: Path, output_root: Path | None = None,
    report_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Refresh base manifests after a byte-identical input relocation."""
    config = load_config(config_path)
    source_metadata, reference_metadata = _preflight(config, data_root)
    shared = _shared_context(config, config_path, data_root, source_metadata, reference_metadata)
    outputs = output_root or data_root / config["output_directory"]
    manifests = [
        _refresh_source_manifest(key, spec, outputs, shared)
        for key, spec in config["sources"].items()
    ]
    _postflight(config, data_root)
    report = report_path or config_path.parents[1] / config["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(manifests), encoding="utf-8")
    return manifests
