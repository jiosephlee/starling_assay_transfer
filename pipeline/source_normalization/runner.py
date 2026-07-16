"""Top-level five-source normalization orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pipeline.source_normalization.artifacts import (
    build_manifest,
    combination_fields,
    combinations_frame,
    output_hashes,
    render_report,
)
from pipeline.source_normalization.io import (
    artifact_metadata,
    load_config,
    load_mapping,
    load_tdc_exclusions,
    read_source,
    required_identifiers,
    resolve_path,
    sha256_file,
    verify_artifact,
    write_json,
    write_parquet,
)
from pipeline.source_normalization.normalize import annotate_duplicates, normalize_row, rejection_record


def _preflight(config: Mapping[str, Any], data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_metadata = {
        source_id: verify_artifact(spec, data_root)
        for source_id, spec in config["sources"].items()
    }
    reference_metadata = {
        name: verify_artifact(pin, data_root)
        for name, pin in config["references"].items()
    }
    return source_metadata, reference_metadata


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
        "config": config,
        "config_hash": sha256_file(config_path),
        "source_metadata": source_metadata,
        "reference_metadata": references,
        "mapping": mapping,
        "tdc": tdc,
    }


def _normalize_records(
    frame: pd.DataFrame, source_id: str, spec: Mapping[str, Any], shared: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = {
        "source_id": source_id,
        "spec": spec,
        "mapping": shared["mapping"],
        "tdc": shared["tdc"],
        "allowlists": shared["config"]["categorical_allowlists"],
        "input_hash": spec["sha256"],
    }
    records, rejections = [], []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
        record, reasons = normalize_row(row, context, row_number)
        if reasons:
            rejections.append(rejection_record(record, reasons))
        else:
            records.append(record)
    return records, rejections


def _assert_accepted_invariants(
    records: list[dict[str, Any]], spec: Mapping[str, Any], shared: Mapping[str, Any]
) -> None:
    for record in records:
        if record["canonical_smiles"] is None:
            raise RuntimeError("accepted record lacks a canonical structure")
        if record["canonical_smiles"] in shared["tdc"] or record["authoritative_smiles"] in shared["tdc"]:
            raise RuntimeError("accepted record overlaps the TDC exclusion set")
        if spec["structure_mode"] == "mapped":
            identifier = record["global_identifier"]
            if record["authoritative_smiles"] != shared["mapping"].get(identifier):
                raise RuntimeError("Q1--Q4 record did not use the authoritative mapping")


def _write_source(
    source_id: str, spec: Mapping[str, Any], data_root: Path,
    output_root: Path, shared: Mapping[str, Any],
) -> dict[str, Any]:
    frame = read_source(spec, data_root)
    records, rejections = _normalize_records(frame, source_id, spec, shared)
    if len(records) + len(rejections) != spec["rows"]:
        raise RuntimeError(f"failed row reconciliation for {source_id}")
    _assert_accepted_invariants(records, spec, shared)
    fields = combination_fields(spec)
    annotate_duplicates(records, fields)
    source_directory = output_root / source_id
    write_parquet(pd.DataFrame(records), source_directory / "records.parquet")
    write_parquet(pd.DataFrame(rejections), source_directory / "rejections.parquet")
    write_parquet(combinations_frame(records, fields, source_id), source_directory / "combinations.parquet")
    hashes = output_hashes(source_directory)
    input_path = resolve_path(data_root, spec["path"])
    unchanged = artifact_metadata(input_path)["sha256"] == spec["sha256"]
    source_shared = {
        **shared,
        "input_metadata": shared["source_metadata"][source_id],
        "source_directory": source_directory,
    }
    manifest = build_manifest(
        source_id=source_id, spec=spec, records=records, rejections=rejections,
        fields=fields, shared=source_shared, output_hashes=hashes, input_unchanged=unchanged,
    )
    write_json(manifest, source_directory / "manifest.json")
    return manifest


def _postflight(config: Mapping[str, Any], data_root: Path) -> None:
    for pin in [*config["sources"].values(), *config["references"].values()]:
        path = resolve_path(data_root, pin["path"])
        if sha256_file(path) != pin["sha256"]:
            raise RuntimeError(f"input changed during normalization: {path}")


def run_normalization(
    config_path: Path, data_root: Path, output_root: Path | None = None,
    report_path: Path | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    source_metadata, reference_metadata = _preflight(config, data_root)
    shared = _shared_context(config, config_path, data_root, source_metadata, reference_metadata)
    outputs = output_root or data_root / config["output_directory"]
    manifests = [
        _write_source(source_id, spec, data_root, outputs, shared)
        for source_id, spec in config["sources"].items()
    ]
    _postflight(config, data_root)
    report = report_path or config_path.parents[1] / config["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(manifests), encoding="utf-8")
    return manifests
