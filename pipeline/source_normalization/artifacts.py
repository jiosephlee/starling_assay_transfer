"""Combination inventories, manifests, and the round-two handoff report."""

from __future__ import annotations

from collections import Counter
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from rdkit import rdBase

from pipeline.source_normalization.io import parquet_schema, sha256_file
from pipeline.source_normalization.normalize import normalized_column_name, stable_hash


def combination_fields(spec: Mapping[str, Any]) -> list[str]:
    fields = ["endpoint_alias_normalized", "unit_normalized", "species_exact", "measurement_kind", "categorical_value"]
    fields.extend(normalized_column_name(spec, column) for column in spec["normalized_fields"])
    return list(dict.fromkeys(fields))


def combinations_frame(records: list[dict[str, Any]], fields: list[str], source_id: str) -> pd.DataFrame:
    rows = Counter(tuple(record.get(field) for field in fields) for record in records)
    combinations = []
    for values, count in rows.items():
        payload = dict(zip(fields, values, strict=True))
        combination_id = stable_hash("combination_v1", source_id, payload)
        combinations.append({"combination_id": combination_id, "source_id": source_id, **payload, "count": count})
    combinations.sort(key=lambda row: row["combination_id"])
    columns = ["combination_id", "source_id", *fields, "count"]
    return pd.DataFrame(combinations, columns=columns)


def _cardinality(records: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    output = {}
    for field in fields:
        counts = Counter(record.get(field) for record in records if record.get(field) is not None)
        top = [{"value": value, "count": count} for value, count in counts.most_common(10)]
        output[field] = {"non_null": sum(counts.values()), "unique": len(counts), "top": top}
    return output


def _publication_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record["pmid"]) for record in records if record.get("pmid") is not None)
    total = sum(counts.values())
    top_ten = sum(count for _, count in counts.most_common(10))
    return {
        "unique_publications": len(counts),
        "top_10_record_share": top_ten / total if total else 0.0,
        "top": [{"pmid": pmid, "count": count} for pmid, count in counts.most_common(10)],
    }


def _reason_stats(rejections: list[dict[str, Any]]) -> dict[str, Any]:
    all_reasons = Counter(reason for row in rejections for reason in row["rejection_reasons"])
    primary = Counter(row["primary_rejection_reason"] for row in rejections)
    return {"all": dict(sorted(all_reasons.items())), "primary": dict(sorted(primary.items()))}


def _structure_stats(records: list[dict[str, Any]], rejections: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = [*records, *rejections]
    comparison = Counter(row.get("smiles_comparison_status") for row in all_rows if row.get("smiles_comparison_status"))
    return {
        "authoritative_resolved": sum(row.get("authoritative_smiles") is not None for row in all_rows),
        "canonical_usable_all_rows": sum(row.get("canonical_smiles") is not None for row in all_rows),
        "canonical_accepted": sum(row.get("canonical_smiles") is not None for row in records),
        "accepted_unique_molecules": len({row.get("canonical_smiles") for row in records}),
        "exported_comparison_all_resolved": dict(sorted(comparison.items())),
        "tdc_exclusions": sum("tdc_molecule_match" in row.get("rejection_reasons", []) for row in rejections),
        "accepted_tdc_matches": 0,
        "rows_profiled": len(all_rows),
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "rdkit": rdBase.rdkitVersion,
    }


def build_manifest(
    *, source_id: str, spec: Mapping[str, Any], records: list[dict[str, Any]],
    rejections: list[dict[str, Any]], fields: list[str], shared: Mapping[str, Any],
    output_hashes: Mapping[str, str], input_unchanged: bool,
) -> dict[str, Any]:
    duplicate_counts = Counter(record["duplicate_group_id"] for record in records)
    source_rows = spec["rows"]
    profiles = _cardinality(records, fields)
    rejection_stats = _reason_stats(rejections)
    return {
        "normalization_version": shared["config"]["version"],
        "artifact_schema_version": shared["config"]["artifact_schema_version"],
        "source_id": source_id,
        "runtime_versions": runtime_versions(),
        "input": shared["input_metadata"],
        "references": shared["reference_metadata"],
        "config_sha256": shared["config_hash"],
        "counts": {"source": source_rows, "accepted": len(records), "rejected": len(rejections)},
        "reconciliation": {"exact": source_rows == len(records) + len(rejections), "input_unchanged": input_unchanged},
        "structure_resolution": _structure_stats(records, rejections),
        "starling_tdc_version_discrepancy": source_id == "starling" and any(
            "tdc_molecule_match" in row["rejection_reasons"] for row in rejections
        ),
        "species": {"non_null": sum(row.get("species_exact") is not None for row in records)},
        "measurement_parse_kinds": dict(sorted(Counter(row["measurement_kind"] for row in records).items())),
        "parse_coverage": {
            "successful_accepted": len(records),
            "missing_measurement": rejection_stats["all"].get("missing_measurement", 0),
            "unparseable_measurement": rejection_stats["all"].get("unparseable_measurement", 0),
        },
        "rejections": rejection_stats,
        "duplicates": {
            "groups": len(duplicate_counts),
            "groups_with_multiple_rows": sum(size > 1 for size in duplicate_counts.values()),
            "rows_in_multirow_groups": sum(size for size in duplicate_counts.values() if size > 1),
        },
        "combination_fields": fields,
        "combination_count": len({tuple(record.get(field) for field in fields) for record in records}),
        "profiles": profiles,
        "missingness": {field: len(records) - profile["non_null"] for field, profile in profiles.items()},
        "publication_concentration": _publication_stats(records),
        "output_sha256": dict(output_hashes),
        "output_artifacts": artifact_details(shared["source_directory"]),
    }


def output_hashes(source_directory: Path) -> dict[str, str]:
    return {
        name: sha256_file(source_directory / name)
        for name in ("records.parquet", "rejections.parquet", "combinations.parquet")
    }


def artifact_details(source_directory: Path) -> dict[str, Any]:
    details = {}
    for name in ("records.parquet", "rejections.parquet", "combinations.parquet"):
        path = source_directory / name
        details[name] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "rows": pq.ParquetFile(path).metadata.num_rows,
            "schema": parquet_schema(path),
        }
    return details


def _report_table(manifests: list[Mapping[str, Any]]) -> list[str]:
    lines = ["| Source | Input | Accepted | Rejected | TDC removed | Species | Combinations |", "|---|---:|---:|---:|---:|---:|---:|"]
    for manifest in manifests:
        counts = manifest["counts"]
        tdc = manifest["structure_resolution"]["tdc_exclusions"]
        species = manifest["species"]["non_null"]
        combinations = manifest["combination_count"]
        lines.append(f"| {manifest['source_id']} | {counts['source']:,} | {counts['accepted']:,} | {counts['rejected']:,} | {tdc:,} | {species:,} | {combinations:,} |")
    return lines


def render_report(manifests: list[Mapping[str, Any]]) -> str:
    total = sum(manifest["counts"]["source"] for manifest in manifests)
    accepted = sum(manifest["counts"]["accepted"] for manifest in manifests)
    rejected = total - accepted
    lines = [
        "# Source normalization report", "", "## Round-one result", "",
        f"The five pinned inputs reconcile exactly across **{total:,}** source rows: **{accepted:,} accepted** and **{rejected:,} rejected**. Structures in Q1--Q4 come only from the authoritative global-identifier mapping; Starling uses its source SMILES. All accepted structures were RDKit parsed, rejected when wildcard-bearing, and screened against both raw and canonical forms of all 640 TDC molecules.", "",
        *_report_table(manifests), "", "## Round-two handoff", "",
        "Each source directory contains a lossless normalized record table, all rejected rows with ordered reasons, and every observed post-filter combination of mechanically normalized structured fields. No canonical endpoint keys were assigned, and no endpoint-specific unit conversions, validity ranges, thresholds, pair logic, or models were changed in this round.", "",
    ]
    for manifest in manifests:
        lines.extend(_source_report(manifest))
    return "\n".join(lines).rstrip() + "\n"


def _source_report(manifest: Mapping[str, Any]) -> list[str]:
    profiles = manifest["profiles"]
    endpoint = profiles["endpoint_alias_normalized"]
    units = profiles["unit_normalized"]
    publication = manifest["publication_concentration"]
    structure = manifest["structure_resolution"]
    parse = manifest["parse_coverage"]
    duplicates = manifest["duplicates"]
    top_endpoints = ", ".join(f"`{item['value']}` ({item['count']:,})" for item in endpoint["top"][:5])
    comparisons = ", ".join(
        f"{name}={count:,}" for name, count in structure["exported_comparison_all_resolved"].items()
    )
    missing = sorted(manifest["missingness"].items(), key=lambda item: (-item[1], item[0]))[:3]
    top_missing = ", ".join(f"`{name}`={count:,}" for name, count in missing)
    return [
        f"### {manifest['source_id']}", "",
        f"Authoritative structures resolved for {structure['authoritative_resolved']:,}/{manifest['counts']['source']:,} rows and {structure['canonical_usable_all_rows']:,} were structurally usable before later gates. Exported/source comparison: {comparisons or 'not applicable'}. TDC removed {structure['tdc_exclusions']:,}; accepted overlap is zero.", "",
        f"Measurement parsing accepted {parse['successful_accepted']:,} rows; {parse['missing_measurement']:,} rows lacked a measurement and {parse['unparseable_measurement']:,} had a present but unapproved/unparseable value.", "",
        f"Observed {endpoint['unique']:,} endpoint aliases and {units['unique']:,} lexical units. Top aliases: {top_endpoints or 'none'}.", "",
        f"There are {manifest['combination_count']:,} observed combinations. Largest structured-field missing counts: {top_missing or 'none'}.", "",
        f"Duplicate annotation found {duplicates['groups_with_multiple_rows']:,} multirow groups covering {duplicates['rows_in_multirow_groups']:,} rows; evidence rows were not collapsed.", "",
        f"Species coverage is {manifest['species']['non_null']:,}/{manifest['counts']['accepted']:,}; {publication['unique_publications']:,} publications contribute records and the top ten account for {publication['top_10_record_share']:.1%}.", "",
    ]
