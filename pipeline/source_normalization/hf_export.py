"""Deterministic Hugging Face exports for cleaned canonical scalar bases."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.source_normalization.io import sha256_file, write_json
from pipeline.source_normalization.normalize import normalized_column_name

EXPORT_VERSION = "hf_cleaned_v2"
INTERNAL_Q_IDENTIFIERS = {"extraction_id", "global_identifier"}
IDENTITY_COLUMNS = (
    "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
    "direction", "target", "kinetic_parameter", "auc_window", "defining_timepoint",
)
COMMON_COLUMNS = (
    "source_id", "source_name", "record_id", "source_row_number", "input_sha256",
    "parent_provenance_id", "child_id", "scalar_value", "scalar_is_approximate",
    "variation_value", "variation_type", "accompanying_interval_lower",
    "accompanying_interval_upper", "unit_normalized", "species_exact", *IDENTITY_COLUMNS,
)
FLOAT_COLUMNS = {
    "scalar_value", "variation_value", "accompanying_interval_lower",
    "accompanying_interval_upper",
}
INTEGER_COLUMNS = {"source_row_number"}
BOOLEAN_COLUMNS = {"scalar_is_approximate"}
SOURCE_AUDIT_COLUMNS = ("pmid", "confidence")
NARRATIVE_COLUMNS = ("extra_details", "support_text", "paragraph_idx")
SCIENTIFIC_COMMON_COLUMNS = (
    "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
    "direction", "target", "kinetic_parameter", "auc_window", "defining_timepoint",
    "species_exact", "scalar_value", "unit_normalized", "scalar_is_approximate",
    "variation_value", "variation_type", "accompanying_interval_lower",
    "accompanying_interval_upper",
)


def source_columns(spec: Mapping[str, Any]) -> list[str]:
    """Return the declared source columns in their original order."""
    schema = spec.get("expected_columns") or spec.get("schema")
    return list(schema)


def replaced_columns(spec: Mapping[str, Any]) -> dict[str, str]:
    """Map public source fields to their cleaned canonical counterparts."""
    replacements = {"smiles": "canonical_smiles"}
    for column in spec["normalized_fields"]:
        replacements[column] = normalized_column_name(spec, column)
    if spec.get("endpoint_column"):
        replacements[spec["endpoint_column"]] = "canonical_endpoint_key"
    replacements[spec["measurement_column"]] = "scalar_value"
    if spec.get("unit_column"):
        replacements[spec["unit_column"]] = "unit_normalized"
    return replacements


def public_columns(spec: Mapping[str, Any], source_id: str | None = None) -> list[str]:
    """Return a scientific-first public schema with narrative fields at the end."""
    columns = source_columns(spec)
    if source_id in {"q1", "q2", "q3", "q4"}:
        columns = [column for column in columns if column not in INTERNAL_Q_IDENTIFIERS]
    primary = ["smiles", spec.get("endpoint_column"), spec["measurement_column"]]
    if spec.get("unit_column"):
        primary.append(spec["unit_column"])
    reserved = set(primary) | set(SOURCE_AUDIT_COLUMNS) | set(NARRATIVE_COLUMNS)
    metadata = [column for column in columns if column not in reserved]
    provenance = [*SOURCE_AUDIT_COLUMNS, *COMMON_COLUMNS]
    ordered = [*primary, *SCIENTIFIC_COMMON_COLUMNS, *metadata, *provenance, *NARRATIVE_COLUMNS]
    return [name for index, name in enumerate(ordered) if name and name in columns + list(COMMON_COLUMNS)
            and name not in ordered[:index]]


def _column_type(name: str, spec: Mapping[str, Any]) -> pa.DataType:
    if name == spec["measurement_column"]:
        return pa.float64()
    if name in FLOAT_COLUMNS:
        return pa.float64()
    if name in INTEGER_COLUMNS:
        return pa.int64()
    if name in BOOLEAN_COLUMNS:
        return pa.bool_()
    return pa.large_string()


def _drop_all_null(table: pa.Table) -> pa.Table:
    names = [name for name in table.column_names if table[name].null_count < table.num_rows]
    return table.select(names)


def clean_table(
    table: pa.Table, spec: Mapping[str, Any], source_id: str | None = None,
) -> pa.Table:
    """Select public fields and substitute each validated cleaned source value."""
    replacements = replaced_columns(spec)
    missing = set(replacements.values()) - set(table.column_names)
    if missing:
        raise KeyError(f"missing canonical replacement columns: {sorted(missing)}")
    arrays = {}
    for output in public_columns(spec, source_id):
        input_name = replacements.get(output, output)
        if input_name not in table.column_names:
            raise KeyError(f"missing export column {input_name!r} for {output!r}")
        arrays[output] = pa.array(table[input_name].to_pylist(), type=_column_type(output, spec))
    return _drop_all_null(pa.table(arrays))


def omitted_columns(
    table: pa.Table, spec: Mapping[str, Any], source_id: str,
) -> list[dict[str, str]]:
    """List policy and dataset-specific null omissions in stable source order."""
    omitted = []
    if source_id in {"q1", "q2", "q3", "q4"}:
        omitted.extend(
            {"column": name, "reason": "internal_identifier"}
            for name in source_columns(spec) if name in INTERNAL_Q_IDENTIFIERS
        )
    allowed = public_columns(spec, source_id)
    omitted.extend(
        {"column": name, "reason": "all_null_in_dataset"}
        for name in allowed if name not in table.column_names
    )
    return omitted


def _validate_replacements(
    clean: pa.Table, records: pa.Table, spec: Mapping[str, Any], source_id: str,
) -> None:
    for output, source in replaced_columns(spec).items():
        if output not in public_columns(spec, source_id):
            continue
        expected = pa.array(records[source].to_pylist(), type=_column_type(output, spec))
        if output not in clean.column_names:
            if expected.null_count != len(expected):
                raise ValueError(f"non-null replacement column was omitted: {output}")
        elif not clean[output].combine_chunks().equals(expected):
            raise ValueError(f"cleaned replacement mismatch: {output} <- {source}")


def _validate_ledger(clean: pa.Table, base_directory: Path) -> None:
    ledger = pq.read_table(base_directory / "parent_ledger.parquet")
    accepted = sum(value or 0 for value in ledger["accepted_child_count"].to_pylist())
    if accepted != clean.num_rows:
        raise ValueError(f"accepted-child ledger count {accepted} != export rows {clean.num_rows}")
    actual: dict[str, int] = {}
    for parent in clean["parent_provenance_id"].to_pylist():
        actual[parent] = actual.get(parent, 0) + 1
    expected = {
        parent: count for parent, count in zip(
            ledger["parent_provenance_id"].to_pylist(),
            ledger["accepted_child_count"].to_pylist(), strict=True,
        ) if count
    }
    if actual != expected:
        raise ValueError("accepted-child ledger parent counts do not reconcile with export")


def validate_clean_export(
    clean: pa.Table, records: pa.Table, spec: Mapping[str, Any], source_id: str,
    base_directory: Path,
) -> None:
    """Assert the compact export preserves all scientific and audit invariants."""
    if clean.num_rows != records.num_rows:
        raise ValueError("export row count differs from canonical base")
    if source_id in {"q1", "q2", "q3", "q4"}:
        leaked = INTERNAL_Q_IDENTIFIERS.intersection(clean.column_names)
        if leaked:
            raise ValueError(f"internal identifiers leaked: {sorted(leaked)}")
    null_columns = [name for name in clean.column_names if clean[name].null_count == clean.num_rows]
    if null_columns:
        raise ValueError(f"all-null columns remain: {null_columns}")
    if not all(math.isfinite(value) for value in clean[spec["measurement_column"]].to_pylist()):
        raise ValueError("non-finite scalar measurement in export")
    for name in ("smiles", "canonical_endpoint_key"):
        if any(not str(value).strip() for value in clean[name].to_pylist()):
            raise ValueError(f"empty {name} in export")
    _validate_replacements(clean, records, spec, source_id)
    _validate_ledger(clean, base_directory)


def _source_manifest(base_directory: Path) -> dict[str, Any]:
    return json.loads((base_directory / "manifest.json").read_text(encoding="utf-8"))


def export_manifest(
    clean: pa.Table, omissions: list[dict[str, str]], source_id: str,
    source_name: str, base_directory: Path, output: Path,
) -> dict[str, Any]:
    """Describe reproducible local export inputs and outputs."""
    manifest = _source_manifest(base_directory)
    data = output / "data" / "train-00000-of-00001.parquet"
    return {
        "export_version": EXPORT_VERSION,
        "source_id": source_id,
        "source_name": source_name,
        "rows": clean.num_rows,
        "columns": {field.name: str(field.type) for field in clean.schema},
        "omitted_columns": omissions,
        "condition_key_included": False,
        "input_records": {
            "path": str(base_directory / "records.parquet"),
            "sha256": sha256_file(base_directory / "records.parquet"),
            "manifest_sha256": sha256_file(base_directory / "manifest.json"),
        },
        "canonical_base_yields": manifest["stage_yields"],
        "artifacts": {"data/train-00000-of-00001.parquet": sha256_file(data)},
    }


def readme_text(
    source_name: str, rows: int, replacements: Mapping[str, str],
    omissions: list[dict[str, str]],
) -> str:
    """Render a compact deterministic data card for one cleaned source."""
    fields = "\n".join(f"- `{old}` → `{new}`" for old, new in replacements.items())
    omitted = "\n".join(
        f"- `{item['column']}`: `{item['reason']}`" for item in omissions
    ) or "- None"
    return f"""---
pretty_name: {source_name.replace('_', ' ').title()} Cleaned
tags:
- chemistry
- pharmacokinetics
- tabular
---

# {source_name.replace('_', ' ').title()} Cleaned

This dataset contains {rows:,} accepted finite scalar measurement children from the
versioned canonical-base pipeline. A parent row with several unambiguous measurements
can produce several child rows. Rejected, malformed, ambiguous, bounded, range-only,
TDC-overlapping, and endpoint-unassignable records are not published here.

## Cleaning

The following source columns are replaced in place with validated canonical or lexical
normalizations:

{fields}

The public column names on the left are retained; the names on the right identify the
validated internal values used to replace them. Columns omitted from this dataset:

{omitted}

`canonical_endpoint_key` is the endpoint identity field. No `condition_key` is added.
`species_exact` is populated only for explicit, unambiguous species references. Narrative
and other source fields without a validated normalization remain source text.
"""


def write_clean_export(
    source_id: str, spec: Mapping[str, Any], base_root: Path, output_root: Path,
) -> dict[str, Any]:
    """Stage one source as a local HF dataset repository."""
    source_name = spec["semantic_name"]
    base_directory = base_root / source_name
    output = output_root / f"{source_name}_cleaned"
    if output.exists():
        shutil.rmtree(output)
    records = pq.read_table(base_directory / "records.parquet")
    clean = clean_table(records, spec, source_id)
    validate_clean_export(clean, records, spec, source_id, base_directory)
    omissions = omitted_columns(clean, spec, source_id)
    output_data = output / "data" / "train-00000-of-00001.parquet"
    output_data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(clean, output_data, compression="zstd", use_dictionary=True)
    readme = readme_text(source_name, clean.num_rows, replaced_columns(spec), omissions)
    (output / "README.md").write_text(readme, encoding="utf-8")
    manifest = export_manifest(
        clean, omissions, source_id, source_name, base_directory, output,
    )
    manifest["artifacts"]["README.md"] = sha256_file(output / "README.md")
    write_json(manifest, output / "metadata" / "export_manifest.json")
    return {"output": output, **manifest}


def write_all_clean_exports(
    config: Mapping[str, Any], base_root: Path, output_root: Path,
) -> list[dict[str, Any]]:
    """Stage every configured canonical source in semantic-name order."""
    return [
        write_clean_export(source_id, spec, base_root, output_root)
        for source_id, spec in config["sources"].items()
    ]


def upload_export(output: Path, repo_id: str) -> str:
    """Create or update one public dataset repository from a staged export."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    staged = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    remote = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    stale = [path for path in remote if path not in staged and path != ".gitattributes"]
    commit = api.upload_folder(
        folder_path=str(output), repo_id=repo_id, repo_type="dataset", delete_patterns=stale or None,
    )
    return str(commit.oid)
