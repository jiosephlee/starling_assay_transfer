"""Adapt any complete TxAgent Starling v7 task into v11 eligible records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.v11_contract import (
    TXAGENT_ROOT,
    artifact_paths,
    artifact_status,
    eligible_projection_manifest,
    heldout_paths,
    load_registry,
    source_backed_fields,
    source_config,
    source_constants,
    task_artifact_root,
    task_config,
    validate_registry,
)
from pipeline.v11_targets import geometry_value, record_percentile, value_calibration
from pipeline.v3_policy import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
RESERVATION_CACHE_DIR = REPO_ROOT / "datasets/eligible/_reserved_molecules_cache"
# key -> reserved molecules, so repeated calls inside one process skip even the JSON read
_RESERVATION_MEMO: dict[str, set[str]] = {}
FIXED_ELIGIBLE_FIELDS = (
    pa.field("child_id", pa.string()),
    pa.field("parent_provenance_id", pa.string()),
    pa.field("record_id", pa.string()),
    pa.field("task_id", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("input_sha256", pa.string()),
    pa.field("source_smiles", pa.string()),
    pa.field("canonical_smiles", pa.string()),
    pa.field("canonical_endpoint_key", pa.string()),
    pa.field("pair_bucket_key", pa.string()),
    pa.field("assay_concept", pa.string()),
    pa.field("metric_type", pa.string()),
    pa.field("measurement_kind", pa.string()),
    pa.field("geometry_value", pa.float64()),
    pa.field("finite_scalar_value", pa.float64()),
    pa.field("canonical_category_id", pa.string()),
    pa.field("canonical_category_rank", pa.float64()),
    pa.field("canonical_measurement_scale_id", pa.string()),
    pa.field("canonical_unit_text", pa.string()),
    pa.field("raw_measurement_text", pa.string()),
    pa.field("raw_unit_text", pa.string()),
    pa.field("value_percentile", pa.float64()),
    pa.field("calibration_sample_standard_deviation", pa.float64()),
    pa.field("calibration_standard_deviation_ddof", pa.int64()),
    pa.field("calibration_standard_deviation_value_field", pa.string()),
    pa.field("unit_basis", pa.string()),
)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_calibrations(path: Path) -> tuple[dict[str, Any], dict[str, Any], Counter]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    accepted: dict[str, Any] = {}
    rejected: Counter = Counter()
    for key, entry in document["buckets"].items():
        if not entry.get("calibration_valid"):
            rejected[str(entry.get("calibration_reason") or "invalid")] += 1
            continue
        try:
            accepted[str(key)] = value_calibration(str(key), entry)
        except ValueError as exc:
            reason = "invalid_value_cdf_or_geometry"
            rejected[reason] += 1
    return document, accepted, rejected


def _load_joined(paths: Mapping[str, Path]) -> pd.DataFrame:
    records = pd.read_parquet(paths["records"])
    buckets = pd.read_parquet(paths["pair_buckets"])
    if not records["canonical_record_id"].is_unique:
        raise ValueError("v7 canonical_record_id values are not unique")
    if not buckets["canonical_record_id"].is_unique or len(records) != len(buckets):
        raise ValueError("v7 record/pair-bucket coverage mismatch")
    bucket_columns = [
        "canonical_record_id", "pair_bucket_key", "bucket_eligible",
        "bucket_exclusion_reason",
    ]
    return records.merge(
        buckets[bucket_columns], on="canonical_record_id", how="left", validate="one_to_one"
    )


def _present(value: Any) -> bool:
    if value is None or value == "":
        return False
    return not isinstance(value, float) or not math.isnan(value)


def _nullable(value: Any) -> Any:
    return value if _present(value) else None


def _prompt_source_values(
    row: Mapping[str, Any], task_id: str, registry: Mapping[str, Any]
) -> dict[str, Any]:
    config = source_config(task_id, str(row["source_id"]), registry)
    names = source_backed_fields(config)
    return {name: row.get(name) if _present(row.get(name)) else None for name in names}


def _validate_constants(
    row: Mapping[str, Any], task_id: str, registry: Mapping[str, Any]
) -> None:
    config = source_config(task_id, str(row["source_id"]), registry)
    for field, expected in source_constants(config).items():
        actual = row.get(field)
        matches = not _present(actual) if expected is None else actual == expected
        if not matches:
            raise ValueError(
                f"{task_id}/{row['source_id']} schema constant drift: "
                f"{field}={actual!r}, expected {expected!r}"
            )


def prompt_field_union(task_id: str, registry: Mapping[str, Any]) -> list[str]:
    """Every prompt field any source of this task can bind, in a stable order.

    The eligible artifact is one table over all sources, so each row must carry the whole union --
    a row that omits another source's fields would otherwise disappear from the written schema.
    Constants are excluded: they live in the registry and are injected at render time.
    """
    names: list[str] = []
    for source_id in sorted(task_config(task_id, registry)["sources"]):
        config = source_config(task_id, source_id, registry)
        for field in source_backed_fields(config):
            if field not in names:
                names.append(field)
    return sorted(names)


def _eligible_row(
    row: Mapping[str, Any], task_id: str, calibration: Any,
    registry: Mapping[str, Any], prompt_fields: list[str],
) -> dict[str, Any]:
    _validate_constants(row, task_id, registry)
    record_id = str(row["canonical_record_id"])
    value = geometry_value(row)
    percentile = record_percentile(row, calibration)
    endpoint = str(row.get("canonical_endpoint_name") or "")
    if not endpoint:
        raise ValueError(f"{task_id}/{record_id} lacks canonical_endpoint_name")
    output = {
        "child_id": f"{task_id}:{record_id}",
        "parent_provenance_id": str(row.get("duplicate_group_id") or record_id),
        "record_id": str(row.get("source_record_id") or record_id),
        "task_id": task_id,
        "source_id": str(row["source_id"]),
        "input_sha256": _stable_hash(f"{task_id}:{record_id}"),
        "source_smiles": str(row["smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "canonical_endpoint_key": endpoint,
        "pair_bucket_key": str(row["pair_bucket_key"]),
        "assay_concept": str(row["source_id"]),
        "metric_type": str(row["source_id"]),
        "measurement_kind": str(row["measurement_kind"]),
        "geometry_value": value,
        "finite_scalar_value": _nullable(row.get("finite_scalar_value")),
        "canonical_category_id": _nullable(row.get("canonical_category_id")),
        "canonical_category_rank": _nullable(row.get("canonical_category_rank")),
        "canonical_measurement_scale_id": _nullable(row.get("canonical_measurement_scale_id")),
        "canonical_unit_text": _nullable(row.get("canonical_unit_text")),
        "raw_measurement_text": _nullable(row.get("measurement_text")),
        "raw_unit_text": _nullable(row.get("unit_text")),
        "value_percentile": percentile,
        "calibration_sample_standard_deviation": calibration.sample_standard_deviation,
        "calibration_standard_deviation_ddof": calibration.standard_deviation_ddof,
        "calibration_standard_deviation_value_field": calibration.standard_deviation_value_field,
        "unit_basis": str(_nullable(row.get("canonical_unit_text")) or ""),
    }
    # Fields this source does not declare stay null: the projection is per-source, so borrowing a
    # value another source happens to carry would leak context the contract never bound.
    output.update({field: None for field in prompt_fields})
    output.update(_prompt_source_values(row, task_id, registry))
    return output


def _rejection_reason(row: Mapping[str, Any], calibrations: Mapping[str, Any]) -> str | None:
    if not bool(row.get("bucket_eligible")):
        return str(row.get("bucket_exclusion_reason") or "bucket_ineligible")
    bucket = str(row.get("pair_bucket_key") or "")
    if bucket not in calibrations:
        return "calibration_unavailable"
    if str(row.get("canonicalization_status") or "") != "valid":
        return "canonicalization_invalid"
    if not _present(row.get("smiles")):
        return "missing_source_smiles"
    if not _present(row.get("canonical_smiles")):
        return "missing_canonical_smiles"
    return None


def build_eligible_records(
    task_id: str, *, artifact_root: Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = dict(registry or load_registry())
    validate_registry(contract, tasks=(task_id,))
    root = artifact_root or task_artifact_root(task_id, contract)
    status = artifact_status(root)
    if status["status"] != "ready":
        raise FileNotFoundError(f"incomplete v7 artifact for {task_id}: {status['missing']}")
    paths = artifact_paths(root)
    calibration_doc, calibrations, rejected_buckets = _load_calibrations(paths["calibration"])
    joined = _load_joined(paths)
    prompt_fields = prompt_field_union(task_id, contract)
    rows: list[dict[str, Any]] = []
    rejected_records: Counter = Counter()
    for row in joined.to_dict("records"):
        reason = _rejection_reason(row, calibrations)
        if reason:
            rejected_records[reason] += 1
            continue
        calibration = calibrations[str(row["pair_bucket_key"])]
        rows.append(_eligible_row(row, task_id, calibration, contract, prompt_fields))
    stats = _build_stats(joined, rows, rejected_records, rejected_buckets)
    stats["prompt_field_coverage"] = _prompt_field_coverage(rows, task_id, contract)
    manifest = _manifest(task_id, root, paths, calibration_doc, stats, contract)
    return rows, manifest


def _coverage_counts(
    rows: list[dict[str, Any]], task_id: str, registry: Mapping[str, Any],
) -> tuple[Counter, dict[str, Counter]]:
    row_counts: Counter = Counter()
    counts: dict[str, Counter] = {}
    for row in rows:
        source_id = str(row["source_id"])
        row_counts[source_id] += 1
        source_counts = counts.setdefault(source_id, Counter())
        config = source_config(task_id, source_id, registry)
        for field in source_backed_fields(config):
            source_counts[field] += int(_present(row.get(field)))
    return row_counts, counts


def _validate_source_coverage(
    task_id: str, source_id: str, total: int, fields: Mapping[str, int],
    config: Mapping[str, Any],
) -> None:
    if total == 0:
        return
    expected_null = set(config.get("expected_all_null_fields", ()))
    unexpectedly_present = [name for name in expected_null if fields[name] != 0]
    unexpectedly_empty = [
        name for name in source_backed_fields(config)
        if name not in expected_null and fields[name] == 0
    ]
    if unexpectedly_present:
        raise ValueError(
            f"{task_id}/{source_id} expected-all-null fields became populated: "
            f"{sorted(unexpectedly_present)}"
        )
    if unexpectedly_empty:
        raise ValueError(
            f"{task_id}/{source_id} prompt fields have zero coverage: "
            f"{sorted(unexpectedly_empty)}"
        )


def _prompt_field_coverage(
    rows: list[dict[str, Any]], task_id: str, registry: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    row_counts, counts = _coverage_counts(rows, task_id, registry)
    coverage: dict[str, dict[str, Any]] = {}
    for source_id in sorted(task_config(task_id, registry)["sources"]):
        config = source_config(task_id, source_id, registry)
        fields = {name: counts.get(source_id, Counter())[name]
                  for name in source_backed_fields(config)}
        total = row_counts[source_id]
        _validate_source_coverage(task_id, source_id, total, fields, config)
        coverage[source_id] = {
            "eligible_records": total,
            "status": "validated" if total else "no_eligible_records",
            "fields": dict(sorted(fields.items())),
            "expected_all_null_fields": sorted(config.get("expected_all_null_fields", ())),
        }
    return coverage


def _build_stats(
    joined: pd.DataFrame, rows: list[dict[str, Any]], rejected_records: Counter,
    rejected_buckets: Counter,
) -> dict[str, Any]:
    return {
        "input_records": len(joined),
        "eligible_records": len(rows),
        "eligible_by_source": dict(sorted(Counter(row["source_id"] for row in rows).items())),
        "eligible_by_measurement_kind": dict(
            sorted(Counter(row["measurement_kind"] for row in rows).items())
        ),
        "record_rejection_counts": dict(sorted(rejected_records.items())),
        "bucket_rejection_counts": dict(sorted(rejected_buckets.items())),
    }


def _manifest(
    task_id: str, root: Path, paths: Mapping[str, Path],
    calibration: Mapping[str, Any], stats: Mapping[str, Any], registry: Mapping[str, Any],
) -> dict[str, Any]:
    hashed = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in paths.items()
    }
    return {
        "stage": "starling_txagent_eligible_v7",
        "task_id": task_id,
        "artifact_root": str(root),
        "record_contract_version": calibration["record_contract_version"],
        "calibration_version": calibration["calibration_version"],
        "eligible_projection": eligible_projection_manifest(task_id, registry),
        "target_geometry": "empirical_value_cdf_separation_from_stage05",
        "inputs": hashed,
        "stats": dict(stats),
    }


def eligible_schema(
    task_id: str, registry: Mapping[str, Any], source_schema: pa.Schema,
) -> pa.Schema:
    fields = list(FIXED_ELIGIBLE_FIELDS)
    missing = [name for name in prompt_field_union(task_id, registry)
               if name not in source_schema.names]
    if missing:
        raise ValueError(f"{task_id} source parquet lacks prompt fields: {sorted(missing)}")
    for name in prompt_field_union(task_id, registry):
        fields.append(pa.field(name, source_schema.field(name).type))
    return pa.schema(fields)


def eligible_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    """Build the eligible table under the explicit task-wide schema."""
    table = pa.Table.from_pylist(rows, schema=schema)
    if not table.schema.equals(schema, check_metadata=False):
        raise ValueError("eligible table schema differs from its explicit contract")
    return table


def _write_eligible_table(table: pa.Table, output: Path) -> None:
    temporary = output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    written = pq.read_schema(temporary)
    if not written.equals(table.schema, check_metadata=False):
        temporary.unlink(missing_ok=True)
        raise ValueError("written eligible parquet schema differs from explicit contract")
    temporary.replace(output)


def write_eligible_records(
    task_id: str, output_dir: Path, *, artifact_root: Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = dict(registry or load_registry())
    rows, manifest = build_eligible_records(
        task_id, artifact_root=artifact_root, registry=contract
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    root = artifact_root or task_artifact_root(task_id, contract)
    source_schema = pq.read_schema(artifact_paths(root)["records"])
    schema = eligible_schema(task_id, contract, source_schema)
    table = eligible_table(rows, schema)
    _write_eligible_table(table, output_dir / "records.parquet")
    manifest["eligible_records_sha256"] = file_sha256(output_dir / "records.parquet")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _identity_backend_version() -> str:
    """RDKit version, because molecule identity -- and so the reservation -- depends on it."""
    try:
        import rdkit

        return str(rdkit.__version__)
    except Exception:  # pragma: no cover - identity backend is resolved lazily
        return "unknown"


def _reservation_cache_key(task_id: str, molecules: list[str], paths: list[Path]) -> str:
    """Content-address the reservation by its inputs: molecules, heldout files, identity backend.

    Keying on content rather than on the eligible manifest keeps the cache correct for callers
    holding rows in memory (verify) as well as those holding a parquet path (the builders).
    """
    digest = hashlib.sha256()
    digest.update(f"v11-reserved:{task_id}:{len(molecules)}\n".encode("utf-8"))
    digest.update(f"rdkit:{_identity_backend_version()}\n".encode("utf-8"))
    for smiles in molecules:
        digest.update(f"{smiles}\n".encode("utf-8"))
    for path in paths:
        digest.update(f"{path.name}:{file_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _load_cached_reservation(path: Path, key: str) -> set[str] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("cache_key") != key:
        return None
    return {str(value) for value in document.get("reserved_molecules", ())}


def _store_reservation(path: Path, task_id: str, key: str, reserved: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "stage": "starling_txagent_eligible_v7.heldout_union_molecules",
        "task_id": task_id,
        "cache_key": key,
        "reserved_molecules": sorted(reserved),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compute_heldout_union(molecules: list[str], paths: list[Path]) -> set[str]:
    if str(TXAGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(TXAGENT_ROOT))
    from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity
    from tools.chembl_tool.common.starling.heldout_index import load_heldout_identity_keys

    keys: set[str] = set()
    for path in paths:
        keys |= load_heldout_identity_keys(path)
    output: set[str] = set()
    for smiles in molecules:
        identity = normalize_molecule_identity(smiles)
        parent = identity.parent_inchi_key or identity.parent_smiles
        if parent in keys:
            output.add(smiles)
    return output


def heldout_union_molecules(
    task_id: str, rows: list[dict[str, Any]], registry: Mapping[str, Any] | None = None,
    *, cache_dir: Path | None = None,
) -> set[str]:
    """Scaffold-reserved molecules, memoized on disk.

    The RDKit identity pass costs ~2 minutes per task and is re-run by both builders and by
    verify. Deduplicating rows to unique molecules first does not help -- the upstream identity
    cache already absorbs the row-level repeats -- so the result itself is cached.
    """
    contract = dict(registry or load_registry())
    paths = list(heldout_paths(task_id, contract))
    molecules = sorted({str(row["canonical_smiles"]) for row in rows})
    key = _reservation_cache_key(task_id, molecules, paths)
    memoized = _RESERVATION_MEMO.get(key)
    if memoized is not None:
        return set(memoized)
    path = (cache_dir or RESERVATION_CACHE_DIR) / f"{task_id}.json"
    reserved = _load_cached_reservation(path, key)
    if reserved is None:
        reserved = _compute_heldout_union(molecules, paths)
        _store_reservation(path, task_id, key, reserved)
    _RESERVATION_MEMO[key] = set(reserved)
    return set(reserved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(load_registry()["tasks"]))
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = write_eligible_records(
        args.task, args.output_dir, artifact_root=args.artifact_root
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
