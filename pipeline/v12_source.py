"""Build leakage-safe V12 eligible records from TxAgent normalized evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.source_normalization import starling_txagent_eligible_v7 as v7_source
from pipeline.v11_targets import ValueCalibration, _cdf_payload, geometry_value
from pipeline.v12_contract import (
    DEFAULT_REGISTRY, gold_split_paths, load_registry,
    resolve_txagent_root, task_artifact_root, task_config, validate_registry,
)
from pipeline.v3_policy import file_sha256


MINIMUM_TRAIN_RECORDS = 25
VALUE_CALIBRATION_SCHEMA = "assay_transfer_train_value_calibration.v12"
DISTANCE_CALIBRATION_SCHEMA = "assay_transfer_train_percentile_distance_cdf.v12.1"
SPLIT_SCHEMA = "assay_transfer_molecule_split.v12"


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_gzip_json(document: Mapping[str, Any], path: Path) -> None:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)


def _source_sample(
    rows: list[dict[str, Any]], task_id: str, task: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = task["construction"].get("source_record_sampling", {})
    retained: set[str] = set()
    report: dict[str, Any] = {}
    for source, rule in rules.items():
        selected = [row for row in rows if str(row["source_id"]) == source]
        namespace = f"v12-source-sample:{task_id}:{source}"
        selected.sort(key=lambda row: _hash(f"{namespace}:{row['canonical_record_id']}"))
        wanted = len(selected) * int(rule["numerator"]) // int(rule["denominator"])
        retained.update(str(row["canonical_record_id"]) for row in selected[:wanted])
        report[source] = {
            **dict(rule), "input_records": len(selected), "retained_records": wanted,
            "seed_namespace": namespace,
        }
    output = [
        row for row in rows
        if str(row["source_id"]) not in rules
        or str(row["canonical_record_id"]) in retained
    ]
    return output, report


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _base_reason(row: Mapping[str, Any], global_entries: Mapping[str, Any]) -> str | None:
    if not bool(row.get("bucket_eligible")):
        return str(row.get("bucket_exclusion_reason") or "bucket_ineligible")
    entry = global_entries.get(str(row.get("pair_bucket_key") or ""))
    if not entry:
        return "global_measurement_metadata_unavailable"
    if str(row.get("measurement_kind") or "") not in {"continuous", "binary", "ordinal"}:
        return "unsupported_measurement_kind"
    if str(row.get("canonicalization_status") or "") != "valid":
        return "canonicalization_invalid"
    if not v7_source._present(row.get("smiles")):
        return "missing_source_smiles"
    if not v7_source._present(row.get("canonical_smiles")):
        return "missing_canonical_smiles"
    try:
        geometry_value(row)
    except (TypeError, ValueError):
        return "nonfinite_geometry_value"
    return None


def _identity_normalizer(txagent_root: Path):
    if str(txagent_root) not in sys.path:
        sys.path.insert(0, str(txagent_root))
    from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity

    return normalize_molecule_identity


def _attach_parent_keys(
    rows: list[dict[str, Any]], txagent_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    normalize = _identity_normalizer(txagent_root)
    cache: dict[str, str | None] = {}
    output = []
    for row in rows:
        smiles = str(row["canonical_smiles"])
        if smiles not in cache:
            identity = normalize(smiles)
            key = identity.parent_inchi_key or identity.parent_smiles
            cache[smiles] = str(key) if key else None
        if cache[smiles] is None:
            continue
        row["normalized_parent_identity_key"] = cache[smiles]
        output.append(row)
    return output, len(rows) - len(output)


def _gold_keys(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys = {str(row.get("molecule_identity_key") or "") for row in rows}
    if not keys or "" in keys or len(keys) != len(rows):
        raise ValueError(f"invalid gold molecule identities: {path}")
    return keys


def _assign_splits(
    rows: list[dict[str, Any]], task_id: str, paths: Mapping[str, Path],
) -> dict[str, Any]:
    evidence = {str(row["normalized_parent_identity_key"]) for row in rows}
    gold = {split: _gold_keys(path) for split, path in paths.items()}
    if (gold["train"] & gold["valid"]) or (gold["train"] & gold["test"]) or (gold["valid"] & gold["test"]):
        raise ValueError("gold train/valid/test molecule identities overlap")
    test = evidence & (gold["valid"] | gold["test"])
    validation_pool = sorted(
        evidence & gold["train"],
        key=lambda key: _hash(f"assay-transfer-v12-validation-parent:{task_id}:{key}"),
    )
    validation = set(validation_pool[: min(len(validation_pool), len(test) // 2)])
    for row in rows:
        key = str(row["normalized_parent_identity_key"])
        row["dataset_split"] = "test" if key in test else "validation" if key in validation else "train"
    counts = Counter(str(row["dataset_split"]) for row in rows)
    parent_counts = Counter()
    for split in ("train", "validation", "test"):
        parent_counts[split] = len({
            str(row["normalized_parent_identity_key"])
            for row in rows if row["dataset_split"] == split
        })
    return {
        "schema_version": SPLIT_SCHEMA,
        "method": "molecule-disjoint splits grouped by TxAgent normalized-parent identity",
        "normalizer_version": "rdkit_fragment_parent.v1",
        "test_parent_source": "eligible intersection with gold valid+test union",
        "validation_parent_source": "stable-hash sample from eligible gold-train parents",
        "validation_seed_namespace": "assay-transfer-v12-validation-parent",
        "validation_target_formula": "floor(test_parent_count/2)",
        "record_counts": dict(sorted(counts.items())),
        "parent_counts": dict(sorted(parent_counts.items())),
        "gold_paths": {split: str(path) for split, path in paths.items()},
        "gold_sha256": {split: file_sha256(path) for split, path in paths.items()},
        "_test_parents": sorted(test), "_validation_order": validation_pool,
    }


def _continuous_cdf(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(float(row["finite_scalar_value"]) for row in rows)
    support, total, before = sorted(counts), len(rows), 0
    midranks = []
    for value in support:
        midranks.append((before + 0.5 * counts[value]) / total)
        before += counts[value]
    return {
        "contract_version": "empirical_value_cdf.v1",
        "measurement_field": "finite_scalar_value", "tie_convention": "midrank",
        "unseen_value_rule": "records_strictly_less_than_value / total_records",
        "outside_support": "saturate_to_zero_or_one", "total_record_count": total,
        "distinct_value_count": len(support), "support_values": support,
        "support_counts": [counts[value] for value in support],
        "support_midranks_0_1": midranks,
    }


def _ordinal_cdf(
    rows: list[Mapping[str, Any]], domain: list[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = Counter(str(row["canonical_category_id"]) for row in rows)
    total, before, categories = len(rows), 0, []
    for item in sorted(domain, key=lambda value: float(value["rank"])):
        category_id, count = str(item["category_id"]), observed[str(item["category_id"])]
        categories.append({
            "category_id": category_id, "rank": float(item["rank"]),
            "observed_count": count, "midrank_0_1": (before + 0.5 * count) / total,
        })
        before += count
    if before != total:
        raise ValueError("train ordinal values fall outside the declared scale")
    return {
        "contract_version": "empirical_ordinal_category_cdf.v1",
        "measurement_field": "canonical_category_rank",
        "category_identity_field": "canonical_category_id",
        "canonical_measurement_scale_id": rows[0]["canonical_measurement_scale_id"],
        "tie_convention": "midrank", "total_record_count": total,
        "categories": categories,
    }


def _local_entry(
    rows: list[Mapping[str, Any]], global_entry: Mapping[str, Any],
    category_domain: list[Mapping[str, Any]],
) -> dict[str, Any]:
    kind = str(rows[0]["measurement_kind"])
    value_cdf = _continuous_cdf(rows) if kind == "continuous" else None
    category_cdf = _ordinal_cdf(rows, category_domain) if kind == "ordinal" else None
    return {
        "source_id": str(rows[0]["source_id"]), "measurement_kind": kind,
        "canonical_measurement_scale_id": rows[0].get("canonical_measurement_scale_id"),
        "record_count": len(rows), "minimum_record_count": MINIMUM_TRAIN_RECORDS,
        "minimum_support_met": True, "calibration_valid": True,
        "calibration_reason": "valid_train_only",
        "observed_sample_standard_deviation": global_entry["observed_sample_standard_deviation"],
        "standard_deviation_ddof": global_entry["standard_deviation_ddof"],
        "standard_deviation_value_field": global_entry["standard_deviation_value_field"],
        "standard_deviation_scope": "global_txagent_stage05_measurement_metadata_only",
        "standard_deviation_reason": global_entry.get("standard_deviation_reason"),
        "value_cdf_valid": value_cdf is not None, "value_cdf": value_cdf,
        "category_cdf_valid": category_cdf is not None, "category_cdf": category_cdf,
    }


def _fit_value_calibrations(
    rows: list[dict[str, Any]], global_entries: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["dataset_split"] == "train":
            grouped[str(row["pair_bucket_key"])].append(row)
    accepted, rejected, rejected_buckets = {}, Counter(), {}
    domains = _category_domains(rows, global_entries)
    for bucket in sorted({str(row["pair_bucket_key"]) for row in rows}):
        values = grouped.get(bucket, [])
        if len(values) < MINIMUM_TRAIN_RECORDS:
            rejected["fewer_than_25_train_records"] += 1
            rejected_buckets[bucket] = {
                "reason": "fewer_than_25_train_records", "train_record_count": len(values),
            }
            continue
        accepted[bucket] = _local_entry(
            values, global_entries[bucket], domains.get(bucket, [])
        )
    document = {
        "schema_version": VALUE_CALIBRATION_SCHEMA, "fit_split": "train",
        "weighting": "record_weighted", "minimum_train_records": MINIMUM_TRAIN_RECORDS,
        "source_sampling_precedes_fit": True, "buckets": accepted,
        "rejected_buckets": rejected_buckets,
        "summary": {"calibrated_buckets": len(accepted), "rejected": dict(rejected)},
    }
    return document, {key: _parse_local_calibration(key, entry) for key, entry in accepted.items()}


def _category_domains(
    rows: list[Mapping[str, Any]], global_entries: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    domains = {}
    inferred: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for bucket, entry in global_entries.items():
        categories = (entry.get("category_cdf") or {}).get("categories")
        if categories:
            domains[str(bucket)] = list(categories)
    for row in rows:
        if row["measurement_kind"] != "ordinal":
            continue
        bucket = str(row["pair_bucket_key"])
        category = str(row["canonical_category_id"])
        inferred[bucket][category] = {
            "category_id": category, "rank": float(row["canonical_category_rank"]),
        }
    for bucket, categories in inferred.items():
        domains.setdefault(bucket, list(categories.values()))
    return domains


def _parse_local_calibration(
    bucket: str, entry: Mapping[str, Any],
) -> ValueCalibration:
    kind = str(entry["measurement_kind"])
    value_cdf, category_cdf = _cdf_payload(kind, entry)
    return ValueCalibration(
        bucket, kind, entry.get("observed_sample_standard_deviation"),
        int(entry["standard_deviation_ddof"]),
        str(entry["standard_deviation_value_field"]), value_cdf, category_cdf,
    )


def _distance_entry(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(float(row["value_percentile"]) for row in rows)
    ordered = sorted(counts)
    total = len(rows)
    positions = np.asarray([round(value * 2 * total) for value in ordered], dtype=np.int64)
    frequencies = np.asarray([counts[value] for value in ordered], dtype=np.int64)
    distance_counts = np.zeros(2 * total + 1, dtype=np.int64)
    distance_counts[0] = sum(count * (count - 1) // 2 for count in frequencies.tolist())
    for index in range(len(positions) - 1):
        distances = positions[index + 1:] - positions[index]
        weights = frequencies[index + 1:] * frequencies[index]
        distance_counts += np.bincount(distances, weights=weights, minlength=len(distance_counts)).astype(np.int64)
    present = np.flatnonzero(distance_counts)
    pair_count = total * (total - 1) // 2
    if int(distance_counts.sum()) != pair_count:
        raise ValueError("percentile-distance pair accounting mismatch")
    return {
        "measurement_kind": str(rows[0]["measurement_kind"]), "record_count": total,
        "distinct_parent_count": len({str(row["normalized_parent_identity_key"]) for row in rows}),
        "distinct_record_unordered_pair_count": pair_count,
        "pair_scope": "all_distinct_record_train_pairs_including_same_parent",
        "support_values": [float(value / (2 * total)) for value in present.tolist()],
        "support_counts": [int(distance_counts[value]) for value in present.tolist()],
    }


def build_distance_calibration(rows: Iterable[Mapping[str, Any]], task_id: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["dataset_split"] == "train" and row["measurement_kind"] in {"continuous", "ordinal"}:
            grouped[str(row["pair_bucket_key"])].append(row)
    return {
        "schema_version": DISTANCE_CALIBRATION_SCHEMA, "task_id": task_id,
        "fit_split": "train", "weighting": "all_distinct_record_unordered_pairs",
        "buckets": {bucket: _distance_entry(values) for bucket, values in sorted(grouped.items())},
    }


def _finalize_split_manifest(
    split: dict[str, Any], rows: list[Mapping[str, Any]],
) -> None:
    split["precalibration_record_counts"] = split.pop("record_counts")
    split["precalibration_parent_counts"] = split.pop("parent_counts")
    split["record_counts"] = dict(sorted(Counter(
        str(row["dataset_split"]) for row in rows
    ).items()))
    split["parent_counts"] = {
        name: len({
            str(row["normalized_parent_identity_key"])
            for row in rows if row["dataset_split"] == name
        })
        for name in ("train", "validation", "test")
    }


def _project_rows(
    raw_rows: list[dict[str, Any]], task_id: str, calibrations: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prompt_fields = v7_source.prompt_field_union(task_id, registry)
    output = []
    for raw in raw_rows:
        calibration = calibrations.get(str(raw["pair_bucket_key"]))
        if calibration is None:
            continue
        row = v7_source._eligible_row(raw, task_id, calibration, registry, prompt_fields)
        row["normalized_parent_identity_key"] = raw["normalized_parent_identity_key"]
        row["dataset_split"] = raw["dataset_split"]
        output.append(row)
    return output


def _apply_parent_splits(
    rows: list[dict[str, Any]], test: set[str], validation: set[str],
) -> None:
    for row in rows:
        parent = str(row["normalized_parent_identity_key"])
        row["dataset_split"] = (
            "test" if parent in test else "validation" if parent in validation else "train"
        )


def _stabilize_split_and_calibration(
    rows: list[dict[str, Any]], task_id: str, split: dict[str, Any],
    global_entries: Mapping[str, Any], registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    test = set(split.pop("_test_parents"))
    order = list(split.pop("_validation_order"))
    validation = set(order[: len(test) // 2])
    for iteration in range(10):
        _apply_parent_splits(rows, test, validation)
        document, calibrations = _fit_value_calibrations(rows, global_entries)
        accepted = set(calibrations)
        surviving = {
            str(row["normalized_parent_identity_key"])
            for row in rows if str(row["pair_bucket_key"]) in accepted
        }
        desired = len(test & surviving) // 2
        viable = [parent for parent in order if parent in surviving]
        updated = set(viable[:desired])
        if updated == validation:
            projected = _project_rows(rows, task_id, calibrations, registry)
            split["fixed_point_iterations"] = iteration + 1
            return projected, document, calibrations
        validation = updated
    raise RuntimeError("validation-parent and train-bucket eligibility did not converge")


def build_eligible_records(
    task_id: str, registry: Mapping[str, Any], txagent_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = task_artifact_root(task_id, registry, txagent_root)
    paths = v7_source.artifact_paths(root)
    global_document = _read_gzip_json(paths["calibration"])
    joined = v7_source._load_joined(paths).to_dict("records")
    rejected = Counter()
    raw_rows = []
    for row in joined:
        reason = _base_reason(row, global_document["buckets"])
        if reason:
            rejected[reason] += 1
        else:
            raw_rows.append(row)
    raw_rows, sampling = _source_sample(raw_rows, task_id, task_config(task_id, registry))
    raw_rows, identity_rejections = _attach_parent_keys(raw_rows, txagent_root)
    if identity_rejections:
        rejected["parent_identity_unavailable"] += identity_rejections
    split = _assign_splits(raw_rows, task_id, gold_split_paths(task_id, registry, txagent_root))
    rows, value_document, calibrations = _stabilize_split_and_calibration(
        raw_rows, task_id, split, global_document["buckets"], registry
    )
    _finalize_split_manifest(split, rows)
    distance = build_distance_calibration(rows, task_id)
    stats = {
        "input_records": len(joined), "precalibration_records": len(raw_rows),
        "eligible_records": len(rows), "base_rejection_counts": dict(sorted(rejected.items())),
        "source_record_sampling": sampling,
        "eligible_by_split": dict(sorted(Counter(row["dataset_split"] for row in rows).items())),
        "eligible_by_measurement_kind": dict(sorted(Counter(row["measurement_kind"] for row in rows).items())),
    }
    return rows, value_document, distance, {"split": split, "stats": stats, "paths": paths}


def write_eligible_records(
    task_id: str, output_dir: Path, registry_path: Path = DEFAULT_REGISTRY,
    txagent_root: Path | None = None,
) -> dict[str, Any]:
    registry, root = load_registry(registry_path), txagent_root or resolve_txagent_root()
    validate_registry(registry, root, tasks=(task_id,))
    rows, value_document, distance, details = build_eligible_records(task_id, registry, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_schema = pq.read_schema(details["paths"]["records"])
    schema = v7_source.eligible_schema(task_id, registry, source_schema)
    schema = pa.schema(list(schema) + [
        pa.field("normalized_parent_identity_key", pa.string()),
        pa.field("dataset_split", pa.string()),
    ])
    v7_source._write_eligible_table(pa.Table.from_pylist(rows, schema=schema), output_dir / "records.parquet")
    _write_gzip_json(value_document, output_dir / "train_value_calibration.json.gz")
    _write_gzip_json(distance, output_dir / "train_percentile_distance_calibration.json.gz")
    split_path = output_dir / "split_manifest.json"
    split_path.write_text(json.dumps(details["split"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "stage": "starling_txagent_eligible_v12", "task_id": task_id,
        "lineage": task_config(task_id, registry)["gold_lineage"],
        "split_contract": details["split"], "stats": details["stats"],
        "prompt_projection": {
            "path": str(registry_path), "sha256": file_sha256(registry_path),
        },
        "calibration_contract": registry["calibration_contract"],
        "global_stage05_sd_policy": (
            "copied unchanged, including null, as measurement-only metadata"
        ),
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in details["paths"].items()
        },
        "artifacts": {},
    }
    for name in (
        "records.parquet", "train_value_calibration.json.gz",
        "train_percentile_distance_calibration.json.gz", "split_manifest.json",
    ):
        manifest["artifacts"][name] = file_sha256(output_dir / name)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "DISTANCE_CALIBRATION_SCHEMA", "MINIMUM_TRAIN_RECORDS", "SPLIT_SCHEMA",
    "VALUE_CALIBRATION_SCHEMA", "build_distance_calibration", "build_eligible_records",
    "write_eligible_records",
]
