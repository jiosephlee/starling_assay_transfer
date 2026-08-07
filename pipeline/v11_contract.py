"""Executable prompt-projection and artifact contract for task-separated v11."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.v3_policy import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
TXAGENT_ROOT = Path("/data1/joseph/TxAgent")
DEFAULT_REGISTRY = REPO_ROOT / "configs/assay_transfer/v11/prompt_projection.json"
TEMPLATE_ROOT = REPO_ROOT / "templates/assay_transfer_v11_intern"
FORBIDDEN_PROMPT_FIELDS = frozenset({"global_context", "global_species_context"})
ELIGIBLE_PROJECTION_SCHEMA_VERSION = "assay_transfer_eligible_projection.v1"
TASK_SCHEMA_MODULES = {
    task: f"tools.chembl_tool.tasks.{task}.starling_schema"
    for task in ("bioavailability_ma", "bbb_martins", "skin_reaction")
}


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "assay_transfer_prompt_projection.v11":
        raise ValueError("unexpected v11 prompt-projection schema")
    if document.get("record_contract_version") != "starling_record_contract.v7":
        raise ValueError("v11 requires the TxAgent v7 record contract")
    return document


def categorical_ablation_config(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the shared train/evaluation policy for the paired V11 variants."""
    try:
        return dict(registry["categorical_ablation_release"])
    except KeyError as exc:
        raise ValueError("v11 registry lacks categorical-ablation release policy") from exc


def task_config(task_id: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return dict(registry["tasks"][task_id])
    except KeyError as exc:
        raise ValueError(f"unknown v11 task: {task_id!r}") from exc


def eligible_projection_manifest(
    task_id: str, registry: Mapping[str, Any],
) -> dict[str, str]:
    """Digest the task contract that controls eligible prompt-field materialization."""
    task = task_config(task_id, registry)
    payload = {
        "schema_version": ELIGIBLE_PROJECTION_SCHEMA_VERSION,
        "task_id": task_id,
        "record_contract_version": registry["record_contract_version"],
        "source_contract_version": registry.get("source_contract_version"),
        "artifact_relpath": task["artifact_relpath"],
        "sources": task["sources"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": ELIGIBLE_PROJECTION_SCHEMA_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def source_config(
    task_id: str, source_id: str, registry: Mapping[str, Any]
) -> dict[str, Any]:
    task = task_config(task_id, registry)
    try:
        return dict(task["sources"][source_id])
    except KeyError as exc:
        raise ValueError(f"unknown v11 source: {task_id}/{source_id}") from exc


def source_backed_fields(config: Mapping[str, Any]) -> list[str]:
    """Prompt fields materialized from source rows, in declaration order."""
    return list(config.get("both", ())) + list(config.get("retrieval_only", ()))


def source_constants(config: Mapping[str, Any]) -> dict[str, Any]:
    """All constants, including values that are retrieval-only."""
    output = dict(config.get("constants", {}))
    output.update(config.get("retrieval_only_constants", {}))
    return output


def task_artifact_root(
    task_id: str,
    registry: Mapping[str, Any],
    txagent_root: Path = TXAGENT_ROOT,
) -> Path:
    return txagent_root / task_config(task_id, registry)["artifact_relpath"]


def artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "manifest.json",
        "records": root / "03_records/records.parquet",
        "pair_buckets": root / "04_pair_buckets/pair_bucket_records.parquet",
        "pair_bucket_metadata": root / "04_pair_buckets/pair_bucket_metadata.json",
        "calibration": root / "05_distance_calibration/pair_bucket_distance_calibration.json.gz",
    }


def artifact_status(root: Path) -> dict[str, Any]:
    paths = artifact_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    return {
        "status": "ready" if not missing else "pending_artifact",
        "root": str(root),
        "missing": missing,
    }


def heldout_paths(
    task_id: str,
    registry: Mapping[str, Any],
    txagent_root: Path = TXAGENT_ROOT,
) -> list[Path]:
    task = task_config(task_id, registry)
    reservation = task["heldout_reservation"]
    return [txagent_root / reservation["labels_relpath"]]


def _heldout_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"heldout reservation file is empty: {path}")
    return rows


def _heldout_identity_keys(path: Path, txagent_root: Path) -> set[str]:
    if str(txagent_root) not in sys.path:
        sys.path.insert(0, str(txagent_root))
    from tools.chembl_tool.common.starling.heldout_index import load_heldout_identity_keys

    return load_heldout_identity_keys(path)


def heldout_reservation_manifest(
    task_id: str, registry: Mapping[str, Any], txagent_root: Path = TXAGENT_ROOT,
) -> dict[str, Any]:
    reservation = task_config(task_id, registry).get("heldout_reservation", {})
    if reservation.get("methodology") != "scaffold":
        raise ValueError(f"{task_id!r} must use scaffold heldout reservation")
    included = list(reservation.get("included_splits", []))
    if included != ["valid", "test"]:
        raise ValueError(f"{task_id!r} must reserve scaffold valid and test")
    relpath = str(reservation.get("labels_relpath") or "")
    if "/scaffold/" not in relpath or "/random/" in relpath:
        raise ValueError(f"{task_id!r} heldout path is not scaffold-only")
    path = txagent_root / relpath
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _heldout_rows(path)
    split_counts = Counter(str(row.get("split") or "") for row in rows)
    if set(split_counts) != set(included):
        raise ValueError(f"{task_id!r} scaffold file has unexpected split rows")
    methods = {str(row.get("split_method") or "") for row in rows}
    if methods != {"bemis_murcko_scaffold_group_subset_sum"}:
        raise ValueError(f"{task_id!r} scaffold file has unexpected split method")
    return {
        "methodology": "scaffold",
        "included_splits": included,
        "labels_path": str(path),
        "labels_sha256": file_sha256(path),
        "split_counts": dict(sorted(split_counts.items())),
        "heldout_parent_identities": len(_heldout_identity_keys(path, txagent_root)),
    }


def _live_record_contract(task_id: str, txagent_root: Path) -> Any:
    if str(txagent_root) not in sys.path:
        sys.path.insert(0, str(txagent_root))
    module = importlib.import_module(TASK_SCHEMA_MODULES[task_id])
    return module.RECORD_CONTRACT


def record_contract_manifest(task_id: str, txagent_root: Path = TXAGENT_ROOT) -> dict:
    return _live_record_contract(task_id, txagent_root).manifest()


def _validate_binding_names(source_id: str, config: Mapping[str, Any]) -> None:
    both = list(config.get("both", []))
    retrieval = list(config.get("retrieval_only", []))
    constants = dict(config.get("constants", {}))
    retrieval_constants = dict(config.get("retrieval_only_constants", {}))
    all_names = both + retrieval + list(constants) + list(retrieval_constants)
    if len(all_names) != len(set(all_names)):
        raise ValueError(f"{source_id!r} repeats a prompt binding")
    forbidden = [name for name in all_names if name.startswith("canonical_")]
    forbidden += [name for name in all_names if name in FORBIDDEN_PROMPT_FIELDS]
    if forbidden:
        raise ValueError(f"{source_id!r} exposes forbidden prompt fields: {forbidden}")
    if "smiles" not in both:
        raise ValueError(f"{source_id!r} must expose source smiles on both sides")
    if "unit_text" in both or "unit_text" in constants:
        raise ValueError(f"{source_id!r} exposes unit_text to the query")
    expected_null = set(config.get("expected_all_null_fields", ()))
    if expected_null - {"unit_text"}:
        raise ValueError(f"{source_id!r} has a non-unit expected-all-null exception")
    if expected_null - set(source_backed_fields(config)):
        raise ValueError(f"{source_id!r} expected-all-null field is not source-backed")


def _validate_source(
    source_id: str, config: Mapping[str, Any], profile: Mapping[str, Any]
) -> None:
    _validate_binding_names(source_id, config)
    visible = set(profile["source_visible_fields"])
    requested = set(source_backed_fields(config))
    missing = requested - visible
    if missing:
        raise ValueError(f"{source_id!r} binds non-source fields: {sorted(missing)}")
    constants = set(source_constants(config))
    if constants & visible:
        raise ValueError(f"{source_id!r} constants are incorrectly source-visible")
    if constants - set(profile["cleaned_source_fields"]):
        raise ValueError(f"{source_id!r} constants are absent from the cleaned schema")


def _validate_construction(task_id: str, configured: Mapping[str, Any]) -> None:
    sources = set(configured["sources"])
    phases = configured.get("priority_phases", [])
    flattened = [source for phase in phases for source in phase]
    if len(flattened) != len(set(flattened)) or set(flattened) != sources:
        raise ValueError(f"{task_id!r} priority phases must partition its sources")
    construction = configured.get("construction", {})
    for name in ("ranking_anchor_width", "record_degree_cap", "ordinary_degree_cap"):
        if int(construction.get(name, 0)) <= 0:
            raise ValueError(f"{task_id!r} has invalid {name}")
    ranking = construction.get("ranking_anchors", {})
    quota_maps = [ranking.get(split, {}) for split in ("validation", "test")]
    quota_maps.append(construction.get("ordinary_pairs_per_label", {}))
    if any(set(quotas) != sources for quotas in quota_maps):
        raise ValueError(f"{task_id!r} construction quotas do not cover every source")
    if any(int(value) < 0 for quotas in quota_maps for value in quotas.values()):
        raise ValueError(f"{task_id!r} construction quotas must be nonnegative")


def _validate_categorical_ablation(registry: Mapping[str, Any]) -> None:
    release = categorical_ablation_config(registry)
    if release.get("evaluation_measurement_kinds") != ["continuous"]:
        raise ValueError("categorical-ablation evaluation must be continuous-only")
    if release.get("categorical_measurement_kinds") != ["binary", "ordinal"]:
        raise ValueError("categorical-ablation kinds must be binary plus ordinal")
    components = release.get("train_components", {})
    expected = {"continuous": ["continuous"], "categorical": ["binary", "ordinal"]}
    if set(components) != set(expected):
        raise ValueError("categorical-ablation train components are incomplete")
    for name, kinds in expected.items():
        component = components[name]
        if component.get("measurement_kinds") != kinds:
            raise ValueError(f"{name!r} component has unexpected measurement kinds")
        numeric = ("max_pairs", "query_degree_cap", "retrieval_degree_cap")
        if any(int(component.get(field, 0)) <= 0 for field in numeric):
            raise ValueError(f"{name!r} component has invalid limits")
    variants = release.get("variants", {})
    if variants != {
        "continuous_only": ["continuous"],
        "with_categorical": ["continuous", "categorical"],
    }:
        raise ValueError("categorical-ablation variants do not preserve additive training")


def validate_registry(
    registry: Mapping[str, Any], txagent_root: Path = TXAGENT_ROOT,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate the registry; ``tasks`` narrows the per-task checks to the ones being built.

    The inventory check below always covers all three tasks -- only the expensive per-task work
    (live schema import, heldout hashing, RDKit identity keys) is narrowed.
    """
    _validate_categorical_ablation(registry)
    expected_tasks = set(TASK_SCHEMA_MODULES)
    if set(registry["tasks"]) != expected_tasks:
        raise ValueError("v11 task registry does not match the supported task inventory")
    selected = expected_tasks if tasks is None else set(tasks)
    unknown = selected - expected_tasks
    if unknown:
        raise ValueError(f"unknown v11 task(s) requested for validation: {sorted(unknown)}")
    validate_template_bindings(registry, tasks=selected)
    report: dict[str, Any] = {}
    for task_id in sorted(selected):
        manifest = record_contract_manifest(task_id, txagent_root)
        configured = task_config(task_id, registry)
        _validate_construction(task_id, configured)
        reservation = heldout_reservation_manifest(task_id, registry, txagent_root)
        if set(configured["sources"]) != set(manifest["sources"]):
            raise ValueError(f"{task_id!r} source inventory mismatch")
        for source_id, config in configured["sources"].items():
            _validate_source(source_id, config, manifest["sources"][source_id])
        root = task_artifact_root(task_id, registry, txagent_root)
        report[task_id] = {**artifact_status(root), "heldout_reservation": reservation}
    return report


def project_prompt_record(
    record: Mapping[str, Any], task_id: str, side: str, registry: Mapping[str, Any],
    *, config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if side not in {"query", "retrieval"}:
        raise ValueError(f"invalid prompt side: {side!r}")
    binding = config or source_config(task_id, str(record.get("source_id") or ""), registry)
    fields = list(binding["both"])
    if side == "retrieval":
        fields.extend(binding["retrieval_only"])
    output = {field: record.get(field) for field in fields}
    output.update(binding.get("constants", {}))
    if side == "retrieval":
        output.update(binding.get("retrieval_only_constants", {}))
    return output


def template_fields(path: Path) -> dict[str, set[str]]:
    text = path.read_text(encoding="utf-8")
    output = {"query": set(), "retrieval": set()}
    for side in output:
        dotted = re.findall(rf"{side}\.([A-Za-z_][A-Za-z0-9_]*)", text)
        getters = re.findall(rf"{side}\.get\([\"']([^\"']+)", text)
        output[side].update(dotted)
        output[side].discard("get")
        output[side].update(getters)
    return output


def validate_template_bindings(
    registry: Mapping[str, Any], tasks: Iterable[str] | None = None,
) -> None:
    selected = set(registry["tasks"]) if tasks is None else set(tasks)
    for task_id in sorted(selected):
        task = registry["tasks"][task_id]
        for source_id, config in task["sources"].items():
            path = TEMPLATE_ROOT / config["template"]
            if not path.is_file():
                raise FileNotFoundError(path)
            fields = template_fields(path)
            allowed_query = set(config["both"]) | set(config.get("constants", {}))
            allowed_retrieval = allowed_query | set(config["retrieval_only"])
            allowed_retrieval |= set(config.get("retrieval_only_constants", {}))
            if fields["query"] != allowed_query:
                raise ValueError(f"{task_id}/{source_id} query template binding mismatch")
            if fields["retrieval"] != allowed_retrieval:
                raise ValueError(f"{task_id}/{source_id} retrieval template binding mismatch")


__all__ = [
    "DEFAULT_REGISTRY",
    "TEMPLATE_ROOT",
    "TXAGENT_ROOT",
    "artifact_paths",
    "artifact_status",
    "categorical_ablation_config",
    "eligible_projection_manifest",
    "heldout_reservation_manifest",
    "heldout_paths",
    "load_registry",
    "project_prompt_record",
    "record_contract_manifest",
    "source_backed_fields",
    "source_config",
    "source_constants",
    "task_artifact_root",
    "task_config",
    "validate_registry",
    "validate_template_bindings",
]
