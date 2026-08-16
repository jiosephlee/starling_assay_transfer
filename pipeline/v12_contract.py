"""Executable lineage, split, and projection contract for assay-transfer V12."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline import v11_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "configs/assay_transfer/v12/prompt_projection.json"
TEMPLATE_ROOT = v11_contract.TEMPLATE_ROOT
SUPPORTED_TASKS = frozenset(("bioavailability_ma", "bbb_martins", "skin_reaction"))


def resolve_txagent_root() -> Path:
    configured = os.environ.get("STARLING_TXAGENT_ROOT")
    candidates = [
        Path(configured) if configured else None,
        Path("/data1/joseph/TxAgent"),
        REPO_ROOT.parent / "TxAgent",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("TxAgent checkout not found; set STARLING_TXAGENT_ROOT")


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "assay_transfer_prompt_projection.v12":
        raise ValueError("unexpected V12 prompt-projection schema")
    base_path = REPO_ROOT / str(document["base_registry_relpath"])
    base = v11_contract.load_registry(base_path)
    tasks = copy.deepcopy(base["tasks"])
    for task_id, override in document["task_overrides"].items():
        tasks[task_id] = _merge(tasks[task_id], override)
        tasks[task_id]["construction"]["ranking_anchors"] = copy.deepcopy(
            override["construction"]["ranking_anchors"]
        )
        tasks[task_id].pop("heldout_reservation", None)
    return {
        **document,
        "tasks": tasks,
        "base_registry_path": str(base_path),
        "source_contract_version": base["source_contract_version"],
    }


def task_config(task_id: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return dict(registry["tasks"][task_id])
    except KeyError as exc:
        raise ValueError(f"unknown V12 task: {task_id!r}") from exc


def source_config(
    task_id: str, source_id: str, registry: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return dict(task_config(task_id, registry)["sources"][source_id])
    except KeyError as exc:
        raise ValueError(f"unknown V12 source: {task_id}/{source_id}") from exc


def release_config(registry: Mapping[str, Any]) -> dict[str, Any]:
    return dict(registry["categorical_ablation_release"])


def task_artifact_root(
    task_id: str, registry: Mapping[str, Any], txagent_root: Path | None = None,
) -> Path:
    root = txagent_root or resolve_txagent_root()
    return root / task_config(task_id, registry)["artifact_relpath"]


def gold_split_paths(
    task_id: str, registry: Mapping[str, Any], txagent_root: Path | None = None,
) -> dict[str, Path]:
    root = txagent_root or resolve_txagent_root()
    gold_root = root / task_config(task_id, registry)["gold_lineage"]["root_relpath"]
    return {
        split: gold_root / f"{split}_molecule_labels.jsonl"
        for split in ("train", "valid", "test")
    }


def _validate_task(task_id: str, registry: Mapping[str, Any], root: Path) -> None:
    task = task_config(task_id, registry)
    sources = set(task["sources"])
    phases = [source for phase in task["priority_phases"] for source in phase]
    if len(phases) != len(set(phases)) or set(phases) != sources:
        raise ValueError(f"{task_id}: priority phases must partition sources")
    construction = task["construction"]
    if int(construction["ranking_anchor_width"]) != 24:
        raise ValueError(f"{task_id}: V12 ranking width must be 24")
    for split in ("validation", "test"):
        families = construction["ranking_anchors"][split]
        if set(families) != {"continuous", "categorical"}:
            raise ValueError(f"{task_id}: incomplete {split} ranking quotas")
        if any(set(families[family]) != sources for family in families):
            raise ValueError(f"{task_id}: {split} quotas do not cover sources")
    paths = gold_split_paths(task_id, registry, root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{task_id}: missing gold lineage files: {missing}")
    source_manifest = v11_contract.record_contract_manifest(task_id, root)
    if set(task["sources"]) != set(source_manifest["sources"]):
        raise ValueError(f"{task_id}: TxAgent source inventory mismatch")
    for source_id, config in task["sources"].items():
        v11_contract._validate_source(
            source_id, config, source_manifest["sources"][source_id]
        )
    status = v11_contract.artifact_status(task_artifact_root(task_id, registry, root))
    if status["status"] != "ready":
        raise FileNotFoundError(f"{task_id}: incomplete TxAgent normalized V7 artifact")


def validate_registry(
    registry: Mapping[str, Any], txagent_root: Path | None = None,
    tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    if registry.get("record_contract_version") != "starling_record_contract.v7":
        raise ValueError("V12 requires the TxAgent V7 record contract")
    if set(registry["tasks"]) != SUPPORTED_TASKS:
        raise ValueError("V12 task inventory is incomplete")
    release = release_config(registry)
    for component in release["train_components"].values():
        if (int(component["query_degree_cap"]), int(component["retrieval_degree_cap"])) != (6, 6):
            raise ValueError("V12 train query/retrieval degree caps must be 6/6")
    selected = SUPPORTED_TASKS if tasks is None else set(tasks)
    if selected - SUPPORTED_TASKS:
        raise ValueError(f"unknown V12 tasks: {sorted(selected - SUPPORTED_TASKS)}")
    root = txagent_root or resolve_txagent_root()
    v11_contract.validate_template_bindings(registry, tasks=selected)
    for task_id in sorted(selected):
        _validate_task(task_id, registry, root)
    return {task_id: {"status": "ready"} for task_id in sorted(selected)}


__all__ = [
    "DEFAULT_REGISTRY", "TEMPLATE_ROOT", "gold_split_paths", "load_registry",
    "release_config", "resolve_txagent_root", "source_config", "task_artifact_root",
    "task_config", "validate_registry",
]
