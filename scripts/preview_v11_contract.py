#!/usr/bin/env python3
"""Create temporary real/synthetic previews for the task-separated v11 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from pipeline.pair_core import stable_hash
from pipeline.source_normalization.starling_txagent_eligible_v7 import build_eligible_records
from pipeline.v11_contract import (
    artifact_status,
    load_registry,
    task_artifact_root,
    task_config,
    validate_registry,
    validate_template_bindings,
)
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v11_targets import target_for


DEFAULT_OUTPUT = Path("tmp/v11_multitask_prompt_contract/previews")


def _real_pairs(rows: list[dict], source_ids: set[str]) -> dict[str, tuple[dict, dict]]:
    first: dict[tuple[str, str], dict] = {}
    pairs: dict[str, tuple[dict, dict]] = {}
    ordered = sorted(rows, key=lambda row: stable_hash(str(row["child_id"])))
    for row in ordered:
        source = str(row["source_id"])
        if source not in source_ids or source in pairs:
            continue
        key = (source, str(row["pair_bucket_key"]))
        prior = first.setdefault(key, row)
        if prior["canonical_smiles"] != row["canonical_smiles"]:
            pairs[source] = (row, prior)
    return pairs


def _synthetic_record(
    task_id: str, source_id: str, config: Mapping[str, Any], side: str
) -> dict[str, Any]:
    record: dict[str, Any] = {"task_id": task_id, "source_id": source_id}
    fields = list(config["both"])
    if side == "retrieval":
        fields += list(config["retrieval_only"])
    for field in fields:
        if field == "smiles":
            record[field] = "CCO" if side == "retrieval" else "CCN"
        elif field == "needs_more_context":
            record[field] = False
        elif field in {"positive_count", "total_tested"}:
            record[field] = 1 if field == "positive_count" else 10
        else:
            record[field] = f"example {field.replace('_', ' ')}"
    record.update(config.get("constants", {}))
    return record


def _real_preview(task_id: str, registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows, manifest = build_eligible_records(task_id, registry=registry)
    sources = set(task_config(task_id, registry)["sources"])
    pairs = _real_pairs(rows, sources)
    output = []
    for source_id in sorted(pairs):
        query, retrieval = pairs[source_id]
        target = target_for(query, retrieval)
        output.append(
            {
                "preview_kind": "real_artifact",
                "task_id": task_id,
                "source_id": source_id,
                "query_record_id": query["child_id"],
                "retrieval_record_id": retrieval["child_id"],
                "prompt": render_prompt(query, retrieval),
                **target,
            }
        )
    configs = task_config(task_id, registry)["sources"]
    for source_id in sorted(sources - set(pairs)):
        query = _synthetic_record(task_id, source_id, configs[source_id], "query")
        retrieval = _synthetic_record(task_id, source_id, configs[source_id], "retrieval")
        output.append(
            {
                "preview_kind": "synthetic_contract_preview_no_calibrated_pair",
                "task_id": task_id,
                "source_id": source_id,
                "prompt": render_prompt(query, retrieval),
                "target": None,
            }
        )
    output.sort(key=lambda row: row["source_id"])
    output[0]["eligible_stats"] = manifest["stats"]
    return output


def _synthetic_preview(task_id: str, registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    sources = task_config(task_id, registry)["sources"]
    for source_id, config in sorted(sources.items()):
        query = _synthetic_record(task_id, source_id, config, "query")
        retrieval = _synthetic_record(task_id, source_id, config, "retrieval")
        output.append(
            {
                "preview_kind": "synthetic_contract_preview",
                "task_id": task_id,
                "source_id": source_id,
                "prompt": render_prompt(query, retrieval),
                "target": None,
            }
        )
    return output


def _write_preview(task_id: str, rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / f"{task_id}.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    sections = [f"# {task_id} v11 prompt previews\n"]
    for row in rows:
        sections.append(f"## {row['source_id']} ({row['preview_kind']})\n\n```text\n{row['prompt']}\n```\n")
    (output / f"{task_id}.md").write_text("\n".join(sections), encoding="utf-8")


def create_previews(
    task_ids: list[str], output: Path, require_artifacts: bool = False
) -> dict[str, str]:
    registry = load_registry()
    statuses = validate_registry(registry)
    validate_template_bindings(registry)
    report = {}
    for task_id in task_ids:
        status = artifact_status(task_artifact_root(task_id, registry))["status"]
        if status != "ready" and require_artifacts:
            raise FileNotFoundError(f"{task_id} v7 artifact is not complete")
        rows = _real_preview(task_id, registry) if status == "ready" else _synthetic_preview(task_id, registry)
        _write_preview(task_id, rows, output)
        report[task_id] = statuses[task_id]["status"]
    return report


def _parse_args() -> argparse.Namespace:
    choices = sorted(load_registry()["tasks"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=choices, default=choices)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(create_previews(args.tasks, args.output, args.require_artifacts), indent=2))


if __name__ == "__main__":
    main()
