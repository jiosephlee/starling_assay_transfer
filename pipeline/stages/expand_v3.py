#!/usr/bin/env python3
"""Create a portable train-only expansion bundle for a frozen v3 release."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from pipeline.v3_policy import V3Policies, file_sha256, resolve_path


def _train_records(eligible: Path, splits: Path) -> pa.Table:
    split_table = pq.read_table(splits / "molecule_splits.parquet")
    train = {row["canonical_smiles"] for row in split_table.to_pylist() if row["split"] == "train"}
    rows = [row for row in pq.read_table(eligible / "records.parquet").to_pylist()
            if row["canonical_smiles"] in train]
    return pa.Table.from_pylist(rows)


def _selected_ids(selection: Path) -> pa.Table:
    rows = pq.read_table(selection / "selected" / "train" / "selected.parquet")
    return pa.table({"candidate_id": rows["candidate_id"]})


def _snapshot_policies(stage: Path, release: Path, policies: V3Policies) -> None:
    policy_dir = stage / "policies"
    policy_dir.mkdir(parents=True)
    refs = policies.release["policies"]
    rewritten = dict(policies.release)
    rewritten["policies"] = {}
    for name, value in refs.items():
        source = policies._policy_path(value)
        target = policy_dir / f"{name}.yaml"
        shutil.copy2(source, target)
        rewritten["policies"][name] = f"policies/{name}.yaml"
    (stage / "release.yaml").write_text(yaml.safe_dump(rewritten, sort_keys=False))


def _archive(stage: Path, output_dir: Path) -> Path:
    archive = output_dir / "train_expansion_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(stage.rglob("*")):
            handle.add(path, arcname=path.relative_to(stage))
    return archive


def build(args: argparse.Namespace) -> dict[str, Any]:
    policies = V3Policies(resolve_path(args.release))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = args.output_dir / "bundle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    pq.write_table(_train_records(args.eligible, args.split_dir), stage / "eligible_train_records.parquet")
    pq.write_table(_selected_ids(args.selection_dir), stage / "selected_candidate_ids.parquet")
    shutil.copy2(args.split_dir / "molecule_splits.parquet", stage / "molecule_splits.parquet")
    shutil.copytree(args.template_dir, stage / "templates")
    _snapshot_policies(stage, resolve_path(args.release), policies)
    bundle_manifest = {"schema_version": policies.release["artifact_schema_version"],
                       "policy_versions": policies.version_bundle, "query_caps": args.query_caps,
                       "ordering_seed": int(policies.sampling["seed"]),
                       "selected_train_rows": pq.read_table(stage / "selected_candidate_ids.parquet").num_rows,
                       "eligible_train_records": pq.read_table(stage / "eligible_train_records.parquet").num_rows}
    (stage / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2))
    archive = _archive(stage, args.output_dir)
    manifest = {**bundle_manifest, "archive": str(archive), "archive_sha256": file_sha256(archive)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-caps", type=json.loads, default={})
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
