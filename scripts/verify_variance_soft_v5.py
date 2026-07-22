#!/usr/bin/env python3
"""Verify V5 membership, schema, target math, prompts, and rebuild hashes."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_split(source: Path, split: str):
    columns = ["candidate_id", "split", "binary_label"]
    table = pq.read_table(source, columns=columns)
    return table.filter(pc.equal(table["split"], split))


def _target_arrays(path: Path) -> dict[str, np.ndarray]:
    columns = ["metadata.target_distribution", "metadata.target_temperature",
               "metadata.completion_is_tie_anchor", "metadata.evidence.n_records",
               "metadata.evidence.n_transfer", "metadata.evidence.n_nontransfer",
               "metadata.evidence.n_ambiguous"]
    table = pq.read_table(path, columns=columns)
    target = table["target_distribution"].combine_chunks()
    values = {"transfer": target.field("transfer").to_numpy(),
              "nontransfer": target.field("nontransfer").to_numpy()}
    for name in ("target_temperature", "completion_is_tie_anchor", "n_records",
                 "n_transfer", "n_nontransfer", "n_ambiguous"):
        values[name] = table[name].to_numpy()
    return values


def _verify_targets(path: Path) -> None:
    data = _target_arrays(path)
    total = data["n_records"].astype(float)
    expected_t = np.sqrt(total) * (1.0 + data["n_ambiguous"] / total)
    delta = (data["n_transfer"] - data["n_nontransfer"]) / expected_t
    expected_a = 1.0 / (1.0 + np.exp(-delta))
    if not np.all(np.isfinite(data["transfer"])):
        raise AssertionError(f"non-finite targets in {path}")
    if not np.allclose(data["transfer"] + data["nontransfer"], 1.0, atol=1e-6):
        raise AssertionError(f"targets do not sum to one in {path}")
    if not np.allclose(data["transfer"], expected_a, atol=1e-6):
        raise AssertionError(f"target formula mismatch in {path}")
    if not np.allclose(data["target_temperature"], expected_t, atol=1e-6):
        raise AssertionError(f"temperature mismatch in {path}")
    ties = data["n_transfer"] == data["n_nontransfer"]
    if not np.array_equal(data["completion_is_tie_anchor"], ties):
        raise AssertionError(f"tie-anchor mismatch in {path}")


def _verify_split(source: Path, artifact: Path, split: str, intern: bool) -> int:
    path = artifact / split / "data.parquet"
    source_rows = _source_split(source, split)
    columns = ["prompt", "completion", "metadata.candidate_id",
               "metadata.binary_label", "metadata.schema_version",
               "metadata.target_policy_version"]
    rendered = pq.read_table(path, columns=columns)
    if rendered.column_names != ["prompt", "completion", "candidate_id", "binary_label",
                                  "schema_version", "target_policy_version"]:
        raise AssertionError(f"unexpected projected schema in {path}")
    if not rendered["candidate_id"].equals(source_rows["candidate_id"]):
        raise AssertionError(f"candidate membership/order mismatch in {path}")
    labels_match = np.array_equal(rendered["binary_label"].to_numpy(),
                                  source_rows["binary_label"].to_numpy())
    if split != "train" and not labels_match:
        raise AssertionError(f"held-out binary-label mismatch in {path}")
    completions = set(rendered["completion"].to_pylist())
    if completions - ({"(A)", "(B)"} if intern else {"A", "B"}):
        raise AssertionError(f"non-binary completion in {path}: {completions}")
    if pc.any(pc.match_substring(rendered["prompt"], "\n(C) ambiguous")).as_py():
        raise AssertionError(f"C choice leaked into prompts in {path}")
    if set(rendered["schema_version"].to_pylist()) != {"assay_transfer_variance_soft_v5"}:
        raise AssertionError(f"schema version mismatch in {path}")
    if set(rendered["target_policy_version"].to_pylist()) != {"variance_temperature_binary_v5"}:
        raise AssertionError(f"target policy mismatch in {path}")
    _verify_targets(path)
    return rendered.num_rows


def verify(source: Path, standard: Path, intern: Path, rebuild: Path | None) -> None:
    for artifact, is_intern in ((standard, False), (intern, True)):
        counts = {split: _verify_split(source, artifact, split, is_intern) for split in SPLITS}
        print(f"{artifact.name}: {counts}")
    if rebuild:
        for artifact in (standard, intern):
            other = rebuild / artifact.name
            for split in SPLITS:
                left = artifact / split / "data.parquet"
                right = other / split / "data.parquet"
                if _sha256(left) != _sha256(right):
                    raise AssertionError(f"nondeterministic rebuild: {left}")
        print("all six rebuild hashes match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--intern", type=Path, required=True)
    parser.add_argument("--rebuild-root", type=Path)
    args = parser.parse_args()
    verify(args.source, args.standard, args.intern, args.rebuild_root)


if __name__ == "__main__":
    main()
