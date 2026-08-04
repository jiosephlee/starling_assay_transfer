#!/usr/bin/env python3
"""Exhaustively verify the V10 Intern raw-pair release and its accepted Fg shortfall."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.source_normalization.starling_txagent_eligible_v6 import _heldout_union_molecules
from pipeline.v3_policy import file_sha256
import scripts.build_v8_intern_raw_pair as v8
import scripts.build_v10_intern_raw_pair as v10
import scripts.verify_raw_pair as base

SPLITS = ("train", "validation", "validation_ranking", "test", "test_ranking")
EXPECTED_ROWS = {
    "train": 951439, "validation": 900, "validation_ranking": 2120,
    "test": 900, "test_ranking": 4400,
}
EXPECTED_RECORDS = {"train": 158747, "validation": 38235, "test": 38447}
EXPECTED_MOLECULES = {"train": 17858, "validation": 466, "test": 470}
EXPECTED_RANKING = {
    "validation": {"Fg": 6, "Fh": 25, "Fa": 25, "oral_bioavailability": 25, "oral_exposure": 25},
    "test": {"Fg": 20, "Fh": 50, "Fa": 50, "oral_bioavailability": 50, "oral_exposure": 50},
}
SOURCE_SHA256 = "5e520b2564e92d4c7707d2a7bdc9f0acccfe7a803a2b2703e3c3008656310be9"
HUB_DATASET_ID = "jiosephlee/assay-transfer-raw-pair-v10-intern"
CALIBRATION_SHA256 = "860082eb4af851ea013542c8b6219eb5d412ffb15b26aac312cd637e7a14cdf7"


def _load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != "v10-txagent-raw-pair":
        raise AssertionError("manifest does not identify the V10 TxAgent raw-pair release")
    return manifest


def _verify_manifest(root: Path, manifest: dict) -> None:
    if manifest.get("hub_dataset_id") != HUB_DATASET_ID:
        raise AssertionError("unexpected Hub dataset id")
    if manifest.get("eligible_source_sha256") != SOURCE_SHA256:
        raise AssertionError("manifest source hash mismatch")
    if manifest.get("rows") != EXPECTED_ROWS:
        raise AssertionError(f"row-count mismatch: {manifest.get('rows')}")
    if manifest.get("record_counts") != EXPECTED_RECORDS:
        raise AssertionError(f"source split-count mismatch: {manifest.get('record_counts')}")
    if manifest.get("split_quotas", {}).get("molecules") != EXPECTED_MOLECULES:
        raise AssertionError("manifest molecule quotas mismatch")
    diagnostics = manifest.get("diagnostics", {})
    if diagnostics.get("ranking_anchors_achieved") != EXPECTED_RANKING:
        raise AssertionError("manifest ranking achievement mismatch")
    if diagnostics.get("ranking_anchors_target", {}).get("test", {}).get("Fg") != 20:
        raise AssertionError("manifest does not carry the 20-anchor Fg test target")
    calibration = manifest.get("target_calibration", {})
    if calibration.get("artifact_sha256") != CALIBRATION_SHA256:
        raise AssertionError("manifest calibration hash mismatch")
    if calibration.get("percentiles") != [5, 95]:
        raise AssertionError("manifest calibration percentiles mismatch")
    if calibration.get("anchor_probabilities") != [0.95, 0.05]:
        raise AssertionError("manifest calibration probabilities mismatch")
    if calibration.get("sigmoid_tails_clipped") is not False:
        raise AssertionError("manifest does not identify unrestricted sigmoid tails")
    if file_sha256(root / "pairwise_distance_quantiles.json") != CALIBRATION_SHA256:
        raise AssertionError("release calibration artifact hash mismatch")
    for split in SPLITS:
        path = root / split / "data.parquet"
        if pq.ParquetFile(path).metadata.num_rows != EXPECTED_ROWS[split]:
            raise AssertionError(f"{split}: physical row count mismatch")
        if file_sha256(path) != manifest["parquet_sha256"][split]:
            raise AssertionError(f"{split}: Parquet hash mismatch")


def _ordinary_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"transfer": 0, "nontransfer": 0})
    for row in rows:
        label = "transfer" if row["binary_label"] == 1 else "nontransfer"
        counts[str(row["assay_concept"])][label] += 1
    return {concept: dict(labels) for concept, labels in counts.items()}


def _expected_ordinary() -> dict[str, dict[str, int]]:
    return {
        concept: {"transfer": target, "nontransfer": target}
        for concept, target in v8.ORDINARY_PER_CONCEPT_LABEL.items()
    }


def _ranking_counts(rows: list[dict]) -> dict[str, int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["ranking_query_id"])].append(row)
    for query_id, group in groups.items():
        if len(group) != v8.RANKING_ANCHOR_WIDTH:
            raise AssertionError(f"{query_id}: ranking width is {len(group)}")
        if {row["ranking_member_index"] for row in group} != set(range(v8.RANKING_ANCHOR_WIDTH)):
            raise AssertionError(f"{query_id}: ranking member indexes are incomplete")
        if len({row["query_record_id"] for row in group}) != 1:
            raise AssertionError(f"{query_id}: multiple ranking query records")
        if len({row["assay_concept"] for row in group}) != 1:
            raise AssertionError(f"{query_id}: multiple ranking concepts")
    return dict(Counter(str(group[0]["assay_concept"]) for group in groups.values()))


def _unordered_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((str(row["query_record_id"]), str(row["retrieval_record_id"]))))
        for row in rows
    }


def _verify_eval_split(root: Path, split: str, source: dict[str, dict]) -> None:
    ordinary = base._rows(root, split)
    ranking = base._rows(root, split + "_ranking")
    for subset, rows in ((split, ordinary), (split + "_ranking", ranking)):
        if len({row["pair_id"] for row in rows}) != len(rows):
            raise AssertionError(f"{subset}: duplicate oriented pair")
        for row in rows:
            base._verify_row(row, source)
            expected_decisive = v10._is_decisive_v10(row, source[row["query_record_id"]])
            if row["is_decisive"] != expected_decisive:
                raise AssertionError(f"{subset}: is_decisive disagrees with target_a")
    for row in ordinary:
        target_a, label = float(row["target_a"]), int(row["binary_label"])
        if label == 1 and target_a <= v10.ORDINARY_TRANSFER_MIN_PROBABILITY:
            raise AssertionError(f"{split}: transfer row is not above the confidence margin")
        if label == 0 and target_a >= v10.ORDINARY_NONTRANSFER_MAX_PROBABILITY:
            raise AssertionError(f"{split}: nontransfer row is not below the confidence margin")
        expected_completion = "(A)" if label == 1 else "(B)"
        if row["completion"] != expected_completion:
            raise AssertionError(f"{split}: completion disagrees with binary label")
    if _unordered_pairs(ordinary) & _unordered_pairs(ranking):
        raise AssertionError(f"{split}: ordinary/ranking pair overlap")
    if _ordinary_counts(ordinary) != _expected_ordinary():
        raise AssertionError(f"{split}: ordinary concept/label counts mismatch")
    if _ranking_counts(ranking) != EXPECTED_RANKING[split]:
        raise AssertionError(f"{split}: ranking anchor counts mismatch")


def _pair_molecules(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["query_smiles", "retrieval_smiles"])
    return set(table["query_smiles"].to_pylist()) | set(table["retrieval_smiles"].to_pylist())


def _verify_membership(root: Path, source: dict[str, dict], manifest: dict) -> None:
    source_rows = list(source.values())
    reserved = _heldout_union_molecules(source_rows)
    train_pairs = _pair_molecules(root / "train" / "data.parquet")
    validation = (_pair_molecules(root / "validation" / "data.parquet") |
                  _pair_molecules(root / "validation_ranking" / "data.parquet"))
    test = (_pair_molecules(root / "test" / "data.parquet") |
            _pair_molecules(root / "test_ranking" / "data.parquet"))
    if len(reserved) != 852 or not validation <= reserved or not test <= reserved:
        raise AssertionError("evaluation molecules escape the 852-molecule reserved pool")
    if train_pairs & (reserved | validation | test):
        raise AssertionError("generated train pairs leak reserved/evaluation molecules")
    train_source = {str(row["canonical_smiles"]) for row in source_rows} - reserved
    molecule_counts = {"train": len(train_source), "validation": len(validation), "test": len(test)}
    record_counts = {
        "train": sum(str(row["canonical_smiles"]) not in reserved for row in source_rows),
        "validation": sum(str(row["canonical_smiles"]) in validation for row in source_rows),
        "test": sum(str(row["canonical_smiles"]) in test for row in source_rows),
    }
    diagnostics = manifest["diagnostics"]
    if molecule_counts != EXPECTED_MOLECULES or record_counts != EXPECTED_RECORDS:
        raise AssertionError("reconstructed split membership counts mismatch")
    if len(validation & test) != diagnostics["validation_test_molecule_overlap"]:
        raise AssertionError("validation/test molecule overlap mismatch")
    if len(reserved - (validation | test)) != diagnostics["reserved_but_never_touched_molecules"]:
        raise AssertionError("untouched reserved-molecule count mismatch")


def verify(root: Path, source_path: Path, calibration_path: Path) -> None:
    if file_sha256(source_path) != SOURCE_SHA256:
        raise AssertionError("eligible source hash mismatch")
    manifest = _load_manifest(root)
    _verify_manifest(root, manifest)
    v10._configure_v10_globals(calibration_path)
    source = base._source_rows(source_path)
    base.render_prompt = v8._render_prompt_v8
    base.target_for = v10._target_for_v10
    base._verify_train_flat(root, source)
    _verify_eval_split(root, "validation", source)
    _verify_eval_split(root, "test", source)
    _verify_membership(root, source, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v10_intern"))
    parser.add_argument("--source", type=Path,
                        default=Path("datasets/eligible/assay_transfer_starling_txagent_v9/records.parquet"))
    parser.add_argument("--calibration", type=Path, default=v10.CALIBRATION_PATH)
    args = parser.parse_args()
    verify(args.root, args.source, args.calibration)
    print("V10 Intern raw-pair release verification passed")


if __name__ == "__main__":
    main()
