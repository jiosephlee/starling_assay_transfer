from __future__ import annotations

from pipeline.v12_contract import load_registry, task_config
from pipeline.v12_ranking import build_ranking_anchors
from pipeline.v12_source import _fit_value_calibrations, build_distance_calibration
from pipeline.v12_targets import target_for
from scripts.build_v12_intern_raw_pair import eligible, pair_relationship


def _record(
    index: int, *, category: str = "positive", parent: str | None = None,
    split: str = "train", kind: str = "binary",
) -> dict:
    return {
        "child_id": f"record-{index}", "record_id": f"source-{index}",
        "task_id": "task", "source_id": "source", "measurement_kind": kind,
        "canonical_endpoint_key": "endpoint", "canonical_measurement_scale_id": "scale",
        "pair_bucket_key": "bucket", "canonical_category_id": category,
        "canonical_category_rank": 1.0 if category == "positive" else 0.0,
        "canonical_smiles": f"SMILES-{index}",
        "normalized_parent_identity_key": parent or f"parent-{index}",
        "dataset_split": split, "value_percentile": index / 100,
        "calibration_sample_standard_deviation": 1.0,
    }


def test_registry_uses_correct_gold_lineages_and_ranking_width() -> None:
    registry = load_registry()
    expected = {
        "bbb_martins": "processed_starling_experimental_meaningful_cns_access_v2",
        "bioavailability_ma": "processed_starling_record_supported_v2",
        "skin_reaction": "processed_starling_record_supported_v2",
    }
    for task_id, lineage in expected.items():
        task = task_config(task_id, registry)
        assert lineage in task["gold_lineage"]["root_relpath"]
        assert task["construction"]["ranking_anchor_width"] == 24
        assert set(task["construction"]["ranking_anchors"]) == {"validation", "test"}


def test_distinct_same_parent_and_same_smiles_records_are_eligible() -> None:
    left, right = _record(1, parent="same"), _record(2, parent="same")
    right["canonical_smiles"] = left["canonical_smiles"]
    assert eligible(left, right)
    assert not eligible(left, left)
    assert pair_relationship(left, right) == "same_parent_same_canonical_smiles_distinct_records"


def test_value_calibration_uses_only_train_records() -> None:
    rows = []
    for index in range(25):
        row = _record(index, kind="continuous")
        row["finite_scalar_value"] = float(index)
        rows.append(row)
    heldout = _record(99, kind="continuous", split="test")
    heldout["finite_scalar_value"] = 1_000_000.0
    rows.append(heldout)
    global_entry = {
        "measurement_kind": "continuous", "observed_sample_standard_deviation": 123.0,
        "standard_deviation_ddof": 1, "standard_deviation_value_field": "finite_scalar_value",
    }
    document, _ = _fit_value_calibrations(rows, {"bucket": global_entry})
    entry = document["buckets"]["bucket"]
    assert entry["record_count"] == 25
    assert entry["value_cdf"]["total_record_count"] == 25
    assert max(entry["value_cdf"]["support_values"]) == 24.0
    assert entry["observed_sample_standard_deviation"] == 123.0


def test_missing_global_sd_is_measurement_metadata_not_a_training_gate() -> None:
    rows = []
    for index in range(25):
        row = _record(index, kind="continuous")
        row["finite_scalar_value"] = float(index)
        rows.append(row)
    global_entry = {
        "measurement_kind": "continuous", "observed_sample_standard_deviation": None,
        "standard_deviation_ddof": 1, "standard_deviation_value_field": "finite_scalar_value",
    }
    document, calibrations = _fit_value_calibrations(rows, {"bucket": global_entry})
    assert "bucket" in document["buckets"]
    assert calibrations["bucket"].sample_standard_deviation is None
    query = {**rows[0], "value_percentile": 0.02, "geometry_value": 0.0,
             "calibration_sample_standard_deviation": None,
             "calibration_standard_deviation_ddof": 1,
             "calibration_standard_deviation_value_field": "finite_scalar_value"}
    retrieval = {**rows[1], "value_percentile": 0.08, "geometry_value": 1.0,
                 "calibration_sample_standard_deviation": None,
                 "calibration_standard_deviation_ddof": 1,
                 "calibration_standard_deviation_value_field": "finite_scalar_value"}
    assert target_for(query, retrieval)["calibration_sample_standard_deviation"] is None


def test_distance_calibration_counts_all_distinct_records_including_same_parent() -> None:
    rows = [
        _record(0, parent="same", kind="continuous"),
        _record(1, parent="same", kind="continuous"),
        _record(2, parent="other", kind="continuous"),
    ]
    document = build_distance_calibration(rows, "task")
    entry = document["buckets"]["bucket"]
    assert entry["distinct_record_unordered_pair_count"] == 3
    assert sum(entry["support_counts"]) == 3
    assert entry["distinct_parent_count"] == 2
    assert "including_same_parent" in entry["pair_scope"]


def _ranking_task(family: str) -> dict:
    other = "continuous" if family == "categorical" else "categorical"
    return {
        "construction": {
            "ranking_anchor_width": 24, "ranking_record_degree_cap": 48,
            "ranking_anchors": {
                "validation": {family: {"source": 1}, other: {"source": 0}},
                "test": {family: {"source": 1}, other: {"source": 0}},
            },
        },
        "priority_phases": [["source"]],
    }


def test_categorical_ranking_is_24_wide_with_minimum_category_and_parent_constraints() -> None:
    query = _record(100, category="positive", parent="heldout", split="validation")
    candidates = [
        _record(index, category="positive" if index < 3 else "negative",
                parent="p1" if index < 29 else "p2")
        for index in range(30)
    ]
    anchors, _ = build_ranking_anchors(
        [query], candidates, _ranking_task("categorical"), "validation"
    )
    assert len(anchors) == 1 and len(anchors[0].candidates) == 24
    categories = [row["canonical_category_id"] for row in anchors[0].candidates]
    assert categories.count("positive") >= 3 and "negative" in categories
    assert len({row["normalized_parent_identity_key"] for row in anchors[0].candidates}) >= 2


def test_continuous_ranking_does_not_cap_two_candidates_per_parent() -> None:
    query = _record(100, parent="heldout", split="test", kind="continuous")
    candidates = []
    for index in range(30):
        row = _record(index, parent="p1" if index < 29 else "p2", kind="continuous")
        row["geometry_value"] = float(index)
        row["value_percentile"] = index / 29
        candidates.append(row)
    query["geometry_value"], query["value_percentile"] = 15.0, 0.5
    target = lambda left, right: {"target_a": 1 - abs(left["value_percentile"] - right["value_percentile"])}
    anchors, _ = build_ranking_anchors(
        [query], candidates, _ranking_task("continuous"), "test", target
    )
    assert len(anchors) == 1 and len(anchors[0].candidates) == 24
    counts = {}
    for row in anchors[0].candidates:
        counts[row["normalized_parent_identity_key"]] = counts.get(row["normalized_parent_identity_key"], 0) + 1
    assert max(counts.values()) > 2 and set(counts) == {"p1", "p2"}
