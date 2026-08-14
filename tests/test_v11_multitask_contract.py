from __future__ import annotations

import ast
import copy
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.v11_contract import (
    categorical_ablation_config,
    eligible_projection_manifest,
    heldout_reservation_manifest,
    load_registry,
    project_prompt_record,
    validate_registry,
    validate_template_bindings,
)
from pipeline.v11_prompt_rendering import render_prompt, reset_prompt_cache
from pipeline.v11_ranking import build_ranking_anchors
from pipeline.v11_1_targets import (
    CALIBRATION_FIELD,
    REFERENCE_PAIR_LIMIT,
    attach_distance_calibrations,
    build_distance_calibration,
    parsed_calibrations,
    target_for as target_for_v11_1,
    write_distance_calibration,
)
from pipeline.v11_targets import (
    TARGET_EPSILON,
    geometry_value,
    record_percentile,
    target_for,
    value_calibration,
)
from pipeline.v3_policy import file_sha256
from pipeline.source_normalization import starling_txagent_eligible_v7 as eligible_v7
from scripts import build_v7_intern_raw_pair as v7
from scripts import build_raw_pair
from scripts import build_v11_intern_raw_pair as v11
from scripts import build_v11_categorical_ablation as ablation
from scripts import publish_v11_categorical_ablation as publish_ablation
from scripts import verify_v11_categorical_ablation as verify_ablation
from scripts.build_v11_1_from_defined_split import _assert_invariants


NEW_FILES = (
    "pipeline/v11_contract.py",
    "pipeline/v11_prompt_rendering.py",
    "pipeline/v11_ranking.py",
    "pipeline/v11_1_targets.py",
    "pipeline/v11_targets.py",
    "pipeline/source_normalization/starling_txagent_eligible_v7.py",
    "scripts/build_v11_intern_raw_pair.py",
    "scripts/build_v11_1_intern_raw_pair.py",
    "scripts/build_v11_1_categorical_release.py",
    "scripts/build_v11_1_from_defined_split.py",
    "scripts/verify_v11_intern_raw_pair.py",
    "scripts/preview_v11_contract.py",
    "scripts/build_v11_categorical_ablation.py",
    "scripts/verify_v11_categorical_ablation.py",
    "scripts/publish_v11_categorical_ablation.py",
    "scripts/verify_v11_1_intern_raw_pair.py",
    "scripts/verify_v11_1_categorical_release.py",
    "scripts/publish_v11_1_categorical_release.py",
)


def _prompt_pair() -> tuple[dict, dict]:
    base = {
        "task_id": "bioavailability_ma",
        "source_id": "fg",
        "smiles": "CCN",
        "endpoint_name": "efflux",
        "unit_text": None,
        "transporter_or_enzyme": "P-gp",
        "substrate_status": None,
        "assay_system": "Caco-2",
        "intestinal_site": float("nan"),
        "qualifying_conditions": None,
    }
    retrieval = {
        **base,
        "smiles": "CCO",
        "measurement_text": "2.1",
        "support_text": "retrieval evidence",
        "extra_details": "retrieval detail",
    }
    return base, retrieval


def _target_pair(
    query_percentile: float = 0.25, retrieval_percentile: float = 0.75,
    kind: str = "continuous", query_category: str = "low",
    retrieval_category: str = "high",
):
    query = {
        "pair_bucket_key": "bucket",
        "measurement_kind": kind,
        "canonical_measurement_scale_id": "scale",
        "geometry_value": 0.0,
        "value_percentile": query_percentile,
        "canonical_category_id": query_category,
        "calibration_sample_standard_deviation": 2.0,
        "calibration_standard_deviation_ddof": 1,
        "calibration_standard_deviation_value_field": "finite_scalar_value",
    }
    retrieval = {
        **query, "geometry_value": 1.0, "value_percentile": retrieval_percentile,
        "canonical_category_id": retrieval_category,
    }
    return query, retrieval


def test_registry_covers_three_tasks_and_thirteen_sources() -> None:
    registry = load_registry()
    assert set(registry["tasks"]) == {
        "bioavailability_ma", "bbb_martins", "skin_reaction",
    }
    assert sum(len(task["sources"]) for task in registry["tasks"].values()) == 13
    validate_registry(registry)
    validate_template_bindings(registry)


def test_release_has_one_variant_two_families_and_independent_caps() -> None:
    release = categorical_ablation_config(load_registry())
    assert release["evaluation_measurement_kinds"] == ["continuous", "binary", "ordinal"]
    assert release["ranking_families"] == {
        "continuous": ["continuous"], "categorical": ["binary", "ordinal"],
    }
    assert release["variants"] == {"with_categorical": ["continuous", "categorical"]}
    for component in release["train_components"].values():
        assert component["max_pairs"] == 1_250_000
        assert component["query_degree_cap"] == 6
        assert component["retrieval_degree_cap"] == 6


def test_train_query_cap_counts_every_emitted_pair(monkeypatch) -> None:
    query = {
        "child_id": "query", "assay_concept": "source", "pair_bucket_key": "bucket",
        "canonical_endpoint_key": "endpoint", "canonical_smiles": "Q",
    }
    retrievals = [
        {**query, "child_id": f"retrieval-{index}", "canonical_smiles": f"R{index}"}
        for index in range(4)
    ]
    monkeypatch.setattr(v7, "_condition_index_by_bucket", lambda _rows: {
        ("endpoint", "bucket"): retrievals,
    })
    monkeypatch.setattr(v7, "_scheduled_buckets", lambda _rows: (
        {"source": {"bucket": [query]}}, {"source": ["bucket"]},
    ))
    monkeypatch.setattr(v7, "propose_records", lambda *_args: retrievals)
    monkeypatch.setattr(v7, "target_for", lambda _query, row: {
        "target_z": float(row["child_id"].rsplit("-", 1)[-1]),
    })
    monkeypatch.setattr(v7, "row_for", lambda q, r, _split: {
        "query_record_id": q["child_id"], "retrieval_record_id": r["child_id"],
    })
    monkeypatch.setattr(v7, "_is_decisive", lambda *_args: True)
    monkeypatch.setattr(v7, "_fix_metadata", lambda _row: None)
    groups = list(v7._iter_list_groups_capped([query, *retrievals], "train", 3))
    rows = [row for group in groups for row in group]
    assert [len(group) for group in groups] == [4, 2]
    assert Counter(row["query_record_id"] for row in rows) == {"query": 6}


@pytest.mark.parametrize("repeated_column", ["query_record_id", "retrieval_record_id"])
def test_verifier_rejects_component_degree_over_cap(tmp_path, repeated_column) -> None:
    rows = []
    for index in range(7):
        row = {
            "pair_id": f"pair-{index}", "query_record_id": f"q-{index}",
            "retrieval_record_id": f"r-{index}", "measurement_kind": "continuous",
        }
        row[repeated_column] = "repeated"
        rows.append(row)
    path = tmp_path / "train" / "data.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path)
    digest, query_max, retrieval_max, kinds = verify_ablation._range_diagnostics(path, 0, 7)
    component = {
        "achieved_pairs": 7, "max_pairs": 10,
        "query_degree_cap": 6, "retrieval_degree_cap": 6,
        "observed_max_query_degree": query_max,
        "observed_max_retrieval_degree": retrieval_max,
        "pair_id_sha256": digest,
        "achieved_by_measurement_kind": dict(kinds),
    }
    manifest = {
        "rows": {"train": 7},
        "train_components": {
            "continuous": component,
            "categorical": {
                "achieved_pairs": 0, "max_pairs": 10,
                "query_degree_cap": 6, "retrieval_degree_cap": 6,
                "observed_max_query_degree": 0, "observed_max_retrieval_degree": 0,
                "pair_id_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "achieved_by_measurement_kind": {},
            },
        },
        "measurement_kind_policy": {"train_components": ["continuous", "categorical"]},
    }
    with pytest.raises(AssertionError, match="component degree cap exceeded"):
        verify_ablation._verify_components(tmp_path, manifest)


def test_categorical_train_filter_reserves_molecules_across_all_kinds() -> None:
    rows = [
        {"child_id": "continuous-heldout", "measurement_kind": "continuous", "canonical_smiles": "A"},
        {"child_id": "binary-heldout", "measurement_kind": "binary", "canonical_smiles": "A"},
        {"child_id": "ordinal-train", "measurement_kind": "ordinal", "canonical_smiles": "B"},
        {"child_id": "continuous-train", "measurement_kind": "continuous", "canonical_smiles": "C"},
    ]
    categorical = ablation._kinds(rows, ["binary", "ordinal"])
    kept = ablation._unreserved(categorical, {"A"})
    assert [row["child_id"] for row in kept] == ["ordinal-train"]


def test_every_task_uses_scaffold_valid_and_test_only() -> None:
    registry = load_registry()
    for task_id, task in registry["tasks"].items():
        reservation = task["heldout_reservation"]
        assert reservation["methodology"] == "scaffold"
        assert reservation["included_splits"] == ["valid", "test"]
        assert "/scaffold/" in reservation["labels_relpath"]
        assert "/random/" not in reservation["labels_relpath"]
        manifest = heldout_reservation_manifest(task_id, registry)
        assert set(manifest["split_counts"]) == {"valid", "test"}
        assert all(count > 0 for count in manifest["split_counts"].values())


def test_projection_hides_query_outcomes_and_canonical_fields() -> None:
    registry = load_registry()
    query, retrieval = _prompt_pair()
    query["measurement_text"] = "QUERY SECRET"
    query["canonical_smiles"] = "CANONICAL SECRET"
    projected = project_prompt_record(query, "bioavailability_ma", "query", registry)
    assert "measurement_text" not in projected
    assert "canonical_smiles" not in projected
    prompt = render_prompt(query, retrieval)
    assert "QUERY SECRET" not in prompt
    assert "CANONICAL SECRET" not in prompt
    assert "retrieval evidence" in prompt
    assert "nan" not in prompt.lower()


def test_eligible_prompt_projection_serializes_missing_text_as_null() -> None:
    registry = load_registry()
    row = {
        "source_id": "fg", "smiles": "CCO", "endpoint_name": "efflux",
        "transporter_or_enzyme": float("nan"), "assay_system": "Caco-2",
        "intestinal_site": float("nan"), "qualifying_conditions": None,
        "measurement_text": "2.1", "substrate_status": float("nan"),
        "support_text": "evidence", "extra_details": float("nan"),
    }
    projected = eligible_v7._prompt_source_values(row, "bioavailability_ma", registry)
    assert projected["assay_system"] == "Caco-2"
    assert projected["transporter_or_enzyme"] is None
    assert projected["extra_details"] is None
    assert eligible_v7._nullable(float("nan")) is None


def test_prompt_prelude_has_no_calibration_policy_language() -> None:
    prompt = render_prompt(*_prompt_pair())
    prelude = prompt.split("\n", 1)[0].lower()
    assert "transfer" in prelude
    assert not any(word in prelude for word in ("threshold", "percentile", "standard deviation", "sd"))


def test_intern_prompt_wraps_both_smiles_in_tags() -> None:
    prompt = render_prompt(*_prompt_pair())
    assert "<SMILES>CCN</SMILES>" in prompt
    assert "<SMILES>CCO</SMILES>" in prompt
    assert not any(line.startswith("SMILES:") for line in prompt.splitlines())


def test_outcome_like_context_is_retrieval_only() -> None:
    fg_query, fg_retrieval = _prompt_pair()
    fg_query.update(substrate_status="QUERY STATUS SECRET", unit_text="QUERY UNIT SECRET")
    fg_retrieval.update(substrate_status="substrate", unit_text="RETRIEVAL UNIT SECRET")
    fg_prompt = render_prompt(fg_query, fg_retrieval)
    assert "substrate status: substrate" in fg_prompt
    assert "QUERY STATUS SECRET" not in fg_prompt
    assert "UNIT SECRET" not in fg_prompt

    oral_query = {
        "task_id": "bioavailability_ma", "source_id": "oral_exposure",
        "smiles": "CCN", "endpoint_name": "bioavailability",
        "unit_text": "QUERY UNIT SECRET",
        "statistic_type": "mean", "oral_dose": "5 mg/kg", "study_context": "rat plasma",
        "qualifying_conditions": None, "comparator_exposure": "QUERY COMPARATOR SECRET",
    }
    oral_retrieval = {
        **oral_query, "smiles": "CCO", "measurement_text": "23.0", "unit_text": "%",
        "comparator_exposure": "IV reference", "support_text": "known evidence",
        "extra_details": None,
    }
    oral_prompt = render_prompt(oral_query, oral_retrieval)
    assert "comparator exposure: IV reference" in oral_prompt
    assert "QUERY COMPARATOR SECRET" not in oral_prompt
    assert "QUERY UNIT SECRET" not in oral_prompt
    assert "unit: %" in oral_prompt


def test_value_cdf_separation_is_the_v11_continuous_soft_target() -> None:
    exact = target_for(*_target_pair(0.25, 0.25))
    middle = target_for(*_target_pair(0.1, 0.6))
    opposite = target_for(*_target_pair(0.0, 1.0))
    assert exact["target_a"] == pytest.approx(1.0 - TARGET_EPSILON)
    assert middle["target_a"] == pytest.approx(0.5)
    assert opposite["target_a"] == pytest.approx(TARGET_EPSILON)
    assert all("target_z" not in result for result in (exact, middle, opposite))


def test_value_cdf_and_sample_sd_are_loaded_from_v7_calibration() -> None:
    entry = {
        "calibration_valid": True,
        "measurement_kind": "continuous",
        "observed_sample_standard_deviation": 2.0,
        "standard_deviation_ddof": 1,
        "standard_deviation_value_field": "finite_scalar_value",
        "value_cdf_valid": True,
        "value_cdf": {
            "support_values": [1.0, 2.0, 4.0],
            "support_counts": [1, 2, 1],
            "support_midranks_0_1": [0.125, 0.5, 0.875],
            "total_record_count": 4,
        },
    }
    calibration = value_calibration("bucket", entry)
    record = {"measurement_kind": "continuous", "finite_scalar_value": 2.0}
    assert calibration.sample_standard_deviation == 2.0
    assert record_percentile(record, calibration) == pytest.approx(0.5)
    record["finite_scalar_value"] = 3.0
    assert record_percentile(record, calibration) == pytest.approx(0.75)


def test_binary_soft_target_uses_category_agreement() -> None:
    same = target_for(*_target_pair(kind="binary", retrieval_category="low"))
    different = target_for(*_target_pair(kind="binary"))
    assert same["target_a"] == pytest.approx(0.95)
    assert different["target_a"] == pytest.approx(0.05)


def _cdf_row(index: int, percentile: float, molecule: str | None = None) -> dict:
    row, _ = _target_pair(percentile, percentile)
    row.update({
        "child_id": f"row-{index}", "task_id": "task",
        "canonical_smiles": molecule or f"C{index}", "value_percentile": percentile,
    })
    return row


def test_v11_1_anchors_the_percentile_distance_cdf_endpoints() -> None:
    rows = [_cdf_row(index, percentile) for index, percentile in enumerate((0.1, 0.2, 0.5, 0.9))]
    document = build_distance_calibration(rows, "task")
    attached, rejected = attach_distance_calibrations(rows, parsed_calibrations(document))
    assert rejected == {}
    same = target_for_v11_1(attached[0], attached[0] | {"canonical_smiles": "other"})
    far = target_for_v11_1(attached[0], attached[-1])
    assert same["target_a"] == pytest.approx(1.0 - TARGET_EPSILON)
    assert far["target_a"] == pytest.approx(TARGET_EPSILON)
    assert same["percentile_distance_cdf"] == pytest.approx(0.0)
    assert far["percentile_distance_cdf"] == pytest.approx(1.0)
    assert "target_z" not in same


def test_v11_1_reference_pairs_exclude_same_molecule() -> None:
    rows = [
        _cdf_row(0, 0.1, "A"), _cdf_row(1, 0.9, "A"),
        _cdf_row(2, 0.2, "B"), _cdf_row(3, 0.8, "C"),
    ]
    entry = next(iter(build_distance_calibration(rows, "task")["buckets"].values()))
    assert entry["admissible_unordered_pair_count"] == 5
    assert entry["reference_pair_count"] == 5
    assert entry["reference_method"] == "exact_unordered_cross_molecule"


def test_v11_1_large_buckets_use_one_hundred_thousand_pairs() -> None:
    rows = [_cdf_row(index, index / 447) for index in range(448)]
    entry = next(iter(build_distance_calibration(rows, "task")["buckets"].values()))
    assert REFERENCE_PAIR_LIMIT == 100_000
    assert entry["admissible_unordered_pair_count"] == 100_128
    assert entry["reference_method"] == "deterministic_hash_sample"
    assert entry["reference_pair_count"] == 100_000


def test_v11_1_calibration_gzip_is_deterministic(tmp_path) -> None:
    document = build_distance_calibration(
        [_cdf_row(index, percentile) for index, percentile in enumerate((0.1, 0.4, 0.9))],
        "task",
    )
    first, second = tmp_path / "first.json.gz", tmp_path / "second.json.gz"
    write_distance_calibration(document, first)
    write_distance_calibration(document, second)
    assert first.read_bytes() == second.read_bytes()


def test_frozen_membership_comparison_allows_only_target_derived_changes() -> None:
    reference = {
        "pair_id": "pair", "prompt": "same", "target_a": 0.2,
        "target_b": 0.8, "target_z": -1.0, "completion": "(B)",
    }
    rebuilt = {
        "pair_id": "pair", "prompt": "same", "target_a": 0.8,
        "target_b": 0.2, "completion": "(A)",
        "percentile_distance": 0.1, "percentile_distance_cdf": 0.2,
    }
    _assert_invariants(reference, rebuilt)
    rebuilt["prompt"] = "changed"
    with pytest.raises(ValueError, match="prompt"):
        _assert_invariants(reference, rebuilt)


def test_v11_1_ordinal_uses_cdf_but_binary_keeps_match_rule() -> None:
    rows = [_cdf_row(index, percentile) for index, percentile in enumerate((0.1, 0.4, 0.9))]
    for row in rows:
        row["measurement_kind"] = "ordinal"
        row["canonical_category_id"] = f"level-{row['child_id']}"
    document = build_distance_calibration(rows, "task")
    attached, _ = attach_distance_calibrations(rows, parsed_calibrations(document))
    assert CALIBRATION_FIELD in attached[0]
    assert target_for_v11_1(attached[0], attached[-1])["target_a"] == pytest.approx(TARGET_EPSILON)
    binary_query, binary_retrieval = _target_pair(kind="binary", retrieval_category="low")
    result = target_for_v11_1(binary_query, binary_retrieval)
    assert result["target_a"] == pytest.approx(0.95)
    assert result["percentile_distance"] is None


def test_geometry_supports_continuous_binary_and_ordinal() -> None:
    assert geometry_value({"measurement_kind": "continuous", "finite_scalar_value": 2.5}) == 2.5
    assert geometry_value({"measurement_kind": "binary", "canonical_category_rank": 1}) == 1.0
    assert geometry_value({"measurement_kind": "ordinal", "canonical_category_rank": 3}) == 3.0


def test_construction_contract_uses_width_twenty_sampling_and_new_quotas() -> None:
    registry = load_registry()
    direct_sources = {
        "bioavailability_ma": "hf_bioavailability",
        "bbb_martins": "direct_bbb",
        "skin_reaction": "direct_skin_reaction",
    }
    task = registry["tasks"]["bioavailability_ma"]
    config = v11.construction_config("bioavailability_ma", registry)
    assert task["priority_phases"] == [
        ["fg", "fh"],
        ["hf_bioavailability", "oral_exposure", "fa"],
    ]
    assert config["ranking_anchor_width"] == 20
    assert config["ranking_record_degree_cap"] == 48
    assert config["source_record_sampling"] == {
        "oral_exposure": {"numerator": 1, "denominator": 2, "method": "stable_hash_exact"}
    }
    assert config["ranking_anchors"]["continuous"]["hf_bioavailability"] == 40
    assert config["ranking_anchors"]["categorical"]["hf_bioavailability"] == 40
    assert config["ranking_anchors"]["continuous"]["fg"] == 10
    for task_id, task in registry["tasks"].items():
        construction = task["construction"]
        assert construction["ranking_anchor_width"] == 20
        for family in ("continuous", "categorical"):
            quotas = construction["ranking_anchors"][family]
            direct = direct_sources[task_id]
            assert quotas[direct] == 40
            for source in set(task["sources"]) - {direct}:
                expected = 10 if task_id == "bioavailability_ma" and source == "fg" else 20
                assert quotas[source] == expected


def test_sft_metadata_contains_soft_targets_groups_and_geometry() -> None:
    row = {
        "target_distribution": {"transfer": 0.7, "nontransfer": 0.3},
        "assay_concept": "direct_bbb",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "measurement_kind": "binary",
        "absolute_geometry_difference": 1.0,
        "calibration_sample_standard_deviation": 2.0,
    }
    v11._fix_metadata_v11(row)
    assert row["metadata"]["soft_targets"] == {"(A)": 0.7, "(B)": 0.3}
    assert row["metadata"]["groups"] == {
        "assay_concept": "direct_bbb",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "measurement_kind": "binary",
    }
    assert row["metadata"]["absolute_geometry_difference"] == 1.0
    assert row["metadata"]["calibration_sample_standard_deviation"] == 2.0
    assert row["metadata"]["target_contract"].startswith("within_pair_bucket_value_cdf")
    assert "target_z" not in row["metadata"]


def _ranking_record(index: int, kind: str, category: str | None = None) -> dict:
    percentile = index / 79 if kind == "continuous" else None
    return {
        "child_id": f"{kind}-{index}", "source_id": "source",
        "measurement_kind": kind, "canonical_endpoint_key": "endpoint",
        "canonical_measurement_scale_id": "scale", "pair_bucket_key": "bucket",
        "canonical_smiles": f"C{index}{kind[0]}", "geometry_value": float(index),
        "value_percentile": percentile, "canonical_category_id": category,
        "calibration_sample_standard_deviation": 2.0,
        "calibration_standard_deviation_ddof": 1,
        "calibration_standard_deviation_value_field": "finite_scalar_value",
    }


def test_ranking_builds_separate_continuous_and_categorical_anchors() -> None:
    rows = [_ranking_record(index, "continuous") for index in range(80)]
    for category_index, category in enumerate(("low", "middle", "high")):
        rows.extend(
            _ranking_record(100 + category_index * 30 + offset, "binary", category)
            for offset in range(30)
        )
    task = {
        "priority_phases": [["source"]],
        "construction": {
            "ranking_anchor_width": 20, "ranking_record_degree_cap": 48,
            "ranking_anchors": {
                "continuous": {"source": 1}, "categorical": {"source": 1},
            },
        },
    }
    release = {"ranking_families": {"continuous": ["continuous"], "categorical": ["binary"]}}
    anchors, diagnostics = build_ranking_anchors(rows, task, release)
    assert Counter(anchor.family for anchor in anchors) == {"continuous": 1, "categorical": 1}
    categorical = next(anchor for anchor in anchors if anchor.family == "categorical")
    query_category = categorical.query["canonical_category_id"]
    same = sum(row["canonical_category_id"] == query_category for row in categorical.candidates)
    assert (len(categorical.candidates), same) == (20, 5)
    assert not diagnostics["families"]["continuous"]["shortfalls"]
    assert not diagnostics["families"]["categorical"]["shortfalls"]


def test_oral_exposure_sampling_is_exact_and_deterministic() -> None:
    rows = [
        {"child_id": f"oral-{index}", "source_id": "oral_exposure"}
        for index in range(9)
    ] + [{"child_id": "fa-0", "source_id": "fa"}]
    task = {"construction": {"source_record_sampling": {
        "oral_exposure": {
            "numerator": 1, "denominator": 2, "method": "stable_hash_exact",
        }
    }}}
    first, report = ablation._sample_configured_sources(rows, task, "bioavailability_ma")
    second, _ = ablation._sample_configured_sources(list(reversed(rows)), task, "bioavailability_ma")
    first_ids, second_ids = {row["child_id"] for row in first}, {row["child_id"] for row in second}
    assert first_ids == second_ids
    assert "fa-0" in first_ids
    assert report["oral_exposure"]["retained_records"] == 4


def test_parquet_flush_preserves_string_type_for_all_null_later_batch(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    writer = build_raw_pair._flush(path, None, [{"scale": "unit"}])
    writer = build_raw_pair._flush(path, writer, [{"scale": None}])
    writer.close()
    table = pq.read_table(path)
    assert table.schema.field("scale").type == pa.string()
    assert table["scale"].to_pylist() == ["unit", None]


def test_existing_eligible_source_requires_matching_provenance(tmp_path) -> None:
    registry = load_registry()
    source = tmp_path / "records.parquet"
    source.write_bytes(b"first")
    manifest = {
        "stage": "starling_txagent_eligible_v7",
        "task_id": "bioavailability_ma",
        "eligible_records_sha256": v11.file_sha256(source),
        "eligible_projection": eligible_projection_manifest("bioavailability_ma", registry),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert v11._prepare_source("bioavailability_ma", source, None, registry) == source
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        v11._prepare_source("bioavailability_ma", source, None, registry)


def test_eligible_projection_digest_tracks_bindings_and_constants_only() -> None:
    registry = load_registry()
    baseline = eligible_projection_manifest("bioavailability_ma", registry)
    changed = copy.deepcopy(registry)
    changed["tasks"]["bioavailability_ma"]["sources"]["fa"]["constants"]["test"] = "x"
    assert eligible_projection_manifest("bioavailability_ma", changed) != baseline
    construction = copy.deepcopy(registry)
    construction["tasks"]["bioavailability_ma"]["construction"][
        "ranking_record_degree_cap"
    ] += 1
    assert eligible_projection_manifest("bioavailability_ma", construction) == baseline


@pytest.mark.parametrize("source_exists", [False, True])
def test_source_regeneration_receives_selected_registry(monkeypatch, tmp_path, source_exists) -> None:
    registry = copy.deepcopy(load_registry())
    registry["custom_registry_marker"] = "selected"
    source = tmp_path / "records.parquet"
    if source_exists:
        source.write_bytes(b"stale")
        manifest = {
            "stage": "starling_txagent_eligible_v7", "task_id": "bioavailability_ma",
            "eligible_records_sha256": file_sha256(source), "eligible_projection": {},
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def fake_write(task_id, output_dir, *, artifact_root=None, registry=None):
        calls.append((task_id, artifact_root, registry["custom_registry_marker"]))
        source.write_bytes(b"fresh")
        manifest = {
            "stage": "starling_txagent_eligible_v7", "task_id": task_id,
            "eligible_records_sha256": file_sha256(source),
            "eligible_projection": eligible_projection_manifest(task_id, registry),
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(v11, "write_eligible_records", fake_write)
    assert v11._prepare_source("bioavailability_ma", source, tmp_path, registry) == source
    assert calls == [("bioavailability_ma", tmp_path, "selected")]


def test_file_sha256_detects_same_size_rewrite_with_restored_mtime(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    status = path.stat()
    before = file_sha256(path)
    path.write_bytes(b"other")
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert path.stat().st_size == status.st_size
    assert path.stat().st_mtime_ns == status.st_mtime_ns
    assert file_sha256(path) != before


def test_stale_upstream_inputs_detects_regenerated_txagent_artifacts(tmp_path) -> None:
    """Skipping the eligible rebuild is only safe while the upstream artifacts are unchanged."""
    manifest_without_provenance = {"stage": "starling_txagent_eligible_v7"}
    assert v11._stale_upstream_inputs(
        manifest_without_provenance, "bioavailability_ma", tmp_path
    ) == []

    artifacts = {}
    for name in ("records", "pair_buckets", "pair_bucket_metadata", "calibration", "manifest"):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = path
    recorded = {
        name: {"path": str(path), "sha256": v11.file_sha256(path)}
        for name, path in artifacts.items()
    }
    manifest = {"inputs": recorded}

    def fake_artifact_paths(_root):
        return artifacts

    original = v11.artifact_paths
    v11.artifact_paths = fake_artifact_paths
    try:
        assert v11._stale_upstream_inputs(manifest, "bioavailability_ma", tmp_path) == []
        artifacts["records"].write_bytes(b"regenerated upstream")
        assert v11._stale_upstream_inputs(
            manifest, "bioavailability_ma", tmp_path
        ) == ["records"]
        artifacts["calibration"].unlink()
        assert v11._stale_upstream_inputs(
            manifest, "bioavailability_ma", tmp_path
        ) == ["calibration", "records"]
    finally:
        v11.artifact_paths = original


def _single_pass_render(query: dict, retrieval: dict, registry) -> str:
    """Oracle: render the whole template in one pass, the way v11 did before block caching."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    from pipeline.v11_contract import TEMPLATE_ROOT, source_config
    from pipeline.v11_prompt_rendering import _clean_projected

    task_id = str(query["task_id"])
    config = source_config(task_id, str(query["source_id"]), registry)
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_ROOT)),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(config["template"]).render(
        query=_clean_projected(project_prompt_record(query, task_id, "query", registry)),
        retrieval=_clean_projected(
            project_prompt_record(retrieval, task_id, "retrieval", registry)
        ),
    ).strip()


def test_block_cached_render_matches_single_pass_render_for_every_template() -> None:
    """The per-record block cache is a pure speedup: prompts must stay byte-identical."""
    registry = load_registry()
    values = [None, "", "  ", "x", "12.5 ± 0.7", "%", "__unknown__", "a|__unknown__|b",
              float("nan"), 0, "multi\nline"]
    checked = 0
    for task_id, task in registry["tasks"].items():
        for source_id, config in task["sources"].items():
            fields = sorted(
                set(config["both"]) | set(config["retrieval_only"])
                | set(config.get("constants", {}))
            )
            for trial in range(40):
                records = []
                for side in (0, 1):
                    record = {"task_id": task_id, "source_id": source_id,
                              "child_id": f"{task_id}:{source_id}:{trial}:{side}"}
                    for index, field in enumerate(fields):
                        record[field] = values[(trial + index + side) % len(values)]
                    record.update(config.get("constants", {}))
                    records.append(record)
                reset_prompt_cache()
                expected = _single_pass_render(records[0], records[1], registry)
                assert render_prompt(records[0], records[1]) == expected
                # again, now served from the cache
                assert render_prompt(records[0], records[1]) == expected
                checked += 1
    assert checked == 13 * 40


def test_every_configured_unit_is_retrieval_only() -> None:
    registry = load_registry()
    checked = 0
    for task_id, task in registry["tasks"].items():
        for source_id, config in task["sources"].items():
            assert "unit_text" not in config["both"]
            assert "unit_text" not in config.get("constants", {})
            if "unit_text" not in config["retrieval_only"]:
                continue
            query = {"task_id": task_id, "source_id": source_id,
                     "child_id": f"{task_id}:{source_id}:query", "unit_text": "QUERY UNIT SECRET"}
            retrieval = {"task_id": task_id, "source_id": source_id,
                         "child_id": f"{task_id}:{source_id}:retrieval",
                         "unit_text": "RETRIEVAL UNIT"}
            for field in set(config["both"]) | set(config["retrieval_only"]):
                query.setdefault(field, f"query-{field}")
                retrieval.setdefault(field, f"retrieval-{field}")
            prompt = render_prompt(query, retrieval)
            assert "QUERY UNIT SECRET" not in prompt
            assert "unit: RETRIEVAL UNIT" in prompt
            checked += 1
    assert checked == 12


def test_retrieval_only_constants_are_side_specific() -> None:
    registry = load_registry()
    config = {
        "both": ["smiles"], "retrieval_only": [], "constants": {"endpoint_name": "x"},
        "retrieval_only_constants": {"unit_text": "%"},
    }
    row = {"source_id": "unused", "smiles": "CCO"}
    query = project_prompt_record(row, "bioavailability_ma", "query", registry, config=config)
    retrieval = project_prompt_record(
        row, "bioavailability_ma", "retrieval", registry, config=config
    )
    assert query == {"smiles": "CCO", "endpoint_name": "x"}
    assert retrieval == {"smiles": "CCO", "endpoint_name": "x", "unit_text": "%"}


def _coverage_registry(*, expected_null: bool = False) -> dict:
    config = {
        "both": ["smiles"], "retrieval_only": ["unit_text", "measurement_text"],
        "constants": {},
    }
    if expected_null:
        config["expected_all_null_fields"] = ["unit_text"]
    return {"tasks": {"task": {"sources": {"source": config}}}}


def test_prompt_coverage_allows_empty_sources_and_declared_null_units() -> None:
    registry = _coverage_registry(expected_null=True)
    empty = eligible_v7._prompt_field_coverage([], "task", registry)
    assert empty["source"]["status"] == "no_eligible_records"
    row = {"source_id": "source", "smiles": "CCO", "unit_text": None,
           "measurement_text": "3.2"}
    populated = eligible_v7._prompt_field_coverage([row], "task", registry)
    assert populated["source"]["status"] == "validated"
    assert populated["source"]["fields"]["unit_text"] == 0


def test_prompt_coverage_rejects_unexpected_empty_or_stale_fields() -> None:
    row = {"source_id": "source", "smiles": "CCO", "unit_text": None,
           "measurement_text": "3.2"}
    with pytest.raises(ValueError, match="zero coverage"):
        eligible_v7._prompt_field_coverage([row], "task", _coverage_registry())
    row["unit_text"] = "%"
    with pytest.raises(ValueError, match="became populated"):
        eligible_v7._prompt_field_coverage(
            [row], "task", _coverage_registry(expected_null=True)
        )
    row.update(unit_text=None, measurement_text=None)
    with pytest.raises(ValueError, match="measurement_text"):
        eligible_v7._prompt_field_coverage(
            [row], "task", _coverage_registry(expected_null=True)
        )


def test_registry_rejects_non_unit_all_null_exceptions() -> None:
    registry = copy.deepcopy(load_registry())
    registry["tasks"]["bioavailability_ma"]["sources"]["fa"][
        "expected_all_null_fields"
    ] = ["endpoint_name"]
    with pytest.raises(ValueError, match="non-unit"):
        validate_registry(registry, tasks=("bioavailability_ma",))


def test_explicit_eligible_schema_preserves_null_first_row_fields() -> None:
    registry = _coverage_registry(expected_null=True)
    source_schema = pa.schema([
        pa.field("smiles", pa.string()), pa.field("unit_text", pa.string()),
        pa.field("measurement_text", pa.string()),
    ])
    schema = eligible_v7.eligible_schema("task", registry, source_schema)
    rows = [
        {"smiles": "CCO", "unit_text": None, "measurement_text": "1"},
        {"smiles": "CCN", "unit_text": "%", "measurement_text": "2"},
    ]
    table = eligible_v7.eligible_table(rows, schema)
    assert table.schema.equals(schema, check_metadata=False)
    assert table.column("unit_text").to_pylist() == [None, "%"]


def test_explicit_eligible_schema_rejects_missing_source_fields() -> None:
    registry = _coverage_registry(expected_null=True)
    source_schema = pa.schema([pa.field("smiles", pa.string())])
    with pytest.raises(ValueError, match="source parquet lacks prompt fields"):
        eligible_v7.eligible_schema("task", registry, source_schema)


def test_publish_tasks_limits_release_scope(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(publish_ablation, "HfApi", object)
    monkeypatch.setattr(
        publish_ablation, "_publish_one",
        lambda _api, root, repo: calls.append((root, repo)) or f"upload:{repo}",
    )
    monkeypatch.setattr(
        publish_ablation, "_verify_remote",
        lambda _api, _root, repo: {"repo_id": repo, "commit": f"remote:{repo}"},
    )
    releases = publish_ablation.publish_tasks(tmp_path, ["bioavailability_ma"])
    assert len(releases) == 1
    assert {root for root, _repo in calls} == {
        tmp_path / "bioavailability_ma" / ablation.VARIANT
    }
    assert all("bioavailability-ma" in repo for _root, repo in calls)


def test_every_new_function_is_at_most_sixty_lines() -> None:
    for filename in NEW_FILES:
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 60, f"{filename}:{node.name}"
