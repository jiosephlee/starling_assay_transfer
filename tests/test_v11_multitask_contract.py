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
from pipeline.v11_targets import geometry_value, raw_calibration, target_for
from pipeline.v3_policy import file_sha256
from pipeline.source_normalization import starling_txagent_eligible_v7 as eligible_v7
from scripts import build_v7_intern_raw_pair as v7
from scripts import build_v11_intern_raw_pair as v11
from scripts import build_v11_categorical_ablation as ablation
from scripts import publish_v11_categorical_ablation as publish_ablation
from scripts import verify_v11_categorical_ablation as verify_ablation


NEW_FILES = (
    "pipeline/v11_contract.py",
    "pipeline/v11_prompt_rendering.py",
    "pipeline/v11_targets.py",
    "pipeline/source_normalization/starling_txagent_eligible_v7.py",
    "scripts/build_v11_intern_raw_pair.py",
    "scripts/verify_v11_intern_raw_pair.py",
    "scripts/preview_v11_contract.py",
    "scripts/build_v11_categorical_ablation.py",
    "scripts/verify_v11_categorical_ablation.py",
    "scripts/publish_v11_categorical_ablation.py",
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


def _target_pair(distance: float, lower: float = 2.0, upper: float = 8.0):
    query = {
        "pair_bucket_key": "bucket",
        "geometry_value": 10.0,
        "calibration_distance_p05": lower,
        "calibration_distance_p95": upper,
    }
    retrieval = {**query, "geometry_value": 10.0 + distance}
    return query, retrieval


def test_registry_covers_three_tasks_and_thirteen_sources() -> None:
    registry = load_registry()
    assert set(registry["tasks"]) == {
        "bioavailability_ma", "bbb_martins", "skin_reaction",
    }
    assert sum(len(task["sources"]) for task in registry["tasks"].values()) == 13
    validate_registry(registry)
    validate_template_bindings(registry)


def test_categorical_ablation_release_is_additive_and_independently_capped() -> None:
    release = categorical_ablation_config(load_registry())
    assert release["evaluation_measurement_kinds"] == ["continuous"]
    assert release["categorical_measurement_kinds"] == ["binary", "ordinal"]
    assert release["variants"] == {
        "continuous_only": ["continuous"],
        "with_categorical": ["continuous", "categorical"],
    }
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
        row = {"query_record_id": f"q-{index}", "retrieval_record_id": f"r-{index}"}
        row[repeated_column] = "repeated"
        rows.append(row)
    path = tmp_path / "train" / "data.parquet"
    path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(rows), path)
    component = {
        "achieved_pairs": 7, "max_pairs": 10,
        "query_degree_cap": 6, "retrieval_degree_cap": 6,
        "observed_max_query_degree": 7 if repeated_column == "query_record_id" else 1,
        "observed_max_retrieval_degree": 7 if repeated_column == "retrieval_record_id" else 1,
    }
    manifest = {
        "rows": {"train": 7}, "train_components": {"continuous": component},
        "measurement_kind_policy": {"train_components": ["continuous"]},
    }
    with pytest.raises(AssertionError, match=f"rows exceed {repeated_column.split('_')[0]}"):
        verify_ablation._verify_caps(tmp_path, manifest)


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


def test_unrestricted_sigmoid_maps_anchors_and_has_open_tails() -> None:
    at_lower = target_for(*_target_pair(2.0))["target_a"]
    at_middle = target_for(*_target_pair(5.0))["target_a"]
    at_upper = target_for(*_target_pair(8.0))["target_a"]
    below = target_for(*_target_pair(0.0))["target_a"]
    above = target_for(*_target_pair(20.0))["target_a"]
    assert at_lower == pytest.approx(0.95)
    assert at_middle == pytest.approx(0.5)
    assert at_upper == pytest.approx(0.05)
    assert 0.95 < below < 1.0
    assert 0.0 < above < 0.05


def test_raw_geometry_reconstruction_matches_standardized_math() -> None:
    knots = [0.0] * 101
    knots[5], knots[95] = 1.5, 4.0
    entry = {
        "calibration_valid": True,
        "measurement_kind": "continuous",
        "distance_calibration": {
            "sample_standard_deviation": 2.0,
            "standardized_distance_percentile_knots": knots,
        },
    }
    calibration = raw_calibration("bucket", entry)
    raw = target_for(*_target_pair(5.0, calibration.distance_p05, calibration.distance_p95))
    standardized = target_for(*_target_pair(2.5, 1.5, 4.0))
    assert calibration.distance_p05 == 3.0
    assert calibration.distance_p95 == 8.0
    assert raw["target_a"] == pytest.approx(standardized["target_a"])


def test_geometry_supports_continuous_binary_and_ordinal() -> None:
    assert geometry_value({"measurement_kind": "continuous", "finite_scalar_value": 2.5}) == 2.5
    assert geometry_value({"measurement_kind": "binary", "canonical_category_rank": 1}) == 1.0
    assert geometry_value({"measurement_kind": "ordinal", "canonical_category_rank": 3}) == 3.0


def test_construction_contract_uses_width_sixteen_and_new_quotas() -> None:
    registry = load_registry()
    task = registry["tasks"]["bioavailability_ma"]
    config = v11.construction_config("bioavailability_ma", registry)
    assert task["priority_phases"] == [
        ["fg", "fh"],
        ["hf_bioavailability", "oral_exposure", "fa"],
    ]
    assert config["ranking_anchor_width"] == 16
    assert config["record_degree_cap"] == config["ordinary_degree_cap"] == 24
    assert config["ranking_anchors"]["test"]["fg"] == 20
    assert config["ranking_anchors"]["validation"]["fg"] == 10
    assert config["ordinary_pairs_per_label"]["fg"] == 50
    for task_id, task in registry["tasks"].items():
        construction = task["construction"]
        assert construction["ranking_anchor_width"] == 16
        for source in task["sources"]:
            if task_id == "bioavailability_ma" and source == "fg":
                continue
            assert construction["ranking_anchors"]["test"][source] == 30
            assert construction["ranking_anchors"]["validation"][source] == 20


def test_sft_metadata_contains_soft_targets_groups_and_geometry() -> None:
    row = {
        "target_distribution": {"transfer": 0.7, "nontransfer": 0.3},
        "assay_concept": "direct_bbb",
        "task_id": "bbb_martins",
        "source_id": "direct_bbb",
        "measurement_kind": "binary",
        "target_z": 0.5,
        "absolute_geometry_difference": 1.0,
        "calibration_distance_p05": 0.0,
        "calibration_distance_p95": 1.0,
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
    assert not any("standard_deviation" in key for key in row["metadata"])


def test_phase_scheduler_runs_test_before_validation(monkeypatch) -> None:
    calls = []

    def fake_run(_rows, split, concepts, *_state):
        calls.append((split, frozenset(concepts)))
        return {"ordinary": [], "ranking": [], "molecules": set(), "ranking_anchor_concepts": {}}

    monkeypatch.setattr(v11.v10, "run_construction", fake_run)
    registry = load_registry()
    v11._construct_phases([], "bioavailability_ma", registry, set())
    assert calls == [
        ("test", frozenset({"fg", "fh"})),
        ("validation", frozenset({"fg", "fh"})),
        ("test", frozenset({"hf_bioavailability", "oral_exposure", "fa"})),
        ("validation", frozenset({"hf_bioavailability", "oral_exposure", "fa"})),
    ]


def test_only_reserved_scaffold_molecules_are_excluded_from_train() -> None:
    rows = [
        {"child_id": "scaffold", "canonical_smiles": "CCO"},
        {"child_id": "random-only", "canonical_smiles": "CCN"},
    ]
    results = {
        "validation": {"molecules": set()},
        "test": {"molecules": set()},
    }
    splits = v11._partition_rows(rows, results, {"CCO"})
    assert [row["child_id"] for row in splits["train"]] == ["random-only"]
    assert not splits["validation"] and not splits["test"]


def test_width_sixteen_ranking_rows_renumber_as_complete_anchors(monkeypatch) -> None:
    monkeypatch.setattr(v11.v8, "RANKING_ANCHOR_WIDTH", 16)
    rows = [
        {"split": "test", "ranking_member_index": index % 16}
        for index in range(32)
    ]
    v11.v10._renumber_ranking_rows(rows)
    assert {row["ranking_query_id"] for row in rows[:16]} == {"test-rank-0000"}
    assert {row["ranking_query_id"] for row in rows[16:]} == {"test-rank-0001"}


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
    construction["tasks"]["bioavailability_ma"]["construction"]["record_degree_cap"] += 1
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
    assert len(releases) == 2
    assert {root for root, _repo in calls} == {
        tmp_path / "bioavailability_ma" / variant for variant in ablation.VARIANTS
    }
    assert all("bioavailability-ma" in repo for _root, repo in calls)


def test_every_new_function_is_at_most_sixty_lines() -> None:
    for filename in NEW_FILES:
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 60, f"{filename}:{node.name}"
