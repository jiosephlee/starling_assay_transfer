import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_v10_intern_raw_pair as v10


def _record(child_id: str, smiles: str) -> dict:
    return {
        "assay_concept": "Fg",
        "canonical_smiles": smiles,
        "child_id": child_id,
    }


def test_ordinary_pool_unions_supplied_and_ranking_touched_records(monkeypatch) -> None:
    records = [_record("reserved-1", "reserved-a"), _record("reserved-2", "reserved-b")]
    ranking_query = _record("ranking-query", "ranking-a")
    ranking_candidate = _record("ranking-candidate", "ranking-b")
    captured: dict[str, object] = {}

    def fake_ranking(_records, _split, _overlap, targets, *_state):
        captured["ranking_target"] = dict(targets)
        anchors = [(ranking_query, [ranking_candidate])]
        return anchors, {"ranking-a", "ranking-b"}

    def fake_ordinary(_records, _split, pool, _targets, _query_degree, _retrieval_degree, forbidden):
        captured["pool"] = set(pool)
        captured["forbidden"] = set(forbidden)
        return []

    monkeypatch.setattr(v10.v8, "_ranking_anchors_v8", fake_ranking)
    monkeypatch.setattr(v10.v8, "_ranking_rows_from_anchors", lambda _anchors, _split: [])
    monkeypatch.setattr(v10, "_ordinary_pairs_v10", fake_ordinary)

    v10.run_construction(records, "test", {"Fg"}, set(), set(), {}, {}, {})

    assert captured["pool"] == {
        "reserved-1", "reserved-2", "ranking-query", "ranking-candidate",
    }
    assert captured["ranking_target"] == {"Fg": 20}
    assert captured["forbidden"] == {v10.v8.unordered_pair_id(ranking_query, ranking_candidate)}
    assert v10.v8.RANKING_ANCHORS_PER_CONCEPT_TEST["Fg"] == 25


def test_persist_manifest_writes_and_prints_same_json(tmp_path, capsys) -> None:
    manifest = {"version": "v10", "rows": {"train": 3}}
    v10._persist_manifest(tmp_path, manifest)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert json.loads(capsys.readouterr().out) == manifest


def test_v10_globals_configure_prompt_before_eval_construction(monkeypatch) -> None:
    monkeypatch.setattr(v10.pair_core, "render_prompt", lambda _query, _retrieval: "old")
    v10._configure_v10_globals()
    assert v10.pair_core.render_prompt is v10.v8._render_prompt_v8
    assert v10.pair_core.target_for is v10._target_for_v10
    assert v10.v7.target_for is v10._target_for_v10
    assert v10.v8.target_for is v10._target_for_v10
    assert v10.v8._candidate_label is v10._candidate_label_v10
    assert v10.v8._is_decisive is v10._is_decisive_v10


def test_ranking_ids_are_unique_after_phase_merge() -> None:
    rows = [{"split": "test", "ranking_query_id": "test-rank-0000",
             "ranking_member_index": index % 20} for index in range(40)]
    v10._renumber_ranking_rows(rows)
    assert {row["ranking_query_id"] for row in rows[:20]} == {"test-rank-0000"}
    assert {row["ranking_query_id"] for row in rows[20:]} == {"test-rank-0001"}


def test_quantile_anchors_map_to_unrestricted_sigmoid_targets() -> None:
    calibration = v10._configure_v10_globals()
    bucket, (lower, upper) = next(iter(v10._CALIBRATION_BY_BUCKET.items()))
    query = {"pair_bucket_key": bucket, "comparison_value": 0.0}
    at_lower = v10._target_for_v10(query, {"comparison_value": lower})
    at_middle = v10._target_for_v10(query, {"comparison_value": (lower + upper) / 2.0})
    at_upper = v10._target_for_v10(query, {"comparison_value": upper})
    below = v10._target_for_v10(query, {"comparison_value": max(0.0, lower / 2.0)})
    above = v10._target_for_v10(query, {"comparison_value": upper * 2.0 + 1.0})
    assert len(calibration["buckets"]) == 289
    assert math.isclose(at_lower["target_a"], 0.95, abs_tol=1e-12)
    assert math.isclose(at_middle["target_a"], 0.5, abs_tol=1e-12)
    assert math.isclose(at_upper["target_a"], 0.05, abs_tol=1e-12)
    assert below["target_a"] >= at_lower["target_a"]
    assert above["target_a"] < at_upper["target_a"]
    assert at_lower["calibration_distance_p05"] == lower
    assert at_upper["calibration_distance_p95"] == upper


def test_ordinary_labels_exclude_uncertain_target_region() -> None:
    v10._configure_v10_globals()
    bucket, (lower, upper) = next(iter(v10._CALIBRATION_BY_BUCKET.items()))
    query = {"pair_bucket_key": bucket, "comparison_value": 0.0}
    assert v10._candidate_label_v10(query, {"comparison_value": lower}) == "transfer"
    assert v10._candidate_label_v10(query, {"comparison_value": upper}) == "nontransfer"
    middle = {"comparison_value": (lower + upper) / 2.0}
    assert v10._candidate_label_v10(query, middle) is None
    assert v10._is_decisive_v10({"target_a": 0.61}, query)
    assert not v10._is_decisive_v10({"target_a": 0.5}, query)


def test_prompt_preludes_do_not_render_numeric_thresholds() -> None:
    template_dir = Path("templates/assay_transfer_v8_intern")
    for path in template_dir.glob("*.jinja"):
        text = path.read_text()
        assert "transfer_criterion_text" not in text
        assert "standard deviation" not in text
        assert "typical variation" in text
        assert "means stronger transfer" in text


def test_v10_rejects_a_source_with_the_wrong_hash(monkeypatch, tmp_path) -> None:
    source = tmp_path / "records.parquet"
    source.write_bytes(b"not the frozen source")
    monkeypatch.setattr(v10, "file_sha256", lambda _path: "unexpected")
    with pytest.raises(ValueError, match="eligible-source hash mismatch"):
        v10._verified_source_hash(source)


def test_manifest_records_the_verified_source_hash(monkeypatch) -> None:
    monkeypatch.setattr(v10, "file_sha256", lambda _path: "calibration-hash")
    monkeypatch.setattr(v10, "_v10_diagnostics", lambda *_args: {})
    monkeypatch.setattr(v10, "_pair_bucket_concentration", lambda _path: {})
    monkeypatch.setattr(v10, "_train_target_diagnostics", lambda _path: {})
    args = SimpleNamespace(source=Path("source"), calibration=Path("calibration"))
    calibration = {
        "schema_version": "pairwise_distance_quantiles_v1",
        "source_policy_sha256": "policy-hash", "lower_percentile": 5,
        "upper_percentile": 95, "target_probability_at_lower": 0.95,
        "target_probability_at_upper": 0.05, "buckets": {"bucket": {}},
    }
    manifest = {"paths": {"train": "train.parquet"}}
    cache = {"results": {}, "reserved": set()}
    v10._enrich_manifest(manifest, args, "verified-source-hash", calibration, {}, cache)
    assert manifest["eligible_source_sha256"] == "verified-source-hash"


def _ranking_records(count: int) -> list[dict]:
    return [
        {
            "assay_concept": "Fg", "pair_bucket_key": "bucket",
            "canonical_endpoint_key": "endpoint", "canonical_smiles": f"smiles-{index}",
            "child_id": f"record-{index}", "comparison_value": float(index),
        }
        for index in range(count)
    ]


def _configure_ranking_test(monkeypatch) -> None:
    def proposed(query, pool, _count, _used, _salt):
        return [row for row in pool if row is not query]

    monkeypatch.setattr(v10.v8, "propose_records", proposed)
    monkeypatch.setattr(
        v10.v8, "target_for",
        lambda _query, retrieval: {"target_z": retrieval["comparison_value"]},
    )


def test_incomplete_ranking_attempts_do_not_consume_degree(monkeypatch) -> None:
    _configure_ranking_test(monkeypatch)
    records, degree = _ranking_records(20), defaultdict(int)
    anchors, _touched = v10.v8._ranking_anchors_v8(
        records, "test", {row["canonical_smiles"] for row in records},
        {"Fg": 1}, set(), degree,
    )
    assert anchors == []
    assert dict(degree) == {}


def test_complete_ranking_anchor_commits_degree_once(monkeypatch) -> None:
    _configure_ranking_test(monkeypatch)
    records, degree = _ranking_records(21), defaultdict(int)
    anchors, _touched = v10.v8._ranking_anchors_v8(
        records, "test", {row["canonical_smiles"] for row in records},
        {"Fg": 1}, set(), degree,
    )
    assert len(anchors) == 1
    query, candidates = anchors[0]
    assert len(candidates) == 20
    assert set(degree) == {row["child_id"] for row in [query, *candidates]}
    assert set(degree.values()) == {1}
