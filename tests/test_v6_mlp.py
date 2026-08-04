from pipeline.v6_mlp import (MLPGroupSampler, assay_paragraph, compact_pair,
                             condition_stats, model_record)


def _row(key, smiles, value, concept="Fa"):
    row = {"child_id": key, "record_id": "source-" + key, "input_sha256": "x",
           "canonical_smiles": smiles, "canonical_endpoint_key": "endpoint",
           "assay_concept": concept, "unit_basis": "percent", "comparison_value": value,
           "transfer_max": 10.0, "not_transfer_min": 30.0,
           "scalar_is_approximate": False}
    row.update({"context_" + name: None for name in __import__("pipeline.pair_core", fromlist=["CONTEXT_FIELDS"]).CONTEXT_FIELDS})
    return row


def test_model_record_hides_all_values():
    row = _row("a", "CC", 20.0)
    record = model_record(row, 0, 0, "train")
    assert not any("value" in key for key in record)
    assert "20" not in record["assay_paragraph"]
    assert "not specified" in assay_paragraph(row)


def test_compact_pair_has_soft_targets_and_only_retrieval_value():
    query, retrieval = _row("a", "CC", 20.0), _row("b", "CCC", 25.0)
    stats = condition_stats([query, retrieval])
    pair = compact_pair(query, retrieval, 0, 0, {"a": 0, "b": 1}, stats)
    assert pair["query_record_index"] == 0 and pair["retrieval_record_index"] == 1
    assert abs(pair["target_a"] + pair["target_b"] - 1.0) < 1e-7
    assert "query_value" not in pair


def test_sampler_returns_complete_distinct_record_group():
    rows = [_row(str(index), "C" * (index + 1), float(index)) for index in range(200)]
    sampler = MLPGroupSampler(rows, group_size=4, proposal_factor=4)
    selected = sampler.next_group("Fa")
    assert selected is not None
    query, candidates = selected
    assert len(candidates) == 4
    assert len({row["child_id"] for row in candidates}) == 4
    assert all(row["canonical_smiles"] != query["canonical_smiles"] for row in candidates)
