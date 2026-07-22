import math

from pipeline.v6_intern import oriented_pair, render_prompt, target_for


def _row(child_id: str, value: float, smiles: str = "C") -> dict:
    row = {"child_id": child_id, "record_id": child_id, "comparison_value": value,
           "scalar_value": value, "canonical_smiles": smiles,
           "canonical_endpoint_key": "endpoint", "assay_concept": "Fa",
           "unit_basis": "percent", "transfer_max": 10.0, "not_transfer_min": 30.0}
    row.update({"context_" + name: None for name in (
        "species_or_population", "report_or_statistic_type", "dose", "study_or_assay_system",
        "measured_process", "biological_context", "medium", "formulation_or_solid_form",
        "transporter_or_enzyme", "substrate_status", "intestinal_site", "molecular_form",
        "enzyme_or_pathway", "qualifying_conditions", "comparator", "extra_details")})
    return row


def test_target_boundaries_are_frozen():
    query = _row("q", 40.0)
    assert math.isclose(target_for(query, _row("a", 50.0))["target_a"], .9)
    assert math.isclose(target_for(query, _row("m", 60.0))["target_a"], .5)
    assert math.isclose(target_for(query, _row("b", 70.0))["target_a"], .1)


def test_direction_is_unique_and_query_value_is_hidden():
    query, retrieval = _row("q", 40.0, "C"), _row("r", 50.0, "CC")
    assert oriented_pair(query, retrieval) != oriented_pair(retrieval, query)
    query_block = render_prompt(query, retrieval).split("Target query record (value hidden)", 1)[1]
    assert "known value" not in query_block
