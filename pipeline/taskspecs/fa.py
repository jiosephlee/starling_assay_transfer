"""Fa task spec — fraction absorbed (intestinal absorption / % bioavailability family).

Fa endpoints in the raw extraction mix several ``property_type`` values across
incompatible units (permeability in cm/s, solubility, etc.). For the first proof we
keep the **percentage** family (oral bioavailability / intestinal absorption / fraction
absorbed) so the value is a bounded 0-100 % and the labeling rule matches oral
bioavailability's absolute-percentage-points rule. Permeability (Papp) is left as a
separate log-scale endpoint for later.
"""

from __future__ import annotations

from typing import Any, Optional

from pipeline.normalize.species_normalization import (
    NORMALIZED_COLUMN,
    normalized_species_or_population,
)

from .base import NumericThresholdTaskSpec, parse_leading_number


def _derive_species(row: dict[str, Any]) -> Any:
    return normalized_species_or_population(row.get("species_or_population"))


class FaTaskSpec(NumericThresholdTaskSpec):
    name = "fa"
    dataset_prefix = "fa"
    label_column = "property_value"

    # Bounded percentage: identity transform, points-apart labeling.
    transfer_threshold = 10.0
    not_transfer_threshold = 30.0

    metadata_columns = [
        "species_or_population",
        "route_medium_site",
        "property_type",
        "qualifying_conditions",
        "extra_details",
    ]

    same_columns = {
        "no_constraints": [],
        "same_species": [NORMALIZED_COLUMN],
        # exact-match "setting": same species, route, and measured property type.
        "condition_key": [NORMALIZED_COLUMN, "route_medium_site", "property_type"],
    }

    column_normalizers = {"species_or_population": normalized_species_or_population}
    derived_columns = {NORMALIZED_COLUMN: _derive_species}

    # -- prepare-stage raw schema --------------------------------------------
    raw_id_column = "global_identifier"
    raw_value_column = "extracted.measured_result"
    raw_units_column = "extracted.property_type"  # drives fraction->% harmonization
    property_filter_column = "extracted.property_type"
    property_filter_values = (
        "oral_bioavailability",
        "intestinal_absorption",
        "fraction_absorbed",
    )
    raw_metadata_map = {
        "species_or_population": "extracted.species_or_model",
        "route_medium_site": "extracted.route_medium_site",
        "property_type": "extracted.property_type",
        "qualifying_conditions": "extracted.qualifying_conditions",
        "extra_details": "extracted.extra_details",
    }
    # A fraction/percentage endpoint must lie in [0, 100]; anything else is a parse
    # artifact (units mixed into measured_result) and is dropped.
    value_range = (0.0, 100.0)

    def parse_value(self, raw: Any, units: Any = None) -> Optional[float]:
        value = parse_leading_number(raw)
        if value is None:
            return None
        # Harmonize onto a 0-100 percentage scale. ``fraction_absorbed`` is often
        # reported as a 0-1 fraction; scale those up. Values already >1 are treated
        # as already-percent.
        if units == "fraction_absorbed" and 0.0 < value <= 1.0:
            value *= 100.0
        return value
