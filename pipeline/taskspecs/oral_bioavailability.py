"""Oral bioavailability task spec.

This wraps the existing oral-bioavailability logic (species normalization + Tianang
condition key + absolute-difference labeling) behind the ``TaskSpec`` interface, so the
current pipeline routes through the abstraction without any behavior change. It is the
reference implementation the new Fa/Fg/Fh specs mirror.
"""

from __future__ import annotations

from typing import Any

# Reuse the existing normalizers from the pipeline.normalize package.
from pipeline.normalize.create_oral_bioavailability_condition_key_base import (
    KEY_COLUMN,
    condition_key,
)
from pipeline.normalize.species_normalization import (
    NORMALIZED_COLUMN,
    normalized_species_or_population,
)

from .base import NumericThresholdTaskSpec


def _derive_species(row: dict[str, Any]) -> Any:
    return normalized_species_or_population(row.get("species_or_population"))


def _derive_condition_key(row: dict[str, Any]) -> Any:
    species = normalized_species_or_population(row.get("species_or_population"))
    return condition_key({**row, NORMALIZED_COLUMN: species})


class OralBioavailabilityTaskSpec(NumericThresholdTaskSpec):
    name = "oral_bioavailability"
    dataset_prefix = "oral_bioavailability"
    label_column = "oral_bioavailability_value"

    # Oral bioavailability is a bounded percentage: identity transform, points-apart rule.
    transfer_threshold = 10.0
    not_transfer_threshold = 30.0

    metadata_columns = [
        "bioavailability_report_type",
        "species_or_population",
        "dose",
        "oral_exposure_mode",
        "qualifying_conditions",
        "comparator",
        "extra_details",
    ]

    same_columns = {
        "no_constraints": [],
        "same_species_v2": [NORMALIZED_COLUMN],
        "condition_key": [KEY_COLUMN],
    }

    column_normalizers = {
        "species_or_population": normalized_species_or_population,
    }

    derived_columns = {
        NORMALIZED_COLUMN: _derive_species,
        KEY_COLUMN: _derive_condition_key,
    }
