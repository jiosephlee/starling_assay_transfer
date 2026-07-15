"""Oral-compatible facade for explicit-only species normalization."""

from __future__ import annotations

from typing import Any

from pipeline.normalize.assay_species_normalization import (
    SPECIES_EXACT_COLUMN,
    SPECIES_OUTPUT_COLUMNS,
    resolve_assay_species,
)
from pipeline.normalize.species_text import NULL_VALUES, normalize_text

NORMALIZED_COLUMN = SPECIES_EXACT_COLUMN


def species_resolution(value: Any) -> str | None:
    """Return the exact species explicitly designated by one source value."""
    return resolve_assay_species(explicit=value)


def normalized_species_or_population(value: Any) -> str | None:
    """Return canonical exact species, or ``None`` when it is unresolved."""
    return species_resolution(value)


def normalized_species_fields(value: Any) -> dict[str, str | None]:
    """Return the single-field normalized species artifact contract."""
    return {SPECIES_EXACT_COLUMN: species_resolution(value)}


__all__ = [
    "NORMALIZED_COLUMN",
    "NULL_VALUES",
    "SPECIES_EXACT_COLUMN",
    "SPECIES_OUTPUT_COLUMNS",
    "normalize_text",
    "normalized_species_fields",
    "normalized_species_or_population",
    "species_resolution",
]
