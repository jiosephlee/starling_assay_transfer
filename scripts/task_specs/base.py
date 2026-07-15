"""Base ``TaskSpec`` abstraction shared by every assay property.

A ``TaskSpec`` is the single place that defines property-specific behavior. The
pipeline (prepare -> base -> pairs -> splits -> hf) reads all of it from here instead
of hard-coding oral-bioavailability columns, thresholds, and normalization logic.

Responsibilities of a spec:

``metadata_columns``
    Columns recorded for the *retrieved* molecule (shown in prompts / used as MLP
    presence features). These retain their original values; derived grouping columns
    never replace them. The query molecule contributes no incidental metadata.

``same_columns``
    Per-mode mapping ``{mode -> [column, ...]}`` of columns that must match (after
    normalization) for a pair to be formed. This is also the "setting" over which the
    query value is marginalized (see the pairs stage).

``label_column``
    Column holding the numeric value used for labeling.

``parse_value`` / ``transform_value``
    ``parse_value`` turns a raw (often messy) string into a float in native units;
    ``transform_value`` maps it onto the comparison scale (identity for a bounded %,
    ``log10`` for order-of-magnitude endpoints like efflux ratio or clearance).

``normalize_column`` / ``derived_columns``
    ``normalize_column`` maps a raw cell value to a grouping equivalence class.
    ``derived_columns`` adds grouping columns such as ``species_exact`` without
    overwriting model-facing raw metadata; ``same_columns`` references these fields.

``label``
    Given the transformed retrieved and query values, returns ``"transfer"``,
    ``"not_transfer"``, or ``None`` (drop / deadband).
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Optional

LabelResult = Optional[str]  # "transfer" | "not_transfer" | None (drop)

# First signed decimal / scientific-notation number in a string; ``search`` (not
# ``match``) so leading qualifiers like ">", "<", "~" are skipped.
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def finite_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to a finite float, or ``None`` if impossible."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_leading_number(value: Any) -> Optional[float]:
    """Parse the first number out of a messy string cell.

    Handles ``%``, ``±`` error, ``>``/``<``/``~`` qualifiers, en/em dashes, and
    scientific notation's mantissa. Returns ``None`` when there is no number.
    (Unicode ``×10^n`` tails are not reconstructed here — those endpoints are handled
    by unit-aware specs or filtered out upstream.)
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    try:
        result = float(match.group())
    except ValueError:
        return None
    return result if math.isfinite(result) else None


class TaskSpec:
    """Property-specific rules for one assay-transfer task."""

    #: Registry key and default dataset directory prefix.
    name: str = ""
    dataset_prefix: str = ""

    #: Column holding the numeric label value.
    label_column: str = "property_value"

    #: Metadata columns recorded for the retrieved molecule.
    metadata_columns: list[str] = []

    #: Per-mode required-same columns (references normalized / derived column names).
    same_columns: dict[str, list[str]] = {}

    #: Optional per-column normalizers: raw value -> equivalence class (or None).
    column_normalizers: dict[str, Callable[[Any], Any]] = {}

    #: Grouping columns to materialize in the prepare stage: name -> row -> value.
    derived_columns: dict[str, Callable[[dict[str, Any]], Any]] = {}

    # -- prepare-stage raw schema (raw CSV -> canonical base parquet) ----------

    #: Raw column holding the molecule id resolved against the SMILES map.
    raw_id_column: str = "global_identifier"

    #: Raw column holding the (messy) value string.
    raw_value_column: str = ""

    #: Optional raw column passed to ``parse_value`` as ``units`` (units / subtype).
    raw_units_column: Optional[str] = None

    #: Optional raw column + allowed values used to filter to one comparable endpoint.
    property_filter_column: Optional[str] = None
    property_filter_values: tuple[str, ...] = ()

    #: Map canonical metadata column -> raw CSV column.
    raw_metadata_map: dict[str, str] = {}

    #: Plausible ``(min, max)`` range for the *native* value; rows outside are dropped
    #: in prepare (guards against messy-string parse artifacts). ``None`` = no bound.
    value_range: Optional[tuple[float, float]] = None

    def in_range(self, native: float) -> bool:
        if self.value_range is None:
            return True
        low, high = self.value_range
        return low <= native <= high

    # -- value handling -------------------------------------------------------

    def parse_value(self, raw: Any, units: Any = None) -> Optional[float]:
        """Parse a raw value cell into a float in native units (default: numeric)."""
        return finite_float(raw)

    def transform_value(self, value: float) -> Optional[float]:
        """Map a native-unit value onto the comparison scale (default: identity)."""
        return value

    def parse_and_transform(self, raw: Any, units: Any = None) -> Optional[float]:
        parsed = self.parse_value(raw, units)
        if parsed is None:
            return None
        return self.transform_value(parsed)

    # -- normalization / grouping --------------------------------------------

    def normalize_column(self, column: str, value: Any) -> Any:
        """Map a raw cell value to its equivalence class (default: identity)."""
        fn = self.column_normalizers.get(column)
        return fn(value) if fn is not None else value

    def derive(self, row: dict[str, Any]) -> dict[str, Any]:
        """Compute the grouping columns for one row."""
        return {name: fn(row) for name, fn in self.derived_columns.items()}

    def same_columns_for_mode(self, mode: str) -> list[str]:
        if mode not in self.same_columns:
            raise KeyError(
                f"task {self.name!r} has no mode {mode!r}; modes: {sorted(self.same_columns)}"
            )
        return list(self.same_columns[mode])

    # -- labeling -------------------------------------------------------------

    def label(self, v_retrieved: float, v_query: float) -> LabelResult:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TaskSpec {self.name!r} modes={sorted(self.same_columns)}>"


class NumericThresholdTaskSpec(TaskSpec):
    """Label by absolute difference on the (transformed) value scale.

    ``diff <= transfer_threshold`` -> transfer; ``diff >= not_transfer_threshold`` ->
    not_transfer; otherwise drop. For log-transformed endpoints an absolute difference
    in log space is a fold-change (e.g. ``log10(2)`` ~ within 2-fold).
    """

    transfer_threshold: float = 10.0
    not_transfer_threshold: float = 30.0

    def label(self, v_retrieved: float, v_query: float) -> LabelResult:
        diff = abs(float(v_retrieved) - float(v_query))
        if diff <= self.transfer_threshold:
            return "transfer"
        if diff >= self.not_transfer_threshold:
            return "not_transfer"
        return None
