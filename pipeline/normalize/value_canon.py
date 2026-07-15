"""Value canonicalizers: raw (value, unit) -> native canonical-unit float, or None.

Each canonicalizer converts one endpoint family's messy reported value onto its canonical
unit (``docs/assay_transfer_design.md`` sections 5.2, 9). It returns ``None`` whenever the
value cannot be resolved without guessing (unparseable number, unaccepted / ambiguous
unit, out of domain) so the caller can *quarantine* the row rather than fabricate a
comparison (``endpoints.yaml: unresolved_unit_or_basis: quarantine``).

The metric-scale transform (e.g. ``log10`` for half-life) is applied separately by
:meth:`pipeline.policy.MetricThreshold.transform_value`; these functions only produce the
native canonical value (minutes, percentage points, fraction, ...).

Unit-heavy families (permeability in cm/s across ``10^-6 cm/s`` variants, clearance across
~1000 unit strings, solubility needing molecular weight) are intentionally not yet
implemented — those endpoints quarantine until their unit tables land.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Optional

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Unit strings that mean "already a percentage 0-100".
_PERCENT_UNITS = {"%", "% of dose", "% of apical dose", "percent", "% remaining", "% absorbed"}
# Unit strings that mean "a 0-1 fraction".
_FRACTION_UNITS = {"fraction", "frac", "ratio", "unitless", "dimensionless", ""}
# Time units -> minutes multiplier.
_TIME_UNITS: dict[str, float] = {
    "min": 1.0, "mins": 1.0, "minute": 1.0, "minutes": 1.0, "min.": 1.0,
    "h": 60.0, "hr": 60.0, "hrs": 60.0, "hour": 60.0, "hours": 60.0, "h.": 60.0,
    "s": 1.0 / 60.0, "sec": 1.0 / 60.0, "secs": 1.0 / 60.0, "second": 1.0 / 60.0, "seconds": 1.0 / 60.0,
    "day": 1440.0, "days": 1440.0, "d": 1440.0,
}


def parse_number(value: Any) -> Optional[float]:
    """First finite number in a messy string cell (skips >, <, ~, ±, dashes)."""
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


def _norm_unit(unit: Any) -> str:
    return "" if unit is None else str(unit).strip().lower()


def percent(raw: Any, unit: Any = None) -> Optional[float]:
    """Canonicalize to percentage points in [0, 100].

    Accepts an explicit percent unit, or a 0-1 fraction (scaled up), or a bare number in
    [0, 100]. Rejects rate-like units (``h^-1``, ``min^-1``, ``fold``) and out-of-range
    values, which are parse artifacts for a bounded-extent endpoint.
    """
    number = parse_number(raw)
    if number is None:
        return None
    u = _norm_unit(unit)
    if u in _FRACTION_UNITS and 0.0 <= number <= 1.0:
        number *= 100.0
    elif u and u not in _PERCENT_UNITS:
        # A named, non-percent unit (rate, concentration, ...) is not a bounded percent.
        return None
    if 0.0 <= number <= 100.0:
        return number
    return None


def fraction(raw: Any, unit: Any = None) -> Optional[float]:
    """Canonicalize to a fraction in [0, 1] (percent units scaled down)."""
    number = parse_number(raw)
    if number is None:
        return None
    u = _norm_unit(unit)
    if u in _PERCENT_UNITS or (u == "" and number > 1.0):
        number /= 100.0
    elif u and u not in _FRACTION_UNITS:
        return None
    if 0.0 <= number <= 1.0:
        return number
    return None


def time_minutes(raw: Any, unit: Any = None) -> Optional[float]:
    """Canonicalize a positive time to minutes; rejects non-time / non-positive units."""
    number = parse_number(raw)
    if number is None or number <= 0:
        return None
    u = _norm_unit(unit)
    multiplier = _TIME_UNITS.get(u)
    if multiplier is None:
        return None
    return number * multiplier


CANONICALIZERS: dict[str, Callable[[Any, Any], Optional[float]]] = {
    "percent": percent,
    "fraction": fraction,
    "time_minutes": time_minutes,
}


def get_canonicalizer(name: str) -> Callable[[Any, Any], Optional[float]]:
    try:
        return CANONICALIZERS[name]
    except KeyError:
        raise KeyError(f"unknown canonicalizer {name!r}; known: {sorted(CANONICALIZERS)}") from None
