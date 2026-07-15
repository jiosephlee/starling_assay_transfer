"""Per-property task specifications for the assay-transfer pipeline.

Each assay property (oral bioavailability, Fa, Fg, Fh, ...) is described by a single
``TaskSpec`` object that centralizes every property-specific rule the pipeline needs:

- which metadata columns are recorded for the retrieved molecule,
- which columns must match to form a pair (per pairing mode),
- how raw values within a column are grouped as "the same" (normalization),
- which column holds the label value and how it is parsed / transformed,
- the rule that turns a (retrieved, query) value pair into transfer / not_transfer / drop.

The concrete spec objects live in this package and are exposed through ``registry``.
"""

from __future__ import annotations

from .base import NumericThresholdTaskSpec, TaskSpec
from .registry import available, get_task, register

__all__ = [
    "TaskSpec",
    "NumericThresholdTaskSpec",
    "get_task",
    "register",
    "available",
]
