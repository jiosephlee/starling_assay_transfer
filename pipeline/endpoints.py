"""Canonical endpoint assignment for raw assay rows (``endpoints.yaml``).

A raw row carries a source-specific endpoint category (``endpoint_category`` for q2,
``gut_wall_process`` for q3, ``metric_type`` for q4). This module maps that category to a
*canonical endpoint* — the unit of the cross-endpoint firewall
(``docs/assay_transfer_design.md`` section 5.1) — and canonicalizes the reported value onto
the endpoint's canonical unit + comparison scale.

``endpoints.yaml`` is the source of each canonical endpoint's ``metric_type``,
``most_specific_schema``, ``rollout_tier`` and its ``raw_families`` (the raw category
strings that route to it). Activation is gated by :data:`ENABLED_CANONICALIZERS`: only
endpoints whose value canonicalizer is implemented are assigned; every other row is
*quarantined* with a reason, so coverage grows additively as unit tables land. Rows that a
category maps to only via disabled endpoints, or whose value/unit cannot be resolved, are
never fabricated into a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

from pipeline.normalize.value_canon import get_canonicalizer
from pipeline.policy import load_metric_policy

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "assay_transfer" / "v1"

# Canonical endpoints activated in this build, mapped to their value canonicalizer.
# Extend as unit-canonicalization tables land (permeability, clearance, solubility, ...).
ENABLED_CANONICALIZERS: dict[str, str] = {
    "q2.fraction_absorbed.percent": "percent",
    "q2.intestinal_absorption.percent": "percent",
    "q3.gut_wall_escape.percent": "percent",
    "q4.extraction_ratio": "fraction",
    "q4.metabolic_half_life": "time_minutes",
}


@dataclass(frozen=True)
class EndpointAssignment:
    """A resolved canonical endpoint + canonicalized value for one raw row."""

    canonical_endpoint_id: str
    metric_type: str
    most_specific_schema: Optional[str]
    rollout_tier: Optional[int]
    native_value: float  # canonical unit (percentage points, fraction, minutes, ...)
    transformed_value: float  # comparison scale (identity or log10)


class EndpointResolver:
    """Assign raw rows to canonical endpoints per ``endpoints.yaml`` + the allowlist."""

    def __init__(self, config: dict[str, Any], enabled: dict[str, str]):
        self.version: str = config.get("version", "")
        self._sources: dict[str, dict[str, Any]] = config.get("source_collections", {}) or {}
        self._endpoints: dict[str, dict[str, Any]] = config.get("canonical_endpoints", {}) or {}
        self._enabled = dict(enabled)
        self._metric_policy = load_metric_policy()
        # (source, raw_family) -> [canonical_endpoint_id], restricted to enabled endpoints.
        self._route: dict[tuple[str, str], list[str]] = {}
        for endpoint_id, spec in self._endpoints.items():
            if endpoint_id not in self._enabled:
                continue
            source = spec.get("source_collection")
            for family in spec.get("raw_families", []):
                self._route.setdefault((source, family), []).append(endpoint_id)

    def source_columns(self, source: str) -> dict[str, Any]:
        spec = self._sources.get(source)
        if spec is None:
            raise KeyError(f"unknown source collection {source!r}; known: {sorted(self._sources)}")
        return spec

    def assign(self, source: str, row: dict[str, Any]) -> tuple[Optional[EndpointAssignment], Optional[str]]:
        cols = self.source_columns(source)
        raw_cat = row.get(cols.get("raw_endpoint_column"))
        raw_cat = (raw_cat or "").strip() if isinstance(raw_cat, str) else raw_cat
        candidates = self._route.get((source, raw_cat), [])
        if not candidates:
            return None, "unmapped_or_disabled_endpoint"
        if len(candidates) > 1:
            return None, "ambiguous_endpoint"
        endpoint_id = candidates[0]
        spec = self._endpoints[endpoint_id]
        metric_type = spec["metric_type"]

        value = row.get(cols.get("raw_value_column"))
        unit_col = cols.get("raw_unit_column")
        unit = row.get(unit_col) if unit_col and unit_col != "embedded_in_measured_value" else None

        canonicalizer = get_canonicalizer(self._enabled[endpoint_id])
        native = canonicalizer(value, unit)
        if native is None:
            return None, "unresolved_value_or_unit"
        transformed = self._metric_policy.for_metric(metric_type).transform_value(native)
        if transformed is None:
            return None, "untransformable_value"
        return (
            EndpointAssignment(
                canonical_endpoint_id=endpoint_id,
                metric_type=metric_type,
                most_specific_schema=spec.get("most_specific_schema"),
                rollout_tier=spec.get("rollout_tier"),
                native_value=native,
                transformed_value=transformed,
            ),
            None,
        )

    def enabled_endpoints(self) -> list[str]:
        return sorted(self._enabled)

    def most_specific_schema(self, canonical_endpoint_id: str) -> Optional[str]:
        spec = self._endpoints.get(canonical_endpoint_id, {})
        return spec.get("most_specific_schema")

    def metric_type(self, canonical_endpoint_id: str) -> str:
        return self._endpoints[canonical_endpoint_id]["metric_type"]


@lru_cache(maxsize=None)
def load_endpoint_resolver(config_root: Optional[str] = None) -> EndpointResolver:
    root = Path(config_root) if config_root else _CONFIG_ROOT
    config = yaml.safe_load((root / "endpoints.yaml").read_text())
    return EndpointResolver(config, ENABLED_CANONICALIZERS)
