"""Load the versioned assay-transfer policy contract (``configs/assay_transfer/v1``).

This module turns the frozen YAML policy files into typed lookups used by the pipeline:

- :class:`MetricPolicy` (``metric_thresholds.yaml``) — per canonical *metric type*, the
  value transform, distance, and the a-priori transfer / non-transfer thresholds. The
  threshold is attached to the metric type, never to a condition-key bucket
  (``docs/assay_transfer_design.md`` section 9).
- :class:`ConditionKeyPolicy` (``condition_keys.yaml``) — the join fields for each
  condition-key profile (``same_endpoint``, ``same_species_same_endpoint``) and each
  endpoint-specific ``most_specific`` schema.

Thresholds are compared on the **canonical (already transformed) value scale**: for a
log10 metric the stored value is ``log10(native)`` and the threshold ``0.301`` means
"within 2-fold". Callers canonicalize values once (see :mod:`pipeline.normalize.value_canon`)
and then use :meth:`MetricThreshold.label` on the transformed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "assay_transfer" / "v1"

LabelResult = Optional[str]  # "transfer" | "not_transfer" | None (deadband / drop)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@dataclass(frozen=True)
class MetricThreshold:
    """Resolved a-priori policy for one canonical metric type."""

    metric_type: str
    transform: str  # "identity" | "log10" | "canonical_category" | "canonical_ordered_bin"
    distance: str  # "absolute" | "equality" | "bin_separation"
    transfer_max: Optional[float] = None
    not_transfer_min: Optional[float] = None
    domain: Optional[tuple[float, float]] = None

    @property
    def is_numeric(self) -> bool:
        return self.distance == "absolute"

    def transform_value(self, native: float) -> Optional[float]:
        """Map a native-unit value onto the comparison (threshold) scale."""
        if self.transform == "identity":
            return native
        if self.transform == "log10":
            if native <= 0 or not math.isfinite(native):
                return None
            return math.log10(native)
        raise ValueError(f"metric {self.metric_type!r} is not a numeric transform ({self.transform})")

    def label(self, v_a: float, v_b: float) -> LabelResult:
        """Two-threshold deadband on already-transformed values (section 8.1)."""
        if not self.is_numeric or self.transfer_max is None or self.not_transfer_min is None:
            raise ValueError(f"metric {self.metric_type!r} has no numeric absolute-distance policy")
        diff = abs(float(v_a) - float(v_b))
        if diff <= self.transfer_max:
            return "transfer"
        if diff >= self.not_transfer_min:
            return "not_transfer"
        return None


class MetricPolicy:
    """``metric_thresholds.yaml`` as a metric-type -> :class:`MetricThreshold` lookup."""

    def __init__(self, config: dict[str, Any]):
        self.version: str = config.get("version", "")
        self._metrics: dict[str, MetricThreshold] = {}
        for name, spec in (config.get("metric_types") or {}).items():
            domain = spec.get("domain")
            domain_tuple = tuple(domain) if isinstance(domain, list) and len(domain) == 2 else None
            self._metrics[name] = MetricThreshold(
                metric_type=name,
                transform=spec.get("transform", "identity"),
                distance=spec.get("distance", "absolute"),
                transfer_max=spec.get("transfer_max"),
                not_transfer_min=spec.get("not_transfer_min"),
                domain=domain_tuple,
            )

    def for_metric(self, metric_type: str) -> MetricThreshold:
        try:
            return self._metrics[metric_type]
        except KeyError:
            raise KeyError(
                f"unknown metric_type {metric_type!r}; known: {sorted(self._metrics)}"
            ) from None

    def metric_types(self) -> list[str]:
        return sorted(self._metrics)


class ConditionKeyPolicy:
    """``condition_keys.yaml`` join-field lookup per profile / most-specific schema."""

    def __init__(self, config: dict[str, Any]):
        self.version: str = config.get("version", "")
        self._profiles: dict[str, dict[str, Any]] = config.get("profiles", {}) or {}
        self._schemas: dict[str, dict[str, Any]] = config.get("most_specific_schemas", {}) or {}

    def profiles(self) -> list[str]:
        return sorted(self._profiles)

    def join_fields(self, profile: str, most_specific_schema: Optional[str] = None) -> list[str]:
        """Join fields for a profile. ``most_specific`` needs the endpoint's schema name."""
        if profile == "most_specific":
            if most_specific_schema is None:
                raise ValueError("most_specific profile requires a most_specific_schema")
            schema = self._schemas.get(most_specific_schema)
            if schema is None:
                raise KeyError(f"unknown most_specific_schema {most_specific_schema!r}")
            return list(schema.get("join_fields", []))
        spec = self._profiles.get(profile)
        if spec is None:
            raise KeyError(f"unknown condition-key profile {profile!r}; known: {self.profiles()}")
        return list(spec.get("join_fields", []))

    def required_non_null(self, profile: str) -> list[str]:
        spec = self._profiles.get(profile, {})
        return list((spec.get("eligibility") or {}).get("required_non_null", []))


@lru_cache(maxsize=None)
def load_metric_policy(config_root: Optional[str] = None) -> MetricPolicy:
    root = Path(config_root) if config_root else _CONFIG_ROOT
    return MetricPolicy(_load_yaml(root / "metric_thresholds.yaml"))


@lru_cache(maxsize=None)
def load_condition_key_policy(config_root: Optional[str] = None) -> ConditionKeyPolicy:
    root = Path(config_root) if config_root else _CONFIG_ROOT
    return ConditionKeyPolicy(_load_yaml(root / "condition_keys.yaml"))
