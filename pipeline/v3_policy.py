"""Version-3 policy loading and canonical endpoint classification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode()).hexdigest()


_TIME_TO_HOURS = {"hour": 1.0, "minute": 1.0 / 60.0, "day": 24.0}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    transform: str
    transfer_max: float
    not_transfer_min: float
    display: str
    domain: Any = None

    def transform_value(self, value: float, unit_basis: Optional[str] = None) -> Optional[float]:
        if not math.isfinite(value):
            return None
        if self.transform == "time_hours":
            factor = _TIME_TO_HOURS.get(unit_basis)
            if factor is None:  # non-time units (percent/ha/hb/...) are not transferable
                return None
            value *= factor
            return value if self.valid_native(value) else None
        if not self.valid_native(value):
            return None
        return math.log10(value) if self.transform == "log10" else value

    def valid_native(self, value: float) -> bool:
        if self.domain == "positive":
            return value > 0
        if isinstance(self.domain, list):
            return float(self.domain[0]) <= value <= float(self.domain[1])
        return True

    def vote(self, left: float, right: float) -> Optional[str]:
        distance = abs(left - right)
        if distance <= self.transfer_max:
            return "transfer"
        if distance >= self.not_transfer_min:
            return "not_transfer"
        return None


class V3Policies:
    """Resolved release, concept, metric, sampling, and similarity policy bundle."""

    def __init__(self, release_path: Path):
        self.release_path = release_path
        self.release = load_yaml(release_path)
        refs = self.release["policies"]
        paths = {name: self._policy_path(value) for name, value in refs.items()}
        self.metrics = load_yaml(paths["metrics"])
        self.concepts = load_yaml(paths["concepts"])
        self.sampling = load_yaml(paths["sampling"])
        self.fingerprint = load_yaml(paths["fingerprint"])
        self.tanimoto = load_yaml(paths["tanimoto"])
        self.target = load_yaml(paths["target"]) if "target" in paths else None
        self._metric_specs = self._build_metric_specs()
        self._metric_rules = self._build_metric_rules()

    def _policy_path(self, value: str) -> Path:
        local = self.release_path.parent / value
        return local if local.exists() else resolve_path(value)

    def _build_metric_specs(self) -> dict[str, MetricSpec]:
        return {
            name: MetricSpec(name=name, **spec)
            for name, spec in self.metrics["thresholds"].items()
        }

    def _build_metric_rules(self) -> list[tuple[str, str, str]]:
        rules = []
        for metric, pairs in self.metrics["rules"].items():
            rules.extend((str(family), str(subtype), metric) for family, subtype in pairs)
        return rules

    def concept_for(self, row: dict[str, Any]) -> Optional[str]:
        matches = []
        for rule in self.concepts["rules"]:
            if rule["source_id"] != row.get("source_id"):
                continue
            family = rule.get("endpoint_family")
            if family is None or family == row.get("endpoint_family"):
                matches.append(rule["concept"])
        if len(matches) > 1:
            raise ValueError(f"multiple assay concepts for row {row.get('child_id')}")
        return matches[0] if matches else None

    def metric_for(self, row: dict[str, Any]) -> Optional[MetricSpec]:
        family, subtype = row.get("endpoint_family"), row.get("endpoint_subtype")
        exact = [m for f, s, m in self._metric_rules if f == family and s == subtype]
        matches = exact or [m for f, s, m in self._metric_rules if f == family and s == "*"]
        if len(set(matches)) > 1:
            raise ValueError(f"multiple metric rules for {family}|{subtype}: {matches}")
        if not matches:
            return None
        metric = matches[0]
        if row.get("unit_basis") == "fraction" and metric == "bounded_percentage":
            metric = "bounded_fraction"
        return self._metric_specs[metric]

    @property
    def version_bundle(self) -> dict[str, str]:
        versions = {
            "release": self.release["version"],
            "metrics": self.metrics["version"],
            "concepts": self.concepts["version"],
            "sampling": self.sampling["version"],
            "fingerprint": self.fingerprint["version"],
            "tanimoto": self.tanimoto["version"],
        }
        if self.target:
            versions["target"] = self.target["version"]
        return versions

    @property
    def candidate_identity_versions(self) -> dict[str, str]:
        frozen = self.release.get("candidate_identity_policy_versions")
        return dict(frozen) if frozen else self.version_bundle
