"""Assign minimal, scientifically safe endpoint identities to scalar children."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable, Mapping

from pipeline.source_normalization.scalar import ScalarEmission
from pipeline.source_normalization.text import normalize_lexical, normalize_unit

ENDPOINT_RESOLVER_VERSION = "canonical_endpoint_resolver_v1"


@dataclass(frozen=True)
class EndpointAssignment:
    canonical_endpoint_key: str
    endpoint_family: str
    endpoint_subtype: str
    unit_basis: str
    direction: str | None = None
    target: str | None = None
    kinetic_parameter: str | None = None
    auc_window: str | None = None
    defining_timepoint: str | None = None

    def columns(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: Any) -> str | None:
    text = normalize_lexical(value)
    if text is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return slug or None


def _unit_basis(emission: ScalarEmission, ratio: bool = False) -> str | None:
    unit = normalize_unit(emission.unit_normalized)
    if unit is None:
        return "dimensionless_ratio" if ratio else None
    compact = unit.replace(" ", "")
    aliases = {
        "%": "percent", "percent": "percent", "per cent": "percent",
        "fraction": "fraction", "unitless": "dimensionless", "ratio": "dimensionless_ratio",
        "fold": "dimensionless_ratio", "times": "dimensionless_ratio",
        "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
        "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
        "s": "second", "sec": "second", "second": "second", "seconds": "second",
        "d": "day", "day": "day", "days": "day",
        "cm/sec": "cm_per_second", "cm/s": "cm_per_second",
    }
    if compact in aliases:
        return aliases[compact]
    if re.fullmatch(r"(?:x?10\^?-?\d+)?cm/(?:s|sec)", compact):
        exponent = re.search(r"10\^?(-?\d+)", compact)
        return f"1e{exponent.group(1)}_cm_per_second" if exponent else "cm_per_second"
    return _slug(unit)


def _direction(text: str) -> str | None:
    normalized = normalize_lexical(text) or ""
    forward = r"(?:a|ap|apical|mucosal)\s*(?:-|→|>|to)\s*(?:b|bl|basolateral|serosal)"
    reverse = r"(?:b|bl|basolateral|serosal)\s*(?:-|→|>|to)\s*(?:a|ap|apical|mucosal)"
    if re.search(forward, normalized):
        return "absorptive"
    if re.search(reverse, normalized):
        return "secretory"
    return None


def _auc_window(label: str) -> str | None:
    compact = re.sub(r"[^a-z0-9∞τ]+", "_", label.lower()).strip("_")
    compact = compact.replace("infinity", "inf").replace("∞", "inf").replace("τ", "tau")
    if not compact.startswith("auc"):
        return None
    suffix = compact[3:].strip("_")
    if suffix in {"inf", "0_inf"}:
        return "zero_to_infinity"
    if suffix in {"last", "0_last", "0_tlast", "0_t_last", "0_t", "t", "0t"}:
        return "zero_to_last"
    if suffix in {"tau", "0_tau", "tau_ss"}:
        return "dosing_interval"
    if suffix in {"ss", "0_7_ss"}:
        return "steady_state"
    match = re.fullmatch(r"(?:0_?)?(\d+(?:_\d+)?)(?:_?(h|hr|d))?", suffix)
    return f"zero_to_{match.group(1).replace('_', '_to_')}{match.group(2) or 'h'}" if match else None


def _key(source: str | None, family: str, subtype: str, **dimensions: str | None) -> str:
    pieces = [source, family, subtype] if source else [family, subtype]
    pieces.extend(value for value in dimensions.values() if value)
    return ".".join(str(value) for value in pieces)


def _assignment(
    source: str | None, family: str, subtype: str, unit: str,
    *, direction: str | None = None, target: str | None = None,
    kinetic: str | None = None, auc: str | None = None, timepoint: str | None = None,
) -> EndpointAssignment:
    dimensions = {
        "unit": unit, "direction": direction, "target": target,
        "kinetic": kinetic if kinetic != subtype else None,
        "auc": auc, "time": _slug(timepoint),
    }
    key = _key(source, family, subtype, **dimensions)
    return EndpointAssignment(key, family, subtype, unit, direction, target, kinetic, auc, _slug(timepoint))


def _target(record: Mapping[str, Any], column: str) -> str | None:
    target = _slug(record.get(column))
    if target in {None, "unspecified", "unknown", "not_reported", "none"}:
        return None
    return target


def _q1_bioavailability(record: Mapping[str, Any], emission: ScalarEmission, alias: str) -> tuple[EndpointAssignment | None, str | None]:
    unit = _unit_basis(emission)
    if "relative" in alias:
        if unit in {"percent", "fraction", "dimensionless_ratio", "dimensionless"}:
            return _assignment("q1", "oral_bioavailability", "relative", unit), None
        return None, "incompatible_or_missing_unit_basis"
    if unit not in {"percent", "fraction"}:
        return None, "incompatible_or_missing_unit_basis"
    comparator = normalize_lexical(record.get("comparator_exposure")) or ""
    explicit = "absolute" in alias or "absolute bioavailability" in comparator
    explicit = explicit or bool(re.search(r"\b(?:intravenous|iv)\b", comparator))
    if explicit:
        return _assignment(None, "oral_bioavailability", "absolute", unit), None
    if "corrected" in alias:
        return _assignment("q1", "oral_bioavailability", "corrected", unit), None
    return None, "missing_absolute_or_relative_identity"


def _q1(source: str, record: Mapping[str, Any], emission: ScalarEmission) -> tuple[EndpointAssignment | None, str | None]:
    alias = normalize_lexical(record.get("endpoint_alias_raw")) or ""
    label = normalize_lexical(emission.measurement_label) or alias
    unit = _unit_basis(emission)
    if "bioavailability" in alias:
        return _q1_bioavailability(record, emission, alias)
    if label.startswith("auc") or label.startswith("aauc"):
        window = _auc_window(label)
        if window is None or unit is None:
            return None, "missing_auc_window_or_unit"
        return _assignment(source, "oral_exposure", "auc", unit, auc=window), None
    concentration = _q1_concentration_subtype(label)
    if concentration:
        if unit is None:
            return None, "incompatible_or_missing_unit_basis"
        return _assignment(source, "oral_exposure", concentration, unit), None
    if label in {"tmax", "t max"}:
        return (_assignment(source, "oral_exposure", "tmax", unit), None) if unit else (None, "incompatible_or_missing_unit_basis")
    if alias == "dose normalized exposure" and unit:
        return _assignment(source, "oral_exposure", "dose_normalized", unit), None
    if alias == "oral iv comparison" and unit == "dimensionless_ratio":
        return _assignment(source, "oral_exposure", "oral_iv_ratio", unit), None
    return None, "unknown_endpoint_alias"


def _q1_concentration_subtype(label: str) -> str | None:
    compact = label.replace(" ", "")
    for prefix, subtype in (("cmax", "cmax"), ("cmin", "cmin"), ("cavg", "cavg"), ("cav", "cavg"), ("ctrough", "ctrough"), ("ctau", "ctau"), ("css", "steady_state_concentration")):
        if compact.startswith(prefix):
            return subtype
    if re.fullmatch(r"c\d+h?", compact):
        return f"concentration_{compact[1:]}"
    return None


def _q2(source: str, record: Mapping[str, Any], emission: ScalarEmission) -> tuple[EndpointAssignment | None, str | None]:
    alias = normalize_lexical(record.get("endpoint_alias_raw")) or ""
    label = normalize_lexical(emission.measurement_label) or ""
    unit = _unit_basis(emission, ratio="ratio" in label)
    if "fraction absorbed" in alias or "intestinal absorption" in alias or alias == "absorption":
        return (_assignment(source, "intestinal_absorption", "fraction_absorbed", unit), None) if unit in {"percent", "fraction"} else (None, "incompatible_or_missing_unit_basis")
    if "effective permeability" in alias:
        if not _is_permeability_unit(unit):
            return None, "incompatible_or_missing_unit_basis"
        return _assignment(source, "intestinal_permeability", "peff", unit), None
    if "permeab" in alias or "permea" in alias:
        return _permeability(source, record, emission, unit)
    if "solub" in alias:
        subtype = _solubility_subtype(record, label)
        return (_assignment(source, "solubility", subtype, unit), None) if subtype and unit else (None, "missing_solubility_subtype_or_unit")
    if "dissolution" in alias or alias == "fraction dissolved":
        return _profile_endpoint(source, "dissolution", "fraction_dissolved", emission, unit)
    if alias == "gi stability":
        return _profile_endpoint(source, "gi_stability", "parent_remaining", emission, unit)
    if alias == "intrinsic dissolution rate" and unit:
        return _assignment(source, "dissolution", "intrinsic_rate", unit), None
    return None, "unknown_endpoint_alias"


def _solubility_subtype(record: Mapping[str, Any], label: str) -> str | None:
    fields = [label, record.get("assay_system"), record.get("condition_medium"), record.get("qualifying_conditions")]
    text = " ".join(normalize_lexical(value) or "" for value in fields)
    if "kinetic" in text:
        return "kinetic"
    if "equilibrium" in text or "thermodynamic" in text:
        return "equilibrium"
    return None


def _profile_endpoint(
    source: str, family: str, subtype: str, emission: ScalarEmission, unit: str | None,
) -> tuple[EndpointAssignment | None, str | None]:
    if unit not in {"percent", "fraction"}:
        return None, "incompatible_or_missing_unit_basis"
    if emission.measurement_timepoint is None:
        return None, "missing_defining_timepoint"
    return _assignment(source, family, subtype, unit, timepoint=emission.measurement_timepoint), None


def _permeability(
    source: str, record: Mapping[str, Any], emission: ScalarEmission, unit: str | None,
) -> tuple[EndpointAssignment | None, str | None]:
    label = normalize_lexical(emission.measurement_label) or ""
    if "efflux ratio" in label:
        ratio_unit = _unit_basis(emission, ratio=True)
        if ratio_unit not in {"dimensionless_ratio", "dimensionless"}:
            return None, "incompatible_ratio_basis"
        return _assignment(source, "intestinal_permeability", "efflux_ratio", ratio_unit, direction="secretory_over_absorptive"), None
    direction = _direction(f"{label} {emission.measurement_text_span}")
    if direction is None:
        return None, "missing_endpoint_direction"
    if not _is_permeability_unit(unit):
        return None, "incompatible_or_missing_unit_basis"
    subtype = "peff" if re.search(r"\bp[_ ]?eff\b", label) else "papp"
    return _assignment(source, "intestinal_permeability", subtype, unit, direction=direction), None


def _is_permeability_unit(unit: str | None) -> bool:
    if unit is None or unit in {"percent", "fraction", "minute", "hour", "second", "day"}:
        return False
    return bool(re.search(r"(?:^|_)(?:cm|nm|um|mm|pm).*?(?:s|sec|min)|cm_per_second", unit))


def _q3(source: str, record: Mapping[str, Any], emission: ScalarEmission) -> tuple[EndpointAssignment | None, str | None]:
    family = normalize_lexical(record.get("endpoint_alias_raw")) or ""
    label = normalize_lexical(emission.measurement_label) or ""
    unit = _unit_basis(emission, ratio="ratio" in label)
    if "efflux ratio" in label:
        return _q3_ratio(source, emission, unit)
    if re.search(r"\b(?:papp|p eff|peff)\b", label) or "bidirectional permeability" in family:
        return _permeability(source, record, emission, unit)
    kinetic = _kinetic_parameter(label)
    if kinetic and family == "intestinal metabolism":
        return _target_kinetic(source, "intestinal_metabolism", kinetic, record, emission, "transporter_or_enzyme")
    if label == "fa x fg":
        return (_assignment(source, "gut_wall_escape", "fg", unit), None) if unit in {"percent", "fraction"} else (None, "incompatible_or_missing_unit_basis")
    if re.fullmatch(r"f[_ ]?g", label) or "gut wall extraction" in family:
        return (_assignment(source, "gut_wall_escape", "fg", unit), None) if unit in {"percent", "fraction"} else (None, "incompatible_or_missing_unit_basis")
    if "auc ratio" in label:
        return _q3_ratio_endpoint(source, "auc_ratio", emission, unit)
    if "relative bioavailability" in label:
        return _q3_ratio_endpoint(source, "relative_bioavailability", emission, unit)
    return None, "unknown_or_underspecified_endpoint"


def _q3_ratio(source: str, emission: ScalarEmission, unit: str | None) -> tuple[EndpointAssignment | None, str | None]:
    if unit not in {"dimensionless_ratio", "dimensionless"}:
        return None, "incompatible_ratio_basis"
    return _assignment(source, "intestinal_transport", "efflux_ratio", unit, direction="secretory_over_absorptive"), None


def _q3_ratio_endpoint(source: str, subtype: str, emission: ScalarEmission, unit: str | None) -> tuple[EndpointAssignment | None, str | None]:
    if unit not in {"dimensionless_ratio", "dimensionless"}:
        return None, "incompatible_ratio_basis"
    auc = _auc_window(emission.measurement_label) if subtype == "auc_ratio" else None
    return _assignment(source, "oral_exposure_change", subtype, unit, auc=auc), None


def _kinetic_parameter(label: str) -> str | None:
    compact = label.replace(" ", "")
    if compact.startswith("vmax"):
        return "vmax"
    if compact == "km" or compact.startswith("km_"):
        return "km"
    if compact.startswith("clint"):
        return "clint"
    return None


def _target_kinetic(
    source: str, family: str, kinetic: str, record: Mapping[str, Any],
    emission: ScalarEmission, target_column: str,
) -> tuple[EndpointAssignment | None, str | None]:
    unit, target = _unit_basis(emission), _target(record, target_column)
    if not _is_kinetic_unit(kinetic, unit):
        return None, "incompatible_or_missing_unit_basis"
    if target is None:
        return None, "missing_endpoint_target"
    return _assignment(source, family, kinetic, unit, target=target, kinetic=kinetic), None


def _is_kinetic_unit(kinetic: str, unit: str | None) -> bool:
    if unit is None or unit in {"percent", "fraction", "dimensionless", "dimensionless_ratio"}:
        return False
    if kinetic == "clint":
        return _is_clearance_unit(unit)
    if kinetic == "km":
        return bool(re.search(r"(?:^|_)(?:m|um|nm|mm|pm|mol|mg_l|ug_ml|ng_ml)(?:_|$)", unit))
    if kinetic == "vmax":
        return bool(re.search(r"(?:^|_)(?:min|h|hour|s|sec)(?:_|$)", unit))
    return False


def _q4(source: str, record: Mapping[str, Any], emission: ScalarEmission) -> tuple[EndpointAssignment | None, str | None]:
    family = normalize_lexical(record.get("endpoint_alias_raw")) or ""
    label = normalize_lexical(emission.measurement_label) or ""
    unit = _unit_basis(emission, ratio="ratio" in label)
    if family in {"intrinsic clearance", "hepatic clearance", "metabolic clearance", "oral clearance", "biliary clearance"}:
        if not _is_clearance_unit(unit):
            return None, "incompatible_or_missing_unit_basis"
        return _assignment(source, "hepatic_clearance", family.replace(" ", "_"), unit), None
    if family == "metabolic half life":
        valid = {"second", "minute", "hour", "day"}
        return (_assignment(source, "hepatic_metabolism", "half_life", unit), None) if unit in valid else (None, "incompatible_or_missing_unit_basis")
    if family in {"extraction ratio", "hepatic extraction"}:
        return (_assignment(source, "hepatic_escape", "extraction_ratio", unit), None) if unit in {"percent", "fraction", "dimensionless"} else (None, "incompatible_or_missing_unit_basis")
    if family in {"cyp metabolism", "ugt metabolism", "fmo metabolism"}:
        return _q4_enzyme(source, family, label, record, emission, unit)
    if "stability" in family:
        return _profile_endpoint(source, "metabolic_stability", family.replace(" ", "_"), emission, unit)
    if family == "substrate depletion" and "half" in label:
        return (_assignment(source, "substrate_depletion", "half_life", unit), None) if unit else (None, "incompatible_or_missing_unit_basis")
    return None, "unknown_or_underspecified_endpoint"


def _is_clearance_unit(unit: str | None) -> bool:
    if unit is None or unit in {"percent", "fraction", "dimensionless", "dimensionless_ratio"}:
        return False
    volume = bool(re.search(r"(?:^|_)(?:u|m)?l(?:_|$)", unit))
    rate = bool(re.search(r"(?:^|_)(?:min|h|hour|s|sec)(?:_|$)", unit))
    return volume and rate


def _q4_enzyme(
    source: str, family: str, label: str, record: Mapping[str, Any],
    emission: ScalarEmission, unit: str | None,
) -> tuple[EndpointAssignment | None, str | None]:
    target = _target(record, "enzyme_or_pathway")
    if target is None:
        return None, "missing_endpoint_target"
    kinetic = _kinetic_parameter(label)
    if kinetic:
        return _target_kinetic(source, family.replace(" ", "_"), kinetic, record, emission, "enzyme_or_pathway")
    if unit == "percent":
        return _assignment(source, family.replace(" ", "_"), "enzyme_contribution", unit, target=target), None
    return None, "missing_kinetic_parameter"


def _starling(source: str, record: Mapping[str, Any], emission: ScalarEmission) -> tuple[EndpointAssignment | None, str | None]:
    report_type = normalize_lexical(record.get("bioavailability_report_type"))
    unit = _unit_basis(emission)
    if unit != "percent":
        return None, "incompatible_or_missing_unit_basis"
    if report_type == "absolute":
        return _assignment(None, "oral_bioavailability", "absolute", unit), None
    if report_type == "systemic availability":
        return _assignment(source, "oral_bioavailability", "systemic_availability", unit), None
    return None, "missing_absolute_or_relative_identity"


_RESOLVERS: dict[str, Callable[[str, Mapping[str, Any], ScalarEmission], tuple[EndpointAssignment | None, str | None]]] = {
    "q1": _q1, "q2": _q2, "q3": _q3, "q4": _q4, "starling": _starling,
}


def assign_canonical_endpoint(
    source: str, record: Mapping[str, Any], emission: ScalarEmission,
) -> tuple[EndpointAssignment | None, str | None]:
    """Assign one key after scalar parsing; never emit a broad fallback key."""
    resolver = _RESOLVERS.get(source)
    if resolver is None:
        return None, "unknown_source"
    assignment, reason = resolver(source, record, emission)
    if assignment is not None and not assignment.canonical_endpoint_key:
        raise RuntimeError("endpoint resolver emitted an empty key")
    return assignment, reason
