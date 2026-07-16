"""Split dedicated measurement fields into finite scalar point estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

from pipeline.source_normalization.measurement import NUMBER
from pipeline.source_normalization.text import measurement_text, normalize_lexical, normalize_unit

PARSER_VERSION = "scalar_measurement_parser_v2"
_LABEL_TEXT = (
    r"efflux\s+ratio|relative\s+bioavailability|absolute\s+bioavailability|auc(?:\s+ratio)?"
    r"(?:[₀-₉0-9∞τ_-]+)?|cmax|tmax|papp|p_eff|peff|vmax|km|clint|half[- ]?life|"
    r"f[_ ]?[agh]|ap(?:ical)?\s*(?:[-→>]|to)\s*b(?:l|asolateral)?|"
    r"b(?:l|asolateral)?\s*(?:[-→>]|to)\s*ap(?:ical)?|"
    r"mucosal\s*(?:[-→>]|to)\s*serosal|serosal\s*(?:[-→>]|to)\s*mucosal"
)
_LABEL = re.compile(rf"\b(?:{_LABEL_TEXT})\b", re.I)
_PLUS_MINUS = re.compile(
    rf"^\s*(?:[:=]\s*)?(?P<approx>≈|~|approx(?:imately)?\s+|about\s+)?"
    rf"(?P<mean>{NUMBER})\s*%?\s*(?:±|\+/-)\s*(?P<variation>{NUMBER})"
    rf"(?:\s*[x×]\s*10\s*\^?\s*(?P<exponent>[-+]?\d+))?",
    re.I,
)
_POINT = re.compile(
    rf"^\s*(?:[:=]\s*)?(?P<approx>≈|~|approx(?:imately)?\s+|about\s+|ca\.?\s+|estimated\s+)?"
    rf"(?P<value>{NUMBER})(?:\s*[x×]\s*10\s*\^?\s*(?P<exponent>[-+]?\d+))?",
    re.I,
)
_BOUND = re.compile(rf"^\s*(?:[:=]\s*)?(?:[<>]=?|≥|≤|at least\s+|at most\s+|no (?:less|more) than\s+)\s*(?:{NUMBER})", re.I)
_RANGE_START = re.compile(rf"^\s*(?:[:=]\s*)?({NUMBER})\s*(?:-|–|—|\bto\b)\s*({NUMBER})", re.I)
_INTERVAL = re.compile(
    rf"(?:\b(?:90|95|99)\s*%?\s*)?(?:ci|confidence interval|range)\s*[:=]?\s*\(?"
    rf"({NUMBER})\s*(?:-|–|—|\bto\b|,)\s*({NUMBER})\)?",
    re.I,
)
_PAREN_INTERVAL = re.compile(rf"\(({NUMBER})\s*(?:-|–|—|\bto\b|,)\s*({NUMBER})\)", re.I)
_MEAN_LOW_HIGH = re.compile(
    rf"^\s*mean\s*[:=]?\s*({NUMBER})\s*;\s*low\s*[:=]?\s*({NUMBER})"
    rf"\s*;\s*high\s*[:=]?\s*({NUMBER})\s*$",
    re.I,
)
_TIMEPOINT = re.compile(rf"\b(?:at|after|within)\s+({NUMBER})\s*(min(?:ute)?s?|h(?:ou)?rs?|days?|seconds?|s)\b", re.I)
_CONCENTRATION = re.compile(rf"\b(?:at|@)\s*(?:a\s+)?(?:concentration\s+(?:of\s+)?)?({NUMBER})\s*((?:p|n|u|µ|μ|m)?m)\b", re.I)
_LEADING_TIME = re.compile(rf"^\s*({NUMBER})\s*(min(?:ute)?s?|h(?:ou)?rs?|days?|seconds?|s)\s*[:=]", re.I)
_LEADING_CONCENTRATION = re.compile(rf"^\s*({NUMBER})\s*((?:p|n|u|µ|μ|m)?m)\s*[:=]", re.I)
_UNIT_TOKEN = re.compile(
    r"%|\b(?:p|n|u|µ|μ|m)?mol\b(?:[\wµμ/^·*⁻−.-]*)|"
    r"\b(?:p|n|u|µ|μ|m)?g\b(?:[\wµμ/^·*⁻−.-]*)|"
    r"\b(?:u|µ|μ|m)?l\b(?:[\wµμ/^·*⁻−.-]*)|"
    r"\b(?:p|n|u|µ|μ|m)m\b|\b(?:n|u|µ|μ|m)?m/s\b|\bcm/(?:s|sec)\b",
    re.I,
)


@dataclass(frozen=True)
class ScalarEmission:
    measurement_label: str
    measurement_text_span: str
    span_start: int
    span_end: int
    unit_raw: str | None
    unit_normalized: str | None
    local_measurement_context: str | None
    measurement_timepoint: str | None
    measurement_concentration: str | None
    scalar_value: float
    scalar_is_approximate: bool
    variation_value: float | None = None
    variation_type: str | None = None
    accompanying_interval_lower: float | None = None
    accompanying_interval_upper: float | None = None
    emission_index: int = 0

    def columns(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FragmentRejection:
    measurement_label: str | None
    measurement_text_span: str
    span_start: int
    span_end: int
    rejection_reason: str

    def columns(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Fragment:
    text: str
    start: int
    end: int
    label: str | None
    label_end: int
    explicit_label: bool


def _float(value: str) -> float:
    return float(value.replace(",", ""))


def _segments(text: str) -> list[tuple[str, int, int]]:
    separator = re.compile(rf"[;\n|]|,\s*(?=(?:{NUMBER})\s+(?:at\b|@))", re.I)
    parts, start = [], 0
    for separator_match in [*separator.finditer(text), None]:
        end = separator_match.start() if separator_match else len(text)
        raw = text[start:end]
        stripped = raw.strip(" ,")
        if not stripped:
            start = separator_match.end() if separator_match else end
            continue
        left = raw.find(stripped)
        offset = start + left
        parts.append((stripped, offset, offset + len(stripped)))
        start = separator_match.end() if separator_match else end
    return parts


def _labeled_fragments(text: str, start: int, end: int) -> list[_Fragment]:
    matches = list(_LABEL.finditer(text))
    if len(matches) < 2:
        label = matches[0].group() if matches else None
        label_end = matches[0].end() if matches else 0
        return [_Fragment(text, start, end, label, label_end, bool(matches))]
    fragments = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start():stop].strip(" ,")
        offset = text.find(chunk, match.start(), stop)
        fragments.append(_Fragment(chunk, start + offset, start + offset + len(chunk), match.group(), match.end() - offset, True))
    return fragments


def _fragments(text: str) -> list[_Fragment]:
    output = []
    for segment, start, end in _segments(text):
        output.extend(_labeled_fragments(segment, start, end))
    return output


def _context(fragment: str) -> tuple[str | None, str | None, str | None]:
    time = _TIMEPOINT.search(fragment) or _LEADING_TIME.search(fragment)
    concentration = _CONCENTRATION.search(fragment) or _LEADING_CONCENTRATION.search(fragment)
    parenthetical = re.findall(r"\(([^()]*)\)", fragment)
    pieces = [item.strip() for item in parenthetical if item.strip()]
    if time:
        pieces.append(time.group())
    if concentration:
        pieces.append(concentration.group())
    local = "; ".join(dict.fromkeys(pieces)) or None
    timepoint = normalize_lexical(f"{time.group(1)} {time.group(2)}") if time else None
    concentration_text = normalize_lexical(f"{concentration.group(1)} {concentration.group(2)}") if concentration else None
    return local, timepoint, concentration_text


def _interval(text: str) -> tuple[float | None, float | None]:
    match = _INTERVAL.search(text) or _PAREN_INTERVAL.search(text)
    if match is None:
        return None, None
    lower, upper = _float(match.group(1)), _float(match.group(2))
    return min(lower, upper), max(lower, upper)


def _embedded_unit(text: str, supplied: Any) -> str | None:
    match = _UNIT_TOKEN.search(text)
    supplied_text = measurement_text(supplied)
    if match is None:
        return supplied_text
    prefix = text[:match.start()]
    if supplied_text and re.search(rf"(?:at|@)\s*(?:{NUMBER})\s*$", prefix, re.I):
        return supplied_text
    if match.group() == "%":
        return "%"
    unit = text[match.start():]
    unit = re.split(r"[(),;]", unit, maxsplit=1)[0].strip()
    return unit or supplied_text


def _estimate(tail: str) -> tuple[float | None, bool, float | None, str | None]:
    variation = _PLUS_MINUS.match(tail)
    if variation:
        factor = 10 ** int(variation.group("exponent")) if variation.group("exponent") else 1.0
        value = _float(variation.group("mean")) * factor
        spread = _float(variation.group("variation")) * factor
        return value, True, spread, "unspecified_variation"
    point = _POINT.match(tail)
    if point:
        factor = 10 ** int(point.group("exponent")) if point.group("exponent") else 1.0
        return _float(point.group("value")) * factor, bool(point.group("approx")), None, None
    return None, False, None, None


def _tail(fragment: _Fragment) -> str:
    tail = fragment.text[fragment.label_end:] if fragment.explicit_label else fragment.text
    direction = re.compile(
        r"^\s*(?:\(?\s*)?(?:a|ap|apical|mucosal|b|bl|basolateral|serosal)"
        r"\s*(?:-|→|>|to)\s*(?:a|ap|apical|mucosal|b|bl|basolateral|serosal)\s*\)?",
        re.I,
    )
    tail = direction.sub("", tail, count=1)
    if not fragment.explicit_label:
        tail = _LEADING_TIME.sub("", tail, count=1)
        tail = _LEADING_CONCENTRATION.sub("", tail, count=1)
    return tail


def _unassociated_numbers(tail: str) -> bool:
    estimate = _PLUS_MINUS.match(tail) or _POINT.match(tail)
    if estimate is None:
        return False
    suffix = tail[estimate.end():]
    suffix = _INTERVAL.sub("", suffix)
    suffix = _PAREN_INTERVAL.sub("", suffix)
    suffix = _TIMEPOINT.sub("", suffix)
    suffix = _CONCENTRATION.sub("", suffix)
    suffix = _UNIT_TOKEN.sub("", suffix)
    suffix = re.sub(r"\b(?:ci|confidence interval|range)\b", "", suffix, flags=re.I)
    return bool(re.search(NUMBER, suffix))


def _reason(fragment: _Fragment, tail: str, scalar: float | None, total: int) -> str | None:
    if _BOUND.match(tail):
        return "bound_only_measurement"
    if _RANGE_START.match(tail):
        return "range_or_interval_only"
    if scalar is None:
        return "qualitative_or_unparseable_measurement"
    if not math.isfinite(scalar):
        return "nonfinite_measurement"
    if _unassociated_numbers(tail):
        return "ambiguous_multiple_measurements"
    if total > 1 and not fragment.explicit_label and not (_TIMEPOINT.search(tail) or _CONCENTRATION.search(tail)):
        return "unlabeled_multiple_measurements"
    if re.search(r"\bfold\b", tail, re.I) and not re.search(r"\b(?:ratio|fold change endpoint)\b", fragment.text, re.I):
        return "comparison_only_fold_change"
    return None


def _emission(fragment: _Fragment, endpoint_alias: str, supplied_unit: Any, index: int) -> tuple[ScalarEmission | None, str | None]:
    tail = _tail(fragment)
    scalar, approximate, variation, variation_type = _estimate(tail)
    reason = _reason(fragment, tail, scalar, 1)
    if reason:
        return None, reason
    lower, upper = _interval(tail)
    local, timepoint, concentration = _context(fragment.text)
    unit = _embedded_unit(tail, supplied_unit)
    label = normalize_lexical(fragment.label or endpoint_alias) or ""
    return ScalarEmission(
        measurement_label=label, measurement_text_span=fragment.text,
        span_start=fragment.start, span_end=fragment.end,
        unit_raw=unit, unit_normalized=normalize_unit(unit),
        local_measurement_context=local, measurement_timepoint=timepoint,
        measurement_concentration=concentration, scalar_value=float(scalar),
        scalar_is_approximate=approximate, variation_value=variation,
        variation_type=variation_type, accompanying_interval_lower=lower,
        accompanying_interval_upper=upper, emission_index=index,
    ), None


def _mean_interval(text: str, endpoint_alias: str, supplied_unit: Any) -> ScalarEmission | None:
    match = _MEAN_LOW_HIGH.match(text)
    if not match:
        return None
    mean, low, high = (_float(match.group(index)) for index in range(1, 4))
    unit = measurement_text(supplied_unit)
    return ScalarEmission(
        measurement_label=normalize_lexical(endpoint_alias) or "", measurement_text_span=text,
        span_start=0, span_end=len(text), unit_raw=unit, unit_normalized=normalize_unit(unit),
        local_measurement_context="reported mean with low/high interval",
        measurement_timepoint=None, measurement_concentration=None, scalar_value=mean,
        scalar_is_approximate=False, accompanying_interval_lower=min(low, high),
        accompanying_interval_upper=max(low, high), emission_index=0,
    )


def split_scalar_measurements(
    value: Any, endpoint_alias: Any, supplied_unit: Any = None,
) -> tuple[list[ScalarEmission], list[FragmentRejection]]:
    """Return deterministic scalar children and audited rejected fragments."""
    text = measurement_text(value)
    alias = measurement_text(endpoint_alias) or ""
    if text is None:
        rejection = FragmentRejection(None, "", 0, 0, "missing_measurement")
        return [], [rejection]
    special = _mean_interval(text, alias, supplied_unit)
    if special:
        return [special], []
    fragments = _fragments(text)
    emissions, rejections = [], []
    for fragment in fragments:
        emission, reason = _emission(fragment, alias, supplied_unit, len(emissions))
        if emission is not None:
            emissions.append(emission)
            continue
        rejections.append(FragmentRejection(
            normalize_lexical(fragment.label), fragment.text, fragment.start,
            fragment.end, reason or "unparseable_measurement",
        ))
    if len(fragments) > 1:
        emissions, rejections = _enforce_association(fragments, emissions, rejections)
    return emissions, rejections


def _enforce_association(
    fragments: list[_Fragment], emissions: list[ScalarEmission], rejections: list[FragmentRejection],
) -> tuple[list[ScalarEmission], list[FragmentRejection]]:
    invalid_spans = {
        (fragment.start, fragment.end) for fragment in fragments
        if not fragment.explicit_label
        and not _has_pairing_context(fragment.text)
    }
    kept = [item for item in emissions if (item.span_start, item.span_end) not in invalid_spans]
    for item in emissions:
        if (item.span_start, item.span_end) in invalid_spans:
            rejections.append(FragmentRejection(
                item.measurement_label, item.measurement_text_span, item.span_start,
                item.span_end, "unlabeled_multiple_measurements",
            ))
    return [ScalarEmission(**{**item.columns(), "emission_index": index}) for index, item in enumerate(kept)], rejections


def _has_pairing_context(text: str) -> bool:
    patterns = (_TIMEPOINT, _CONCENTRATION, _LEADING_TIME, _LEADING_CONCENTRATION)
    return any(pattern.search(text) for pattern in patterns)
