"""Explicit Fg-family parsing for Q3 gut-wall records."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from pipeline.source_normalization.measurement import NUMBER
from pipeline.source_normalization.scalar import FragmentRejection, ScalarEmission
from pipeline.source_normalization.text import measurement_text

_FG = re.compile(
    r"(?<![a-z])%?f\s*[_ ]?g(?:ut)?\b|"
    r"fraction (?:of (?:the )?)?(?:drug )?escaping (?:the )?(?:gut|intestinal)"
    r"(?: (?:wall|metabolism|extraction))?",
    re.I,
)
_COMPOUND = re.compile(r"f\s*(?:a|abs)\s*[·×*x]\s*f\s*[_ ]?g\b", re.I)
_VALUE = re.compile(
    rf"\s*(?P<value>{NUMBER})\s*(?P<percent>%?)\s*"
    rf"(?:±\s*(?P<variation>{NUMBER})\s*%?)?\s*(?P<context>\([^()]*\))?",
    re.I,
)


def split_explicit_fg_measurements(
    value: Any, support_override: str | None = None,
) -> tuple[list[ScalarEmission], list[FragmentRejection]] | None:
    """Parse explicit Fg and Fa×Fg values into the shared Fg assay concept."""
    text = support_override or measurement_text(value)
    if text is None:
        return None
    compounds = list(_COMPOUND.finditer(text))
    labels = [match for match in _FG.finditer(text) if not _in_compound(match, compounds)] + compounds
    if not labels and not compounds:
        return None
    emissions, rejected = [], []
    for label in sorted(labels, key=lambda item: item.start()):
        parsed, fragments = _parse_label(
            text, label, labels, len(emissions), bool(support_override), _label_kind(label, compounds),
        )
        emissions.extend(parsed)
        rejected.extend(fragments)
    return _reindex(emissions), rejected


def _in_compound(label: re.Match[str], compounds: list[re.Match[str]]) -> bool:
    return any(item.start() <= label.start() < item.end() for item in compounds)


def _label_kind(label: re.Match[str], compounds: list[re.Match[str]]) -> str:
    return "fa x fg" if _in_compound(label, compounds) else "fg"


def _parse_label(
    text: str, label: re.Match[str], labels: list[re.Match[str]], index: int, recovered: bool, kind: str,
) -> tuple[list[ScalarEmission], list[FragmentRejection]]:
    stop = _label_stop(text, label, labels)
    segment = text[label.end():stop]
    found = list(_VALUE.finditer(segment))
    emissions, rejected = [], []
    for match in found:
        emission, reason = _fg_emission(text, label, match, index + len(emissions), recovered, kind)
        if emission is not None:
            emissions.append(emission)
        elif reason == "non_fg_quantity_after_fg":
            continue
        else:
            rejected.append(_value_rejection(label, match, reason or "unparseable_measurement", kind))
    if not found:
        rejected.append(FragmentRejection(kind, label.group(), label.start(), label.end(), "qualitative_or_unparseable_measurement"))
    return emissions, rejected


def _label_stop(text: str, label: re.Match[str], labels: list[re.Match[str]]) -> int:
    following = [item.start() for item in labels if item.start() > label.start()]
    boundary = re.search(r";|\n", text[label.end():])
    boundary_start = label.end() + boundary.start() if boundary else None
    if boundary_start is not None and _condition_continues(text[boundary_start + 1:]):
        boundary_start = None
    stops = following + ([boundary_start] if boundary_start is not None else [])
    return min(stops) if stops else len(text)


def _condition_continues(text: str) -> bool:
    match = _VALUE.match(text)
    return match is not None and match.group("context") is not None


def _fg_emission(
    text: str, label: re.Match[str], match: re.Match[str], index: int, recovered: bool, kind: str,
) -> tuple[ScalarEmission | None, str | None]:
    value = float(match.group("value").replace(",", ""))
    percent = bool(match.group("percent"))
    if not percent and not 0 <= value <= 1:
        return None, "incompatible_or_missing_unit_basis"
    if _has_following_unit(text, label.end() + match.end()):
        return None, "non_fg_quantity_after_fg" if index else "incompatible_or_missing_unit_basis"
    scale = 1.0 if percent else 100.0
    variation = match.group("variation")
    context = _context(match.group("context"), recovered)
    return ScalarEmission(
        measurement_label=kind, measurement_text_span=text[label.start():label.end() + match.end()],
        span_start=label.start(), span_end=label.end() + match.end(), unit_raw="%",
        unit_normalized="percent", local_measurement_context=context, measurement_timepoint=None,
        measurement_concentration=None, scalar_value=round(value * scale, 12), scalar_is_approximate=False,
        variation_value=round(float(variation.replace(",", "")) * scale, 12) if variation else None,
        variation_type="unspecified_variation" if variation else None, emission_index=index,
    ), None


def _has_following_unit(text: str, end: int) -> bool:
    return bool(re.match(r"\s*(?:mg|µg|ug|nm|µm|um|mm|ml|l)\b", text[end:], re.I))


def _context(value: str | None, recovered: bool) -> str | None:
    parts = [value.strip("() ")] if value else []
    if recovered:
        parts.append("support text re-extraction")
    return "; ".join(part for part in parts if part) or None


def _value_rejection(label: re.Match[str], match: re.Match[str], reason: str, kind: str) -> FragmentRejection:
    start, end = label.start(), label.end() + match.end()
    return FragmentRejection(kind, label.string[start:end], start, end, reason)


def _reindex(emissions: list[ScalarEmission]) -> list[ScalarEmission]:
    return [replace(item, emission_index=index) for index, item in enumerate(emissions)]
