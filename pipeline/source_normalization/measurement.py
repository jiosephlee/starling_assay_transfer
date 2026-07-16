"""Endpoint-agnostic measurement parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable

from pipeline.source_normalization.text import measurement_text, normalize_lexical

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?"
_SCI = re.compile(rf"({NUMBER})\s*[x×]\s*10\s*\^?\s*([-+]?\d+)", re.I)
_RANGE = re.compile(rf"({NUMBER})\s*(?:-|\bto\b)\s*({NUMBER})", re.I)
_PLUS_MINUS = re.compile(rf"({NUMBER})\s*(?:±|\+/-)\s*({NUMBER})", re.I)
_LOWER = re.compile(rf"(?:>=|≥|>|at least\s+|no less than\s+)\s*({NUMBER})", re.I)
_LOWER_AFTER = re.compile(rf"({NUMBER})\s*(?:or higher|and above|or more)\b", re.I)
_UPPER = re.compile(rf"(?:<=|≤|<|at most\s+|no more than\s+)\s*({NUMBER})", re.I)
_UPPER_AFTER = re.compile(rf"({NUMBER})\s*(?:or lower|and below|or less)\b", re.I)
_APPROX = re.compile(r"(?:≈|~|\bapprox(?:imately)?\b|\babout\b|\bca\.?\b|\bestimated\b)", re.I)
_UNIT = re.compile(
    r"(?:%|(?:p|n|u|µ|μ|m)?mol(?:e)?(?:s)?(?:[/· ](?:l|ml|min|h|s|kg|g|mg|10\^?\d+\s*cells?))+|"
    r"(?:n|u|µ|μ|m)?g(?:[/· ](?:l|ml|min|h|s|kg|g|mg))+|cm(?:\^?-?\d+)?(?:[/· ](?:s|h))+|"
    r"(?:u|µ|μ|m)?l(?:[/· ](?:min|h|s|kg|g|mg))+|(?:min|hours?|hrs?|h|seconds?|sec|s)|fold)",
    re.I,
)


@dataclass(frozen=True)
class ParsedMeasurement:
    scalar_value: float | None = None
    scalar_is_approximate: bool | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None
    categorical_value: str | None = None

    @property
    def successful(self) -> bool:
        values = asdict(self).values()
        return any(value is not None for value in values)

    @property
    def kind(self) -> str | None:
        names = []
        if self.scalar_value is not None:
            names.append("scalar")
        if self.lower_bound is not None or self.upper_bound is not None:
            names.append("bound")
        if self.interval_lower is not None or self.interval_upper is not None:
            names.append("interval")
        if self.categorical_value is not None:
            names.append("categorical")
        return "+".join(names) or None

    def columns(self) -> dict[str, Any]:
        return {**asdict(self), "measurement_kind": self.kind}


def _float(value: str) -> float:
    return float(value.replace(",", ""))


def _collapse_scientific(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{_float(match.group(1)) * 10 ** int(match.group(2)):.17g}"

    return _SCI.sub(replace, text)


def _bounds(text: str) -> tuple[float | None, float | None]:
    lower_match = _LOWER.search(text) or _LOWER_AFTER.search(text)
    upper_match = _UPPER.search(text) or _UPPER_AFTER.search(text)
    lower = _float(lower_match.group(1)) if lower_match else None
    upper = _float(upper_match.group(1)) if upper_match else None
    return lower, upper


def _interval(text: str) -> tuple[float | None, float | None]:
    match = _RANGE.search(text)
    if not match:
        return None, None
    first, second = _float(match.group(1)), _float(match.group(2))
    return min(first, second), max(first, second)


def _scalar(text: str, has_bound: bool) -> tuple[float | None, bool | None]:
    if has_bound:
        return None, None
    plus_minus = _PLUS_MINUS.search(text)
    if plus_minus:
        return _float(plus_minus.group(1)), True
    numbers = re.findall(NUMBER, text)
    if len(numbers) != 1:
        return None, None
    approximate = bool(_APPROX.search(text))
    return _float(numbers[0]), approximate


def approved_category(value: Any, allowlist: Iterable[str]) -> str | None:
    normalized = normalize_lexical(value)
    allowed = {normalize_lexical(item) for item in allowlist}
    return normalized if normalized in allowed else None


def parse_measurement(value: Any, categorical: str | None = None) -> ParsedMeasurement:
    text = measurement_text(value)
    if text is None:
        return ParsedMeasurement(categorical_value=categorical)
    collapsed = _collapse_scientific(text)
    lower, upper = _bounds(collapsed)
    interval_lower, interval_upper = _interval(collapsed)
    scalar, approximate = _scalar(collapsed, lower is not None or upper is not None)
    return ParsedMeasurement(
        scalar_value=scalar,
        scalar_is_approximate=approximate,
        lower_bound=lower,
        upper_bound=upper,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        categorical_value=categorical,
    )


def extract_embedded_unit(value: Any) -> str | None:
    text = measurement_text(value)
    if text is None:
        return None
    matches = _UNIT.findall(text)
    return matches[0].strip() if matches else None

