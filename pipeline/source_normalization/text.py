"""Mechanical text and unit normalization without endpoint semantics."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pipeline.normalize.species_text import NULL_VALUES

_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "‒": "-", "−": "-"})
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def as_nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in NULL_VALUES:
        return None
    return text


def normalize_lexical(value: Any) -> str | None:
    text = as_nonempty_text(value)
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).lower()
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_unit(value: Any) -> str | None:
    text = as_nonempty_text(value)
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = text.translate(_SUPERSCRIPTS).replace("μ", "u").replace("µ", "u")
    text = text.replace("·", "*").replace("×", "x").lower()
    text = re.sub(r"\s*([/*^])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def measurement_text(value: Any) -> str | None:
    text = as_nonempty_text(value)
    if text is None:
        return None
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = text.translate(_SUPERSCRIPTS).replace("−", "-")
    return re.sub(r"\s+", " ", text).strip()

