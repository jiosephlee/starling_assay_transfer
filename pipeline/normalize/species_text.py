"""Shared text normalization primitives for every species resolver."""

from __future__ import annotations

import re
from typing import Any

NULL_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not applicable",
    "not reported",
    "not specified",
    "not stated",
    "null",
    "unknown",
    "unspecified",
}


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return None if text in NULL_VALUES else text
