"""Explicit-only species normalization for oral and Q1-Q4 records.

The normalized contract contains one nullable field, ``species_exact``. It is
populated only from a dedicated species column or an exact species designation in
an approved structured text column. The resolver does not infer species from
population vocabulary, generic taxonomic groups, or cell-line origin.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from species_text import normalize_text

SPECIES_EXACT_COLUMN = "species_exact"
SPECIES_OUTPUT_COLUMNS = (SPECIES_EXACT_COLUMN,)
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "assay_transfer"
    / "v1"
    / "species.yaml"
)


@lru_cache(maxsize=4)
def load_species_config(path: str = str(DEFAULT_CONFIG_PATH)) -> Mapping[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("version") != "species_v1":
        raise ValueError(f"unsupported species config version: {config.get('version')!r}")
    return config


def normalize_assay_text(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace("‑", "-").replace("‒", "-").replace("_", " ")
    return re.sub(r"\s+", " ", text)


def _as_texts(values: Any | Iterable[Any]) -> tuple[str, ...]:
    if values is None:
        return ()
    items = (values,) if isinstance(values, (str, bytes)) else tuple(values)
    normalized = (normalize_assay_text(value) for value in items)
    return tuple(text for text in normalized if text is not None)


@lru_cache(maxsize=4)
def _exact_alias_patterns(
    config_path: str,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    config = load_species_config(config_path)
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for label, aliases in config["exact_aliases"].items():
        for alias in aliases:
            normalized = normalize_assay_text(alias)
            if normalized is None:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
            patterns.append((label, re.compile(pattern)))
    return tuple(patterns)


@lru_cache(maxsize=100_000)
def _matching_labels(text: str, config_path: str) -> tuple[str, ...]:
    matches: list[tuple[str, int, int]] = []
    for label, pattern in _exact_alias_patterns(config_path):
        if label == "human" and re.search(r"\bnon[- ]?human\b", text):
            continue
        for match in pattern.finditer(text):
            matches.append((label, match.start(), match.end()))
    kept = []
    for label, start, end in matches:
        contained = any(
            other_start <= start
            and other_end >= end
            and other_end - other_start > end - start
            for _, other_start, other_end in matches
        )
        if not contained:
            kept.append(label)
    return tuple(sorted(set(kept)))


def matching_exact_species(
    values: Any | Iterable[Any], config_path: str = str(DEFAULT_CONFIG_PATH)
) -> tuple[str, ...]:
    labels: set[str] = set()
    for text in _as_texts(values):
        labels.update(_matching_labels(text, config_path))
    return tuple(sorted(labels))


def _one_exact_or_none(values: Any | Iterable[Any], config_path: str) -> str | None:
    labels = matching_exact_species(values, config_path)
    return labels[0] if len(labels) == 1 else None


def resolve_assay_species(
    *,
    explicit: Any | Iterable[Any] = None,
    explicit_texts: Any | Iterable[Any] = (),
    contexts: Any | Iterable[Any] = (),
    assay_systems: Any | Iterable[Any] = (),
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> str | None:
    """Return one explicitly named exact species, otherwise ``None``.

    A populated dedicated species value is authoritative, including when it is too
    generic or ambiguous to normalize. The other arguments are approved structured
    text fields and are considered only when the dedicated value is absent.
    """
    explicit_values = _as_texts(explicit)
    if explicit_values:
        return _one_exact_or_none(explicit_values, config_path)
    approved_texts = (*_as_texts(explicit_texts), *_as_texts(contexts), *_as_texts(assay_systems))
    return _one_exact_or_none(approved_texts, config_path)


def resolve_species_record(
    row: Mapping[str, Any],
    source_collection: str,
    config_path: str = str(DEFAULT_CONFIG_PATH),
) -> str | None:
    config = load_species_config(config_path)
    try:
        fields = config["source_fields"][source_collection]
    except KeyError as exc:
        raise KeyError(f"unknown source collection: {source_collection!r}") from exc
    explicit_column = fields["explicit_species_column"]
    explicit = row.get(explicit_column) if explicit_column else None
    texts = [row.get(column) for column in fields["explicit_text_columns"]]
    return resolve_assay_species(
        explicit=explicit,
        explicit_texts=texts,
        config_path=config_path,
    )
