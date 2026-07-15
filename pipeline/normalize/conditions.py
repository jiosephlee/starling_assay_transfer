"""Normalize free-text condition fields into equivalence classes for ``most_specific``.

``condition_keys.yaml`` most-specific schemas join on normalized condition fields
(``assay_system_family``, ``administration_route``, ``anatomical_site``,
``formulation_class``, ``intestinal_site_normalized``, ``scaling_method``,
``cofactor_class``). The raw sources for these are extreme free text (thousands of
distinct spellings), which ``docs/assay_transfer_design.md`` section 6.3 forbids from a
join key. This module maps that free text to a small, documented set of equivalence
classes by keyword; anything it cannot confidently classify becomes ``None``, so the row
is excluded from ``most_specific`` (``missing_policy: exclude``; unknown never equals
unknown).

Fields with no reliable structured raw source (``scaling_method``, ``cofactor_class``)
resolve to ``None`` for now — their endpoints' ``most_specific`` universes are therefore
empty until a structured source exists. This is intentional and reported, not silently
filled from ambiguous text.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Normalized condition columns emitted into base for most-specific keying.
CONDITION_COLUMNS = (
    "assay_system_family",
    "administration_route",
    "anatomical_site",
    "formulation_class",
    "intestinal_site_normalized",
    "scaling_method",
    "cofactor_class",
)

# Ordered (specific -> general) keyword -> class rules. First hit wins.
_ASSAY_SYSTEM_FAMILY = [
    ("microsome", "microsome"),
    ("hepatocyte", "hepatocyte"),
    ("s9", "s9_fraction"),
    ("recombinant", "recombinant_enzyme"),
    ("supersome", "recombinant_enzyme"),
    ("baculo", "recombinant_enzyme"),
    ("caco-2", "caco2"),
    ("caco2", "caco2"),
    ("mdck", "mdck"),
    ("pampa", "pampa"),
    ("everted", "everted_gut_sac"),
    ("gut sac", "everted_gut_sac"),
    ("ussing", "ussing_chamber"),
    ("epiintestinal", "intestinal_microtissue"),
    ("microtissue", "intestinal_microtissue"),
    ("in situ", "in_situ_perfusion"),
    ("in-situ", "in_situ_perfusion"),
    ("perfus", "perfusion"),
    ("in vivo", "in_vivo"),
    ("oral administration", "in_vivo"),
    ("oral absorption", "in_vivo"),
    ("pharmacokinetic", "in_vivo"),
    ("deconvolution", "in_vivo"),
    ("plasma", "in_vivo"),
]

_ROUTE = [
    ("intraduoden", "intraduodenal"),
    ("intravenous", "iv"),
    (" i.v", "iv"),
    ("intraperitoneal", "ip"),
    ("oral", "oral"),
    ("p.o", "oral"),
    ("gavage", "oral"),
    ("duodenal infusion", "intraduodenal"),
]

_ANATOMICAL_SITE = [
    ("duoden", "duodenum"),
    ("jejun", "jejunum"),
    ("ileum", "ileum"),
    ("ileal", "ileum"),
    ("colon", "colon"),
    ("large intestine", "colon"),
    ("small intestine", "small_intestine"),
    ("stomach", "stomach"),
    ("gastric", "stomach"),
    ("intestin", "intestine"),
]

_FORMULATION = [
    ("suspension", "suspension"),
    ("tablet", "tablet"),
    ("capsule", "capsule"),
    ("oral solution", "solution"),
    ("aqueous solution", "solution"),
    ("solution", "solution"),
    ("powder", "powder"),
    ("emulsion", "emulsion"),
    ("micelle", "micellar"),
    ("amorphous", "amorphous"),
    ("crystal", "crystalline"),
    ("polymorph", "crystalline"),
    ("salt", "salt"),
    ("free base", "free_base"),
]


def _classify(text: str, rules: list[tuple[str, str]]) -> Optional[str]:
    low = text.lower()
    for needle, label in rules:
        if needle in low:
            return label
    return None


def _first_text(row: dict[str, Any], columns: list[str]) -> str:
    parts = [str(row.get(c)) for c in columns if row.get(c) is not None and str(row.get(c)).strip()]
    return " | ".join(parts)


# Field -> (source columns to search, keyword rules). Fields absent here resolve to None.
_FIELD_RULES: dict[str, tuple[list[str], list[tuple[str, str]]]] = {
    "assay_system_family": (["assay_system"], _ASSAY_SYSTEM_FAMILY),
    "administration_route": (["assay_system", "biological_context"], _ROUTE),
    "anatomical_site": (["biological_context", "assay_system", "intestinal_site"], _ANATOMICAL_SITE),
    "formulation_class": (["formulation_or_solid_form"], _FORMULATION),
    "intestinal_site_normalized": (["intestinal_site", "biological_context"], _ANATOMICAL_SITE),
}


def normalize_conditions(row: dict[str, Any]) -> dict[str, Optional[str]]:
    """Return every normalized condition column for one raw row (None when unresolved)."""
    out: dict[str, Optional[str]] = {c: None for c in CONDITION_COLUMNS}
    for field, (columns, rules) in _FIELD_RULES.items():
        text = _first_text(row, columns)
        if text:
            out[field] = _classify(text, rules)
    # scaling_method, cofactor_class: no reliable structured source yet -> None.
    return out
