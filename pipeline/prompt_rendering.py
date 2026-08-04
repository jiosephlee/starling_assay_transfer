"""Self-contained assay-transfer pairwise prompt rendering.

    from pipeline.prompt_rendering import render_prompt
    prompt = render_prompt(query_row, retrieval_row, template_dir="templates/assay_transfer_v8_intern")

``query_row``/``retrieval_row`` are plain dicts carrying whatever fields the templates in
``template_dir`` reference -- typically ``assay_concept``, ``canonical_smiles``, ``context_*``
fields, ``display_value``/``display_unit``, ``transfer_criterion_text``. This module has no
dependency on ``pipeline.pair_core`` or anything else in this repo's build pipeline -- it can be
copied out (this file + a templates directory) and reused standalone anywhere ``jinja2`` is
available.

Per-concept template selection falls back to ``default.jinja`` when ``template_dir`` has no
``<assay_concept>.jinja`` of its own.

Placeholder-token cleaning: TxAgent's eligible-records schema uses the literal string
``"__unknown__"`` as its "no information" sentinel for several context fields (confirmed via
direct inspection of the v7.1 eligible-records artifact: 534,910 occurrences across context
columns, heavily concentrated in a handful of fields -- e.g. 100% of ``oral_exposure`` rows'
``context_species_or_population``/``context_study_or_assay_system``). Unlike ``None`` or an empty
string, that sentinel is truthy, so a template's ``{{ x or "not specified" }}`` idiom doesn't catch
it and it leaks into prompts verbatim. ``clean_context()`` replaces any such sentinel string with
``None`` before rendering, so every template's existing fallback idiom works without each template
author needing to special-case it per field.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

PLACEHOLDER_TOKENS = frozenset({"__unknown__"})


def _clean_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in PLACEHOLDER_TOKENS:
        return None
    # Some fields (e.g. context_study_or_assay_system) are pipe-delimited compound values
    # ("solubility_assay|aqueous_solubility|__unknown__") where only one segment is a
    # placeholder -- drop just that segment rather than discarding the whole (partly real)
    # value. Confirmed against the v7.1 eligible-records artifact: 20,833 rows have this pattern,
    # concentrated entirely in context_study_or_assay_system.
    if "|" in stripped:
        segments = [segment for segment in stripped.split("|")
                   if segment.strip().lower() not in PLACEHOLDER_TOKENS]
        return "|".join(segments) or None
    return value


def clean_context(row: dict[str, Any]) -> dict[str, Any]:
    """Shallow copy of `row` with placeholder tokens cleaned from every string field -- whole-
    field sentinels become None; a sentinel segment inside a pipe-delimited compound value is
    dropped, keeping the rest."""
    return {key: _clean_value(value) for key, value in row.items()}


@lru_cache(maxsize=None)
def _environment(template_dir: str) -> Environment:
    return Environment(loader=FileSystemLoader(template_dir), undefined=StrictUndefined)


@lru_cache(maxsize=None)
def _template_for(template_dir: str, concept: str) -> Template:
    env = _environment(template_dir)
    available = {path.name for path in Path(template_dir).glob("*.jinja")}
    name = f"{concept}.jinja" if f"{concept}.jinja" in available else "default.jinja"
    return env.get_template(name)


def render_prompt(query: dict[str, Any], retrieval: dict[str, Any], *, template_dir: str) -> str:
    """Render the per-concept Jinja prompt for one query/retrieval pair. The query record's
    value is never templated -- only its context fields are shown."""
    template = _template_for(template_dir, str(query["assay_concept"]))
    return template.render(query=clean_context(query), retrieval=clean_context(retrieval))
