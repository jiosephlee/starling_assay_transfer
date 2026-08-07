"""Render source-faithful v11 assay-transfer prompts from the JSON contract."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline.prompt_rendering import clean_context
from pipeline.v11_contract import (
    DEFAULT_REGISTRY,
    TEMPLATE_ROOT,
    load_registry,
    project_prompt_record,
    source_config,
)

# Every v11 template is the same shape: a shared prelude, a retrieval-only block, a query-only
# block, and a shared answer stanza. The two blocks each depend on exactly one record, so they are
# compiled apart and rendered once per record instead of once per pair -- with ~1.1M pairs over
# ~225k records that is a ~5x reduction in Jinja work. Splitting happens here rather than in the
# template files so the .jinja sources stay the single readable definition of a prompt.
RETRIEVAL_MARKER = "Molecule A (known)\n"
QUERY_MARKER = "\nMolecule B (query)\n"
ANSWER_MARKER = "\n{{ p.answer() }}"
MACRO_IMPORT = '{% import "_macros.jinja" as p %}'
_BLOCK_CACHE_LIMIT = 1_000_000

# (task_id, source_id, side, record identity) -> rendered block
_block_cache: dict[tuple[str, str, str, str], str] = {}


def _clean_projected(record: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = clean_context(dict(record))
    for field, value in cleaned.items():
        if isinstance(value, float) and math.isnan(value):
            cleaned[field] = None
    return cleaned


@lru_cache(maxsize=None)
def _environment(template_root: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_root),
        undefined=StrictUndefined,
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
        auto_reload=False,
    )


@lru_cache(maxsize=None)
def _block_environment(template_root: str) -> Environment:
    """Same settings as ``_environment`` but preserving trailing newlines.

    The split fragments end on newlines that are structural (they separate the two molecule
    blocks); Jinja's default would eat one and silently change every prompt.
    """
    environment = _environment(template_root).overlay(keep_trailing_newline=True)
    environment.auto_reload = False
    return environment


@lru_cache(maxsize=None)
def _template_parts(template_root: str, template_name: str) -> tuple[str, Any, Any, str]:
    """Split one template into (rendered prelude, retrieval block, query block, rendered answer)."""
    environment = _block_environment(template_root)
    source = environment.loader.get_source(environment, template_name)[0]
    try:
        head, rest = source.split(RETRIEVAL_MARKER, 1)
        retrieval_block, rest = rest.split(QUERY_MARKER, 1)
        query_block = rest.split(ANSWER_MARKER, 1)[0]
    except ValueError as exc:
        raise ValueError(f"v11 template {template_name!r} does not match the block layout") from exc
    prelude = environment.from_string(head).render()
    answer = environment.from_string(f"{MACRO_IMPORT}{{{{ p.answer() }}}}").render()
    return (
        prelude,
        environment.from_string(MACRO_IMPORT + retrieval_block),
        environment.from_string(MACRO_IMPORT + query_block),
        answer,
    )


def _render_block(
    record: Mapping[str, Any], task_id: str, side: str, registry: Mapping[str, Any],
    config: Mapping[str, Any], template: Any,
) -> str:
    projected = _clean_projected(
        project_prompt_record(record, task_id, side, registry, config=config)
    )
    return template.render(**{side: projected})


def _block_for(
    record: Mapping[str, Any], task_id: str, source_id: str, side: str,
    registry: Mapping[str, Any], config: Mapping[str, Any], template: Any,
) -> str:
    identity = record.get("child_id")
    if identity is None:
        return _render_block(record, task_id, side, registry, config, template)
    key = (task_id, source_id, side, str(identity))
    cached = _block_cache.get(key)
    if cached is None:
        cached = _render_block(record, task_id, side, registry, config, template)
        if len(_block_cache) >= _BLOCK_CACHE_LIMIT:
            _block_cache.clear()
        _block_cache[key] = cached
    return cached


def reset_prompt_cache() -> None:
    """Drop memoized blocks; call when record contents change under a reused identity."""
    _block_cache.clear()


@lru_cache(maxsize=None)
def _registry(path: str) -> dict[str, Any]:
    return load_registry(Path(path))


@lru_cache(maxsize=None)
def _binding(path: str, task_id: str, source_id: str) -> dict[str, Any]:
    """Resolve and copy one source binding once per renderer process."""
    return source_config(task_id, source_id, _registry(path))


def render_prompt(
    query: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    template_root: Path = TEMPLATE_ROOT,
) -> str:
    """Render one same-task/source pair without exposing the query outcome."""
    task_id = str(query.get("task_id") or "")
    source_id = str(query.get("source_id") or "")
    if task_id != str(retrieval.get("task_id") or ""):
        raise ValueError("v11 prompt pair crosses task datasets")
    if source_id != str(retrieval.get("source_id") or ""):
        raise ValueError("v11 prompt pair crosses source datasets")
    registry_key = str(registry_path)
    registry = _registry(registry_key)
    config = _binding(registry_key, task_id, source_id)
    prelude, retrieval_template, query_template, answer = _template_parts(
        str(template_root), config["template"]
    )
    rendered_retrieval = _block_for(
        retrieval, task_id, source_id, "retrieval", registry, config, retrieval_template
    )
    rendered_query = _block_for(
        query, task_id, source_id, "query", registry, config, query_template
    )
    return (
        f"{prelude}{RETRIEVAL_MARKER}{rendered_retrieval}"
        f"{QUERY_MARKER}{rendered_query}\n{answer}"
    ).strip()


__all__ = ["render_prompt", "reset_prompt_cache"]
