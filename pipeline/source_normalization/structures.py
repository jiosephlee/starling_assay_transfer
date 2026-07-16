"""Authoritative structure resolution and TDC matching."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from rdkit import Chem

from pipeline.source_normalization.text import as_nonempty_text


@lru_cache(maxsize=200_000)
def canonicalize_smiles(smiles: str | None) -> tuple[str | None, str | None]:
    raw = as_nonempty_text(smiles)
    if raw is None:
        return None, "invalid_structure"
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return None, "invalid_structure"
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        return None, "wildcard_structure"
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return canonical, None


def collapse_mapping(rows: Iterable[tuple[str, str]]) -> tuple[dict[str, str], int]:
    grouped: dict[str, set[str]] = {}
    duplicate_rows = 0
    for identifier, smiles in rows:
        if identifier in grouped:
            duplicate_rows += 1
        grouped.setdefault(identifier, set()).add(smiles)
    conflicts = {key: values for key, values in grouped.items() if len(values) > 1}
    if conflicts:
        preview = list(sorted(conflicts))[:5]
        raise ValueError(f"conflicting SMILES mappings for identifiers: {preview}")
    return {key: next(iter(values)) for key, values in grouped.items()}, duplicate_rows


def compare_exported_smiles(exported: str | None, mapped: str) -> str:
    raw = as_nonempty_text(exported)
    if raw is None:
        return "missing"
    if raw == mapped:
        return "exact_match"
    exported_canonical, error = canonicalize_smiles(raw)
    mapped_canonical, mapped_error = canonicalize_smiles(mapped)
    if error is None and mapped_error is None and exported_canonical == mapped_canonical:
        return "canonical_match"
    return "mismatch"


def tdc_match(raw: str | None, canonical: str | None, exclusions: set[str]) -> bool:
    return raw in exclusions or canonical in exclusions

