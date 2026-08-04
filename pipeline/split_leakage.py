"""Detect and prevent cross-split molecule leakage.

A molecule "leaks" if its canonical_smiles appears in the row lists of more than one split (e.g.
train and validation) -- the model could see that exact molecule's structure during training via
one concept while its value for a different, held-out concept is being predicted in eval.

pipeline.pair_core.molecule_splits() is leak-free by construction: it assigns every molecule to
exactly one split, so a plain molecule-identity partition can never produce this on its own.
Leakage only enters when something *overrides* that partition for a subset of rows -- e.g.
relocating one concept's records into a different split than the rest of that molecule's records
sit in. This module exists so such an override can be checked and corrected explicitly (find the
rows it would leak, drop them) instead of silently producing cross-split molecule overlap.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def molecules_by_split(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    return {name: {row["canonical_smiles"] for row in rows} for name, rows in split_rows.items()}


def find_cross_split_molecules(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    """Return {canonical_smiles: {splits it appears in}} for every molecule present in 2+ splits."""
    membership: dict[str, set[str]] = defaultdict(set)
    for split, molecules in molecules_by_split(split_rows).items():
        for smiles in molecules:
            membership[smiles].add(split)
    return {smiles: splits for smiles, splits in membership.items() if len(splits) > 1}


def assert_no_leakage(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    leaked = find_cross_split_molecules(split_rows)
    if leaked:
        sample = dict(list(leaked.items())[:5])
        raise RuntimeError(f"{len(leaked)} molecules appear in more than one split: {sample}"
                           f"{' ...' if len(leaked) > 5 else ''}")


def drop_concept_leaks(split_rows: dict[str, list[dict[str, Any]]], *, concept: str,
                       keep_split: str) -> dict[str, list[dict[str, Any]]]:
    """Given a split assignment where every row of `concept` has already been force-placed into
    `keep_split` (e.g. "shove all Fg rows into train" regardless of where the rest of each
    molecule's rows sit), find which of those placements leak the molecule into a second split --
    i.e. that molecule has non-`concept` rows sitting in some other split -- and drop exactly
    those `concept` rows from `keep_split`. The other splits are untouched: they already hold
    each affected molecule's legitimate (non-`concept`) rows, which this never removes."""
    other_molecules: set[str] = set()
    for split, rows in split_rows.items():
        if split != keep_split:
            other_molecules.update(row["canonical_smiles"] for row in rows)
    cleaned = {name: list(rows) for name, rows in split_rows.items()}
    cleaned[keep_split] = [
        row for row in cleaned[keep_split]
        if not (row["assay_concept"] == concept and row["canonical_smiles"] in other_molecules)
    ]
    return cleaned
