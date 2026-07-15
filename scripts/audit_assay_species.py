#!/usr/bin/env python3
"""Audit explicit-only ``species_exact`` coverage for oral and Q1-Q4."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.normalize.assay_species_normalization import resolve_species_record  # noqa: E402

SOURCE_COLUMNS = {
    "oral_bioavailability": ["species_or_population"],
    "q1": ["study_context"],
    "q2": ["biological_context", "assay_system"],
    "q3": ["intestinal_site", "assay_system"],
    "q4": ["species", "assay_system"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/starling_assays/datasets"),
    )
    parser.add_argument(
        "--oral-path",
        type=Path,
        default=Path("datasets/base/Oral_bioavailability_cleaned/train.parquet"),
    )
    parser.add_argument("--top-unresolved", type=int, default=20)
    return parser.parse_args()


def _row_combinations(table: Any, columns: list[str]) -> Counter[tuple[Any, ...]]:
    values = [table[column].to_pylist() for column in columns]
    return Counter(zip(*values))


def audit_source(path: Path, source: str, top_unresolved: int) -> dict[str, Any]:
    columns = SOURCE_COLUMNS[source]
    table = pq.read_table(path, columns=columns)
    exact: Counter[str] = Counter()
    unresolved: Counter[str] = Counter()
    for values, count in _row_combinations(table, columns).items():
        row = dict(zip(columns, values))
        species_exact = resolve_species_record(row, source)
        if species_exact is not None:
            exact[species_exact] += count
        else:
            label = " | ".join(str(value or "") for value in values)
            unresolved[label] += count
    resolved_rows = sum(exact.values())
    return {
        "rows": table.num_rows,
        "resolved_exact_rows": resolved_rows,
        "unresolved_rows": table.num_rows - resolved_rows,
        "resolved_exact_fraction": resolved_rows / table.num_rows if table.num_rows else 0.0,
        "species_exact": dict(exact.most_common()),
        "top_unresolved": dict(unresolved.most_common(top_unresolved)),
    }


def main() -> None:
    args = parse_args()
    paths = {
        "oral_bioavailability": args.oral_path,
        **{
            source: args.dataset_root / source / "extractions.parquet"
            for source in ("q1", "q2", "q3", "q4")
        },
    }
    report = {
        source: audit_source(path, source, args.top_unresolved)
        for source, path in paths.items()
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
