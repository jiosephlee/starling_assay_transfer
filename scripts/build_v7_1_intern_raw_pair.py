#!/usr/bin/env python3
"""Build v7.1 raw-pair Parquets from the TxAgent-sourced eligible records, now including Fg.

Thin wrapper around ``scripts/build_v7_intern_raw_pair.py`` -- reuses all of its pairing/labeling/
degree-cap/bucket-round-robin/rendering logic unchanged (imported directly, not copied). The only
difference from v7: TxAgent's Fg normalization was rebuilt and now yields 1,056 eligible records
(462 distinct molecules) where it previously had zero (v7's docstring: "all 27,671 fg rows fail
with missing_canonical_unit/unresolved_structure upstream"). v7's other four concepts
(oral_bioavailability/oral_exposure/Fa/Fh) are numerically identical between the two eligible-
records snapshots -- confirmed by comparing per-concept counts before/after the TxAgent rebuild --
so this is purely an additive change.

Per user decision, Fg is train-only: it never appears in validation, test, or either ranking split.
``_ordinary_variable_concepts``/``_ranking_variable_size`` (imported from v7, unmodified) derive
their concept list dynamically from whatever's present in the split they're given, so once Fg is
absent from the validation/test row lists it's automatically excluded from both benchmarks with no
separate filtering needed.

``--split-mode`` controls how the base train/validation/test molecule partition is built, before Fg
is handled at all:

- ``reuse`` (default): freezes validation/test to the *exact same 1,250+1,250 molecules* as the
  original v7 build. ``_old_v7_held_out_molecules()`` recomputes that identity deterministically
  from the untouched ``assay_transfer_starling_txagent_v7`` artifact (same ``molecule_splits()``
  call, same frozen v7 SPLIT_SIZES/SPLIT_RECORDS) rather than re-deriving a fresh split here --
  re-running ``_heldout_subset`` against the new (larger, Fg-inclusive) weight distribution would
  risk picking a *different* held-out set purely because Fg's added record weights shift SHA-256
  tie-breaking priority for molecules that happen to have new Fg records.
- ``recompute``: derives a fresh 1,250/1,250-molecule held-out set from v7.1's own weight
  distribution via ``pipeline.pair_core.molecule_splits()``, the same mechanism v7 used originally.
  Target record totals per split are derived proportionally from the live weight distribution
  (``_fresh_split_targets()``), not hardcoded, since this mode's entire purpose is adapting to
  whatever the current eligible-records snapshot contains. One caveat: a molecule whose *only*
  eligible records are Fg can still be chosen into validation/test purely on Fg weight, then lose
  all its rows once Fg is dropped from eval (see below) -- so under this mode validation/test can
  land a little under the 1,250-molecule target (observed: ~1,238-1,241). ``reuse`` mode does not
  have this issue, since the frozen v7 identity was computed before Fg existed at all.

Either way, this produces the ordinary leak-free molecule partition first (every molecule to exactly
one split, Fg included, no special-casing). Only afterward does ``_route_fg_to_train()`` build a
*candidate* override that force-relocates every Fg row into train and strips it from
validation/test, and ``pipeline.split_leakage.drop_concept_leaks()`` removes exactly the Fg rows
that override would leak -- any molecule whose non-Fg records already sit in a different split --
from train, rather than letting them silently land in two splits. ``split_leakage.assert_no_leakage``
then verifies the result is clean. This is a general, reusable pattern (not Fg-specific): the same
two calls handle any future "force this concept into that split" override safely.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

import pipeline.pair_core as pair_core
import pipeline.split_leakage as split_leakage
import scripts.build_raw_pair as pair_driver
from pipeline.pair_core import record_key
from scripts.build_raw_pair import build
from scripts.build_v7_intern_raw_pair import (
    _eligible_with_pair_bucket, _iter_list_groups_capped, _ordinary_variable_concepts,
    _pair_bucket_concentration, _ranking_variable_size, _render_prompt_v7,
    _train_target_diagnostics, _write_train_flat_variable,
)

# v7's frozen artifact + quotas, needed only to recompute its exact validation/test molecule
# identity for --split-mode=reuse -- untouched by this script.
OLD_V7_SOURCE = Path("datasets/eligible/assay_transfer_starling_txagent_v7/records.parquet")
OLD_V7_SPLIT_SIZES = {"train": 15906, "validation": 1250, "test": 1250}
OLD_V7_SPLIT_RECORDS = {"train": 173853, "validation": 13662, "test": 13662}

FG_CONCEPT = "Fg"


def _old_v7_held_out_molecules() -> tuple[set[str], set[str]]:
    """Recompute v7's exact validation/test molecule sets from its untouched eligible-records
    artifact, so --split-mode=reuse can freeze the same held-out identity instead of deriving one."""
    old_rows = pq.read_table(OLD_V7_SOURCE).to_pylist()
    saved_sizes, saved_records = pair_core.SPLIT_SIZES, pair_core.SPLIT_RECORDS
    pair_core.SPLIT_SIZES, pair_core.SPLIT_RECORDS = OLD_V7_SPLIT_SIZES, OLD_V7_SPLIT_RECORDS
    try:
        split_map = pair_core.molecule_splits(old_rows)
    finally:
        pair_core.SPLIT_SIZES, pair_core.SPLIT_RECORDS = saved_sizes, saved_records
    validation = {smiles for smiles, split in split_map.items() if split == "validation"}
    test = {smiles for smiles, split in split_map.items() if split == "test"}
    return validation, test


def _fresh_split_targets(rows: list[dict], *, validation_molecules: int = 1250,
                         test_molecules: int = 1250) -> tuple[dict[str, int], dict[str, int]]:
    """Proportional record-count targets for a fresh molecule_splits() call, derived from this
    dataset's own weight distribution rather than a hardcoded historical number."""
    weights: dict[str, int] = defaultdict(int)
    for row in rows:
        weights[str(row["canonical_smiles"])] += 1
    total_records, total_molecules = sum(weights.values()), len(weights)
    per_split_records = round(total_records * validation_molecules / total_molecules)
    sizes = {"train": total_molecules - validation_molecules - test_molecules,
             "validation": validation_molecules, "test": test_molecules}
    records = {"train": total_records - 2 * per_split_records,
               "validation": per_split_records, "test": per_split_records}
    return sizes, records


def _base_split_rows(rows: list[dict], *, mode: str) -> dict[str, list[dict]]:
    """The ordinary leak-free molecule partition -- every molecule to exactly one split, every
    concept (Fg included) treated uniformly, no special-casing. See module docstring for modes."""
    if mode == "reuse":
        validation_molecules, test_molecules = _old_v7_held_out_molecules()
        split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
        for row in rows:
            if row["canonical_smiles"] in validation_molecules:
                split_rows["validation"].append(row)
            elif row["canonical_smiles"] in test_molecules:
                split_rows["test"].append(row)
            else:
                split_rows["train"].append(row)
        return split_rows
    if mode == "recompute":
        sizes, records = _fresh_split_targets(rows)
        pair_core.SPLIT_SIZES, pair_core.SPLIT_RECORDS = sizes, records
        split_map = pair_core.molecule_splits(rows)
        split_rows = {"train": [], "validation": [], "test": []}
        for row in rows:
            split_rows[split_map[row["canonical_smiles"]]].append(row)
        return split_rows
    raise ValueError(f"unknown --split-mode {mode!r}")


def _route_fg_to_train(split_rows: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Force every Fg row into train, then use pipeline.split_leakage to drop exactly the ones
    that would leak a held-out molecule's structure into train. General pattern, not Fg-specific:
    any future "force concept X into split Y" override should go through the same two calls."""
    candidate = {name: list(rows) for name, rows in split_rows.items()}
    fg_rows = [row for row in candidate["validation"] + candidate["test"]
              if row["assay_concept"] == FG_CONCEPT]
    candidate["validation"] = [row for row in candidate["validation"] if row["assay_concept"] != FG_CONCEPT]
    candidate["test"] = [row for row in candidate["test"] if row["assay_concept"] != FG_CONCEPT]
    candidate["train"] = candidate["train"] + fg_rows
    cleaned = split_leakage.drop_concept_leaks(candidate, concept=FG_CONCEPT, keep_split="train")
    split_leakage.assert_no_leakage(cleaned)
    return {name: sorted(rows, key=record_key) for name, rows in cleaned.items()}


def _make_split_records_v7_1(mode: str):
    """Memoizing closure: build_raw_pair.build() calls _split_records once per build,
    but main() also needs the resulting counts up front to pass as build()'s expected_split --
    caching avoids repeating a --split-mode=recompute run (~15-20s)."""
    cache: dict[str, dict[str, list[dict]]] = {}

    def _split_records_v7_1(rows: list[dict]) -> dict[str, list[dict]]:
        if not cache:
            cache["result"] = _route_fg_to_train(_base_split_rows(rows, mode=mode))
        return cache["result"]
    return _split_records_v7_1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/eligible/assay_transfer_starling_txagent_v7_1/records.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v7_1_intern"),
    )
    parser.add_argument("--split-mode", choices=("reuse", "recompute"), default="reuse",
                        help="reuse: freeze validation/test to v7's exact held-out molecules "
                             "(default). recompute: derive a fresh 1250/1250 held-out set from "
                             "v7.1's own weight distribution.")
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups; default is flat per-pair (v6_5-style)")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    args = parser.parse_args()

    train_group_iterator = (functools.partial(_iter_list_groups_capped, min_group_size=4)
                            if args.listnet else _iter_list_groups_capped)

    split_records_fn = _make_split_records_v7_1(args.split_mode)
    rows = pq.read_table(args.source).to_pylist()
    expected_records = len(rows)
    split_rows = split_records_fn(rows)
    expected_split = {name: len(values) for name, values in split_rows.items()}
    split_quotas = {"molecules": {name: len({r["canonical_smiles"] for r in values})
                                  for name, values in split_rows.items()},
                    "records": expected_split}

    pair_core.eligible = _eligible_with_pair_bucket
    pair_core.render_prompt = _render_prompt_v7
    pair_core.iter_list_groups = train_group_iterator
    pair_driver._ordinary = _ordinary_variable_concepts
    pair_driver._ranking = _ranking_variable_size
    pair_driver.iter_list_groups = train_group_iterator
    pair_driver._write_train_flat = _write_train_flat_variable
    pair_driver._split_records = split_records_fn

    volume = args.train_groups if args.listnet else args.train_pairs
    manifest = build(
        args.source, args.output, volume, listnet=args.listnet,
        expected_records=expected_records,
        expected_split=expected_split,
    )
    manifest["version"] = "v7.1-txagent-raw-pair"
    manifest["split_mode"] = args.split_mode
    manifest["split_quotas"] = split_quotas
    manifest["notes"] = {
        "fg_train_only": True,
        "validation_test_molecules_frozen_to": (
            "assay_transfer_starling_txagent_v7" if args.split_mode == "reuse" else None),
    }
    manifest["diagnostics"] = {
        "pair_bucket_concentration_by_concept": _pair_bucket_concentration(args.source),
        "train_target_distribution": _train_target_diagnostics(Path(manifest["paths"]["train"])),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
