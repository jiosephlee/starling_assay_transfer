#!/usr/bin/env python3
"""Build v9 raw-pair Parquets from the v6-normalized TxAgent source (``starling_normalized_v6``,
via ``pipeline/source_normalization/starling_txagent_eligible_v6.py``).

Thin wrapper around ``scripts/build_v8_intern_raw_pair.py`` -- reuses its ranking-anchor
construction, ordinary-pair construction, degree caps, split-eval caching, and template
rendering entirely unchanged (imported, not copied), with exactly one override:

**Ordinary-pair candidate *selection*** (deciding which record pairs count as "transfer" vs
"nontransfer" when building the binary ordinary benchmark/train examples,
``_candidate_label`` inside ``_ordinary_pairs_v8``) uses a fixed, bucket-independent pair of SD
thresholds -- ``ORDINARY_CANDIDATE_TRANSFER_MAX_SD = 0.5`` / ``ORDINARY_CANDIDATE_NOT_TRANSFER_MIN_SD
= 1.5`` -- instead of each bucket's own ``(transfer_max=1.0, not_transfer_min=percentile-derived)``
pair. Per user decision: this keeps ordinary-pair selection restricted to visibly clear-cut
examples (near-identical vs. clearly distant) on a single, easy-to-reason-about scale, while
leaving everything else untouched:

- Every row's actual *continuous* target (``target_a``/``target_b``/``completion``/``target_z``,
  computed by ``pipeline/pair_core.py::target_for()`` from each record's own
  ``transfer_max``/``not_transfer_min``) is unaffected -- ``target_for()`` itself is never
  patched here, only the separate binary bucketing that decides which *candidates* get proposed
  as "transfer"/"nontransfer" in the first place.
- Ranking's value-based spread (``_ranking_span``, sorted by ``target_z``) doesn't call
  ``_candidate_label`` at all, so it's untouched by construction.
- Train's flat pair generation (``iter_list_groups``/``_iter_list_groups_capped``) also never
  calls ``_candidate_label`` -- confirmed via ``scripts/build_raw_pair.py::build()``, which only
  invokes ``_ordinary``/``_ranking`` for the ``validation``/``test`` splits, never for ``train``.

Since ``comparison_value`` is already ``raw_value / bucket_sample_standard_deviation`` (see
``starling_txagent_eligible_v6.py``), ``target_for(query, retrieval)["distance"]`` is already
expressed directly in standardized-distance (SD) units, comparable across buckets with no
further rescaling -- a fixed global SD cutoff here is exactly as meaningful as it is anywhere
else in this pipeline.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import pyarrow.parquet as pq

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v8_intern_raw_pair as v8
from pipeline.pair_core import target_for
from scripts.build_raw_pair import build
from scripts.build_v7_intern_raw_pair import _iter_list_groups_capped, _pair_bucket_concentration, _train_target_diagnostics, _write_train_flat_variable

ORDINARY_CANDIDATE_TRANSFER_MAX_SD = 0.5
ORDINARY_CANDIDATE_NOT_TRANSFER_MIN_SD = 1.5


def _candidate_label_fixed_sd(query: dict, retrieval: dict) -> str | None:
    """Ordinary-pair candidate bucketing on a fixed, bucket-independent SD scale -- distinct
    from query["transfer_max"]/["not_transfer_min"], which stay bucket-specific and keep driving
    every row's actual continuous target via target_for(), untouched by this override."""
    distance = target_for(query, retrieval)["distance"]
    if distance <= ORDINARY_CANDIDATE_TRANSFER_MAX_SD:
        return "transfer"
    return "nontransfer" if distance >= ORDINARY_CANDIDATE_NOT_TRANSFER_MIN_SD else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/eligible/assay_transfer_starling_txagent_v9/records.parquet"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v9_intern"),
    )
    parser.add_argument("--listnet", action="store_true",
                        help="build indivisible 4-member ListNet groups; default is flat per-pair (v6_5-style)")
    parser.add_argument("--train-groups", type=int, default=250000)
    parser.add_argument("--train-pairs", type=int, default=1250000)
    args = parser.parse_args()

    train_group_iterator = (functools.partial(_iter_list_groups_capped, min_group_size=4)
                            if args.listnet else _iter_list_groups_capped)

    # The one override: ordinary-pair candidate selection only. See module docstring for why
    # this is safe to patch as a bare module attribute (Python resolves the bare name
    # `_candidate_label` inside `_ordinary_pairs_v8` against this module's globals at call time).
    v8._candidate_label = _candidate_label_fixed_sd

    split_records_fn, cache = v8._make_split_eval_v8()
    rows = pq.read_table(args.source).to_pylist()
    expected_records = len(rows)
    split_rows = split_records_fn(rows)
    expected_split = {name: len(values) for name, values in split_rows.items()}

    validation_molecules, test_molecules = cache["validation"]["molecules"], cache["test"]["molecules"]
    train_molecules = {str(row["canonical_smiles"]) for row in split_rows["train"]}
    assert not (train_molecules & (validation_molecules | test_molecules)), \
        "train leaks molecules already used in validation/test"

    overlap_molecules = v8._overlap_molecules(rows)
    split_quotas = {
        "molecules": {name: len({r["canonical_smiles"] for r in values})
                     for name, values in split_rows.items()},
        "records": expected_split,
        "overlap_molecule_fraction": {
            "validation": round(len(validation_molecules & overlap_molecules) / max(1, len(validation_molecules)), 4),
            "test": round(len(test_molecules & overlap_molecules) / max(1, len(test_molecules)), 4),
        },
    }

    ordinary_fn, ranking_fn = v8._make_eval_readers(cache)
    pair_core.eligible = v8._eligible_with_pair_bucket
    pair_core.render_prompt = v8._render_prompt_v8
    pair_core.iter_list_groups = train_group_iterator
    pair_driver._ordinary = ordinary_fn
    pair_driver._ranking = ranking_fn
    pair_driver.iter_list_groups = train_group_iterator
    pair_driver._write_train_flat = _write_train_flat_variable
    pair_driver._split_records = split_records_fn

    volume = args.train_groups if args.listnet else args.train_pairs
    manifest = build(
        args.source, args.output, volume, listnet=args.listnet,
        expected_records=expected_records,
        expected_split=expected_split,
    )
    manifest["version"] = "v9-txagent-raw-pair"
    manifest["split_quotas"] = split_quotas
    manifest["notes"] = {
        "starling_overlap_pairs_first_val_test": True,
        "fg_in_eval": True,
        "ordinary_candidate_thresholds_sd": {
            "transfer_max": ORDINARY_CANDIDATE_TRANSFER_MAX_SD,
            "not_transfer_min": ORDINARY_CANDIDATE_NOT_TRANSFER_MIN_SD,
        },
    }
    manifest["diagnostics"] = {
        **v8._eval_diagnostics(cache),
        "pair_bucket_concentration_by_concept": _pair_bucket_concentration(args.source),
        "train_target_distribution": _train_target_diagnostics(Path(manifest["paths"]["train"])),
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
