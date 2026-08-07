"""Adapt TxAgent's finished Bioavailability_Ma v6 artifacts into our eligible-records schema.

Sibling of ``starling_txagent_eligible.py`` (v5-sourced, untouched, still used by v7/v7.1/v8's
already-shipped eligible-records artifacts) -- **not a modification of it**. TxAgent's
``starling_normalized_v6`` artifact replaced the entire per-metric calibrated-threshold system v5
had (``endpoint_policies/v2`` registry: per-``endpoint_policy_key`` ``transform``/
``raw_distance_anchor``/``strict``-``primary``-``permissive`` threshold variants, joined via a
separate ``endpoint_policy_assignments.parquet``) with a single universal per-``pair_bucket_key``
rule: ``standardized_distance = abs(left_raw - right_raw) / bucket_sample_standard_deviation``,
eligible for transfer-pairing only if ``standardized_distance <= 1.0``
(``05_assay_transfer_policy/pair_bucket_transfer_policy.json.gz``, ``distance_contract.
transfer_max_standard_deviations``). No per-metric transform, no threshold variant, no pre-scaled
comparison value -- these methodologies are different enough that branching one file around both
would be worse than two clean files.

Our ``pipeline/pair_core.py::target_for()`` needs a *deadband* (``transfer_max``/``not_transfer_min``)
to compute its sigmoid target, but v6 only ships one cutoff. Per user decision, both thresholds are
reconstructed from v6's own per-bucket calibration data, not invented: every ``assay_transfer_eligible``
bucket ships ``distance_policy.standardized_distance_percentile_knots`` (a 101-point percentile CDF,
0..100, of standardized distances between random within-bucket pairs) alongside
``sample_standard_deviation``. We set:

- ``comparison_value = finite_scalar_value / bucket_sample_standard_deviation`` (pre-divides by the
  bucket's own empirical SD, so ``distance = |comparison_value_a - comparison_value_b|`` reproduces
  v6's own standardized-distance formula exactly).
- ``transfer_max = 1.0`` (matches v6's own ``transfer_max_standard_deviations``, the cutoff TxAgent
  itself already trusts).
- ``not_transfer_min = standardized_distance_percentile_knots[p]`` where
  ``p = min(100, round(one_sd_percentile_0_100) + NOT_TRANSFER_MIN_MARGIN)`` -- a percentile
  ``NOT_TRANSFER_MIN_MARGIN`` points *above* wherever this bucket's own 1.0-SD cutoff
  (``transfer_max``) happens to fall in its distribution, rather than a fixed global percentile.
  A fixed percentile (90th, tried first) turned out wrong empirically: ``one_sd_percentile_0_100``
  itself ranges from ~46 to ~99 across the 289 eligible buckets (median ~77), so for 77/289 buckets
  a fixed 90th-percentile floor already sat *below* the 1.0-SD cutoff -- an inverted, degenerate
  deadband that silently discarded 108,301 of 199,595 eligible records. Anchoring the margin to each
  bucket's own ``one_sd_percentile_0_100`` instead guarantees ``not_transfer_min`` percentile-ranks
  strictly above ``transfer_max`` for every bucket (verified: 0/289 buckets degenerate at margin>=5;
  ``NOT_TRANSFER_MIN_MARGIN=20`` used for a comfortable "clearly on the far side" gap). Data-driven
  per bucket, not an arbitrary global multiplier.

``pipeline/pair_core.py::target_for()``'s z-score is scale-invariant to a common linear rescale of
(comparison_value, transfer_max, not_transfer_min), so this reproduces the intended sigmoid shape
while trusting v6's own calibration end to end, the same way the v5 adapter trusted the v5 registry.

v6 has no finer-grained "endpoint_policy_key" than ``pair_bucket_key`` any more (the whole per-metric
registry is gone) -- ``canonical_endpoint_key`` is set equal to ``pair_bucket_key`` here, which makes
``pipeline/pair_core.py::eligible()``'s ``canonical_endpoint_key`` check and
``scripts/build_v7_intern_raw_pair.py::_eligible_with_pair_bucket()``'s extra ``pair_bucket_key``
check redundant but harmless -- both still work unmodified against this adapter's output.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TXAGENT_ROOT = Path("/data1/joseph/TxAgent")
if str(TXAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TXAGENT_ROOT))

from tools.chembl_tool.common.starling.heldout_index import (  # noqa: E402
    identity_key,
    load_heldout_identity_keys,
)
from tools.chembl_tool.common.molecule_identity import normalize_molecule_identity  # noqa: E402

_TXAGENT_V6_DIR = (
    TXAGENT_ROOT
    / "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v6"
)
DEFAULT_RECORDS = _TXAGENT_V6_DIR / "03_records" / "records.parquet"
DEFAULT_PAIR_BUCKETS = _TXAGENT_V6_DIR / "04_pair_buckets" / "pair_bucket_records.parquet"
DEFAULT_POLICY = _TXAGENT_V6_DIR / "05_assay_transfer_policy" / "pair_bucket_transfer_policy.json.gz"
DEFAULT_RANDOM_HELDOUT = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/random/test_molecule_labels.jsonl"
DEFAULT_SCAFFOLD_HELDOUT = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/scaffold/test_molecule_labels.jsonl"
# TxAgent's own authoritative heldout identity, one file per split methodology -- each already a
# pre-combined union of that methodology's valid+test molecule identities (209+209=418 apiece),
# newly added alongside the older train/valid/test_molecule_labels.jsonl files (which DEFAULT_*
# _HELDOUT above only partially reconstructed by reading test_molecule_labels.jsonl alone, missing
# the valid half). Used by _heldout_union_molecules() to let TxAgent's own identity -- not our own
# construction algorithm's touch order -- decide validation/test molecule membership.
DEFAULT_RANDOM_HELDOUT_UNION = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/random/heldout_molecule_labels.jsonl"
DEFAULT_SCAFFOLD_HELDOUT_UNION = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/scaffold/heldout_molecule_labels.jsonl"

# Percentile points added to a bucket's own one_sd_percentile_0_100 to pick the
# standardized_distance_percentile_knots index used as the non-transfer floor. See module
# docstring for rationale (a fixed global percentile does not work: verified empirically to be
# safe from margin=5 upward across all 289 eligible buckets; 20 gives a comfortable margin).
NOT_TRANSFER_MIN_MARGIN = 20

# source_id -> our assay_concept, mirrors configs/assay_transfer/v3/concepts.yaml's q1..q4/starling
# rules -- unchanged from the v5 adapter.
SOURCE_TO_CONCEPT = {
    "direct_hf": "oral_bioavailability",
    "oral_exposure": "oral_exposure",
    "fa": "Fa",
    "fg": "Fg",
    "fh": "Fh",
}

# pipeline/pair_core.py CONTEXT_FIELDS <- TxAgent v6 records.parquet columns. canonical_dose_key/
# canonical_assay_system/canonical_species no longer exist in v6 at all (confirmed via schema diff);
# global_context/global_species_context are v6's coarser replacements, populated only for
# fa/fg/fh (null for direct_hf/oral_exposure) -- added ahead of the raw fallback fields since they
# carry real, if coarser, information where present.
CONTEXT_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "species_or_population": ("species_or_population", "global_species_context", "species"),
    "report_or_statistic_type": ("canonical_bioavailability_report_type", "bioavailability_report_type", "statistic_type"),
    "dose": ("dose", "oral_dose"),
    "study_or_assay_system": ("global_context", "study_context", "oral_exposure_mode", "assay_system"),
    "measured_process": ("endpoint_name",),
    "biological_context": ("biological_context",),
    "medium": ("condition_medium",),
    "formulation_or_solid_form": ("formulation_or_solid_form",),
    "transporter_or_enzyme": ("transporter_or_enzyme",),
    "substrate_status": ("substrate_status",),
    "intestinal_site": ("intestinal_site",),
    "molecular_form": ("molecular_form",),
    "enzyme_or_pathway": ("enzyme_or_pathway",),
    "qualifying_conditions": ("qualifying_conditions",),
    "comparator": ("comparator", "comparator_exposure"),
    "extra_details": ("extra_details",),
    # Populated for every row, but templates must only ever show this for the retrieval record
    # (never the query) -- support_text states the row's own measured value in prose in nearly
    # every sample checked, so showing it for the query would leak the hidden value. It's safe
    # for retrieval only because retrieval's value is already shown anyway.
    "support_text": ("support_text",),
}

REQUIRED_OUTPUT_FIELDS = (
    "child_id", "parent_provenance_id", "record_id", "source_id", "input_sha256",
    "source_smiles", "canonical_smiles", "canonical_endpoint_key", "endpoint_family",
    "endpoint_subtype", "unit_basis", "scalar_value", "scalar_is_approximate",
    "assay_concept", "metric_type", "comparison_value", "transfer_max", "not_transfer_min",
    "threshold_display", "measurement_label",
)


def _first_nonempty(row: pd.Series, fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return value
    return None


def _context(row: pd.Series) -> dict[str, Any]:
    return {
        f"context_{name}": (None if (value := _first_nonempty(row, fields)) is None else str(value))
        for name, fields in CONTEXT_FIELD_SOURCES.items()
    }


def _stable_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_frames(records_path: Path, pair_buckets_path: Path) -> pd.DataFrame:
    """Two-way merge only -- v6 has no per-record policy-assignment file (the whole per-endpoint
    registry it used to join is gone); pair_bucket_records.parquet already carries everything a
    record needs to look its transfer policy up (pair_bucket_key)."""
    records = pd.read_parquet(records_path)
    pair_buckets = pd.read_parquet(
        pair_buckets_path, columns=["normalized_record_id", "pair_bucket_key", "bucket_eligible"],
    )
    if len(records) != len(pair_buckets):
        raise ValueError("records/pair_buckets row-count mismatch")
    return records.merge(
        pair_buckets, on="normalized_record_id", how="left", validate="one_to_one",
    )


def _heldout_parent_keys(heldout_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in heldout_paths:
        keys |= load_heldout_identity_keys(path)
    return keys


def _record_parent_key(row: pd.Series) -> str:
    identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
    return identity.parent_inchi_key or identity.parent_smiles


def _heldout_union_molecules(rows: list[dict[str, Any]]) -> set[str]:
    """Union of random's and scaffold's TxAgent-authoritative heldout molecule identities
    (DEFAULT_RANDOM_HELDOUT_UNION / DEFAULT_SCAFFOLD_HELDOUT_UNION), matched against `rows` by
    chemical parent identity -- returns the matched canonical_smiles set (~852 molecules against
    the current eligible-records pool). `rows` may be plain dicts (e.g. from
    pyarrow.Table.to_pylist(), as scripts/build_v10_intern_raw_pair.py uses) or pandas rows --
    _record_parent_key works against either via .get()."""
    heldout_keys = _heldout_parent_keys([DEFAULT_RANDOM_HELDOUT_UNION, DEFAULT_SCAFFOLD_HELDOUT_UNION])
    parent_key_by_molecule: dict[str, str] = {}
    for row in rows:
        smiles = str(row["canonical_smiles"])
        if smiles not in parent_key_by_molecule:
            parent_key_by_molecule[smiles] = _record_parent_key(row)
    return {smiles for smiles, key in parent_key_by_molecule.items() if key in heldout_keys}


def _direction_note(measurement_subtype: str) -> str:
    """Value-free direction/orientation phrase, derived only from the categorical subtype suffix
    (never from free text) -- safe because a classification label cannot leak the hidden value."""
    if measurement_subtype.endswith("_apical_to_basolateral"):
        return "measured in the apical-to-basolateral direction"
    if measurement_subtype.endswith("_basolateral_to_apical"):
        return "measured in the basolateral-to-apical direction"
    if measurement_subtype.endswith("_b_over_a"):
        return "expressed as a basolateral-over-apical-style directional ratio (B/A)"
    if measurement_subtype.endswith("_a_over_b"):
        return "expressed as an apical-over-basolateral-style directional ratio (A/B)"
    return ""


def _sd_unit(value: float) -> str:
    return "standard deviation" if abs(value - 1.0) < 1e-9 else "standard deviations"


def _transfer_criterion_text(transfer_max_sd: float, not_transfer_min_sd: float) -> str:
    """Plain chemistry statement of the transfer/not-transfer criterion in standardized-distance
    (standard-deviation) units -- v6 has no distance_kind/policy_family/fold-change concept left
    to phrase around, so this is the one wording used for every concept."""
    return (
        f"Two measurements are considered to transfer when they are within {transfer_max_sd:g} "
        f"{_sd_unit(transfer_max_sd)} of typical variation for this measurement, and considered "
        f"not to transfer once they are {not_transfer_min_sd:g} {_sd_unit(not_transfer_min_sd)} "
        "or more apart."
    )


def _load_policy(policy_path: Path) -> dict[str, Any]:
    with gzip.open(policy_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _eligible_bucket(
    row: pd.Series, buckets: dict[str, Any], stats: dict[str, int],
) -> tuple[str, dict[str, Any]] | None:
    if str(row.get("normalization_validity_status") or "") != "valid":
        stats["rejected_normalization_invalid"] += 1
        return None
    if not bool(row.get("bucket_eligible")):
        stats["rejected_bucket_ineligible"] += 1
        return None
    bucket_key = str(row.get("pair_bucket_key") or "")
    bucket = buckets.get(bucket_key)
    if bucket is None or not bool(bucket.get("assay_transfer_eligible")):
        stats["rejected_bucket_transfer_ineligible"] += 1
        return None
    return bucket_key, bucket


def _bucket_calibration(bucket: dict[str, Any]) -> tuple[float, float, int]:
    distance_policy = bucket["distance_policy"]
    sample_std = float(distance_policy["sample_standard_deviation"])
    anchor_percentile = round(float(distance_policy["one_sd_percentile_0_100"]))
    percentile = min(100, anchor_percentile + NOT_TRANSFER_MIN_MARGIN)
    return sample_std, float(distance_policy["standardized_distance_percentile_knots"][percentile]), percentile


def _output_record(
    row: pd.Series, bucket_key: str, sample_std: float, not_transfer_min: float,
    not_transfer_min_percentile: int,
) -> dict[str, Any]:
    scalar_value = float(row["finite_scalar_value"])
    concept = SOURCE_TO_CONCEPT[str(row["source_id"])]
    record_id = str(row.get("source_record_id") or row["normalized_record_id"])
    out = {
        "child_id": str(row["normalized_record_id"]),
        "parent_provenance_id": str(row.get("duplicate_group_id") or row["normalized_record_id"]),
        "record_id": record_id,
        "source_id": str(row["source_id"]),
        "input_sha256": _stable_hash(str(row["normalized_record_id"])),
        "source_smiles": str(row["source_smiles"]),
        "canonical_smiles": str(row["canonical_smiles"]),
        "canonical_endpoint_key": bucket_key,
        "endpoint_family": concept,
        "endpoint_subtype": str(row.get("canonical_measurement") or ""),
        "unit_basis": str(row.get("canonical_unit") or ""),
        "scalar_value": scalar_value,
        "scalar_is_approximate": bool(pd.notna(row.get("variation_value"))),
        "assay_concept": concept,
        "metric_type": concept,
        "comparison_value": scalar_value / sample_std,
        "transfer_max": 1.0,
        "not_transfer_min": not_transfer_min,
        "threshold_display": (
            f"standardized distance <= 1 SD transfers; >= {not_transfer_min:g} SD does not "
            f"(bucket's own {not_transfer_min_percentile}th percentile of random within-bucket "
            "pairwise distances)"
        ),
        "measurement_label": str(row.get("endpoint_name") or ""),
        "measurement_subtype": str(row.get("canonical_measurement") or ""),
        "policy_family": concept,
        "sample_standard_deviation": sample_std,
        "pair_bucket_key": bucket_key,
        "display_value": str(row.get("measurement_text") or ""),
        "display_unit": str(row.get("unit_text") or ""),
        "context_direction": _direction_note(str(row.get("canonical_measurement") or "")),
        "transfer_criterion_text": _transfer_criterion_text(1.0, not_transfer_min),
    }
    out.update(_context(row))
    return out


def build_eligible_records(
    *,
    records_path: Path = DEFAULT_RECORDS,
    pair_buckets_path: Path = DEFAULT_PAIR_BUCKETS,
    policy_path: Path = DEFAULT_POLICY,
    heldout_paths: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # `None` means "use the default heldout files"; `[]` explicitly means "no heldout exclusion"
    # -- these must stay distinguishable (an `or` fallback would treat both the same, since `[]`
    # is falsy).
    if heldout_paths is None:
        heldout_paths = [DEFAULT_RANDOM_HELDOUT, DEFAULT_SCAFFOLD_HELDOUT]
    policy_doc = _load_policy(policy_path)
    buckets = policy_doc["buckets"]

    merged = _load_frames(records_path, pair_buckets_path)
    heldout_keys = _heldout_parent_keys(heldout_paths)

    stats = {
        "input_records": len(merged),
        "rejected_normalization_invalid": 0,
        "rejected_bucket_ineligible": 0,
        "rejected_bucket_transfer_ineligible": 0,
        "rejected_degenerate_deadband": 0,
        "rejected_heldout_excluded": 0,
        "eligible_records": 0,
    }
    output: list[dict[str, Any]] = []
    seen_parent_keys_excluded: set[str] = set()

    for _, row in merged.iterrows():
        eligible_bucket = _eligible_bucket(row, buckets, stats)
        if eligible_bucket is None:
            continue
        bucket_key, bucket = eligible_bucket
        parent_key = _record_parent_key(row)
        if parent_key and parent_key in heldout_keys:
            stats["rejected_heldout_excluded"] += 1
            seen_parent_keys_excluded.add(parent_key)
            continue

        sample_std, not_transfer_min, percentile = _bucket_calibration(bucket)
        if not_transfer_min <= 1.0:
            stats["rejected_degenerate_deadband"] += 1
            continue
        output.append(_output_record(row, bucket_key, sample_std, not_transfer_min, percentile))
        stats["eligible_records"] += 1

    stats["heldout_parent_keys_total"] = len(heldout_keys)
    stats["heldout_parent_keys_matched"] = len(seen_parent_keys_excluded)
    return output, stats


def write_eligible_records(
    output_dir: Path,
    *,
    heldout_paths: list[Path] | None = None,
) -> dict[str, Any]:
    rows, stats = build_eligible_records(heldout_paths=heldout_paths)
    if not rows:
        raise RuntimeError("no eligible TxAgent-sourced records produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / "records.parquet", compression="zstd")
    manifest = {
        "stage": "starling_txagent_eligible_v6",
        "not_transfer_min_margin": NOT_TRANSFER_MIN_MARGIN,
        "stats": stats,
        "sources": sorted(SOURCE_TO_CONCEPT),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/eligible/assay_transfer_starling_txagent_v9"))
    parser.add_argument("--disable-heldout-exclusion", action="store_true",
                        help="include records whose molecule overlaps Starling's own held-out "
                             "test set, instead of excluding them (e.g. to route them into a "
                             "downstream build's own validation/test split)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_paths = [] if args.disable_heldout_exclusion else None
    manifest = write_eligible_records(args.output_dir, heldout_paths=heldout_paths)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
