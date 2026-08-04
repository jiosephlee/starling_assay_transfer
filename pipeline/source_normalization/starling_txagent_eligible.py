"""Adapt TxAgent's finished Bioavailability_Ma v5 artifacts into our eligible-records schema.

TxAgent (``/data1/joseph/TxAgent``) has already normalized, condition-key-bucketed, and
endpoint-calibrated Bioavailability_Ma (five merged sources: ``oral_exposure``, ``fa``, ``fg``,
``fh``, ``direct_hf``). This module consumes those finished artifacts directly and produces the
same "eligible records" shape ``pipeline/stages/compose_v3.py::_eligible_row()`` would produce
from our own sources -- the shape ``scripts/build_raw_pair.py`` /
``pipeline/pair_core.py`` read unmodified. It deliberately skips ``compose_v3.py`` (and our own
``metrics.yaml``/``concepts.yaml``): TxAgent's endpoint-policy-v2 registry already is that
derivation, for a namespace (``endpoint_policy_key``) our own policy tables don't know about.

``comparison_value`` is written pre-scaled into TxAgent's normalized (0-100) distance space
(``comparison_value = 100/raw_distance_anchor * transform(finite_scalar_value)``), and
``transfer_max``/``not_transfer_min`` are taken directly from the registry's
``normalized_thresholds``. ``pipeline/pair_core.py::target_for()``'s z-score is scale-invariant
to a common linear rescale of (a, b, distance), so this reproduces the same sigmoid target as
using TxAgent's raw values, while trusting their calibration end to end -- see
``docs/assay_transfer_design_v4.md`` sibling plan notes for the derivation. The one accepted
inexactness: TxAgent's own ``normalize_policy_distance`` clips at 100 once raw distance reaches
the *permissive* variant's ``not_transfer_min`` (the anchor); this adapter does not clip, so an
extreme outlier pair's distance can exceed 100 here where TxAgent's own reporting would show a
flat 100. Inconsequential -- the sigmoid is already saturated near its floor well before that
point.
"""

from __future__ import annotations

import argparse
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

_TXAGENT_V5_DIR = (
    TXAGENT_ROOT
    / "outputs/chembl_tool/tasks/bioavailability_ma/evidence_library/starling_normalized_v5"
)
DEFAULT_RECORDS = _TXAGENT_V5_DIR / "records.parquet"
DEFAULT_PAIR_BUCKETS = _TXAGENT_V5_DIR / "pair_buckets" / "pair_bucket_records.parquet"
DEFAULT_ASSIGNMENTS = _TXAGENT_V5_DIR / "endpoint_policies" / "v2" / "endpoint_policy_assignments.parquet"
DEFAULT_REGISTRY = _TXAGENT_V5_DIR / "endpoint_policies" / "v2" / "endpoint_policy_registry.json"
DEFAULT_RANDOM_HELDOUT = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/random/test_molecule_labels.jsonl"
DEFAULT_SCAFFOLD_HELDOUT = TXAGENT_ROOT / "data/processed_starling/Bioavailability_Ma/scaffold/test_molecule_labels.jsonl"

# source_id -> our assay_concept, mirrors configs/assay_transfer/v3/concepts.yaml's q1..q4/starling rules.
SOURCE_TO_CONCEPT = {
    "direct_hf": "oral_bioavailability",
    "oral_exposure": "oral_exposure",
    "fa": "Fa",
    "fg": "Fg",
    "fh": "Fh",
}

# pipeline/pair_core.py CONTEXT_FIELDS <- TxAgent records.parquet columns (same names
# compose_v3.py::_context() already maps our own sources onto).
CONTEXT_FIELD_SOURCES: dict[str, tuple[str, ...]] = {
    "species_or_population": ("species_or_population", "canonical_species", "species"),
    "report_or_statistic_type": ("canonical_bioavailability_report_type", "bioavailability_report_type", "statistic_type"),
    "dose": ("canonical_dose_key", "dose", "oral_dose"),
    "study_or_assay_system": ("canonical_assay_system", "study_context", "oral_exposure_mode", "assay_system"),
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


def _load_frames(
    records_path: Path, pair_buckets_path: Path, assignments_path: Path,
) -> pd.DataFrame:
    records = pd.read_parquet(records_path)
    pair_buckets = pd.read_parquet(
        pair_buckets_path, columns=["normalized_record_id", "pair_bucket_key", "bucket_eligible"],
    )
    assignments = pd.read_parquet(assignments_path)
    if not (len(records) == len(pair_buckets) == len(assignments)):
        raise ValueError("records/pair_buckets/assignments row-count mismatch")
    merged = records.merge(
        pair_buckets, on="normalized_record_id", how="left", validate="one_to_one",
    ).merge(
        assignments, on="normalized_record_id", how="left", validate="one_to_one",
    )
    return merged


def _heldout_parent_keys(heldout_paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in heldout_paths:
        keys |= load_heldout_identity_keys(path)
    return keys


def _record_parent_key(row: pd.Series) -> str:
    identity = normalize_molecule_identity(str(row.get("canonical_smiles") or ""))
    return identity.parent_inchi_key or identity.parent_smiles


def _transform(value: float, transform: str) -> float | None:
    if transform == "log10":
        if not math.isfinite(value) or value <= 0:
            return None
        return math.log10(value)
    if transform == "identity":
        return value if math.isfinite(value) else None
    raise ValueError(f"unsupported transform: {transform!r}")


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


def _transfer_criterion_text(distance_kind: str, transfer_max_raw: float, not_transfer_min_raw: float) -> str:
    """Plain chemistry statement of the transfer/not-transfer criterion from raw threshold
    numbers -- no mention of TxAgent, policy_family, or threshold_profile names."""
    if distance_kind in ("absolute_log10_ratio", "absolute_log10_coordinate_difference"):
        # Both express a log10-fold distance; the latter is for values already reported as bare
        # log10 coordinates (transform="identity" upstream), so the raw threshold numbers are
        # already in the same fold-change units either way.
        fold_transfer = round(10.0 ** transfer_max_raw, 2)
        fold_not_transfer = round(10.0 ** not_transfer_min_raw, 2)
        return (
            f"Two measurements are considered to transfer when they are within {fold_transfer:g}-fold "
            f"of each other, and considered not to transfer once they are {fold_not_transfer:g}-fold "
            "or more apart."
        )
    if distance_kind == "absolute_percentage_points":
        return (
            f"Two measurements are considered to transfer when they are within {transfer_max_raw:g} "
            f"percentage points of each other, and considered not to transfer once they are "
            f"{not_transfer_min_raw:g} percentage points or more apart."
        )
    if distance_kind == "absolute_fraction":
        return (
            f"Two measurements are considered to transfer when they are within {transfer_max_raw:g} "
            f"of each other (on a 0-1 scale), and considered not to transfer once they are "
            f"{not_transfer_min_raw:g} or more apart."
        )
    raise ValueError(f"unsupported distance kind: {distance_kind!r}")


def build_eligible_records(
    *,
    records_path: Path = DEFAULT_RECORDS,
    pair_buckets_path: Path = DEFAULT_PAIR_BUCKETS,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    registry_path: Path = DEFAULT_REGISTRY,
    heldout_paths: list[Path] | None = None,
    threshold_variant: str = "primary",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # `None` means "use the default heldout files"; `[]` explicitly means "no heldout exclusion"
    # -- these must stay distinguishable (an `or` fallback would treat both the same, since `[]`
    # is falsy).
    if heldout_paths is None:
        heldout_paths = [DEFAULT_RANDOM_HELDOUT, DEFAULT_SCAFFOLD_HELDOUT]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    policies = registry["endpoint_policies"]

    merged = _load_frames(records_path, pair_buckets_path, assignments_path)
    heldout_keys = _heldout_parent_keys(heldout_paths)

    stats = {
        "input_records": len(merged),
        "rejected_normalization_invalid": 0,
        "rejected_policy_unassigned": 0,
        "rejected_bucket_ineligible": 0,
        "rejected_heldout_excluded": 0,
        "rejected_nonpositive_log_domain": 0,
        "eligible_records": 0,
    }
    output: list[dict[str, Any]] = []
    seen_parent_keys_excluded: set[str] = set()

    for _, row in merged.iterrows():
        if str(row.get("normalization_validity_status") or "") != "valid":
            stats["rejected_normalization_invalid"] += 1
            continue
        policy_key = row.get("endpoint_policy_key")
        if str(row.get("policy_assignment_status") or "") != "assigned" or not policy_key:
            stats["rejected_policy_unassigned"] += 1
            continue
        if not bool(row.get("bucket_eligible")):
            stats["rejected_bucket_ineligible"] += 1
            continue
        parent_key = _record_parent_key(row)
        if parent_key and parent_key in heldout_keys:
            stats["rejected_heldout_excluded"] += 1
            seen_parent_keys_excluded.add(parent_key)
            continue

        policy = policies[policy_key]
        transformed = _transform(float(row["finite_scalar_value"]), policy["transform"])
        if transformed is None:
            stats["rejected_nonpositive_log_domain"] += 1
            continue
        anchor = float(policy["normalization"]["raw_distance_anchor"])
        comparison_value = (100.0 / anchor) * transformed
        thresholds = policy["normalization"]["normalized_thresholds"][threshold_variant]
        raw_thresholds = policy["thresholds"][threshold_variant]
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
            "canonical_endpoint_key": str(policy_key),
            "endpoint_family": concept,
            "endpoint_subtype": str(policy["measurement_subtype"]),
            "unit_basis": str(row.get("canonical_unit") or ""),
            "scalar_value": float(row["finite_scalar_value"]),
            "scalar_is_approximate": bool(pd.notna(row.get("variation_value"))),
            "assay_concept": concept,
            "metric_type": str(policy["policy_family"]),
            "comparison_value": comparison_value,
            "transfer_max": float(thresholds["transfer_max"]),
            "not_transfer_min": float(thresholds["not_transfer_min"]),
            "threshold_display": str(policy["rationale"]),
            "measurement_label": str(row.get("endpoint_name") or ""),
            "measurement_subtype": str(policy["measurement_subtype"]),
            "policy_family": str(policy["policy_family"]),
            "raw_distance_anchor": anchor,
            "threshold_variant": threshold_variant,
            "pair_bucket_key": row.get("pair_bucket_key"),
            # Source-native display fields (not TxAgent's canonical/cleaned columns): the raw
            # (measurement_text, unit_text) pair is self-consistent as originally reported, unlike
            # pairing unit_text with finite_scalar_value (which is folded to canonical_unit's
            # convention -- see module docstring history / plan notes on the ~16K notation-folded
            # rows this would otherwise silently mis-scale).
            "display_value": str(row.get("measurement_text") or ""),
            "display_unit": str(row.get("unit_text") or ""),
            # Value-free (categorical, not leaked from free text) direction/orientation phrase.
            "context_direction": _direction_note(str(policy["measurement_subtype"])),
            # Plain chemistry statement of the transfer criterion; never names TxAgent/policy_family.
            "transfer_criterion_text": _transfer_criterion_text(
                str(policy["distance"]), float(raw_thresholds["transfer_max"]),
                float(raw_thresholds["not_transfer_min"]),
            ),
        }
        out.update(_context(row))
        output.append(out)
        stats["eligible_records"] += 1

    stats["heldout_parent_keys_total"] = len(heldout_keys)
    stats["heldout_parent_keys_matched"] = len(seen_parent_keys_excluded)
    return output, stats


def write_eligible_records(
    output_dir: Path,
    *,
    threshold_variant: str = "primary",
    heldout_paths: list[Path] | None = None,
) -> dict[str, Any]:
    rows, stats = build_eligible_records(threshold_variant=threshold_variant, heldout_paths=heldout_paths)
    if not rows:
        raise RuntimeError("no eligible TxAgent-sourced records produced")
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / "records.parquet", compression="zstd")
    manifest = {
        "stage": "starling_txagent_eligible",
        "threshold_variant": threshold_variant,
        "stats": stats,
        "sources": sorted(SOURCE_TO_CONCEPT),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/eligible/assay_transfer_starling_txagent_v7"))
    parser.add_argument("--threshold-variant", choices=("strict", "primary", "permissive"), default="primary")
    parser.add_argument("--disable-heldout-exclusion", action="store_true",
                        help="include records whose molecule overlaps Starling's own held-out "
                             "test set, instead of excluding them (e.g. to route them into a "
                             "downstream build's own validation/test split)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_paths = [] if args.disable_heldout_exclusion else None
    manifest = write_eligible_records(args.output_dir, threshold_variant=args.threshold_variant,
                                      heldout_paths=heldout_paths)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
