#!/usr/bin/env python3
"""Render the binary v3 materialized artifact as three-column HF Parquet splits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import StrictUndefined, Template

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.soft_evidence import modal_completion, target_distribution
from pipeline.v3_policy import file_sha256

SPLITS = ("train", "validation", "test")
CONTEXT_NAMES = (
    "molecule_name", "species_or_population", "report_or_statistic_type", "dose",
    "study_or_assay_system", "measured_process", "biological_context", "medium",
    "formulation_or_solid_form", "transporter_or_enzyme", "substrate_status",
    "intestinal_site", "molecular_form", "enzyme_or_pathway", "qualifying_conditions",
    "comparator", "extra_details",
)


def _target_type() -> pa.StructType:
    return pa.struct([pa.field("transfer", pa.float32()),
                      pa.field("nontransfer", pa.float32()),
                      pa.field("ambiguous", pa.float32())])


def _metadata_type(soft_evidence: bool = False) -> pa.StructType:
    strings = ("schema_version", "candidate_id", "split", "assay_concept",
               "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
               "metric_type", "threshold_display", "k_profile", "setting_key",
               "retrieval_record_id", "retrieved_source_id", "retrieved_smiles", "query_smiles",
               "retrieved_measurement_label",
               "retrieved_original_smiles", "query_original_smiles", "tanimoto_bucket", "template_id")
    if soft_evidence:
        strings += ("target_policy_version",)
    fields = [pa.field(name, pa.string()) for name in strings]
    fields += [pa.field("retrieved_value", pa.float64()), pa.field("transfer_max", pa.float64()),
               pa.field("not_transfer_min", pa.float64()), pa.field("binary_label", pa.int8()),
               pa.field("tanimoto", pa.float32())]
    evidence = pa.struct([pa.field(name, pa.int32()) for name in
                          ("n_records", "n_transfer", "n_nontransfer", "n_ambiguous")] +
                         [pa.field("majority_margin", pa.float32())] +
                         ([pa.field(name, pa.float32()) for name in
                           ("transfer_fraction", "nontransfer_fraction", "ambiguous_fraction")]
                          if soft_evidence else []))
    context = pa.struct([pa.field(name, pa.string()) for name in CONTEXT_NAMES])
    provenance = pa.struct([pa.field(name, pa.string()) for name in
                            ("parent_provenance_id", "record_id", "input_sha256", "child_id")])
    nested = [pa.field("evidence", evidence), pa.field("retrieval_context", context),
              pa.field("provenance", provenance)]
    if soft_evidence:
        nested.append(pa.field("target_distribution", _target_type()))
    return pa.struct(fields + nested)


HF_SCHEMA = pa.schema([pa.field("prompt", pa.large_string()), pa.field("completion", pa.string()),
                       pa.field("metadata", _metadata_type())])


def _hf_schema(soft_evidence: bool) -> pa.Schema:
    fields = [pa.field("prompt", pa.large_string()), pa.field("completion", pa.string())]
    fields.append(pa.field("metadata", _metadata_type(soft_evidence)))
    return pa.schema(fields)


def _load_templates(template_dir: Path) -> dict[str, Template]:
    return {concept: Template((template_dir / f"{concept}.jinja").read_text(),
                              undefined=StrictUndefined, keep_trailing_newline=True)
            for concept in ("oral_bioavailability", "oral_exposure", "Fa", "Fg", "Fh")}


def _evidence(row: dict[str, Any], soft_evidence: bool = False) -> dict[str, Any]:
    names = ["n_records", "n_transfer", "n_nontransfer", "n_ambiguous", "majority_margin"]
    if soft_evidence:
        names += ["transfer_fraction", "nontransfer_fraction", "ambiguous_fraction"]
    return {name: row.get(name) for name in names}


def _context(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row.get(f"retrieved_context_{name}") for name in CONTEXT_NAMES}


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {"parent_provenance_id": row.get("retrieved_parent_provenance_id"),
            "record_id": row.get("retrieved_record_id"), "input_sha256": row.get("retrieved_input_sha256"),
            "child_id": row.get("retrieval_record_id")}


def _metadata(row: dict[str, Any], template_id: str, schema_version: str,
              soft_evidence: bool = False, target_policy_version: str | None = None) -> dict[str, Any]:
    names = ("candidate_id", "split", "assay_concept", "canonical_endpoint_key",
             "endpoint_family", "endpoint_subtype", "unit_basis", "metric_type",
             "threshold_display", "k_profile", "setting_key", "retrieval_record_id",
             "retrieved_source_id", "retrieved_smiles", "query_smiles", "retrieved_measurement_label", "retrieved_value",
             "retrieved_original_smiles", "query_original_smiles", "transfer_max",
             "not_transfer_min", "binary_label", "tanimoto", "tanimoto_bucket")
    meta = {name: row.get(name) for name in names}
    meta.update({"schema_version": schema_version, "template_id": template_id,
                 "evidence": _evidence(row, soft_evidence), "retrieval_context": _context(row),
                 "provenance": _provenance(row)})
    if soft_evidence:
        meta["target_policy_version"] = target_policy_version
        meta["target_distribution"] = target_distribution(row)
    return meta


def _render(row: dict[str, Any], templates: dict[str, Template], variant: str = "standard",
            schema_version: str = "assay_transfer_binary_v3", soft_evidence: bool = False,
            target_policy_version: str | None = None) -> dict[str, Any]:
    label = row.get("binary_label")
    if not soft_evidence and label not in (0, 1):
        raise ValueError(f"invalid binary label for {row.get('candidate_id')}: {label}")
    concept = row["assay_concept"]
    version = "v4" if soft_evidence else "v3"
    suffix = f"intern_mcqa_{version}" if variant == "intern" else f"mcqa_{version}"
    template_id = f"{concept}_{suffix}"
    labels = {name: f"({name})" if variant == "intern" else name for name in "ABC"}
    completion = modal_completion(row) if soft_evidence else ("A" if label == 1 else "B")
    record = {"prompt": templates[concept].render(row=row), "completion": labels[completion],
              "metadata": _metadata(row, template_id, schema_version, soft_evidence,
                                    target_policy_version)}
    return record


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = pq.read_table(args.dataset).to_pylist()
    if not rows:
        raise RuntimeError("no materialized rows to render")
    templates = _load_templates(args.template_dir)
    variant = getattr(args, "template_variant", "standard")
    schema_version = getattr(args, "schema_version", "assay_transfer_binary_v3")
    soft_evidence = bool(getattr(args, "soft_evidence", False))
    target_version = getattr(args, "target_policy_version", None)
    if soft_evidence and not target_version:
        raise ValueError("soft-evidence rendering requires target_policy_version")
    by_split = {split: [] for split in SPLITS}
    for row in rows:
        if row.get("split") in by_split:
            by_split[row["split"]].append(_render(row, templates, variant, schema_version,
                                                   soft_evidence, target_version))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written, hashes, labels = {}, {}, {}
    for split, records in by_split.items():
        if not records:
            continue
        path = args.output_dir / split / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records, schema=_hf_schema(soft_evidence)),
                       path, compression="zstd")
        written[split], hashes[split] = len(records), file_sha256(path)
        labels[split] = dict(Counter(record["completion"] for record in records))
    features = ["prompt", "completion", "metadata"]
    completion_map = ({"rule": "unique_argmax_c_on_tie"} if soft_evidence else
                      ({"1": "(A)", "0": "(B)"} if variant == "intern" else {"1": "A", "0": "B"}))
    info = {"stage": "render_hf", "schema_version": schema_version,
            "top_level_features": features, "rows_per_split": written,
            "completion_map": completion_map, "completion_counts": labels,
            "parquet_sha256": hashes, "template_dir": str(args.template_dir)}
    info["template_variant"] = variant
    if soft_evidence:
        info["target_policy_version"] = target_version
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-variant", choices=("standard", "intern"), default="standard")
    parser.add_argument("--schema-version", default="assay_transfer_binary_v3")
    parser.add_argument("--soft-evidence", action="store_true")
    parser.add_argument("--target-policy-version")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
