#!/usr/bin/env python3
"""Render the binary v3 materialized artifact as three-column HF Parquet splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import StrictUndefined, Template

from pipeline.v3_policy import file_sha256

SPLITS = ("train", "validation", "test")
CONTEXT_NAMES = (
    "molecule_name", "species_or_population", "report_or_statistic_type", "dose",
    "study_or_assay_system", "measured_process", "biological_context", "medium",
    "formulation_or_solid_form", "transporter_or_enzyme", "substrate_status",
    "intestinal_site", "molecular_form", "enzyme_or_pathway", "qualifying_conditions",
    "comparator", "extra_details",
)


def _metadata_type() -> pa.StructType:
    strings = ("schema_version", "candidate_id", "split", "assay_concept",
               "canonical_endpoint_key", "endpoint_family", "endpoint_subtype", "unit_basis",
               "metric_type", "threshold_display", "k_profile", "setting_key",
               "retrieval_record_id", "retrieved_source_id", "retrieved_smiles", "query_smiles",
               "retrieved_measurement_label",
               "retrieved_original_smiles", "query_original_smiles", "tanimoto_bucket", "template_id")
    fields = [pa.field(name, pa.string()) for name in strings]
    fields += [pa.field("retrieved_value", pa.float64()), pa.field("transfer_max", pa.float64()),
               pa.field("not_transfer_min", pa.float64()), pa.field("binary_label", pa.int8()),
               pa.field("tanimoto", pa.float32())]
    evidence = pa.struct([pa.field(name, pa.int32()) for name in
                          ("n_records", "n_transfer", "n_nontransfer", "n_ambiguous")] +
                         [pa.field("majority_margin", pa.float32())])
    context = pa.struct([pa.field(name, pa.string()) for name in CONTEXT_NAMES])
    provenance = pa.struct([pa.field(name, pa.string()) for name in
                            ("parent_provenance_id", "record_id", "input_sha256", "child_id")])
    return pa.struct(fields + [pa.field("evidence", evidence), pa.field("retrieval_context", context),
                               pa.field("provenance", provenance)])


HF_SCHEMA = pa.schema([pa.field("prompt", pa.large_string()), pa.field("completion", pa.string()),
                       pa.field("metadata", _metadata_type())])


def _load_templates(template_dir: Path) -> dict[str, Template]:
    return {concept: Template((template_dir / f"{concept}.jinja").read_text(),
                              undefined=StrictUndefined, keep_trailing_newline=True)
            for concept in ("oral_bioavailability", "oral_exposure", "Fa", "Fg", "Fh")}


def _evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row.get(name) for name in
            ("n_records", "n_transfer", "n_nontransfer", "n_ambiguous", "majority_margin")}


def _context(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row.get(f"retrieved_context_{name}") for name in CONTEXT_NAMES}


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    return {"parent_provenance_id": row.get("retrieved_parent_provenance_id"),
            "record_id": row.get("retrieved_record_id"), "input_sha256": row.get("retrieved_input_sha256"),
            "child_id": row.get("retrieval_record_id")}


def _metadata(row: dict[str, Any], template_id: str, schema_version: str) -> dict[str, Any]:
    names = ("candidate_id", "split", "assay_concept", "canonical_endpoint_key",
             "endpoint_family", "endpoint_subtype", "unit_basis", "metric_type",
             "threshold_display", "k_profile", "setting_key", "retrieval_record_id",
             "retrieved_source_id", "retrieved_smiles", "query_smiles", "retrieved_measurement_label", "retrieved_value",
             "retrieved_original_smiles", "query_original_smiles", "transfer_max",
             "not_transfer_min", "binary_label", "tanimoto", "tanimoto_bucket")
    meta = {name: row.get(name) for name in names}
    meta.update({"schema_version": schema_version, "template_id": template_id,
                 "evidence": _evidence(row), "retrieval_context": _context(row),
                 "provenance": _provenance(row)})
    return meta


def _render(row: dict[str, Any], templates: dict[str, Template], variant: str = "standard",
            schema_version: str = "assay_transfer_binary_v3") -> dict[str, Any]:
    label = row.get("binary_label")
    if label not in (0, 1):
        raise ValueError(f"invalid binary label for {row.get('candidate_id')}: {label}")
    concept = row["assay_concept"]
    suffix = "intern_mcqa_v3" if variant == "intern" else "mcqa_v3"
    template_id = f"{concept}_{suffix}"
    labels = ("(A)", "(B)") if variant == "intern" else ("A", "B")
    return {"prompt": templates[concept].render(row=row), "completion": labels[0] if label == 1 else labels[1],
            "metadata": _metadata(row, template_id, schema_version)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows = pq.read_table(args.dataset).to_pylist()
    if not rows:
        raise RuntimeError("no materialized rows to render")
    templates = _load_templates(args.template_dir)
    variant = getattr(args, "template_variant", "standard")
    schema_version = getattr(args, "schema_version", "assay_transfer_binary_v3")
    by_split = {split: [] for split in SPLITS}
    for row in rows:
        if row.get("split") in by_split:
            by_split[row["split"]].append(_render(row, templates, variant, schema_version))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written, hashes, labels = {}, {}, {}
    for split, records in by_split.items():
        if not records:
            continue
        path = args.output_dir / split / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records, schema=HF_SCHEMA), path, compression="zstd")
        written[split], hashes[split] = len(records), file_sha256(path)
        labels[split] = dict(Counter(record["completion"] for record in records))
    info = {"stage": "render_hf", "schema_version": schema_version,
            "top_level_features": ["prompt", "completion", "metadata"], "rows_per_split": written,
            "completion_map": ({"1": "(A)", "0": "(B)"} if variant == "intern"
                               else {"1": "A", "0": "B"}), "completion_counts": labels,
            "parquet_sha256": hashes, "template_dir": str(args.template_dir)}
    info["template_variant"] = variant
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-variant", choices=("standard", "intern"), default="standard")
    parser.add_argument("--schema-version", default="assay_transfer_binary_v3")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2))


if __name__ == "__main__":
    main()
