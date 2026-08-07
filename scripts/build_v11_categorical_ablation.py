#!/usr/bin/env python3
"""Build paired continuous-only and with-categorical V11 Intern datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.build_v7_intern_raw_pair as v7
from pipeline.pair_core import record_key
from pipeline.source_normalization.starling_txagent_eligible_v7 import heldout_union_molecules
from pipeline.v11_contract import (
    DEFAULT_REGISTRY,
    categorical_ablation_config,
    heldout_reservation_manifest,
    load_registry,
    validate_registry,
)
from pipeline.v11_prompt_rendering import reset_prompt_cache
from pipeline.v3_policy import file_sha256
from scripts import build_v11_intern_raw_pair as v11
from scripts.build_raw_pair import build


DEFAULT_OUTPUT_ROOT = Path("datasets/hf_parquet/assay_transfer_raw_pair_v11_categorical_ablation")
DEFAULT_COMPONENT_ROOT = Path("tmp/v11_categorical_ablation_release/components")
VARIANTS = ("continuous_only", "with_categorical")
EVAL_SPLITS = ("validation", "validation_ranking", "test", "test_ranking")
CONCAT_BATCH_ROWS = 32_768
TRAIN_ZSTD_LEVEL = 3
TASK_TITLES = {
    "bioavailability_ma": "Bioavailability",
    "bbb_martins": "Blood-Brain Barrier",
    "skin_reaction": "Skin Reaction",
}
HUB_DATASET_IDS = {
    "bioavailability_ma": {
        "continuous_only": "jiosephlee/assay-transfer-raw-pair-v11-bioavailability-ma-continuous-only-intern",
        "with_categorical": "jiosephlee/assay-transfer-raw-pair-v11-bioavailability-ma-with-categorical-intern",
    },
    "bbb_martins": {
        "continuous_only": "jiosephlee/assay-transfer-raw-pair-v11-bbb-martins-continuous-only-intern",
        "with_categorical": "jiosephlee/assay-transfer-raw-pair-v11-bbb-martins-with-categorical-intern",
    },
    "skin_reaction": {
        "continuous_only": "jiosephlee/assay-transfer-raw-pair-v11-skin-reaction-continuous-only-intern",
        "with_categorical": "jiosephlee/assay-transfer-raw-pair-v11-skin-reaction-with-categorical-intern",
    },
}


def _kinds(rows: Iterable[dict], allowed: Iterable[str]) -> list[dict]:
    accepted = set(allowed)
    return [row for row in rows if str(row["measurement_kind"]) in accepted]


def _unreserved(rows: Iterable[dict], reserved: set[str]) -> list[dict]:
    return [row for row in rows if str(row["canonical_smiles"]) not in reserved]


def _fixed_split_reader(split_rows: Mapping[str, list[dict]]):
    def split_records(_rows: list[dict]) -> dict[str, list[dict]]:
        return dict(split_rows)

    return split_records


def _eval_cache(
    task_id: str, rows: list[dict], registry: Mapping[str, Any], reserved: set[str]
) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    release = categorical_ablation_config(registry)
    continuous = _kinds(rows, release["evaluation_measurement_kinds"])
    results = v11._construct_phases(continuous, task_id, registry, reserved)
    split_rows = v11._partition_rows(continuous, results, reserved)
    cache = {"results": results, "reserved": reserved, "split_rows": split_rows}
    return cache, split_rows


def _build_continuous(
    task_id: str, source: Path, output: Path, rows: list[dict],
    registry: Mapping[str, Any], reserved: set[str], cap: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache, split_rows = _eval_cache(task_id, rows, registry, reserved)
    split_reader = _fixed_split_reader(split_rows)
    v11._configure_driver(cache, split_reader, v7._iter_list_groups_capped)
    expected = {name: len(values) for name, values in split_rows.items()}
    manifest = build(
        source, output, cap, listnet=False,
        expected_records=len(rows), expected_split=expected, records=rows,
    )
    return manifest, cache


def _write_categorical_component(
    task_id: str, root: Path, rows: list[dict], reserved: set[str],
    registry: Mapping[str, Any], cap: int, schema: pa.Schema,
) -> tuple[Path, int]:
    release = categorical_ablation_config(registry)
    categorical = _unreserved(_kinds(rows, release["categorical_measurement_kinds"]), reserved)
    path_text, count = v7._write_train_flat_variable(root, categorical, cap, schema)
    path = Path(path_text)
    if pq.ParquetFile(path).metadata.num_rows != count:
        raise RuntimeError(f"{task_id}: categorical component row-count mismatch")
    return path, count


def _copy_evaluation(source_root: Path, target_root: Path) -> None:
    for split in EVAL_SPLITS:
        source = source_root / split / "data.parquet"
        target = target_root / split / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _concat_parquets(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    schema = pq.ParquetFile(inputs[0]).schema_arrow
    writer = pq.ParquetWriter(
        temporary, schema, compression="zstd", compression_level=TRAIN_ZSTD_LEVEL
    )
    count = 0
    try:
        for path in inputs:
            for batch in pq.ParquetFile(path).iter_batches(CONCAT_BATCH_ROWS):
                writer.write_table(pa.Table.from_batches([batch], schema=schema))
                count += batch.num_rows
    finally:
        writer.close()
    temporary.replace(output)
    return count


def _pair_id_sha256(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    seen = 0
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["pair_id"]):
        for pair_id in batch.column(0).to_pylist():
            if limit is not None and seen >= limit:
                return digest.hexdigest()
            digest.update(str(pair_id).encode("utf-8") + b"\n")
            seen += 1
    return digest.hexdigest()


def _parquet_kind_counts(path: Path) -> dict[str, int]:
    counts: Counter = Counter()
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=["measurement_kind"]):
        counts.update(str(value) for value in batch.column(0).to_pylist())
    return dict(sorted(counts.items()))


def _degree_maxima(path: Path) -> dict[str, int]:
    query: Counter = Counter()
    retrieval: Counter = Counter()
    columns = ["query_record_id", "retrieval_record_id"]
    for batch in pq.ParquetFile(path).iter_batches(8192, columns=columns):
        query.update(str(value) for value in batch.column(0).to_pylist())
        retrieval.update(str(value) for value in batch.column(1).to_pylist())
    return {
        "observed_max_query_degree": max(query.values(), default=0),
        "observed_max_retrieval_degree": max(retrieval.values(), default=0),
    }


def _record_inventory(rows: list[dict], reserved: set[str]) -> dict[str, Any]:
    all_counts = Counter(str(row["measurement_kind"]) for row in rows)
    heldout = [row for row in rows if str(row["canonical_smiles"]) in reserved]
    train = [row for row in rows if str(row["canonical_smiles"]) not in reserved]
    return {
        "all_eligible_by_measurement_kind": dict(sorted(all_counts.items())),
        "scaffold_reserved_by_measurement_kind": dict(sorted(Counter(
            str(row["measurement_kind"]) for row in heldout
        ).items())),
        "train_candidate_by_measurement_kind": dict(sorted(Counter(
            str(row["measurement_kind"]) for row in train
        ).items())),
    }


def _split_artifacts(root: Path) -> tuple[dict[str, str], dict[str, int]]:
    names = ("train", *EVAL_SPLITS)
    paths = {name: root / name / "data.parquet" for name in names}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    counts = {name: pq.ParquetFile(path).metadata.num_rows for name, path in paths.items()}
    return hashes, counts


def _component_manifest(
    name: str, config: Mapping[str, Any], path: Path, achieved: int,
) -> dict[str, Any]:
    degrees = _degree_maxima(path)
    query_cap = int(config["query_degree_cap"])
    retrieval_cap = int(config["retrieval_degree_cap"])
    if degrees["observed_max_query_degree"] > query_cap:
        raise RuntimeError(f"{name}: emitted rows exceed query degree cap")
    if degrees["observed_max_retrieval_degree"] > retrieval_cap:
        raise RuntimeError(f"{name}: emitted rows exceed retrieval degree cap")
    return {
        "measurement_kinds": list(config["measurement_kinds"]),
        "max_pairs": int(config["max_pairs"]),
        "achieved_pairs": achieved,
        "achieved_by_measurement_kind": _parquet_kind_counts(path),
        "query_degree_cap": query_cap,
        "retrieval_degree_cap": retrieval_cap,
        "pair_id_sha256": _pair_id_sha256(path),
        "component_parquet_sha256": file_sha256(path),
        "component_name": name,
        **degrees,
    }


def _variant_manifest(
    variant: str, root: Path, base: Mapping[str, Any], task_id: str,
    source: Path, registry_path: Path, release: Mapping[str, Any],
    components: Mapping[str, dict], inventory: Mapping[str, Any], cache: Mapping[str, Any],
) -> dict[str, Any]:
    hashes, counts = _split_artifacts(root)
    included = list(release["variants"][variant])
    manifest = dict(base)
    manifest.update(
        version="v11-txagent-v7-categorical-ablation",
        dataset_variant=variant,
        hub_dataset_id=HUB_DATASET_IDS[task_id][variant],
        task_id=task_id,
        source=str(source),
        eligible_source_sha256=file_sha256(source),
        eligible_source_manifest_sha256=file_sha256(source.parent / "manifest.json"),
        prompt_projection={
            "schema_version": "assay_transfer_prompt_projection.v11",
            "path": str(registry_path), "sha256": file_sha256(registry_path),
        },
        construction=v11.construction_config(task_id, load_registry(registry_path)),
        measurement_kind_policy={
            "reservation": "all_eligible_measurement_kinds",
            "evaluation": list(release["evaluation_measurement_kinds"]),
            "train_components": included,
            "categorical_is_train_only": True,
        },
        train_components={name: components[name] for name in included},
        eligible_record_inventory=dict(inventory),
        heldout_reservation=heldout_reservation_manifest(task_id, load_registry(registry_path)),
        diagnostics=v11._diagnostics(cache, v11.construction_config(task_id, load_registry(registry_path))),
        rows=counts, parquet_sha256=hashes,
        paths={name: str(root / name / "data.parquet") for name in counts},
    )
    return manifest


def _dataset_card(task_id: str, variant: str, manifest: Mapping[str, Any]) -> str:
    title = TASK_TITLES[task_id]
    label = "Continuous Only" if variant == "continuous_only" else "With Categorical"
    rows = manifest["rows"]
    components = manifest["train_components"]
    component_text = ", ".join(
        f"{name}={details['achieved_pairs']:,}" for name, details in components.items()
    )
    return f"""---
pretty_name: Assay Transfer Raw Pair V11 Intern — {title} — {label}
task_categories:
  - text-classification
language:
  - en
configs:
  - config_name: default
    data_files:
      - split: train
        path: train/data.parquet
      - split: validation
        path: validation/data.parquet
      - split: test
        path: test/data.parquet
      - split: validation_ranking
        path: validation_ranking/data.parquet
      - split: test_ranking
        path: test_ranking/data.parquet
---

# Assay Transfer Raw Pair V11 Intern — {title} — {label}

This is the `{variant}` member of the V11 categorical-ablation pair for `{task_id}`.
Validation and test are constructed only from continuous measurements and are byte-identical
across both variants. Scaffold valid/test identities are reserved using all eligible records, so
binary and ordinal records from held-out molecules cannot leak into train.

Train contains {rows['train']:,} pairs ({component_text}). Each component has an independent
1,250,000-pair ceiling and independent query/retrieval record degree caps of six. The
with-categorical variant is the exact continuous train component followed by a separately built
binary-plus-ordinal component; categorical records never appear in evaluation.

Target calibration remains independent within each `pair_bucket_key`: raw pairwise-distance p05
maps to transfer probability 0.95 and p95 maps to 0.05 through the unrestricted sigmoid. Prompts
use the source-native V11 templates and wrap both structures in `<SMILES>...</SMILES>` tags.

Split rows: validation={rows['validation']:,}, validation_ranking={rows['validation_ranking']:,},
test={rows['test']:,}, test_ranking={rows['test_ranking']:,}. Full construction provenance,
shortfalls, hashes, and component policies are frozen in `manifest.json`.
"""


def _write_release_files(root: Path, task_id: str, variant: str, manifest: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        _dataset_card(task_id, variant, manifest), encoding="utf-8"
    )


def build_task_pair(
    task_id: str, source: Path, output_root: Path, component_root: Path,
    registry_path: Path = DEFAULT_REGISTRY, rebuild_source: bool = False,
) -> dict[str, dict]:
    registry = load_registry(registry_path)
    validate_registry(registry, tasks=(task_id,))
    v11.configure_engine(task_id, registry, registry_path)
    if rebuild_source:
        v11.write_eligible_records(task_id, source.parent, registry=registry)
    # Otherwise _prepare_source regenerates only when the artifact is absent, and always
    # validates its manifest stage/task/hash -- so an unconditional rebuild is pure waste.
    source = v11._prepare_source(task_id, source, None, registry)
    # Prompt blocks are memoized by child_id; a regenerated source can reuse an id with new
    # contents, so drop anything cached before this build's records are read.
    reset_prompt_cache()
    rows = pq.read_table(source).to_pylist()
    release = categorical_ablation_config(registry)
    continuous_config = release["train_components"]["continuous"]
    categorical_config = release["train_components"]["categorical"]
    reserved = heldout_union_molecules(task_id, rows, registry)
    continuous_root = output_root / task_id / "continuous_only"
    base, cache = _build_continuous(
        task_id, source, continuous_root, rows, registry, reserved,
        int(continuous_config["max_pairs"]),
    )
    continuous_path = continuous_root / "train" / "data.parquet"
    schema = pq.ParquetFile(continuous_path).schema_arrow
    categorical_path, categorical_count = _write_categorical_component(
        task_id, component_root / task_id, rows, reserved, registry,
        int(categorical_config["max_pairs"]), schema,
    )
    with_root = output_root / task_id / "with_categorical"
    _copy_evaluation(continuous_root, with_root)
    _concat_parquets([continuous_path, categorical_path], with_root / "train" / "data.parquet")
    components = {
        "continuous": _component_manifest(
            "continuous", continuous_config, continuous_path, base["rows"]["train"]
        ),
        "categorical": _component_manifest(
            "categorical", categorical_config, categorical_path, categorical_count
        ),
    }
    inventory = _record_inventory(rows, reserved)
    manifests = {
        variant: _variant_manifest(
            variant, output_root / task_id / variant, base, task_id, source,
            registry_path, release, components, inventory, cache,
        ) for variant in VARIANTS
    }
    for variant, manifest in manifests.items():
        _write_release_files(output_root / task_id / variant, task_id, variant, manifest)
    return manifests


def _parse_args() -> argparse.Namespace:
    registry = load_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(registry["tasks"]))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--component-root", type=Path, default=DEFAULT_COMPONENT_ROOT)
    parser.add_argument(
        "--rebuild-source", action="store_true",
        help="regenerate the eligible records even when a matching artifact already exists",
    )
    args = parser.parse_args()
    args.source = args.source or v11.DEFAULT_ELIGIBLE_ROOT / args.task / "records.parquet"
    return args


def main() -> None:
    args = _parse_args()
    manifests = build_task_pair(
        args.task, args.source, args.output_root, args.component_root, args.registry,
        args.rebuild_source,
    )
    summary = {variant: manifest["rows"] for variant, manifest in manifests.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
