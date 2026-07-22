#!/usr/bin/env python3
"""Run the v3 canonical-base-to-Hugging-Face build DAG."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.stages import compose_v3, expand_v3, materialize, pairs, render_hf, selection, split  # noqa: E402
from pipeline.v3_policy import V3Policies, resolve_path  # noqa: E402

STAGES = ("compose", "split", "pairs", "select", "materialize", "render_hf", "expand")
SPLITS = ("train", "validation", "test")


def _paths(root: Path, build: str) -> dict[str, Path]:
    return {"eligible": root / "eligible" / build, "split": root / "splits" / build,
            "pairs": root / "pairs" / build / "same_endpoint",
            "select": root / "select" / build / "same_endpoint",
            "parquet": root / "parquet" / build / "same_endpoint",
            "hf": root / "hf_parquet" / build, "expand": root / "expand" / build}


def _compose(config: dict, paths: dict, release: Path) -> dict:
    return compose_v3.build(Namespace(base=config["base_inputs"], release=str(release),
                                      output_dir=paths["eligible"]))


def _split(paths: dict, policies: V3Policies) -> dict:
    fractions = policies.sampling["split"]
    return split.build(Namespace(base=[paths["eligible"]], output_dir=paths["split"],
                                 seed=int(policies.sampling["seed"]),
                                 train_frac=float(fractions["train"]),
                                 val_frac=float(fractions["validation"]),
                                 test_frac=float(fractions["test"])))


def _pair(paths: dict, release: Path, caps: dict[str, int]) -> dict:
    return pairs.build(Namespace(base=[paths["eligible"]], split_dir=paths["split"],
                                 release=str(release), output_dir=paths["pairs"],
                                 query_caps=caps, max_queries=None))


def _select(paths: dict, release: Path, quotas: dict[str, int], output: Path,
            targets: dict | None = None) -> dict:
    return selection.build(Namespace(candidates=[paths["pairs"]], output_dir=output,
                                     train_quota=quotas["train"], val_quota=quotas["validation"],
                                     test_quota=quotas["test"], release=str(release),
                                     stratum_targets=targets))


def _deficient(report: dict) -> set[str]:
    concepts = set()
    for data in report["splits"].values():
        concepts.update(key.split("|", 1)[0] for key in data["underfilled_strata"])
    return concepts


def _deficient_strata(report: dict) -> set[str]:
    strata = set()
    for data in report["splits"].values():
        strata.update(data["underfilled_strata"])
    return strata


def _actionable_deficits(report: dict, policies: V3Policies) -> set[str]:
    sparse = policies.sampling["sparse_strata"]
    train_sparse = set(sparse["train_all_available"])
    heldout_sparse = set(sparse["heldout_matched_min"])
    backfill_concepts = set(sparse.get("heldout_backfill_within_concept", []))
    actionable = set()
    for split, data in report["splits"].items():
        for stratum in data["underfilled_strata"]:
            concept = stratum.split("|", 1)[0]
            ignored = (split == "train" and stratum in train_sparse) or (
                split != "train" and stratum in heldout_sparse
            ) or (
                split != "train" and concept in backfill_concepts
            )
            if not ignored:
                actionable.add(stratum)
    return actionable


def _exhausted_strata(pair_report: dict, capacity_report: dict) -> list[str]:
    saturation = pair_report.get("enumeration_saturation", {})
    deficient = set()
    for split in capacity_report["splits"].values():
        deficient.update(split["underfilled_strata"])
    exhausted = []
    for stratum in sorted(deficient):
        concept, bucket = stratum.split("|", 1)
        if saturation.get(f"{concept}|{bucket}|capped_molecules", 0) == 0:
            exhausted.append(stratum)
    return exhausted


def _capacity(paths: dict, release: Path, policies: V3Policies) -> tuple[dict, dict]:
    caps = {key: int(value) for key, value in
            policies.sampling["initial_queries_per_retrieval_per_bucket"].items()}
    quotas = {key: int(value * float(policies.sampling["capacity_headroom"]))
              for key, value in policies.sampling["quotas"].items()}
    capacity_dir = paths["select"].parent / "capacity_check"
    failure_path = capacity_dir / "capacity_failure.json"
    failure_path.unlink(missing_ok=True)
    for attempt in range(6):
        pair_report = _pair(paths, release, caps)
        capacity_report = _select(paths, release, quotas, capacity_dir)
        deficient_strata = _actionable_deficits(capacity_report, policies)
        deficient = {stratum.split("|", 1)[0] for stratum in deficient_strata}
        if not deficient_strata:
            return caps, {"pairs": pair_report, "capacity": capacity_report}
        exhausted = [name for name in _exhausted_strata(pair_report, capacity_report)
                     if name in deficient_strata]
        if exhausted:
            failure = {"status": "blocked", "reason": "candidate_universe_exhausted",
                       "exhausted_strata": exhausted, "query_caps": caps,
                       "pairs": pair_report, "capacity": capacity_report}
            capacity_dir.mkdir(parents=True, exist_ok=True)
            (capacity_dir / "capacity_failure.json").write_text(json.dumps(failure, indent=2))
            raise RuntimeError(f"v3 candidate universe exhausted for strata: {exhausted}")
        for concept in deficient:
            caps[concept] = 0 if attempt == 4 else max(1, caps[concept] * 2)
    raise RuntimeError(f"v3 capacity exhausted for concepts: {sorted(deficient)}")


def _resolved_targets(policies: V3Policies, capacity: dict) -> dict[str, dict[str, int]]:
    concepts = policies.concepts["concepts"]
    strata = [f"{concept}|{bucket}" for concept in concepts for bucket in ("low", "high")]
    quotas = {key: int(value) for key, value in policies.sampling["quotas"].items()}
    targets = {split: {stratum: quotas[split] // len(strata) for stratum in strata}
               for split in SPLITS}
    available = {split: capacity["splits"][split]["per_stratum_available"] for split in SPLITS}
    sparse = policies.sampling["sparse_strata"]
    for stratum in sparse["train_all_available"]:
        targets["train"][stratum] = available["train"].get(stratum, 0)
    for stratum in sparse["heldout_matched_min"]:
        matched = min(available["validation"].get(stratum, 0), available["test"].get(stratum, 0))
        targets["validation"][stratum] = matched
        targets["test"][stratum] = matched
    _apply_heldout_backfill(targets, available, sparse, quotas, len(concepts))
    return targets


def _apply_heldout_backfill(targets: dict, available: dict, sparse: dict,
                            quotas: dict, concept_count: int) -> None:
    """Keep concept totals fixed when one held-out similarity bucket is exhausted."""
    concept_target = quotas["validation"] // concept_count
    for split in ("validation", "test"):
        for concept in sparse.get("heldout_backfill_within_concept", []):
            high, low = f"{concept}|high", f"{concept}|low"
            high_target = min(targets[split][high], available[split].get(high, 0))
            low_target = concept_target - high_target
            if available[split].get(low, 0) < low_target:
                raise RuntimeError(f"insufficient {split} backfill capacity for {concept}")
            targets[split][high] = high_target
            targets[split][low] = low_target


def _target_totals(targets: dict[str, dict[str, int]]) -> dict[str, int]:
    return {split: sum(values.values()) for split, values in targets.items()}


def resume_after_capacity(config: dict[str, Any], root: Path) -> dict[str, Any]:
    build_name, release = config["build"], resolve_path(config["release"])
    policies, paths = V3Policies(release), _paths(root, build_name)
    capacity = json.loads((paths["select"].parent / "capacity_check/manifest.json").read_text())
    pair_report = json.loads((paths["pairs"] / "manifest.json").read_text())
    targets = _resolved_targets(policies, capacity)
    results = {"build": build_name, "release": str(release), "stages":
               {"pairs": pair_report, "capacity": capacity}}
    results["stages"]["select"] = _select(paths, release, _target_totals(targets),
                                             paths["select"], targets)
    results["stages"]["materialize"] = _materialize(paths, policies)
    results["stages"]["render_hf"] = _render(config, paths)
    caps = pair_report["query_caps"]
    results["stages"]["expand"] = _expand(config, paths, release, caps)
    results.update({"resolved_query_caps": caps, "resolved_stratum_targets": targets})
    output = root / "builds" / f"{build_name}.run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    return results


def _materialize(paths: dict, policies: V3Policies) -> dict:
    selected = [paths["select"] / "selected" / name for name in SPLITS]
    return materialize.build(Namespace(pairs=selected, base=[paths["eligible"]],
                                       output_dir=paths["parquet"],
                                       allow_null_train=bool(
                                           policies.release.get("soft_evidence_primary"))))


def _render(config: dict, paths: dict) -> dict:
    template_dir = resolve_path(config["hf"]["template_dir"])
    policies = V3Policies(resolve_path(config["release"]))
    schema_version = policies.release["artifact_schema_version"]
    soft_evidence = bool(policies.release.get("soft_evidence_primary"))
    target_version = policies.target["version"] if policies.target else None
    return render_hf.build(Namespace(dataset=paths["parquet"] / "dataset.parquet",
                                     template_dir=template_dir, output_dir=paths["hf"],
                                     schema_version=schema_version, soft_evidence=soft_evidence,
                                     target_policy_version=target_version))


def _render_v5_variant(config: dict, root: Path, variant: str) -> dict:
    build_name = config["build"] + ("_intern" if variant == "intern" else "")
    source = resolve_path(config["frozen_materialized_source"])
    template_key = "intern_template_dir" if variant == "intern" else "template_dir"
    target = V3Policies(resolve_path(config["release"])).target
    return render_hf.build(Namespace(
        dataset=source,
        template_dir=resolve_path(config["hf"][template_key]),
        output_dir=root / "hf_parquet" / build_name,
        template_variant=variant,
        schema_version="assay_transfer_variance_soft_v5",
        soft_evidence=False,
        variance_soft_binary=True,
        target_policy_version=target["version"],
    ))


def _run_frozen_v5(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = resolve_path(config["frozen_materialized_source"])
    if not source.exists():
        raise FileNotFoundError(f"frozen V4 materialized source does not exist: {source}")
    source_rows = pq.read_table(source, columns=["candidate_id"]).num_rows
    results = {"build": config["build"], "release": str(resolve_path(config["release"])),
               "frozen_materialized_source": str(source), "source_rows": source_rows,
               "stages": {}}
    results["stages"]["render_hf"] = _render_v5_variant(config, root, "standard")
    results["stages"]["render_hf_intern"] = _render_v5_variant(config, root, "intern")
    output = root / "builds" / f"{config['build']}.run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    return results


def _expand(config: dict, paths: dict, release: Path, caps: dict) -> dict:
    return expand_v3.build(Namespace(eligible=paths["eligible"], split_dir=paths["split"],
                                     selection_dir=paths["select"], release=str(release),
                                     template_dir=resolve_path(config["hf"]["template_dir"]),
                                     output_dir=paths["expand"], query_caps=caps))


def run_build(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if config.get("frozen_materialized_source"):
        return _run_frozen_v5(config, root)
    build_name, release = config["build"], resolve_path(config["release"])
    policies, paths = V3Policies(release), _paths(root, build_name)
    results: dict[str, Any] = {"build": build_name, "release": str(release), "stages": {}}
    results["stages"]["compose"] = _compose(config, paths, release)
    results["stages"]["split"] = _split(paths, policies)
    caps, capacity = _capacity(paths, release, policies)
    results["stages"].update(capacity)
    targets = _resolved_targets(policies, capacity["capacity"])
    quotas = _target_totals(targets)
    results["stages"]["select"] = _select(paths, release, quotas, paths["select"], targets)
    if _deficient(results["stages"]["select"]):
        raise RuntimeError("final v3 selection underfilled after successful capacity check")
    results["stages"]["materialize"] = _materialize(paths, policies)
    results["stages"]["render_hf"] = _render(config, paths)
    results["stages"]["expand"] = _expand(config, paths, release, caps)
    results["resolved_query_caps"] = caps
    results["resolved_stratum_targets"] = targets
    output = root / "builds" / f"{build_name}.run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--resume-after-capacity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = resume_after_capacity(config, args.root) if args.resume_after_capacity else run_build(config, args.root)
    print(json.dumps({"build": result["build"], "stages": list(result["stages"])}, indent=2))


if __name__ == "__main__":
    main()
