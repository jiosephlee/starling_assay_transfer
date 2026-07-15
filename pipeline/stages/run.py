#!/usr/bin/env python3
"""Build DAG runner: a build config -> the full assay-transfer pipeline.

Executes the stages in dependency order over the stage-first ``datasets/`` layout
(``dataset_system_design.md`` sections 2, 7):

    prepare (per source) -> split (composed) -> pairs (per K profile)
        -> materialize (per profile) -> render_hf (per profile)

Outputs are keyed by build name so progress is visible per stage directory:

    <root>/base/<source>/                 (per source, reusable)
    <root>/splits/<build>/                (per build)
    <root>/pairs/<build>/<profile>/
    <root>/parquet/<build>/<profile>/
    <root>/hf_parquet/<build>/<profile>/

Each stage writes its own manifest.json. ``--from`` / ``--only`` restrict which stages run.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages import materialize, pairs, prepare, render_hf, split  # noqa: E402

STAGES = ("prepare", "split", "pairs", "materialize", "render_hf")
DEFAULT_SOURCE_DIR = _REPO_ROOT / "datasets" / "starling_assays" / "datasets"


def _stages_to_run(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [args.only]
    start = STAGES.index(args.from_stage) if args.from_stage else 0
    return list(STAGES[start:])


def run_build(config: dict[str, Any], root: Path, source_dir: Path, stages: list[str]) -> dict[str, Any]:
    build_name = config["build"]
    sources = config["sources"]
    split_cfg = config.get("split", {})
    pairs_cfg = config.get("pairs", {})
    hf_cfg = config.get("hf", {})
    profiles = pairs_cfg.get("profiles", ["same_endpoint"])

    base_dirs = [root / "base" / s for s in sources]
    split_dir = root / "splits" / build_name
    results: dict[str, Any] = {"build": build_name, "stages": {}}

    if "prepare" in stages:
        results["stages"]["prepare"] = {}
        for s in sources:
            r = prepare.build(
                Namespace(source=s, input=None, source_dir=source_dir,
                          output_dir=root / "base" / s, compression="zstd")
            )
            results["stages"]["prepare"][s] = {"records_kept": r["records_kept"]}

    if "split" in stages:
        results["stages"]["split"] = split.build(
            Namespace(base=base_dirs, output_dir=split_dir,
                      seed=split_cfg.get("seed", 17),
                      train_frac=split_cfg.get("train_frac", 0.8),
                      val_frac=split_cfg.get("val_frac", 0.1),
                      test_frac=split_cfg.get("test_frac", 0.1))
        )

    for profile in profiles:
        pairs_dir = root / "pairs" / build_name / profile
        parquet_dir = root / "parquet" / build_name / profile
        hf_dir = root / "hf_parquet" / build_name / profile
        if "pairs" in stages:
            results["stages"].setdefault("pairs", {})[profile] = pairs.build(
                Namespace(base=base_dirs, split_dir=split_dir, profile=profile,
                          output_dir=pairs_dir, seed=pairs_cfg.get("seed", 17),
                          max_queries=pairs_cfg.get("max_queries", 64))
            )
        if "materialize" in stages:
            results["stages"].setdefault("materialize", {})[profile] = materialize.build(
                Namespace(pairs=pairs_dir / "pairs.parquet", base=base_dirs, output_dir=parquet_dir)
            )
        if "render_hf" in stages:
            template = Path(hf_cfg.get("template", render_hf.DEFAULT_TEMPLATE))
            results["stages"].setdefault("render_hf", {})[profile] = render_hf.build(
                Namespace(dataset=parquet_dir / "dataset.parquet", template=template, output_dir=hf_dir)
            )

    (root / "builds").mkdir(parents=True, exist_ok=True)
    (root / "builds" / f"{build_name}.run.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="configs/builds/<build>.yaml")
    parser.add_argument("--root", type=Path, default=_REPO_ROOT / "datasets")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default=None)
    parser.add_argument("--only", choices=STAGES, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    results = run_build(config, args.root, args.source_dir, _stages_to_run(args))
    print(json.dumps({"build": results["build"], "stages": list(results["stages"])}, indent=2))


if __name__ == "__main__":
    main()
