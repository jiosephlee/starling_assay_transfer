#!/usr/bin/env python3
"""Run the canonical Oral Bioavailability v3 dataset pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def phase_base(config: dict, python: str, overwrite: bool) -> None:
    base = config["base"]
    cmd = [
        python,
        "scripts/build_oral_bioavailability_base.py",
        "--input",
        base["input"],
        "--output-dir",
        base["output_dir"],
        "--tdc-dir",
        base["tdc_dir"],
        "--tdc-task",
        base["tdc_task"],
        "--reference-dir",
        base["reference_dir"],
        "--smiles-column",
        base.get("smiles_column", "smiles"),
        "--tdc-splits",
        *base.get("tdc_splits", ["train", "valid", "test"]),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run(cmd)


def phase_pairs(config: dict, python: str, overwrite: bool) -> None:
    base = config["base"]
    pairs = config["pairs"]
    for mode, mode_cfg in pairs["modes"].items():
        cmd = [
            python,
            "scripts/create_oral_bioavailability_pairs.py",
            "--input",
            base["output_dir"],
            "--mode",
            mode,
            "--output-root",
            pairs["output_root"],
            "--output-dir",
            str(Path(pairs["output_root"]) / mode_cfg["output_dir"]),
            "--metadata-columns",
            *pairs["metadata_columns"],
            "--transfer-threshold",
            str(pairs.get("transfer_threshold", 10.0)),
            "--not-transfer-threshold",
            str(pairs.get("not_transfer_threshold", 30.0)),
            "--similarity-buckets",
            str(pairs.get("similarity_buckets", 5)),
            "--seed",
            str(pairs.get("seed", 13)),
            "--workers",
            str(pairs.get("workers", 1)),
            "--tasks-per-worker",
            str(pairs.get("tasks_per_worker", 4)),
            "--progress-every",
            str(pairs.get("progress_every", 1_000_000)),
        ]
        if pairs.get("similarity_thresholds"):
            cmd.extend(["--similarity-thresholds", *[str(value) for value in pairs["similarity_thresholds"]]])
        if pairs.get("enumerate_all", True):
            cmd.append("--enumerate-all")
        if overwrite:
            cmd.append("--overwrite")
        run(cmd)


SPLITS_SCRIPTS = {
    "v1": "scripts/create_oral_bioavailability_splits.py",
    "v2": "scripts/create_oral_bioavailability_splits_v2.py",
    "community": "scripts/create_oral_bioavailability_splits_v2.py",
}


def phase_splits(config: dict, python: str, overwrite: bool) -> None:
    base = config["base"]
    pairs = config["pairs"]
    splits = config["splits"]
    strategy = str(splits.get("selection_strategy", "v1")).lower()
    if strategy not in SPLITS_SCRIPTS:
        raise ValueError(f"unknown splits.selection_strategy: {strategy!r}")
    cmd = [
        python,
        SPLITS_SCRIPTS[strategy],
        "--base-input",
        base["output_dir"],
        "--condition-key-pairs",
        str(Path(pairs["output_root"]) / pairs["modes"]["condition_key"]["output_dir"]),
        "--same-species-pairs",
        str(Path(pairs["output_root"]) / pairs["modes"]["same_species_v2"]["output_dir"]),
        "--no-constraints-pairs",
        str(Path(pairs["output_root"]) / pairs["modes"]["no_constraints"]["output_dir"]),
        "--output-root",
        splits["output_root"],
        "--train-direction-mode",
        "both",
        "--eval-directions-per-subset",
        str(splits.get("eval_directions_per_subset", 10_000)),
        "--shared-eval-compatibility-column",
        splits.get("shared_eval_compatibility_column", "species_or_population_normalized"),
    ]
    if "universes" in splits:
        cmd.extend(["--universes", *splits["universes"]])
    if "candidate_pool_multiplier" in splits:
        cmd.extend(["--candidate-pool-multiplier", str(splits["candidate_pool_multiplier"])])
    if strategy == "v1":
        if "preferred_no_train_molecules" in splits:
            cmd.extend(["--preferred-no-train-molecules", str(splits["preferred_no_train_molecules"])])
    else:
        for key, flag in (
            ("community_method", "--community-method"),
            ("community_resolution", "--community-resolution"),
            ("holdout_cost_weight", "--holdout-cost-weight"),
            ("holdout_supply_multiplier", "--holdout-supply-multiplier"),
            ("holdout_cost_pairs", "--holdout-cost-pairs"),
        ):
            if key in splits:
                cmd.extend([flag, str(splits[key])])
    if "arrow_cpu_count" in splits:
        cmd.extend(["--arrow-cpu-count", str(splits["arrow_cpu_count"])])
    if splits.get("disjoint_eval_molecules", False):
        cmd.append("--disjoint-eval-molecules")
    if overwrite:
        cmd.append("--overwrite")
    run(cmd)


def phase_hf(config: dict, python: str, overwrite: bool) -> None:
    for variant in config["hf"]["variants"]:
        cmd = [
            python,
            "scripts/create_oral_bioavailability_hf.py",
            "--universe",
            "all",
            "--variant",
            variant,
            "--validate-unidirectional-train",
        ]
        if overwrite:
            cmd.append("--overwrite")
        run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/oral_bioavailability_v3.yaml")
    parser.add_argument("--phase", choices=("base", "pairs", "splits", "hf", "all"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    phases = ("base", "pairs", "splits", "hf") if args.phase == "all" else (args.phase,)
    for phase in phases:
        globals()[f"phase_{phase}"](config, args.python, args.overwrite)


if __name__ == "__main__":
    main()
