#!/usr/bin/env python3
"""Canonical Oral Bioavailability HF prompt parquet renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INTERNAL_DIR = SCRIPT_DIR / "internal"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(INTERNAL_DIR) not in sys.path:
    sys.path.insert(0, str(INTERNAL_DIR))

import create_hf_parquets_from_splits as hf_render  # noqa: E402


SPLIT_DIR_PREFIX = {
    "source_value": "shared_eval_full",
    "no_source_value": "shared_eval_unidirectional_full",
}
TEMPLATES = {
    "generic": {
        "source_value": Path("templates/generic_transfer_classification.jinja"),
        "no_source_value": Path("templates/generic_transfer_classification_no_source_value.jinja"),
    },
    "intern": {
        "source_value": Path("templates/generic_transfer_classification_intern.jinja"),
        "no_source_value": Path("templates/generic_transfer_classification_no_source_value_intern.jinja"),
    },
}


def default_split_dir(universe: str, variant: str, split_version: str) -> Path:
    suffix = f"{SPLIT_DIR_PREFIX[variant]}_{split_version}"
    return Path("datasets/pairs_split_full") / f"oral_bioavailability_{universe}_{suffix}"


def default_output_dir(universe: str, variant: str, split_version: str, template_family: str) -> Path:
    suffix = "source_value" if variant == "source_value" else "no_source_value"
    template_part = "" if template_family == "generic" else f"_{template_family}"
    return Path("datasets/pairs_split_hf") / f"oral_bioavailability_{universe}_{suffix}{template_part}_{split_version}"


def build_one(args: argparse.Namespace, universe: str, variant: str) -> dict:
    render_args = argparse.Namespace(
        split_dir=args.split_dir or default_split_dir(universe, variant, args.split_version),
        output_dir=args.output_dir or default_output_dir(
            universe,
            variant,
            args.split_version,
            args.template_family,
        ),
        input_format=args.input_format,
        template=args.template or TEMPLATES[args.template_family][variant],
        variant=variant,
        splits=args.splits,
        max_rows_per_split=args.max_rows_per_split,
        train_sample_fraction=args.train_sample_fraction,
        train_sampling_strategy=args.train_sampling_strategy,
        train_source_pair_sample_fraction=args.train_source_pair_sample_fraction,
        dedupe_opposite_directions=args.dedupe_opposite_directions,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        workers=args.workers,
        parquet_row_group_size=args.parquet_row_group_size,
        parquet_compression=args.parquet_compression,
        progress_every_seconds=args.progress_every_seconds,
        overwrite=args.overwrite,
    )
    if variant == "no_source_value" and args.validate_unidirectional_train:
        metadata = render_args.split_dir / "metadata.json"
        if metadata.exists():
            payload = json.loads(metadata.read_text())
            if payload.get("train_direction_mode") != "unidirectional":
                raise ValueError(f"{render_args.split_dir} is not a unidirectional train split")
    return hf_render.build(render_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("condition_key", "same_species_v2", "no_constraints", "all"), default="all")
    parser.add_argument("--variant", choices=("source_value", "no_source_value", "both"), default="both")
    parser.add_argument("--split-version", choices=("v3", "v3_v2"), default="v3_v2")
    parser.add_argument("--template-family", choices=("generic", "intern"), default="generic")
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-format", choices=("auto", "jsonl", "parquet"), default="auto")
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=hf_render.SPLITS, default=list(hf_render.SPLITS))
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--train-sample-fraction", type=float, default=1.0)
    parser.add_argument("--train-sampling-strategy", choices=("random", "label_stratified"), default="random")
    parser.add_argument("--train-source-pair-sample-fraction", type=float, default=None)
    parser.add_argument("--dedupe-opposite-directions", action="store_true")
    parser.add_argument("--validate-unidirectional-train", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parquet-row-group-size", type=int, default=50_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--progress-every-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universes = ("condition_key", "same_species_v2", "no_constraints") if args.universe == "all" else (args.universe,)
    variants = ("source_value", "no_source_value") if args.variant == "both" else (args.variant,)
    if (len(universes) > 1 or len(variants) > 1) and (args.split_dir or args.output_dir):
        raise SystemExit("--split-dir/--output-dir can only be used with one universe and one variant")
    out = {}
    for universe in universes:
        for variant in variants:
            out[f"{universe}/{variant}"] = build_one(args, universe, variant)
    print(json.dumps({key: value["rendered_counts"] for key, value in out.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
