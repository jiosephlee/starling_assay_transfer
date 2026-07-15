"""Convert a TDC official_tianang task (SMILES + binary Y) into the record-native
KNN eval layout, plus a base parquet for embedding precompute.

TDC rows only carry ``drug`` (SMILES) and ``Y`` (0/1) -- no assay/condition metadata and
no oral bioavailability scalar. We therefore leave the 7 model metadata fields null (they
become "missing" -> learned missing_emb) and set ``oral_bioavailability_value`` to NaN
(only used for the ob_bin slice label, never by the model / no-source-value scorer).

The base parquet concatenates train then valid molecules so a single precomputed embedding
table (indexed by position == row_index) covers both the neighbor pool and the queries.

Outputs (relative to repo root):
  datasets/base/<name>/base.parquet
  datasets/starling_eval/<name>_record_splits_hf/
    data/full_metadata/train/part-00000.parquet          (neighbor "sources")
    data/full_metadata/validation_1/part-00000.parquet   (queries; TDC valid split)
    metadata/split_manifest.json

Usage:
    python ml/scripts/build_tdc_record_splits.py --task Bioavailability_Ma
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The 7 free-text metadata fields the model expects in the base parquet (all null for TDC).
METADATA_FIELDS = (
    "molecule_name",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
)


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def record_frame(smiles: list[str], labels: list[int], start_index: int) -> pd.DataFrame:
    n = len(smiles)
    return pd.DataFrame(
        {
            "row_index": np.arange(start_index, start_index + n, dtype=np.int64),
            "smiles": smiles,
            "Y": np.asarray(labels, dtype=np.int64),
            "oral_bioavailability_value": np.full(n, np.nan, dtype=np.float64),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Bioavailability_Ma")
    parser.add_argument("--tdc-dir", default="tdc/official_tianang")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    tdc = root / args.tdc_dir
    name = f"tdc_{args.task}" if not args.task.startswith("tdc_") else args.task

    train = load_jsonl(tdc / "train" / f"{args.task}.jsonl")
    valid = load_jsonl(tdc / "valid" / f"{args.task}.jsonl")
    print(f"[tdc] {args.task}: train={len(train)} valid={len(valid)}")

    # Base parquet: train rows first (row_index 0..), then valid rows -- one embedding table.
    all_smiles = list(train["drug"]) + list(valid["drug"])
    base = pd.DataFrame({"smiles": all_smiles})
    for f in METADATA_FIELDS:
        base[f] = pd.Series([None] * len(base), dtype="object")
    base_path = root / "datasets" / "base" / name / "base.parquet"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(base_path, index=False)
    print(f"[base] wrote {base_path} ({len(base)} molecules)")

    # Record splits: train = sources (row_index 0..), validation_1 = queries (row_index after train).
    train_rec = record_frame(list(train["drug"]), list(train["Y"]), start_index=0)
    valid_rec = record_frame(list(valid["drug"]), list(valid["Y"]), start_index=len(train))

    ds_root = root / "datasets" / "starling_eval" / f"{name}_record_splits_hf"
    for split, frame in (("train", train_rec), ("validation_1", valid_rec)):
        out = ds_root / "data" / "full_metadata" / split / "part-00000.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        print(f"[record] wrote {out} ({len(frame)} rows, row_index "
              f"{int(frame['row_index'].min())}..{int(frame['row_index'].max())})")

    manifest = {
        "source": "TDC official_tianang",
        "task": args.task,
        "config": "full_metadata",
        "splits": {"train": len(train_rec), "validation_1": len(valid_rec)},
        "base_parquet": str(base_path.relative_to(root)),
        "note": "no metadata / no source value; all metadata fields missing",
    }
    manifest_path = ds_root / "metadata" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
