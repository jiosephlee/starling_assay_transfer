#!/usr/bin/env python3
"""Render generic transfer split JSONL files into HF-style Parquet datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from jinja2 import Environment, Template

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_transfer import SPLITS, read_jsonl_gz, stable_priority, utc_now, write_json  # noqa: E402


DEFAULT_SPLIT_DIR = Path("datasets/pairs_split/generic_transfer_pair_splits")
DEFAULT_OUTPUT_DIR = Path("datasets/pairs_split_hf/generic_transfer_hf_parquet")
DEFAULT_TEMPLATE = Path("templates/generic_transfer_classification.jinja")
TRANSFER_COMPLETIONS = {"transfer": "A", "not_transfer": "B"}
HF_SCHEMA_VERSION = "generic_transfer_hf_parquet_v1"
SAMPLE_DENOMINATOR = 1_000_000
VARIANTS = ("source_value", "no_source_value")
SAFE_METADATA_FIELDS = {
    "molecule_name",
    "species_or_population",
    "dose",
    "oral_exposure_mode",
    "qualifying_conditions",
    "comparator",
    "extra_details",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressLogger:
    def __init__(self, phase: str, total: int | None, interval_seconds: float) -> None:
        self.phase = phase
        self.total = total if total and total > 0 else None
        self.interval_seconds = interval_seconds
        self.start = datetime.now(timezone.utc).timestamp()
        self.last = 0.0
        self.update(0, force=True)

    def update(self, current: int, *, force: bool = False, extra: str = "") -> None:
        if self.interval_seconds <= 0 and not force:
            return
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - self.last < self.interval_seconds:
            return
        elapsed = max(now - self.start, 1e-9)
        rate = current / elapsed
        if self.total:
            pct = 100.0 * current / self.total
            remaining = max(self.total - current, 0)
            eta = remaining / rate if rate > 0 else None
            rows = f"{current:,}/{self.total:,} ({pct:.2f}%)"
        else:
            eta = None
            rows = f"{current:,}"
        suffix = f" {extra}" if extra else ""
        print(
            f"[{utc_now()}] {self.phase}: rows={rows} "
            f"rate={rate:,.0f}/s elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        self.last = now

    def finish(self, current: int, *, extra: str = "") -> None:
        self.update(current, force=True, extra=extra)


def compile_template(path: Path) -> Template:
    env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    return env.from_string(path.read_text())


def render_template(template: Template, values: dict[str, Any]) -> str:
    return template.render(**values).strip() + "\n"


def row_label(row: dict[str, Any]) -> str:
    label = row.get("transfer_label")
    if label not in TRANSFER_COMPLETIONS:
        raise ValueError(f"invalid transfer label for {row.get('pair_id')}: {label!r}")
    return str(label)


def prompt_metadata(metadata: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "source_value":
        return metadata
    return {key: metadata.get(key) for key in SAFE_METADATA_FIELDS}


def build_values(row: dict[str, Any], variant: str) -> dict[str, Any]:
    molecule_a_metadata = row["molecule_a"].get("metadata") or {}
    molecule_b_metadata = row["molecule_b"].get("metadata") or {}
    values = {
        "molecule_a": {
            "canonical_smiles": row["molecule_a"]["canonical_smiles"],
            "metadata": prompt_metadata(molecule_a_metadata, variant),
        },
        "molecule_b": {
            "canonical_smiles": row["molecule_b"]["canonical_smiles"],
            "metadata": prompt_metadata(molecule_b_metadata, variant),
        },
        "group_id": row.get("group_id"),
    }
    if variant == "source_value":
        source_value = row.get("source_oral_bioavailability_value")
        if source_value is None:
            raise ValueError(f"missing source value for {row.get('pair_id')}")
        values["source_oral_bioavailability_value"] = source_value
    return values


def serialized_molecule_metadata(row: dict[str, Any], molecule: str, variant: str) -> str:
    metadata = row[molecule].get("metadata") or {}
    if variant == "no_source_value":
        metadata = prompt_metadata(metadata, variant)
    return json.dumps(metadata, sort_keys=True)


def metadata_for_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    metadata = {
        "sample_id": str(row["pair_id"]),
        "pair_id": str(row["pair_id"]),
        "split": row.get("split"),
        "split_version": row.get("split_version"),
        "eval_subset": row.get("eval_subset"),
        "group_id": row.get("group_id"),
        "record_id_a": str(row.get("record_id_a")),
        "record_id_b": str(row.get("record_id_b")),
        "canonical_smiles_a": row["molecule_a"]["canonical_smiles"],
        "canonical_smiles_b": row["molecule_b"]["canonical_smiles"],
        "direction": row.get("direction"),
        "source_pair_id": row.get("source_pair_id"),
        "transfer_label": row.get("transfer_label"),
        "weighted_tanimoto": (
            None if row.get("weighted_tanimoto") is None else float(row["weighted_tanimoto"])
        ),
        "value_difference": (
            None if row.get("value_difference") is None else float(row["value_difference"])
        ),
        "metadata_a_json": serialized_molecule_metadata(row, "molecule_a", variant),
        "metadata_b_json": serialized_molecule_metadata(row, "molecule_b", variant),
        "tool_version": HF_SCHEMA_VERSION,
    }
    if variant == "source_value":
        metadata["source_oral_bioavailability_value"] = float(row["source_oral_bioavailability_value"])
    return metadata


def hf_row(row: dict[str, Any], template: Template, variant: str) -> dict[str, Any]:
    label = row_label(row)
    return {
        "prompt": render_template(template, build_values(row, variant)),
        "completion": TRANSFER_COMPLETIONS[label],
        "metadata": metadata_for_row(row, variant),
    }


def validate_rendered_row(row: dict[str, Any], variant: str) -> None:
    prompt = row["prompt"]
    required = ["Molecule A", "Molecule B", "(A) transfer", "(B) not transfer", "Answer:"]
    missing = [text for text in required if text not in prompt]
    if missing:
        raise AssertionError(f"rendered prompt missing required text: {missing}")
    forbidden = [
        "transfer_label",
        "value_difference",
        "oral_bioavailability_value",
        "source_oral_bioavailability_value",
        "T_transfer",
        "T_not_transfer",
        "weighted_tanimoto",
        "support_text",
        "bioavailability_report_type",
    ]
    exposed = [text for text in forbidden if text in prompt]
    if exposed:
        raise AssertionError(f"rendered prompt exposes target/source leakage fields: {exposed}")
    if variant == "source_value" and "known oral bioavailability" not in prompt:
        raise AssertionError("source-value prompt is missing known oral bioavailability")
    if variant == "no_source_value":
        no_source_forbidden = [
            "known oral bioavailability",
            "source value",
            "source-value",
            "source_value",
            "source_oral_bioavailability_value",
        ]
        exposed_no_source = [text for text in no_source_forbidden if text in prompt]
        if exposed_no_source:
            raise AssertionError(f"no-source prompt exposes source value text: {exposed_no_source}")
        metadata = row["metadata"]
        if "source_oral_bioavailability_value" in metadata:
            raise AssertionError("no-source metadata includes source_oral_bioavailability_value")
        serialized = f"{metadata.get('metadata_a_json', '')}\n{metadata.get('metadata_b_json', '')}"
        metadata_forbidden = [
            "support_text",
            "bioavailability_report_type",
            "oral_bioavailability_value",
            "source_oral_bioavailability_value",
        ]
        exposed_metadata = [text for text in metadata_forbidden if text in serialized]
        if exposed_metadata:
            raise AssertionError(f"no-source metadata exposes leaky fields: {exposed_metadata}")
    if row["completion"] not in {"A", "B"}:
        raise AssertionError(f"invalid completion: {row['completion']}")


def schema(variant: str) -> pa.Schema:
    metadata_fields = [
        ("sample_id", pa.large_string()),
        ("pair_id", pa.large_string()),
        ("split", pa.string()),
        ("split_version", pa.string()),
        ("eval_subset", pa.string()),
        ("group_id", pa.large_string()),
        ("record_id_a", pa.large_string()),
        ("record_id_b", pa.large_string()),
        ("canonical_smiles_a", pa.large_string()),
        ("canonical_smiles_b", pa.large_string()),
        ("direction", pa.string()),
        ("source_pair_id", pa.large_string()),
    ]
    if variant == "source_value":
        metadata_fields.append(("source_oral_bioavailability_value", pa.float64()))
    metadata_fields.extend(
        [
            ("transfer_label", pa.string()),
            ("weighted_tanimoto", pa.float64()),
            ("value_difference", pa.float64()),
            ("metadata_a_json", pa.large_string()),
            ("metadata_b_json", pa.large_string()),
            ("tool_version", pa.string()),
        ]
    )
    metadata_type = pa.struct(metadata_fields)
    return pa.schema(
        [
            ("prompt", pa.large_string()),
            ("completion", pa.string()),
            ("metadata", metadata_type),
        ]
    )


class ParquetRowWriter:
    def __init__(self, path: Path, *, row_group_size: int, compression: str, variant: str) -> None:
        self.schema = schema(variant)
        self.writer = pq.ParquetWriter(path, self.schema, compression=compression)
        self.row_group_size = row_group_size
        self.buffer: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.row_group_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=self.schema))
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def __enter__(self) -> "ParquetRowWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def prepare_output(output_dir: Path, splits: Iterable[str], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"{split}.parquet" for split in splits] + [output_dir / split for split in splits]
    expected.append(output_dir / "dataset_info.json")
    present = [path for path in expected if path.exists()]
    if present and not overwrite:
        formatted = "\n".join(str(path) for path in present)
        raise FileExistsError(f"output files exist; pass --overwrite:\n{formatted}")
    for path in present:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def source_metadata(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parquet_split_path(split_dir: Path, split: str) -> Path:
    path = split_dir / split
    if path.exists():
        return path
    parquet_file = split_dir / f"{split}.parquet"
    if parquet_file.exists():
        return parquet_file
    raise FileNotFoundError(f"no Parquet split found for {split} under {split_dir}")


def parquet_row_count(path: Path) -> int:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    return sum(pq.ParquetFile(file_path).metadata.num_rows for file_path in files)


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    return files


def split_sample_fraction(args: argparse.Namespace, split: str) -> float:
    if split == "train":
        return args.train_sample_fraction
    return 1.0


def selected_direction(source_pair_id: str, seed: int) -> str:
    return "a_to_b" if stable_priority(seed, "direction", source_pair_id) % 2 == 0 else "b_to_a"


def source_pair_sample_fraction(args: argparse.Namespace, split: str) -> float | None:
    if split != "train":
        return None
    if args.train_source_pair_sample_fraction is not None:
        return args.train_source_pair_sample_fraction
    if args.dedupe_opposite_directions:
        return 1.0
    return None


def train_sampling_key(row: dict[str, Any], strategy: str, base_id: str) -> tuple[str, ...]:
    if strategy == "label_stratified":
        return ("train_label", str(row.get("transfer_label")), base_id)
    return ("train", base_id)


def keep_sampled_row(
    row: dict[str, Any],
    split: str,
    fraction: float,
    seed: int,
    *,
    train_source_pair_sample_fraction: float | None = None,
    dedupe_opposite_directions: bool = False,
    train_sampling_strategy: str = "random",
) -> bool:
    if split == "train" and (
        train_source_pair_sample_fraction is not None or dedupe_opposite_directions
    ):
        source_pair_id = str(row.get("source_pair_id") or row.get("pair_id"))
        source_fraction = 1.0 if train_source_pair_sample_fraction is None else train_source_pair_sample_fraction
        if source_fraction < 1.0:
            threshold = int(source_fraction * SAMPLE_DENOMINATOR)
            priority = stable_priority(
                seed,
                *train_sampling_key(row, train_sampling_strategy, source_pair_id),
            ) % SAMPLE_DENOMINATOR
            if priority >= threshold:
                return False
        if dedupe_opposite_directions:
            direction = row.get("direction")
            if direction in {"a_to_b", "b_to_a"}:
                return direction == selected_direction(source_pair_id, seed)
        return True

    if fraction >= 1.0:
        return True
    threshold = int(fraction * SAMPLE_DENOMINATOR)
    pair_id = str(row.get("pair_id"))
    if split == "train":
        return stable_priority(
            seed,
            *train_sampling_key(row, train_sampling_strategy, pair_id),
        ) % SAMPLE_DENOMINATOR < threshold
    return stable_priority(seed, split, pair_id) % SAMPLE_DENOMINATOR < threshold


def iter_split_rows(args: argparse.Namespace, split: str) -> Iterator[dict[str, Any]]:
    if args.input_format == "jsonl":
        yield from read_jsonl_gz(
            args.split_dir / f"{split}.jsonl.gz",
            max_rows=args.max_rows_per_split,
        )
        return

    path = parquet_split_path(args.split_dir, split)
    dataset = ds.dataset(path, format="parquet")
    sample_fraction = split_sample_fraction(args, split)
    source_fraction = source_pair_sample_fraction(args, split)
    yielded = 0
    for batch in dataset.to_batches(batch_size=args.batch_size):
        for row in batch.to_pylist():
            if not keep_sampled_row(
                row,
                split,
                sample_fraction,
                args.sample_seed,
                train_source_pair_sample_fraction=source_fraction,
                dedupe_opposite_directions=args.dedupe_opposite_directions,
                train_sampling_strategy=args.train_sampling_strategy,
            ):
                continue
            yield row
            yielded += 1
            if args.max_rows_per_split is not None and yielded >= args.max_rows_per_split:
                return


def infer_input_format(split_dir: Path, splits: list[str]) -> str:
    if all((split_dir / f"{split}.jsonl.gz").exists() for split in splits):
        return "jsonl"
    if all((split_dir / split).exists() or (split_dir / f"{split}.parquet").exists() for split in splits):
        return "parquet"
    raise FileNotFoundError(f"could not infer split input format under {split_dir}")


def render_parquet_file_worker(
    *,
    input_file: str,
    output_file: str,
    split: str,
    template_path: str,
    variant: str,
    train_sample_fraction: float,
    train_source_pair_sample_fraction: float | None,
    dedupe_opposite_directions: bool,
    train_sampling_strategy: str,
    sample_seed: int,
    batch_size: int,
    parquet_row_group_size: int,
    parquet_compression: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    template = compile_template(Path(template_path))
    input_path = Path(input_file)
    count = 0
    source_rows = 0
    completions: Counter[str] = Counter()
    sample_fraction = train_sample_fraction if split == "train" else 1.0
    source_fraction = train_source_pair_sample_fraction if split == "train" else None
    dedupe = dedupe_opposite_directions and split == "train"
    with ParquetRowWriter(
        Path(output_file),
        row_group_size=parquet_row_group_size,
        compression=parquet_compression,
        variant=variant,
    ) as writer:
        parquet_file = pq.ParquetFile(input_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            source_rows += batch.num_rows
            for source_row in batch.to_pylist():
                if not keep_sampled_row(
                    source_row,
                    split,
                    sample_fraction,
                    sample_seed,
                    train_source_pair_sample_fraction=source_fraction,
                    dedupe_opposite_directions=dedupe,
                    train_sampling_strategy=train_sampling_strategy,
                ):
                    continue
                row = hf_row(source_row, template, variant)
                if count < 1000:
                    validate_rendered_row(row, variant)
                writer.write(row)
                count += 1
                completions[row["completion"]] += 1
                if max_rows is not None and count >= max_rows:
                    break
    return {
        "input_file": input_file,
        "output_file": output_file,
        "split": split,
        "source_rows": source_rows,
        "rendered_rows": count,
        "completion_counts": dict(sorted(completions.items())),
    }


def build_parallel_parquet(args: argparse.Namespace, template: Template) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    del template
    rendered_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    completion_counts: dict[str, dict[str, int]] = {}
    files_by_split: dict[str, int] = {}
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for split in args.splits:
            split_path = parquet_split_path(args.split_dir, split)
            files = parquet_files(split_path)
            file_row_counts = [pq.ParquetFile(path).metadata.num_rows for path in files]
            source_counts[split] = sum(file_row_counts)
            output_split_dir = args.output_dir / split
            output_split_dir.mkdir(parents=True, exist_ok=True)
            files_by_split[split] = len(files)
            remaining_max = args.max_rows_per_split
            for part_id, (input_file, file_row_count) in enumerate(zip(files, file_row_counts, strict=True)):
                if remaining_max is not None:
                    if remaining_max <= 0:
                        break
                    max_rows_for_file = min(remaining_max, file_row_count)
                    remaining_max -= max_rows_for_file
                else:
                    max_rows_for_file = None
                futures.append(
                    executor.submit(
                        render_parquet_file_worker,
                        input_file=str(input_file),
                        output_file=str(output_split_dir / f"part-{part_id:05d}.parquet"),
                        split=split,
                        template_path=str(args.template),
                        variant=args.variant,
                        train_sample_fraction=args.train_sample_fraction,
                        train_source_pair_sample_fraction=args.train_source_pair_sample_fraction,
                        dedupe_opposite_directions=args.dedupe_opposite_directions,
                        train_sampling_strategy=args.train_sampling_strategy,
                        sample_seed=args.sample_seed,
                        batch_size=args.batch_size,
                        parquet_row_group_size=args.parquet_row_group_size,
                        parquet_compression=args.parquet_compression,
                        max_rows=max_rows_for_file,
                    )
                )
        completed = 0
        progress = ProgressLogger("render parquet shards", len(futures), args.progress_every_seconds)
        split_completion_counts: dict[str, Counter[str]] = {split: Counter() for split in args.splits}
        split_rendered_counts: Counter[str] = Counter()
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            split = result["split"]
            split_rendered_counts[split] += int(result["rendered_rows"])
            split_completion_counts[split].update(result["completion_counts"])
            progress.update(
                completed,
                extra=(
                    f"last={split}:{result['rendered_rows']:,} "
                    f"rendered_total={sum(split_rendered_counts.values()):,}"
                ),
            )
        progress.finish(completed, extra=f"rendered_total={sum(split_rendered_counts.values()):,}")
    for split in args.splits:
        rendered_counts[split] = int(split_rendered_counts[split])
        completion_counts[split] = dict(sorted(split_completion_counts[split].items()))
    return rendered_counts, completion_counts, source_counts, files_by_split


def validate_template_for_variant(path: Path, variant: str) -> None:
    text = path.read_text()
    if variant == "source_value" and "source_oral_bioavailability_value" not in text:
        raise ValueError("source_value variant requires a template that renders source_oral_bioavailability_value")
    if variant == "no_source_value":
        forbidden = [
            "source_oral_bioavailability_value",
            "known oral bioavailability",
        ]
        exposed = [token for token in forbidden if token in text]
        if exposed:
            raise ValueError(f"no_source_value template contains source-value text: {exposed}")


def metadata_field_names(parquet_schema: pa.Schema) -> set[str]:
    metadata_field = parquet_schema.field("metadata")
    return {field.name for field in metadata_field.type}


def verify_output(args: argparse.Namespace) -> None:
    for split in args.splits:
        path = args.output_dir / split
        if not path.exists():
            path = args.output_dir / f"{split}.parquet"
        for file_path in parquet_files(path):
            parquet_file = pq.ParquetFile(file_path)
            fields = metadata_field_names(parquet_file.schema_arrow)
            if args.variant == "source_value":
                if "source_oral_bioavailability_value" not in fields:
                    raise AssertionError(f"{file_path}: missing source value metadata field")
            else:
                if "source_oral_bioavailability_value" in fields:
                    raise AssertionError(f"{file_path}: no-source artifact has source value metadata field")
            sample_table = parquet_file.read_row_group(0) if parquet_file.metadata.num_row_groups else None
            if sample_table is None or sample_table.num_rows == 0:
                continue
            for row in sample_table.slice(0, min(sample_table.num_rows, 1000)).to_pylist():
                validate_rendered_row(row, args.variant)
            break


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    validate_template_for_variant(args.template, args.variant)
    if args.input_format == "auto":
        args.input_format = infer_input_format(args.split_dir, list(args.splits))
    if args.input_format == "jsonl":
        for split in args.splits:
            path = args.split_dir / f"{split}.jsonl.gz"
            if not path.exists():
                raise FileNotFoundError(path)
    else:
        for split in args.splits:
            parquet_split_path(args.split_dir, split)
    prepare_output(args.output_dir, args.splits, args.overwrite)
    template = compile_template(args.template)
    rendered_counts: dict[str, int] = {}
    completion_counts: dict[str, dict[str, int]] = {}
    source_counts: dict[str, int] = {}
    files_by_split: dict[str, int] = {}

    if args.input_format == "parquet" and args.workers > 1:
        rendered_counts, completion_counts, source_counts, files_by_split = build_parallel_parquet(args, template)
    else:
        for split in args.splits:
            count = 0
            seen = 0
            completions: Counter[str] = Counter()
            total_rows = (
                None
                if args.input_format == "jsonl"
                else parquet_row_count(parquet_split_path(args.split_dir, split))
            )
            sample_fraction = split_sample_fraction(args, split)
            source_fraction = source_pair_sample_fraction(args, split)
            progress_total = (
                None
                if total_rows is None
                else min(
                    total_rows,
                    int(
                        total_rows
                        * (
                            source_fraction
                            * (0.5 if args.dedupe_opposite_directions and split == "train" else 1.0)
                            if source_fraction is not None
                            else sample_fraction
                        )
                    )
                    + 1,
                )
            )
            progress = ProgressLogger(f"render {split}", progress_total, args.progress_every_seconds)
            with ParquetRowWriter(
                args.output_dir / f"{split}.parquet",
                row_group_size=args.parquet_row_group_size,
                compression=args.parquet_compression,
                variant=args.variant,
            ) as writer:
                for source_row in iter_split_rows(args, split):
                    seen += 1
                    row = hf_row(source_row, template, args.variant)
                    if count < 1000:
                        validate_rendered_row(row, args.variant)
                    writer.write(row)
                    count += 1
                    completions[row["completion"]] += 1
                    progress.update(seen, extra=f"rendered={count:,}")
            progress.finish(seen, extra=f"rendered={count:,}")
            source_counts[split] = int(total_rows or seen)
            rendered_counts[split] = count
            completion_counts[split] = dict(sorted(completions.items()))
            files_by_split[split] = 1

    verify_output(args)

    metadata = {
        "schema_version": HF_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "source_split_dir": str(args.split_dir),
        "source_split_metadata": source_metadata(args.split_dir),
        "output_dir": str(args.output_dir),
        "input_format": args.input_format,
        "variant": args.variant,
        "source_value_visible": args.variant == "source_value",
        "included_splits": list(args.splits),
        "template": str(args.template),
        "completion_mapping": dict(TRANSFER_COMPLETIONS),
        "row_schema": ["prompt", "completion", "metadata"],
        "source_counts": source_counts,
        "rendered_counts": rendered_counts,
        "completion_counts": completion_counts,
        "max_rows_per_split": args.max_rows_per_split,
        "sampling": {
            "train_sample_fraction": args.train_sample_fraction,
            "train_source_pair_sample_fraction": args.train_source_pair_sample_fraction,
            "dedupe_opposite_directions": args.dedupe_opposite_directions,
            "train_sampling_strategy": args.train_sampling_strategy,
            "effective_train_directional_row_fraction": (
                None
                if args.train_source_pair_sample_fraction is None
                else args.train_source_pair_sample_fraction
                * (0.5 if args.dedupe_opposite_directions else 1.0)
            ),
            "sample_seed": args.sample_seed,
            "sample_denominator": SAMPLE_DENOMINATOR,
            "policy": (
                "deterministic stable hash by source_pair_id with at most one direction per source pair"
                if args.train_source_pair_sample_fraction is not None or args.dedupe_opposite_directions
                else (
                    "deterministic stable hash by train transfer_label and pair_id; "
                    "validation/test always full"
                    if args.train_sampling_strategy == "label_stratified"
                    else "deterministic stable hash by split and pair_id; validation/test always full"
                )
            ),
        },
        "parquet": {
            "row_group_size": args.parquet_row_group_size,
            "compression": args.parquet_compression,
            "files_by_split": files_by_split,
        },
        "prompt_leakage_policy": (
            "validation rejects target label/value/similarity field names in rendered prompt; "
            "no_source_value also omits source value metadata and sanitizes serialized molecule metadata"
        ),
    }
    write_json(args.output_dir / "dataset_info.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-format", choices=("auto", "jsonl", "parquet"), default="auto")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--variant", choices=VARIANTS, default="source_value")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--train-sample-fraction", type=float, default=1.0)
    parser.add_argument("--train-sampling-strategy", choices=("random", "label_stratified"), default="random")
    parser.add_argument("--train-source-pair-sample-fraction", type=float, default=None)
    parser.add_argument("--dedupe-opposite-directions", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parquet-row-group-size", type=int, default=50_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--progress-every-seconds", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_rows_per_split is not None and args.max_rows_per_split < 1:
        parser.error("--max-rows-per-split must be positive")
    if args.parquet_row_group_size < 1:
        parser.error("--parquet-row-group-size must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0 < args.train_sample_fraction <= 1:
        parser.error("--train-sample-fraction must be in (0, 1]")
    if args.train_source_pair_sample_fraction is not None and not 0 < args.train_source_pair_sample_fraction <= 1:
        parser.error("--train-source-pair-sample-fraction must be in (0, 1]")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "output_dir": metadata["output_dir"],
                "rendered_counts": metadata["rendered_counts"],
                "completion_counts": metadata["completion_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
