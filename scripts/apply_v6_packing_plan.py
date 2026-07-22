#!/usr/bin/env python3
"""Join exact V6 token/packing assignments onto retained raw training rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _plan(path: Path) -> dict[tuple[int, int], dict]:
    rows = pq.read_table(path).to_pylist()
    return {(row["listnet_group_index"], row["listnet_member_index"]): row for row in rows}


def apply_plan(source: Path, plan_path: Path, output: Path) -> int:
    plan, writer, count = _plan(plan_path), None, 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=8192):
        output_rows = []
        for row in batch.to_pylist():
            key = (row["listnet_group_index"], row["listnet_member_index"])
            assignment = plan.get(key)
            if assignment is None:
                continue
            row.update({name: assignment[name] for name in (
                "templated_token_count", "packed_chunk_index", "optimizer_batch_index",
                "optimizer_batch_position")})
            output_rows.append(row)
        if output_rows:
            table = pa.Table.from_pylist(output_rows)
            writer = writer or pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(output_rows)
    if writer is None:
        raise RuntimeError("packing plan retained no training rows")
    writer.close()
    if count != len(plan):
        raise RuntimeError(f"packing-plan join mismatch: wrote {count}, expected {len(plan)}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print({"rows": apply_plan(args.source, args.plan, args.output)})


if __name__ == "__main__":
    main()
