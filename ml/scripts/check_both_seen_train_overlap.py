
#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('datasets/pairs_split_full')
DATASETS = [
    ('condition_key', ROOT / 'oral_bioavailability_condition_key_shared_eval_full'),
    ('same_species_v2', ROOT / 'oral_bioavailability_same_species_v2_shared_eval_full'),
    ('no_constraints', ROOT / 'oral_bioavailability_no_constraints_shared_eval_full'),
]
BATCH_SIZE = 2_000_000


def read_val_both_seen(path: Path):
    rows = []
    for f in sorted((path / 'validation').glob('*.parquet')):
        table = pq.read_table(f, columns=['row_index_a', 'row_index_b', 'eval_subset'])
        a = table['row_index_a'].to_numpy(zero_copy_only=False)
        b = table['row_index_b'].to_numpy(zero_copy_only=False)
        subset = table['eval_subset'].to_pylist()
        mask = np.array([x == 'both_seen' for x in subset], dtype=bool)
        rows.extend(zip(a[mask].tolist(), b[mask].tolist()))
    directed = set((int(a), int(b)) for a, b in rows)
    unordered = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in rows)
    mols = sorted({x for pair in unordered for x in pair})
    return rows, directed, unordered, np.array(mols, dtype=np.uint32), set(mols)


def pack_pairs(a: np.ndarray, b: np.ndarray, *, unordered: bool) -> np.ndarray:
    x = a.astype(np.uint64, copy=False)
    y = b.astype(np.uint64, copy=False)
    if unordered:
        lo = np.minimum(x, y)
        hi = np.maximum(x, y)
        return (lo << np.uint64(32)) | hi
    return (x << np.uint64(32)) | y


def add_counts(counter: Counter, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    unique, counts = np.unique(values, return_counts=True)
    for value, count in zip(unique.tolist(), counts.tolist()):
        counter[int(value)] += int(count)


def scan_train(path: Path, directed, unordered, mols_np, mols_set):
    train_files = sorted((path / 'train').glob('*.parquet'))
    directed_keys = np.array(
        [((int(a) << 32) | int(b)) for a, b in directed],
        dtype=np.uint64,
    )
    unordered_keys = np.array(
        [((int(a) << 32) | int(b)) for a, b in unordered],
        dtype=np.uint64,
    )
    directed_overlap = 0
    unordered_overlap = 0
    rows = 0
    incident_rows = 0
    both_endpoints_in_both_seen_set = 0
    deg = Counter()
    file_stats = []
    t0 = time.time()
    for i, f in enumerate(train_files, 1):
        pf = pq.ParquetFile(f)
        f_rows = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=['row_index_a', 'row_index_b']):
            a = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            b = batch.column(1).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
            n = len(a)
            rows += n
            f_rows += n
            directed_overlap += int(np.count_nonzero(np.isin(pack_pairs(a, b, unordered=False), directed_keys)))
            unordered_overlap += int(np.count_nonzero(np.isin(pack_pairs(a, b, unordered=True), unordered_keys)))
            ma = np.isin(a, mols_np, assume_unique=False)
            mb = np.isin(b, mols_np, assume_unique=False)
            incident_rows += int(np.count_nonzero(ma | mb))
            both_endpoints_in_both_seen_set += int(np.count_nonzero(ma & mb))
            add_counts(deg, a[ma])
            add_counts(deg, b[mb])
        file_stats.append((str(f), f_rows))
        elapsed = time.time() - t0
        print(f'[progress] {path.name} file {i}/{len(train_files)} rows={rows:,} elapsed={elapsed:.1f}s', flush=True)
    return {
        'train_rows_scanned': rows,
        'directed_exact_overlap_rows': directed_overlap,
        'unordered_exact_overlap_rows': unordered_overlap,
        'train_rows_incident_to_both_seen_molecules': incident_rows,
        'train_rows_with_both_endpoints_in_both_seen_molecule_set': both_endpoints_in_both_seen_set,
        'degree_values': [deg[m] for m in mols_set],
    }


def summarize_degrees(vals):
    arr = np.array(vals, dtype=np.float64)
    if len(arr) == 0:
        return {}
    return {
        'min': int(np.min(arr)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.percentile(arr, 50)),
        'mean': float(np.mean(arr)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'max': int(np.max(arr)),
    }


def main():
    out = {}
    for key, path in DATASETS:
        meta = json.loads((path / 'metadata.json').read_text())
        val_rows, directed, unordered, mols_np, mols_set = read_val_both_seen(path)
        stats = scan_train(path, directed, unordered, mols_np, mols_set)
        deg_vals = stats.pop('degree_values')
        out[key] = {
            'rows_by_split': meta['rows_by_split'],
            'val_both_seen_rows': len(val_rows),
            'val_both_seen_directed_unique_pairs': len(directed),
            'val_both_seen_unordered_unique_pairs': len(unordered),
            'val_both_seen_unique_molecules': len(mols_set),
            **stats,
            'directed_exact_overlap_rate_vs_val_both_seen_rows': stats['directed_exact_overlap_rows'] / max(1, len(val_rows)),
            'unordered_exact_overlap_rate_vs_val_both_seen_unordered': stats['unordered_exact_overlap_rows'] / max(1, len(unordered)),
            'incident_rows_per_train_row': stats['train_rows_incident_to_both_seen_molecules'] / max(1, stats['train_rows_scanned']),
            'both_seen_molecule_degree': summarize_degrees(deg_vals),
        }
        print('[result]', key, json.dumps(out[key], sort_keys=True), flush=True)
    out_path = Path('ml/results/both_seen_train_overlap_source_value.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f'[done] wrote {out_path}', flush=True)

if __name__ == '__main__':
    main()
