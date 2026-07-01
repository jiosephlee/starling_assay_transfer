#!/usr/bin/env python3
"""Create shared directed full eval splits (v2 marginal-cost reuse holdout).

This is a v2 selection strategy. It leaves the v1 script untouched and reuses
its pure helpers (molecule index, eval-pool collection, universe writing,
staging, validation). The only behavioural change vs v1 is *which* molecules
are held out for the unseen eval subsets.

v1 prefers globally high-degree molecules, which keeps the held-out set small
(good reuse) but deletes the most training pairs. The first v2 attempt instead
ordered by absolute per-molecule cost, which scattered the held-out set across
many cheap low-degree molecules with almost no reuse, blowing up |S| (and thus
training-pair loss) ~4x.

This v2 uses a *marginal-cost reuse-aware greedy*: when filling each eval
subset it strongly prefers candidate pairs that reuse molecules already held
out (zero marginal training-pair cost), then pairs that extend the current
held-out cluster by a single cheap molecule, and only opens a fresh cheap pair
when neither is available. This builds a tight, low-conductance, low-cost
held-out cluster, keeping |S| small (like v1) while skewing it toward molecules
that are cheap to remove from the target (no_constraints) universe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_oral_bioavailability_shared_eval_splits as v1  # noqa: E402

# Aliases so the copied support-accounting logic runs unchanged against v1.
pair_key = v1.pair_key
molecule_pair_key = v1.molecule_pair_key
direction_endpoints = v1.direction_endpoints
selected_direction = v1.selected_direction
choose_stratum = v1.choose_stratum
row_priority = v1.row_priority
stratum_for_row = v1.stratum_for_row
stratum_to_string = v1.stratum_to_string
serializable_counter = v1.serializable_counter
stable_priority = v1.stable_priority
utc_now = v1.utc_now
write_json = v1.write_json
EVAL_SPLITS = v1.EVAL_SPLITS
EVAL_SUBSETS = v1.EVAL_SUBSETS

V2_SPLIT_VERSION = "oral_bioavailability_shared_eval_directed_v2"
HOLDOUT_STRATEGY = "marginal_cost_reuse_v2"


def mode_metadata_stem_v2(train_direction_mode: str) -> str:
    if train_direction_mode == "unidirectional":
        return "oral_bioavailability_shared_eval_v3_v2_unidirectional"
    return "oral_bioavailability_shared_eval_v3_v2"


def compute_molecule_costs(
    args: argparse.Namespace,
    molecule_index: v1.MoleculeIndex,
    *,
    pairs_dir: Path,
    label: str,
) -> tuple[np.ndarray, int]:
    """Per-molecule incident compact-pair count over a pair universe (the cost
    of holding the molecule out, i.e. training pairs it would delete)."""
    n_molecules = len(molecule_index.molecule_canonical_smiles)
    cost = np.zeros(n_molecules, dtype=np.int64)
    record_to_molecule = molecule_index.record_to_molecule
    input_path = v1.pair_dir_records(pairs_dir)
    total_rows = v1.parquet_row_count(input_path)
    dataset = ds.dataset(input_path, format="parquet")
    progress = v1.ProgressLogger(f"compute {label} molecule cost", total_rows, args.progress_every_seconds)
    processed = 0
    for batch in dataset.to_batches(columns=["row_index_a", "row_index_b"], batch_size=args.batch_size):
        left = batch.column("row_index_a").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        right = batch.column("row_index_b").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        left_molecule = record_to_molecule[left].astype(np.int64, copy=False)
        right_molecule = record_to_molecule[right].astype(np.int64, copy=False)
        cost += np.bincount(left_molecule, minlength=n_molecules).astype(np.int64, copy=False)
        cost += np.bincount(right_molecule, minlength=n_molecules).astype(np.int64, copy=False)
        processed += batch.num_rows
        progress.update(processed)
    progress.finish(processed)
    return cost, total_rows


def choose_stratum_filler(
    remaining: Counter[tuple[Any, ...]],
    allocation: dict[tuple[Any, ...], int],
    available: Any,
    seed: int,
    split: str,
    subset: str,
    step: int,
) -> tuple[Any, ...] | None:
    """choose_stratum variant whose selectable count comes from a filler."""
    best: tuple[float, int, int, int] | None = None
    best_stratum: tuple[Any, ...] | None = None
    for stratum, need in remaining.items():
        if need <= 0:
            continue
        selectable = available(stratum)
        if selectable <= 0:
            continue
        quota = max(1, allocation.get(stratum, 1))
        score = (
            need / quota,
            need,
            selectable,
            -stable_priority(seed, split, subset, "stratum", step, repr(stratum)),
        )
        if best is None or score > best:
            best = score
            best_stratum = stratum
    return best_stratum


class NoOverlapFiller:
    """Reuse-aware pair source for the no_overlap subset of one split.

    Pairs are served in priority order: both endpoints already held out
    (ready2, zero marginal cost) -> exactly one endpoint held out (ready1,
    opens one cheap molecule, extends the cluster) -> neither endpoint held out
    (static, cheapest-cost-sum first, opens a fresh pair).
    """

    def __init__(self, rows_by_stratum, molecule_index, cost, seed, split):
        self.meta: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.mol_pairs: dict[int, list[tuple[int, int]]] = {}
        self.static: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        self.sptr: dict[tuple[Any, ...], int] = {}
        self.ready2: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        self.ready1: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        for stratum, rows in rows_by_stratum.items():
            ordered = sorted(
                rows,
                key=lambda row: (
                    int(cost[molecule_index.molecule_id(row["row_index_a"])])
                    + int(cost[molecule_index.molecule_id(row["row_index_b"])]),
                    row_priority(seed, split, "no_overlap", row),
                ),
            )
            keys: list[tuple[int, int]] = []
            for row in ordered:
                pk = pair_key(row["row_index_a"], row["row_index_b"])
                if pk in self.meta:
                    continue
                left, right = molecule_index.pair_molecules(pk)
                self.meta[pk] = (stratum, left, right, row)
                self.mol_pairs.setdefault(left, []).append(pk)
                self.mol_pairs.setdefault(right, []).append(pk)
                keys.append(pk)
            self.static[stratum] = keys
            self.sptr[stratum] = 0
            self.ready2[stratum] = []
            self.ready1[stratum] = []

    def row(self, pk: tuple[int, int]) -> dict[str, Any]:
        return self.meta[pk][3]

    def available(self, stratum) -> int:
        return (
            len(self.ready2.get(stratum, ()))
            + len(self.ready1.get(stratum, ()))
            + (len(self.static.get(stratum, ())) - self.sptr.get(stratum, 0))
        )

    def on_added(self, molecule: int, holdout: set[int]) -> None:
        for pk in self.mol_pairs.get(molecule, ()):  # promote pairs touching molecule
            stratum, left, right, _row = self.meta[pk]
            other = right if left == molecule else left
            if other in holdout:
                self.ready2[stratum].append(pk)
            else:
                self.ready1[stratum].append(pk)

    def next(self, stratum, consumed: set[tuple[int, int]]) -> tuple[int, int] | None:
        for bucket in (self.ready2[stratum], self.ready1[stratum]):
            while bucket:
                pk = bucket.pop()
                if pk not in consumed:
                    return pk
        arr = self.static[stratum]
        index = self.sptr[stratum]
        while index < len(arr):
            pk = arr[index]
            index += 1
            if pk not in consumed:
                self.sptr[stratum] = index
                return pk
        self.sptr[stratum] = index
        return None


class ASeenFiller:
    """Reuse-aware pair source for the a_seen_only subset of one split.

    The held-out (target) molecule is fixed per pair by the deterministic
    direction. Pairs whose target is already held out are served first (zero
    marginal cost: reuse the no_overlap cluster); otherwise cheapest target
    first.
    """

    def __init__(self, rows_by_stratum, molecule_index, cost, seed, split):
        self.meta: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.target_pairs: dict[int, list[tuple[int, int]]] = {}
        self.static: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        self.sptr: dict[tuple[Any, ...], int] = {}
        self.ready: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        for stratum, rows in rows_by_stratum.items():
            entries: list[tuple[int, int, tuple[int, int]]] = []
            for row in rows:
                pk = pair_key(row["row_index_a"], row["row_index_b"])
                if pk in self.meta:
                    continue
                direction = selected_direction(seed, split, "a_seen_only", pk)
                source_record, target_record = direction_endpoints(pk, direction)
                source = molecule_index.molecule_id(source_record)
                target = molecule_index.molecule_id(target_record)
                self.meta[pk] = (stratum, row, source, target, direction)
                self.target_pairs.setdefault(target, []).append(pk)
                entries.append((int(cost[target]), row_priority(seed, split, "a_seen_only", row), pk))
            entries.sort()
            self.static[stratum] = [pk for _cost, _priority, pk in entries]
            self.sptr[stratum] = 0
            self.ready[stratum] = []

    def info(self, pk: tuple[int, int]) -> tuple[Any, ...]:
        return self.meta[pk]

    def available(self, stratum) -> int:
        return len(self.ready.get(stratum, ())) + (len(self.static.get(stratum, ())) - self.sptr.get(stratum, 0))

    def seed_ready(self, holdout: set[int]) -> None:
        for pk, (stratum, _row, _source, target, _direction) in self.meta.items():
            if target in holdout:
                self.ready[stratum].append(pk)

    def on_added(self, molecule: int) -> None:
        for pk in self.target_pairs.get(molecule, ()):
            self.ready[self.meta[pk][0]].append(pk)

    def next(self, stratum, consumed: set[tuple[int, int]]) -> tuple[int, int] | None:
        bucket = self.ready[stratum]
        while bucket:
            pk = bucket.pop()
            if pk not in consumed:
                return pk
        arr = self.static[stratum]
        index = self.sptr[stratum]
        while index < len(arr):
            pk = arr[index]
            index += 1
            if pk not in consumed:
                self.sptr[stratum] = index
                return pk
        self.sptr[stratum] = index
        return None


def cost_report(
    cost: np.ndarray,
    total_pairs: int,
    no_train_molecules: set[int],
    pools: dict[str, dict[str, dict[tuple[Any, ...], list[dict[str, Any]]]]],
    molecule_index: v1.MoleculeIndex,
    *,
    cost_pairs_dir: Path,
) -> dict[str, Any]:
    selected_cost = int(sum(int(cost[molecule]) for molecule in no_train_molecules))
    degrees = v1.eval_candidate_degrees(pools, molecule_index)
    baseline_set = {molecule for molecule, _count in degrees.most_common(len(no_train_molecules))}
    baseline_cost = int(sum(int(cost[molecule]) for molecule in baseline_set))
    return {
        "metric": "holdout_incident_pairs_upper_bound",
        "cost_pairs_dir": str(cost_pairs_dir),
        "total_cost_universe_pairs": int(total_pairs),
        "selected_holdout_molecules": len(no_train_molecules),
        "selected_incident_cost_upper_bound": selected_cost,
        "baseline_top_degree_incident_cost_upper_bound": baseline_cost,
        "projected_train_pairs_kept_lower_bound": int(max(0, total_pairs - selected_cost)),
        "baseline_projected_train_pairs_kept_lower_bound": int(max(0, total_pairs - baseline_cost)),
    }


def select_eval_rows_v2(
    args: argparse.Namespace, molecule_index: v1.MoleculeIndex
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools, source_strata, allocation = v1.collect_eval_pools(args)
    support_graph = v1.load_train_support_graph(args, molecule_index)
    support_counts = {molecule: sum(neighbors.values()) for molecule, neighbors in support_graph.items()}
    removed_train_pairs: set[tuple[int, int]] = set()
    removed_molecule_pairs: Counter[tuple[int, int]] = Counter()
    cost_pairs_dir = args.holdout_cost_pairs or args.no_constraints_pairs
    cost, total_pairs = compute_molecule_costs(args, molecule_index, pairs_dir=cost_pairs_dir, label="holdout_cost")

    def cost_sum(row: dict[str, Any]) -> int:
        left, right = molecule_index.pair_molecules(pair_key(row["row_index_a"], row["row_index_b"]))
        return int(cost[left]) + int(cost[right])

    for split in EVAL_SPLITS:
        for rows in pools[split]["both_seen"].values():
            rows.sort(key=lambda row: (cost_sum(row), row_priority(args.seed, split, "both_seen", row)))
    no_overlap_fillers = {
        split: NoOverlapFiller(pools[split]["no_overlap"], molecule_index, cost, args.seed, split)
        for split in EVAL_SPLITS
    }
    a_seen_fillers = {
        split: ASeenFiller(pools[split]["a_seen_only"], molecule_index, cost, args.seed, split)
        for split in EVAL_SPLITS
    }

    selected: list[dict[str, Any]] = []
    selected_pairs: set[tuple[int, int]] = set()
    no_train_molecules: set[int] = set()
    train_required_molecules: set[int] = set()
    eval_molecules: set[int] = set()
    eval_molecules_by_split: dict[str, set[int]] = {split: set() for split in EVAL_SPLITS}
    reuse_stats = {"no_overlap_reuse": 0, "no_overlap_open": 0, "a_seen_reuse": 0, "a_seen_open": 0}
    stats: dict[str, Any] = {
        "target_directions_per_eval_split_subset": args.eval_directions_per_subset,
        "molecule_identity": molecule_index.stats,
        "selection_policy": {
            "holdout_strategy": HOLDOUT_STRATEGY,
            "holdout_method": "marginal_cost_reuse_greedy",
            "stratification_source": (
                "condition-key shared-compatible compact pair universe before validation/test molecule removal"
            ),
            "stratification_fields": ["transfer_label", "similarity_bucket", *args.pair_metadata_columns],
            "stratum_allocation": "largest_remainder_proportional_by_stratum",
            "underfilled_quota_policy": "leave_underfilled_no_backfill",
            "disjoint_eval_molecules": bool(args.disjoint_eval_molecules),
            "holdout_cost_metric": "incident_pairs_in_cost_universe",
            "holdout_cost_pairs_dir": str(cost_pairs_dir),
        },
        "source_strata": serializable_counter(source_strata, args.pair_metadata_columns),
        "allocation_per_eval_split_subset": serializable_counter(Counter(allocation), args.pair_metadata_columns),
        "selected_counts": {split: Counter() for split in EVAL_SPLITS},
        "pool_sizes": {
            split: {
                subset: int(sum(len(rows) for rows in stratum_rows.values()))
                for subset, stratum_rows in subset_rows.items()
            }
            for split, subset_rows in pools.items()
        },
        "pool_strata": {
            split: {
                subset: int(sum(1 for rows in stratum_rows.values() if rows))
                for subset, stratum_rows in subset_rows.items()
            }
            for split, subset_rows in pools.items()
        },
        "reuse": reuse_stats,
        "skipped": Counter(),
        "selected_strata": {
            split: {subset: Counter() for subset in EVAL_SUBSETS}
            for split in EVAL_SPLITS
        },
        "unfilled_allocation": {
            split: {subset: Counter(allocation) for subset in EVAL_SUBSETS}
            for split in EVAL_SPLITS
        },
    }

    def pair_removed(pair: tuple[int, int]) -> bool:
        return pair in removed_train_pairs

    def removed_count_between(left: int, right: int) -> int:
        return int(removed_molecule_pairs[molecule_pair_key(left, right)])

    def support_between(left: int, right: int) -> int:
        return max(0, int(support_graph.get(left, {}).get(right, 0)) - removed_count_between(left, right))

    def support_after_pair_removal(molecule: int, pair: tuple[int, int]) -> int:
        value = support_counts.get(molecule, 0)
        pair_molecules = molecule_index.pair_molecules(pair)
        if molecule in pair_molecules and not pair_removed(pair):
            other = pair_molecules[1] if pair_molecules[0] == molecule else pair_molecules[0]
            if other not in no_train_molecules:
                value -= 1
        return value

    def no_train_decrements(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> Counter[int]:
        decrements: Counter[int] = Counter()
        additions = {int(molecule) for molecule in molecules}
        extra_pair_molecules = molecule_index.pair_molecules(extra_removed_pair) if extra_removed_pair else None
        for molecule in additions:
            if molecule in no_train_molecules:
                continue
            for neighbor, count in support_graph.get(molecule, {}).items():
                decrement = int(count) - removed_count_between(molecule, neighbor)
                if extra_pair_molecules and molecule in extra_pair_molecules and neighbor in extra_pair_molecules:
                    decrement = max(0, decrement - 1)
                if decrement <= 0:
                    continue
                if neighbor in no_train_molecules or neighbor in additions:
                    continue
                decrements[neighbor] += decrement
        return decrements

    def can_mark_no_train(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> bool:
        decrements = no_train_decrements(molecules, extra_removed_pair=extra_removed_pair)
        for molecule, decrement in decrements.items():
            if molecule in train_required_molecules and support_counts.get(molecule, 0) - decrement <= 0:
                return False
        return True

    def remove_train_pair(pair: tuple[int, int]) -> None:
        if pair_removed(pair):
            return
        removed_train_pairs.add(pair)
        left, right = molecule_index.pair_molecules(pair)
        removed_molecule_pairs[molecule_pair_key(left, right)] += 1
        if left not in no_train_molecules:
            support_counts[left] = support_counts.get(left, 0) - 1
        if right != left and right not in no_train_molecules:
            support_counts[right] = support_counts.get(right, 0) - 1

    def mark_no_train(molecules: Iterable[int], *, extra_removed_pair: tuple[int, int] | None = None) -> None:
        decrements = no_train_decrements(molecules, extra_removed_pair=extra_removed_pair)
        for molecule, decrement in decrements.items():
            support_counts[molecule] = support_counts.get(molecule, 0) - decrement
        for molecule in molecules:
            no_train_molecules.add(int(molecule))
            support_counts[int(molecule)] = 0

    def disjoint_conflict(split: str, molecules: set[int]) -> bool:
        if not args.disjoint_eval_molecules:
            return False
        other = eval_molecules_by_split["test" if split == "validation" else "validation"]
        return bool(molecules & other)

    def record_selection(row, pk, split, subset, direction, stratum) -> None:
        left_record, right_record = pk
        out = dict(row)
        out["split"] = split
        out["eval_subset"] = subset
        out["direction"] = direction
        out["molecule_id_a"] = int(molecule_index.molecule_id(left_record))
        out["molecule_id_b"] = int(molecule_index.molecule_id(right_record))
        selected.append(out)
        selected_pairs.add(pk)
        stats["selected_counts"][split][subset] += 1
        stats["selected_strata"][split][subset][stratum] += 1

    def finalize_unfilled(split, subset, remaining, exhausted) -> None:
        unfilled = Counter({stratum: count for stratum, count in remaining.items() if count > 0})
        unfilled.update(exhausted)
        stats["unfilled_allocation"][split][subset] = unfilled
        if stats["selected_counts"][split][subset] < args.eval_directions_per_subset:
            print(
                f"[{utc_now()}] WARNING: underfilled {split}/{subset}: "
                f"selected={stats['selected_counts'][split][subset]:,} "
                f"target={args.eval_directions_per_subset:,} "
                f"unfilled={sum(unfilled.values()):,}",
                file=sys.stderr,
                flush=True,
            )

    def fill_no_overlap(split: str) -> None:
        filler = no_overlap_fillers[split]
        remaining = Counter(allocation)
        exhausted: Counter[tuple[Any, ...]] = Counter()
        step = 0
        while remaining:
            stratum = choose_stratum_filler(remaining, allocation, filler.available, args.seed, split, "no_overlap", step)
            if stratum is None:
                stats["skipped"]["no_selectable_remaining_stratum"] += 1
                break
            picked = False
            while True:
                pk = filler.next(stratum, selected_pairs)
                if pk is None:
                    break
                left_record, right_record = pk
                left, right = molecule_index.pair_molecules(pk)
                if disjoint_conflict(split, {left, right}):
                    stats["skipped"]["cross_eval_split_molecule_conflict"] += 1
                    continue
                if left in train_required_molecules or right in train_required_molecules:
                    stats["skipped"]["no_overlap_train_required_conflict"] += 1
                    continue
                if not can_mark_no_train((left, right), extra_removed_pair=pk):
                    stats["skipped"]["no_overlap_train_support_conflict"] += 1
                    continue
                left_new = left not in no_train_molecules
                right_new = right not in no_train_molecules
                if left_new or right_new:
                    reuse_stats["no_overlap_open"] += 1
                else:
                    reuse_stats["no_overlap_reuse"] += 1
                direction = selected_direction(args.seed, split, "no_overlap", pk)
                remove_train_pair(pk)
                mark_no_train((left, right), extra_removed_pair=pk)
                eval_molecules.update((left, right))
                eval_molecules_by_split[split].update((left, right))
                # Only rescan molecules that were genuinely newly held out, so
                # high-reuse molecules are not re-promoted O(degree) times each.
                if left_new:
                    filler.on_added(left, no_train_molecules)
                if right_new:
                    filler.on_added(right, no_train_molecules)
                record_selection(filler.row(pk), pk, split, "no_overlap", direction, stratum)
                remaining[stratum] -= 1
                if remaining[stratum] <= 0:
                    del remaining[stratum]
                picked = True
                break
            if not picked:
                stats["skipped"]["stratum_exhausted"] += 1
                exhausted[stratum] += remaining[stratum]
                del remaining[stratum]
            step += 1
        finalize_unfilled(split, "no_overlap", remaining, exhausted)

    def fill_a_seen(split: str) -> None:
        filler = a_seen_fillers[split]
        filler.seed_ready(no_train_molecules)
        remaining = Counter(allocation)
        exhausted: Counter[tuple[Any, ...]] = Counter()
        step = 0
        while remaining:
            stratum = choose_stratum_filler(remaining, allocation, filler.available, args.seed, split, "a_seen_only", step)
            if stratum is None:
                stats["skipped"]["no_selectable_remaining_stratum"] += 1
                break
            picked = False
            while True:
                pk = filler.next(stratum, selected_pairs)
                if pk is None:
                    break
                _stratum, row, source, target, direction = filler.info(pk)
                if disjoint_conflict(split, {source, target}):
                    stats["skipped"]["cross_eval_split_molecule_conflict"] += 1
                    continue
                if source == target:
                    stats["skipped"]["a_seen_only_same_molecule_conflict"] += 1
                    continue
                if source in no_train_molecules or target in train_required_molecules:
                    stats["skipped"]["a_seen_only_seen_status_conflict"] += 1
                    continue
                if support_counts.get(source, 0) - support_between(source, target) <= 0:
                    stats["skipped"]["a_seen_only_source_without_train_support"] += 1
                    continue
                if not can_mark_no_train((target,)):
                    stats["skipped"]["a_seen_only_target_train_support_conflict"] += 1
                    continue
                target_new = target not in no_train_molecules
                if target_new:
                    reuse_stats["a_seen_open"] += 1
                else:
                    reuse_stats["a_seen_reuse"] += 1
                remove_train_pair(pk)
                train_required_molecules.add(source)
                mark_no_train((target,))
                eval_molecules.update((source, target))
                eval_molecules_by_split[split].update((source, target))
                if target_new:
                    filler.on_added(target)
                record_selection(row, pk, split, "a_seen_only", direction, stratum)
                remaining[stratum] -= 1
                if remaining[stratum] <= 0:
                    del remaining[stratum]
                picked = True
                break
            if not picked:
                stats["skipped"]["stratum_exhausted"] += 1
                exhausted[stratum] += remaining[stratum]
                del remaining[stratum]
            step += 1
        finalize_unfilled(split, "a_seen_only", remaining, exhausted)

    def fill_both_seen(split: str) -> None:
        rows_by_stratum = pools[split]["both_seen"]
        remaining = Counter(allocation)
        exhausted: Counter[tuple[Any, ...]] = Counter()
        pointers = {stratum: 0 for stratum in rows_by_stratum}
        step = 0
        while remaining:
            stratum = choose_stratum(remaining, allocation, rows_by_stratum, pointers, args.seed, split, "both_seen", step)
            if stratum is None:
                stats["skipped"]["no_selectable_remaining_stratum"] += 1
                break
            rows = rows_by_stratum.get(stratum) or []
            picked = False
            while pointers[stratum] < len(rows):
                row = rows[pointers[stratum]]
                pointers[stratum] += 1
                pk = pair_key(row["row_index_a"], row["row_index_b"])
                if pk in selected_pairs:
                    stats["skipped"]["already_selected_pair"] += 1
                    continue
                left, right = molecule_index.pair_molecules(pk)
                if disjoint_conflict(split, {left, right}):
                    stats["skipped"]["cross_eval_split_molecule_conflict"] += 1
                    continue
                if left in no_train_molecules or right in no_train_molecules:
                    stats["skipped"]["both_seen_no_train_molecule_conflict"] += 1
                    continue
                if support_after_pair_removal(left, pk) <= 0 or support_after_pair_removal(right, pk) <= 0:
                    stats["skipped"]["both_seen_without_train_support"] += 1
                    continue
                direction = selected_direction(args.seed, split, "both_seen", pk)
                remove_train_pair(pk)
                train_required_molecules.update((left, right))
                eval_molecules.update((left, right))
                eval_molecules_by_split[split].update((left, right))
                record_selection(row, pk, split, "both_seen", direction, stratum)
                remaining[stratum] -= 1
                if remaining[stratum] <= 0:
                    del remaining[stratum]
                picked = True
                break
            if not picked:
                stats["skipped"]["stratum_exhausted"] += 1
                exhausted[stratum] += remaining[stratum]
                del remaining[stratum]
            step += 1
        finalize_unfilled(split, "both_seen", remaining, exhausted)

    selection_order = [
        (split, subset)
        for subset in ("no_overlap", "a_seen_only", "both_seen")
        for split in EVAL_SPLITS
    ]
    stats["selection_policy"]["selection_order"] = [
        {"split": split, "subset": subset} for split, subset in selection_order
    ]
    fillers = {"no_overlap": fill_no_overlap, "a_seen_only": fill_a_seen, "both_seen": fill_both_seen}
    for split, subset in selection_order:
        fillers[subset](split)

    stats["selected_counts"] = {
        split: dict(sorted(counter.items())) for split, counter in stats["selected_counts"].items()
    }
    stats["selected_strata"] = {
        split: {
            subset: serializable_counter(counter, args.pair_metadata_columns)
            for subset, counter in subset_counters.items()
        }
        for split, subset_counters in stats["selected_strata"].items()
    }
    stats["unfilled_allocation"] = {
        split: {
            subset: serializable_counter(counter, args.pair_metadata_columns)
            for subset, counter in subset_counters.items()
        }
        for split, subset_counters in stats["unfilled_allocation"].items()
    }
    stats["selected_pairs"] = len(selected_pairs)
    stats["selected_directions"] = len(selected)
    stats["no_train_molecules"] = len(no_train_molecules)
    stats["train_required_molecules"] = len(train_required_molecules)
    stats["eval_molecules"] = len(eval_molecules)
    stats["eval_molecules_by_split"] = {
        split: len(molecules) for split, molecules in sorted(eval_molecules_by_split.items())
    }
    stats["holdout_cost_report"] = cost_report(
        cost, total_pairs, no_train_molecules, pools, molecule_index, cost_pairs_dir=cost_pairs_dir
    )
    stats["skipped"] = dict(sorted(stats["skipped"].items()))
    return selected, stats


def write_eval_selection_v2(
    selected_rows: list[dict[str, Any]],
    output_root: Path,
    stats: dict[str, Any],
    train_direction_mode: str,
) -> None:
    stratification_fields = stats["selection_policy"]["stratification_fields"][2:]
    payload = {
        "schema_version": f"{V2_SPLIT_VERSION}_selection",
        "created_at_utc": utc_now(),
        "selection_stats": stats,
        "rows": [
            {
                "row_index_a": int(row["row_index_a"]),
                "row_index_b": int(row["row_index_b"]),
                "molecule_id_a": int(row["molecule_id_a"]),
                "molecule_id_b": int(row["molecule_id_b"]),
                "split": row["split"],
                "eval_subset": row["eval_subset"],
                "direction": row["direction"],
                "transfer_label": int(row["transfer_label"]),
                "value_difference": float(row["value_difference"]),
                "weighted_tanimoto": float(row["weighted_tanimoto"]),
                "similarity_bucket": int(row["similarity_bucket"]),
                "stratum": stratum_to_string(stratum_for_row(row, stratification_fields), stratification_fields),
            }
            for row in selected_rows
        ],
    }
    write_json(output_root / f"{mode_metadata_stem_v2(train_direction_mode)}_selection.json", payload)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.arrow_cpu_count:
        pa.set_cpu_count(args.arrow_cpu_count)
        pa.set_io_thread_count(args.arrow_cpu_count)
    # Tag materialized rows/metadata as v2 (process-local override; v1 untouched).
    v1.SPLIT_VERSION = V2_SPLIT_VERSION
    universes = v1.pair_universes(args)
    outputs = {
        key: value
        for key, value in v1.split_output_dirs(args.output_root, args.train_direction_mode, args.output_name_suffix).items()
        if key in universes
    }
    v1.prepare_outputs(outputs, args.overwrite, args.splits)
    base_table = v1.load_base_table(args)
    molecule_index = v1.build_molecule_index(base_table, args.smiles_column)
    selected_rows, selection_stats = select_eval_rows_v2(args, molecule_index)

    universe_metadata: dict[str, Any] = {}
    staged_outputs = v1.make_staged_output_dirs(outputs)
    installed = False
    try:
        for universe, pair_dir in universes.items():
            universe_metadata[universe] = v1.write_universe(
                universe=universe,
                pair_dir=pair_dir,
                output_dir=staged_outputs[universe],
                public_output_dir=outputs[universe],
                selected_rows=selected_rows,
                base_table=base_table,
                molecule_index=molecule_index,
                args=args,
            )
        v1.install_staged_output_dirs(staged_outputs, outputs, args.overwrite)
        installed = True
    finally:
        if not installed:
            v1.cleanup_staged_output_dirs(staged_outputs.values())

    write_eval_selection_v2(selected_rows, args.output_root, selection_stats, args.train_direction_mode)

    metadata = {
        "schema_version": V2_SPLIT_VERSION,
        "holdout_strategy": HOLDOUT_STRATEGY,
        "created_at_utc": utc_now(),
        "base_input": args.base_input,
        "condition_key_pair_dir": str(args.condition_key_pairs),
        "same_species_pair_dir": str(args.same_species_pairs),
        "no_constraints_pair_dir": str(args.no_constraints_pairs),
        "train_direction_mode": args.train_direction_mode,
        "shared_eval_compatibility_column": args.shared_eval_compatibility_column,
        "molecule_identity": molecule_index.stats,
        "shared_eval_compatibility_policy": (
            "validation/test candidates are selected from condition-key pairs only when both rows have "
            "identical non-null values in this column, guaranteeing presence in same_species_v2; "
            "no_constraints compatibility is automatic"
        ),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "selection": selection_stats,
        "universe_rows_by_split": {
            key: value["rows_by_split"] for key, value in universe_metadata.items()
        },
        "universe_validation": {
            key: value["validation"] for key, value in universe_metadata.items()
        },
    }
    write_json(args.output_root / f"{mode_metadata_stem_v2(args.train_direction_mode)}_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=v1.DEFAULT_BASE_INPUT)
    parser.add_argument("--condition-key-pairs", type=Path, default=v1.DEFAULT_CONDITION_KEY_PAIRS)
    parser.add_argument("--same-species-pairs", type=Path, default=v1.DEFAULT_SAME_SPECIES_PAIRS)
    parser.add_argument("--no-constraints-pairs", type=Path, default=v1.DEFAULT_NO_CONSTRAINTS_PAIRS)
    parser.add_argument(
        "--universes",
        nargs="+",
        choices=("condition_key", "same_species_v2", "no_constraints"),
        default=["condition_key", "same_species_v2", "no_constraints"],
        help="Pair universes to materialize. Selection is still based on condition_key.",
    )
    parser.add_argument("--output-root", type=Path, default=v1.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name-suffix", default="")
    parser.add_argument("--splits", nargs="+", choices=v1.SPLITS, default=list(v1.SPLITS))
    parser.add_argument(
        "--train-direction-mode",
        choices=v1.TRAIN_DIRECTION_MODES,
        default="bidirectional",
        help="Whether train writes both pair directions or one deterministic direction per source_pair_id.",
    )
    parser.add_argument("--metadata-columns", nargs="+", default=v1.DEFAULT_METADATA_COLUMNS)
    parser.add_argument("--shared-eval-compatibility-column", default=v1.DEFAULT_SHARED_COMPATIBILITY_COLUMN)
    parser.add_argument(
        "--pair-metadata-columns",
        nargs="+",
        default=[
            "bioavailability_report_type",
            "species_or_population",
            "dose",
            "oral_exposure_mode",
            "qualifying_conditions",
            "comparator",
            "extra_details",
        ],
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--value-column", default="oral_bioavailability_value")
    parser.add_argument("--eval-directions-per-subset", type=int, default=10_000)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=25)
    parser.add_argument(
        "--holdout-cost-pairs",
        type=Path,
        default=None,
        help="Pair universe whose per-molecule incident degree defines holdout cost (default: no_constraints).",
    )
    parser.add_argument("--holdout-cost-weight", type=float, default=1.0, help=argparse.SUPPRESS)
    # Deprecated/no-op (kept for wrapper/runner compatibility).
    parser.add_argument("--community-method", choices=("louvain", "label_propagation"), default="louvain", help=argparse.SUPPRESS)
    parser.add_argument("--community-resolution", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--holdout-supply-multiplier", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--parquet-file-row-limit", type=int, default=10_000_000)
    parser.add_argument("--progress-every-seconds", type=float, default=300.0)
    parser.add_argument("--arrow-cpu-count", type=int, default=None)
    parser.add_argument(
        "--disjoint-eval-molecules",
        action="store_true",
        help="Require validation and test eval pairs to use disjoint RDKit-canonical molecule groups.",
    )
    parser.add_argument("--transfer-threshold", type=float, default=10.0)
    parser.add_argument("--not-transfer-threshold", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.eval_directions_per_subset < 1:
        parser.error("--eval-directions-per-subset must be positive")
    if args.candidate_pool_multiplier < 1:
        parser.error("--candidate-pool-multiplier must be positive")
    if args.holdout_cost_weight < 0:
        parser.error("--holdout-cost-weight cannot be negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.parquet_file_row_limit < 1:
        parser.error("--parquet-file-row-limit must be positive")
    if args.progress_every_seconds < 0:
        parser.error("--progress-every-seconds cannot be negative")
    if args.arrow_cpu_count is not None and args.arrow_cpu_count < 1:
        parser.error("--arrow-cpu-count must be positive")
    return args


def main() -> None:
    metadata = build(parse_args())
    print(
        json.dumps(
            {
                "outputs": metadata["outputs"],
                "selection": metadata["selection"],
                "universe_rows_by_split": metadata["universe_rows_by_split"],
                "universe_validation": metadata["universe_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
