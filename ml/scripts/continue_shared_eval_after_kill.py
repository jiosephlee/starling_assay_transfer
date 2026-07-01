#!/usr/bin/env python3
"""Continue the 3000-step shared-eval finals after skipping one killed run."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import run_shared_eval_benchmark as bench


def is_alive(pid: int) -> bool:
    return subprocess.run(
        ["ps", "-p", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def wait_for_pids(pids: list[int]) -> None:
    while any(is_alive(pid) for pid in pids):
        time.sleep(30)


def run_uploads(
    python: str,
    final_run_suffix: str,
    *,
    public_hf: bool,
    skip_no_source_condition_key: bool,
) -> None:
    for lane in bench.LANES:
        winner_by_universe = {row["universe"]: row for row in bench.read_winners(lane)}
        for universe in bench.UNIVERSES:
            if skip_no_source_condition_key and lane.key == "no_source_value" and universe.key == "condition_key":
                print("[skip] upload no_source_value/condition_key", flush=True)
                continue
            run_name = bench.final_run_name(lane, universe, winner_by_universe[universe.key], final_run_suffix)
            run_dir = lane.final_root / run_name
            if not (run_dir / "best" / "best_metric.json").exists():
                raise FileNotFoundError(f"missing best checkpoint metadata: {run_dir / 'best' / 'best_metric.json'}")
            bench.run_upload(lane, universe, run_dir, python, public_hf=public_hf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--final-run-suffix", required=True)
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--public-hf", action="store_true")
    parser.add_argument("--skip-no-source-condition-key", action="store_true")
    args = parser.parse_args()

    no_source = next(lane for lane in bench.LANES if lane.key == "no_source_value")
    winners = {row["universe"]: row for row in bench.read_winners(no_source)}

    for universe in bench.UNIVERSES[1:]:
        print(f"[run] no_source_value/{universe.key}", flush=True)
        bench.run_final(
            no_source,
            universe,
            winners[universe.key],
            args.python,
            final_run_suffix=args.final_run_suffix,
            rebuild_memmap=False,
            upload_hf=False,
            public_hf=args.public_hf,
        )

    if args.wait_pid:
        print(f"[wait] pids {args.wait_pid}", flush=True)
        wait_for_pids(args.wait_pid)

    print("[upload] completed finals", flush=True)
    run_uploads(
        args.python,
        args.final_run_suffix,
        public_hf=args.public_hf,
        skip_no_source_condition_key=args.skip_no_source_condition_key,
    )
    print("[done] continuation complete", flush=True)


if __name__ == "__main__":
    main()
