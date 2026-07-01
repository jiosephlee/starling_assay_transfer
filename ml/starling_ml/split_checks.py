"""Package entrypoint for shared-eval split integrity checks."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("both_seen_overlap", "checks"), default="checks")
    args = parser.parse_args()
    if args.phase in {"both_seen_overlap", "checks"}:
        script = Path(__file__).resolve().parents[1] / "scripts" / "check_both_seen_train_overlap.py"
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
