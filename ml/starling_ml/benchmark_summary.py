"""Package entrypoint for shared-eval benchmark summaries.

The implementation still lives in the historical script while v3 parity checks
are running; this module gives downstream callers a stable package import path.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_shared_eval_models_and_baselines.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
