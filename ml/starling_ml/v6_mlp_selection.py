"""Validation-ranking architecture selection for the V6.5 100M ablation."""

from __future__ import annotations

import json
from pathlib import Path


MODES = ("concat", "difference", "difference_product")


def _run_metrics(path: Path, step: int) -> dict:
    completed = json.loads((path / "completed.json").read_text())
    if completed["step"] != step:
        raise ValueError(f"expected run to end at step {step}: {path}")
    return {"ndcg_at_5": completed["best_ndcg_at_5"],
            "ndcg_at_10": completed["best_ndcg_at_10"],
            "best_step": completed["best_step"]}


def select_fusion(runs: Path, output: Path, step: int = 1000) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {output}")
    metrics = {mode: _run_metrics(runs / mode, step) for mode in MODES}
    order = {mode: index for index, mode in enumerate(MODES)}
    winner = min(MODES, key=lambda mode: (-metrics[mode]["ndcg_at_5"],
                                          -metrics[mode]["ndcg_at_10"], order[mode]))
    result = {"version": "v6_5_mlp_100m_fusion_selection_v2", "selection_step": step,
              "selection_split": "validation_ranking_fixed_100_queries",
              "primary_metric": "ndcg_at_5",
              "tie_breakers": ["ndcg_at_10", "fixed_mode_order"],
              "winner": winner, "metrics": metrics,
              "resume_checkpoint": str(runs / winner / "last.pt"),
              "selected_checkpoint": str(runs / winner / "best.pt"),
              "test_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
