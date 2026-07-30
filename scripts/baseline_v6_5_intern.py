#!/usr/bin/env python3
"""V6.5 frozen-benchmark baselines: majority + train-tuned Morgan-Tanimoto (binary),
Morgan-Tanimoto ranking. Writes a long-format TSV to tables/.

Similarity is ECFP4 (Morgan radius 2, 2048 bits) Tanimoto between the query and retrieval
molecules. Rationale: structurally similar molecules should share the measured value (transfer).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---- similarity ----------------------------------------------------------------
def fingerprints(smiles: set[str]) -> dict[str, object]:
    out = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if smi else None
        out[smi] = _GEN.GetFingerprint(mol) if mol is not None else None
    return out


def similarities(query, retrieval, fps) -> np.ndarray:
    sims = np.zeros(len(query), dtype=float)
    for i, (q, r) in enumerate(zip(query, retrieval)):
        fq, fr = fps.get(q), fps.get(r)
        sims[i] = DataStructs.TanimotoSimilarity(fq, fr) if (fq is not None and fr is not None) else 0.0
    return sims


# ---- binary metrics ------------------------------------------------------------
def binary_eval(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float, object]:
    """Return (accuracy, macro_f1, transfer_precision). transfer_precision is None when the
    predictor makes no transfer (class 1) predictions."""
    labels, preds = np.asarray(labels).astype(int), np.asarray(preds).astype(int)
    acc = float((labels == preds).mean())

    def prf(c: int) -> tuple[float, float]:
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r

    (p0, r0), (p1, r1) = prf(0), prf(1)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    transfer_precision = p1 if int(((preds == 1)).sum()) else None  # undefined with no positives
    return acc, 0.5 * (f1(p0, r0) + f1(p1, r1)), transfer_precision


# ---- ranking metrics (copied from starling_ml.intern_v6 to match model eval) ----
def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    lr, rr = _rankdata(left), _rankdata(right)
    if np.std(lr) == 0 or np.std(rr) == 0:
        return 0.0
    return float(np.corrcoef(lr, rr)[0, 1])


def _list_metrics(relevance: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    predicted = np.argsort(-scores, kind="stable")
    ideal = np.argsort(-relevance, kind="stable")
    discounts = 1.0 / np.log2(np.arange(2, min(k, len(relevance)) + 2))
    gains = np.exp2(relevance) - 1.0
    dcg = float(np.sum(gains[predicted[:k]] * discounts))
    ideal_dcg = float(np.sum(gains[ideal[:k]] * discounts))
    best = float(relevance[ideal[0]])
    top = predicted[:k]
    return {"ndcg_at_10": 0.0 if ideal_dcg == 0 else dcg / ideal_dcg,
            "top1_regret": best - float(relevance[predicted[0]]),
            "best_in_top10_regret": best - float(np.max(relevance[top])),
            "mean_top10_relevance": float(np.mean(relevance[top])),
            "spearman": _spearman(relevance, scores)}


def ranking_metrics(relevance, scores, group_ids, k: int = 10) -> dict[str, float]:
    rel, sc = np.asarray(relevance), np.asarray(scores)
    groups: dict[object, list[int]] = {}
    for idx, g in enumerate(group_ids):
        groups.setdefault(g, []).append(idx)
    vals = [_list_metrics(rel[rows], sc[rows], k) for rows in groups.values()]
    return {name: float(np.mean([v[name] for v in vals])) for name in vals[0]}


# ---- driver --------------------------------------------------------------------
def _load(root: Path, split: str, cols):
    return pq.read_table(root / split / "data.parquet", columns=cols).to_pydict()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("datasets/hf_parquet/assay_transfer_raw_pair_v6_5_2_intern"))
    ap.add_argument("--out", type=Path, default=Path("tables/assay_transfer_v6_5_baselines.tsv"))
    ap.add_argument("--tune-sample", type=int, default=200_000)
    args = ap.parse_args()

    # tuning set: train pairs, binary label = transfer (target_a >= 0.5)
    tr = _load(args.root, "train", ["query_smiles", "retrieval_smiles", "target_a"])
    n = len(tr["target_a"])
    idx = random.Random(4878).sample(range(n), min(args.tune_sample, n))
    tq = [tr["query_smiles"][i] for i in idx]
    trr = [tr["retrieval_smiles"][i] for i in idx]
    tlab = np.array([1 if tr["target_a"][i] >= 0.5 else 0 for i in idx])

    ordinary = {s: _load(args.root, s, ["query_smiles", "retrieval_smiles", "binary_label"])
                for s in ("validation", "test")}
    ranking = {s: _load(args.root, s, ["query_smiles", "retrieval_smiles", "target_z", "ranking_query_id"])
               for s in ("validation_ranking", "test_ranking")}

    # one fingerprint per unique SMILES across everything we touch
    smis = set(tq) | set(trr)
    for d in list(ordinary.values()) + list(ranking.values()):
        smis |= set(d["query_smiles"]) | set(d["retrieval_smiles"])
    fps = fingerprints(smis)
    unparsed = sum(1 for s in smis if fps[s] is None)
    print(f"unique SMILES: {len(smis)}  unparsed: {unparsed}")

    # tune Tanimoto threshold on train (maximize macro-F1)
    tsim = similarities(tq, trr, fps)
    grid = np.linspace(0.0, 1.0, 201)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        _, mf1, _ = binary_eval(tlab, tsim >= t)
        if mf1 > best_f1:
            best_f1, best_t = mf1, float(t)
    majority = int(round(tlab.mean()))
    print(f"tuned Tanimoto threshold={best_t:.3f} (train macro-F1={best_f1:.4f}); "
          f"train transfer-rate={tlab.mean():.3f} -> majority class={majority}")

    rows = []

    def add(baseline, eval_kind, split, n_rows, **m):
        rows.append({"baseline": baseline, "eval": eval_kind, "split": split, "n": n_rows,
                     "threshold": m.get("threshold", ""), "accuracy": m.get("accuracy", ""),
                     "macro_f1": m.get("macro_f1", ""), "transfer_precision": m.get("transfer_precision", ""),
                     "ndcg_at_10": m.get("ndcg_at_10", ""),
                     "spearman": m.get("spearman", ""), "top1_regret": m.get("top1_regret", ""),
                     "best_in_top10_regret": m.get("best_in_top10_regret", ""),
                     "mean_top10_relevance": m.get("mean_top10_relevance", "")})

    # binary evals
    for split, d in ordinary.items():
        lab = np.asarray(d["binary_label"]).astype(int)
        acc_m, f1_m, tp_m = binary_eval(lab, np.full(len(lab), majority))
        add("majority", "binary", split, len(lab), accuracy=acc_m, macro_f1=f1_m,
            transfer_precision="NA" if tp_m is None else tp_m)
        sim = similarities(d["query_smiles"], d["retrieval_smiles"], fps)
        acc_t, f1_t, tp_t = binary_eval(lab, sim >= best_t)
        add("tanimoto_train_tuned", "binary", split, len(lab), threshold=best_t, accuracy=acc_t,
            macro_f1=f1_t, transfer_precision="NA" if tp_t is None else tp_t)

    # ranking eval
    for split, d in ranking.items():
        sim = similarities(d["query_smiles"], d["retrieval_smiles"], fps)
        m = ranking_metrics(d["target_z"], sim, d["ranking_query_id"], k=10)
        add("morgan_tanimoto", "ranking", split, len(sim), **m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["baseline", "eval", "split", "n", "threshold", "accuracy", "macro_f1",
              "transfer_precision", "ndcg_at_10", "spearman", "top1_regret",
              "best_in_top10_regret", "mean_top10_relevance"]
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", args.out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
