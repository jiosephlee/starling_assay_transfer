"""Compatibility exports for record-native KNN evaluation.

Active KNN evaluation lives in ``knn_data``, ``knn_retrieval``,
``knn_scorers``, and ``knn_pipeline``. Pair-derived record builders were
removed from this module.
"""
from __future__ import annotations

from .knn_data import (
    CANONICAL_QUERY_SPLITS,
    FULL_METADATA_CONFIG,
    MISSING_FIELDS,
    canonicalize_smiles,
    dataset_manifest_path,
    load_record_queries,
    load_record_sources,
    normalize_record_split,
    read_yaml,
    record_split_path,
    resolve_path,
)
from .knn_pipeline import compute_metrics, load_starling_model, timestamped_run_dir, weighted_vote, write_csv
from .knn_retrieval import fingerprints, ranked_candidates, weighted_similarity

__all__ = [
    "CANONICAL_QUERY_SPLITS",
    "FULL_METADATA_CONFIG",
    "MISSING_FIELDS",
    "canonicalize_smiles",
    "compute_metrics",
    "dataset_manifest_path",
    "fingerprints",
    "load_record_queries",
    "load_record_sources",
    "load_starling_model",
    "normalize_record_split",
    "ranked_candidates",
    "read_yaml",
    "record_split_path",
    "resolve_path",
    "timestamped_run_dir",
    "weighted_similarity",
    "weighted_vote",
    "write_csv",
]
