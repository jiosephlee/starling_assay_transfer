import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from starling_ml.train_v6_mlp_100m import _checkpoint, _optimizer, _resume
from starling_ml.v6_embedding_cache import (build_embedding_cache, reindex_embedding_cache,
                                             validate_embedding_cache)
from starling_ml.v6_mlp_100m import FUSION_SPECS, SwiGLUBlock, V6FusionMLP100M
from starling_ml.v6_mlp_100m_data import (build_group_schedule, build_pair_cache,
                                          ranking_subset_indices, validate_group_schedule)
from starling_ml.v6_mlp_metrics import ordinary_metrics, ranking_metrics, wandb_metrics
from starling_ml.v6_mlp_selection import select_fusion


def test_fusion_parameter_contracts_and_depth():
    for mode, spec in FUSION_SPECS.items():
        model = V6FusionMLP100M(mode)
        assert len(model.blocks) == 8
        assert sum(parameter.numel() for parameter in model.parameters()) == spec.parameters
        del model


def test_swiglu_block_has_finite_gradients():
    block = SwiGLUBlock(16, 32, 0.0)
    values = torch.randn(3, 16, requires_grad=True)
    block(values).square().mean().backward()
    assert values.grad is not None
    assert all(parameter.grad is not None for parameter in block.parameters())


def _fake_records(path):
    rows = [{"record_index": 0, "record_id": "r0", "molecule_index": 0, "canonical_smiles": "C",
             "assay_paragraph": "assay one", "assay_concept": "Fa"},
            {"record_index": 1, "record_id": "r1", "molecule_index": 1, "canonical_smiles": "CC",
             "assay_paragraph": "assay two", "assay_concept": "Fg"},
            {"record_index": 2, "record_id": "r2", "molecule_index": 0, "canonical_smiles": "C",
             "assay_paragraph": "assay three", "assay_concept": "Fh"}]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_embedding_cache_is_offline_indexed_and_hash_validated(tmp_path):
    records, output = tmp_path / "records.parquet", tmp_path / "cache"
    _fake_records(records)

    def molecules(values, device, budget):
        return np.ones((len(values), 768), np.float16), np.array([3, 4], np.uint16)

    def assays(values, device, batch):
        return np.full((len(values), 768), 2, np.float16), np.array([5, 6, 7], np.uint16)

    manifest = build_embedding_cache(records, output, "cpu", molecules, assays)
    assert manifest["molecules"] == 2
    assert np.load(output / "record_molecule_index.npy").tolist() == [0, 1, 0]
    assert validate_embedding_cache(output, records)["records"] == 3
    with (output / "molecule_token_lengths.npy").open("ab") as handle:
        handle.write(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_embedding_cache(output)


def test_embedding_cache_reindex_reuses_only_identical_records(tmp_path):
    parent_records, new_records = tmp_path / "parent.parquet", tmp_path / "new.parquet"
    parent, output = tmp_path / "parent_cache", tmp_path / "new_cache"
    _fake_records(parent_records)
    table = pq.read_table(parent_records)
    pq.write_table(table.take(pa.array([0, 1])), new_records)
    rewritten = pq.read_table(new_records).set_column(
        0, "record_index", pa.array([0, 1], type=pa.int64()))
    pq.write_table(rewritten, new_records)

    def molecules(values, device, budget):
        values = np.arange(len(values) * 768, dtype=np.float16).reshape(len(values), 768)
        return values, np.array([3, 4], np.uint16)

    def assays(values, device, batch):
        values = np.arange(len(values) * 768, dtype=np.float16).reshape(len(values), 768)
        return values, np.array([5, 6, 7], np.uint16)

    build_embedding_cache(parent_records, parent, "cpu", molecules, assays)
    manifest = reindex_embedding_cache(parent, parent_records, new_records, output)
    assert manifest["reuse"]["removed_records"] == 1
    assert np.array_equal(np.load(output / "assay_embeddings.npy"),
                          np.load(parent / "assay_embeddings.npy")[[0, 1]])
    assert validate_embedding_cache(output, new_records)["records"] == 2


def _pair_rows(rows: int, ranking: bool = False) -> list[dict]:
    output = []
    for index in range(rows):
        row = {"query_record_index": index % 2, "retrieval_record_index": (index + 1) % 2,
               "retrieval_value": 0.5, "target_z": float(index), "target_a": 0.5,
               "group_index": index // 20 if ranking else index // 40,
               "member_index": index % (20 if ranking else 40),
               "retrieval_is_approximate": True}
        if ranking:
            row["ranking_query_id"] = f"query-{index // 20:04d}"
        output.append(row)
    return output


def test_pair_only_cache_is_bound_and_excludes_approximate_flag(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "pair_cache"
    dataset.mkdir()
    pq.write_table(pa.Table.from_pylist([{"record_index": 0}, {"record_index": 1}]),
                   dataset / "records.parquet")
    (dataset / "manifest.json").write_text("{}")
    for split in ("train", "validation", "test"):
        pq.write_table(pa.Table.from_pylist(_pair_rows(40)), dataset / f"{split}.parquet")
    for split in ("validation_ranking", "test_ranking"):
        pq.write_table(pa.Table.from_pylist(_pair_rows(40, True)), dataset / f"{split}.parquet")
    manifest = build_pair_cache(dataset, output)
    assert manifest["splits"]["train"]["rows"] == 40
    assert not list(output.rglob("*approximate*"))
    assert np.load(output / "validation_ranking" / "ranking_query_id.npy").shape == (40,)


def test_ranking_subset_is_complete_balanced_and_deterministic():
    group_ids = np.repeat(np.arange(10), 20)
    concepts = np.repeat(np.array([0] * 2 + [1] * 3 + [2] * 5, dtype=np.uint8), 20)
    first = ranking_subset_indices(group_ids, concepts, maximum=6, seed=42)
    second = ranking_subset_indices(group_ids, concepts, maximum=6, seed=42)
    assert np.array_equal(first, second)
    assert len(first) == 120
    assert len(np.unique(group_ids[first])) == 6
    assert set(np.unique(concepts[first])) == {0, 1, 2}


def test_ordinary_metrics_and_wandb_namespaces():
    probability = np.array([0.9, 0.8, 0.1, 0.4])
    target = np.array([0.9, 0.1, 0.1, 0.9])
    concepts = np.array([0, 0, 1, 1], dtype=np.uint8)
    metrics = ordinary_metrics(probability, target, concepts)
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["transfer_precision"] == 0.5
    assert metrics["overall"]["soft_mae"] == pytest.approx(0.3)
    flat = wandb_metrics("validation", metrics)
    assert "eval/validation/overall/binary_macro_f1" in flat
    assert "eval/validation/assay_concept/Fa/binary_accuracy" in flat
    assert flat["eval/validation/overall/binary_parse_rate"] == 1.0


def test_ranking_metrics_use_linear_gain_and_maximum_tie_sets():
    target_z = np.concatenate(([2.0, 2.0], np.zeros(18), [3.0], np.ones(19)))
    target_a = np.concatenate(([0.9, 0.9], np.full(18, 0.1), [0.9], np.full(19, 0.2)))
    scores = np.concatenate(([0.5, 0.4], np.linspace(0.3, 0.0, 18),
                             [0.5, 0.9], np.linspace(0.8, 0.0, 18)))
    groups = np.repeat([0, 1], 20)
    concepts = np.repeat(np.array([0, 1], dtype=np.uint8), 20)
    metrics = ranking_metrics(target_z, target_a, scores, groups, concepts)
    assert metrics["overall"]["query_count"] == 2
    assert metrics["assay_concept"]["Fa"]["top1_hit"] == 1.0
    assert metrics["assay_concept"]["Fg"]["top1_hit"] == 0.0
    assert metrics["assay_concept"]["Fg"]["best_in_top10_hit"] == 1.0
    assert "eval/ranking_validation/overall/ndcg_at_5" in wandb_metrics(
        "ranking_validation", metrics)


def test_group_schedule_is_stable_and_validated(tmp_path):
    pairs, train = tmp_path / "pairs", tmp_path / "pairs" / "train"
    train.mkdir(parents=True)
    np.save(train / "target_a.npy", np.zeros(400, dtype=np.float32))
    (pairs / "manifest.json").write_text(json.dumps({
        "version": "v6_5_mlp_100m_pair_cache_v2", "sha256": {},
        "splits": {"train": {"source_sha256": "source"}}}))
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    build_group_schedule(pairs, first, steps=5, batch_groups=3, group_size=40)
    build_group_schedule(pairs, second, steps=5, batch_groups=3, group_size=40)
    left, manifest = validate_group_schedule(first, expected_steps=5)
    right, _ = validate_group_schedule(second, expected_steps=5)
    assert np.array_equal(left, right)
    assert manifest["seed"] == 4878


def test_one_pass_schedule_is_unique_and_drops_only_tail_groups(tmp_path):
    pairs, train = tmp_path / "pairs", tmp_path / "pairs" / "train"
    train.mkdir(parents=True)
    np.save(train / "target_a.npy", np.zeros(440, dtype=np.float32))
    (pairs / "manifest.json").write_text(json.dumps({
        "version": "v6_5_mlp_100m_pair_cache_v2", "sha256": {},
        "splits": {"train": {"source_sha256": "source"}}}))
    path = tmp_path / "one_pass.npy"
    manifest = build_group_schedule(pairs, path, steps=2, batch_groups=5,
                                    group_size=40, sampling_mode="one_pass")
    schedule, _ = validate_group_schedule(path, expected_steps=2, pair_cache=pairs)
    assert len(np.unique(schedule)) == 10
    assert manifest["training_groups"] == 11
    assert manifest["retained_groups"] == 10
    assert manifest["dropped_groups"] == 1
    assert manifest["sampling_mode"] == "one_pass"


def test_checkpoint_restores_optimizer_scheduler_and_rng(tmp_path):
    args = argparse.Namespace(learning_rate=1e-4, weight_decay=0.01,
                              warmup_steps=2, schedule_steps=10)
    model, restored = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer, scheduler = _optimizer(model, args)
    other_optimizer, other_scheduler = _optimizer(restored, args)
    state = {"step": 3, "best_macro_f1": 0.5}
    path = tmp_path / "checkpoint.pt"
    _checkpoint(path, model, optimizer, scheduler, state, argparse.Namespace(test=True))
    loaded = _resume(path, restored, other_optimizer, other_scheduler)
    assert loaded == state
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def test_selection_uses_exact_step_and_frozen_tie_breakers(tmp_path):
    runs = tmp_path / "runs"
    scores = {"concat": (0.7, 0.2), "difference": (0.7, 0.3),
              "difference_product": (0.6, 0.9)}
    for mode, (ndcg5, ndcg10) in scores.items():
        path = runs / mode
        path.mkdir(parents=True)
        (path / "completed.json").write_text(json.dumps({"step": 1000,
            "best_ndcg_at_5": ndcg5, "best_ndcg_at_10": ndcg10, "best_step": 800}))
    result = select_fusion(runs, tmp_path / "selection.json")
    assert result["winner"] == "difference"
    assert result["test_used"] is False
    assert result["primary_metric"] == "ndcg_at_5"


def test_new_functions_stay_within_sixty_lines():
    root = Path(__file__).parents[1]
    paths = list((root / "starling_ml").glob("v6_*100m*.py"))
    paths.extend((root / "starling_ml").glob("v6_embedding_cache.py"))
    paths.extend((root / "starling_ml").glob("train_v6_mlp_100m.py"))
    paths.extend((root / "scripts").glob("*v6_mlp_100m*.py"))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 60, (path, node.name)
