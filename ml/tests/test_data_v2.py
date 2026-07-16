"""v2 data<->model contract: materialized parquet -> memmap -> dataset -> collate -> model."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

_ML = Path(__file__).resolve().parents[1]
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader

    from starling_ml import data as datamod
    from starling_ml.config import LossConfig, ModelConfig
    from starling_ml.model import TransferPairModel

    _OK = True
except Exception:  # pragma: no cover
    _OK = False

FIELDS = datamod.DEFAULT_METADATA_FIELDS if _OK else ()


def _dataset_rows():
    def row(rs, qs, y, tgt, blabel, asys):
        r = {
            "split": "train",
            "retrieved_smiles": rs,
            "query_smiles": qs,
            "retrieved_value": y,
            "continuous_target": tgt,
            "binary_label": blabel,  # 1 / 0 / None
        }
        for f in FIELDS:
            r[f] = None
        r["retrieved_assay_system"] = asys
        return r

    return [
        row("CCO", "CCCO", 50.0, 2.0, 1, "human liver microsomes"),
        row("CCO", "c1ccccc1", 50.0, 20.0, None, "human liver microsomes"),
        row("CCCO", "CCO", 55.0, 3.0, 0, "rat liver microsomes"),
    ]


@unittest.skipUnless(_OK, "torch/pyarrow not available")
class DataV2ContractTest(unittest.TestCase):
    def test_end_to_end_contract(self) -> None:
        rows = _dataset_rows()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ds_parquet = d / "dataset.parquet"
            pq.write_table(pa.Table.from_pylist(rows), ds_parquet)

            # Build the precompute-style index maps (dedup structures + metadata tuples).
            smiles = sorted({r["retrieved_smiles"] for r in rows} | {r["query_smiles"] for r in rows})
            smiles_to_row = {s: i for i, s in enumerate(smiles)}
            keys = sorted({datamod.metadata_key(r, FIELDS) for r in rows})
            meta_to_row = {k: i for i, k in enumerate(keys)}

            meta = datamod.build_split_memmap(str(ds_parquet), str(d / "mm"), "train",
                                              smiles_to_row, meta_to_row)
            self.assertEqual(meta["count"], 3)
            self.assertEqual(meta["hard_binary_count"], 2)  # one null-binary row

            ds = datamod.PairDataset(str(d / "mm"), "train")
            loader = DataLoader(ds, batch_size=3, collate_fn=datamod.collate_pairs)
            batch = next(iter(loader))
            self.assertEqual(set(batch), {"a_idx", "b_idx", "meta_a_idx", "source_value", "distance", "labels"})
            # nullable binary is NaN where absent.
            self.assertEqual(int(torch.isnan(batch["labels"]).sum()), 1)

            # Feed the real v2 model with random embedding tables sized to the maps.
            mcfg = ModelConfig(mol_hidden=16, mol_out=8, meta_field_proj=4, d_model=16, d_ff=32, n_blocks=1)
            n_smiles, n_meta = len(smiles_to_row), len(meta_to_row)
            model = TransferPairModel(
                mcfg, LossConfig(aux_binary_weight=0.5),
                np.random.randn(n_smiles, 768).astype("float32"),
                np.random.randn(n_meta, len(FIELDS), 384).astype("float32"),
                np.ones((n_meta, len(FIELDS)), dtype="float32"),
            )
            out = model(**batch)
            self.assertEqual(out["distance"].shape, (3,))
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()


if __name__ == "__main__":
    unittest.main()
