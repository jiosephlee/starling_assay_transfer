"""v2 precompute dedup/index logic (stub encoders) chained into data + model."""

from __future__ import annotations

import json
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

    from starling_ml import data as datamod
    from starling_ml import precompute_embeddings as pre
    from starling_ml.config import Config, LossConfig, ModelConfig
    from starling_ml.model import TransferPairModel

    _OK = True
except Exception:  # pragma: no cover
    _OK = False

FIELDS = list(datamod.DEFAULT_METADATA_FIELDS) if _OK else []


def _rows():
    def r(rs, qs, asys, blabel):
        row = {"split": "train", "retrieved_smiles": rs, "query_smiles": qs,
               "retrieved_value": 50.0, "continuous_target": 2.0, "binary_label": blabel}
        for f in FIELDS:
            row[f] = None
        row["retrieved_assay_system"] = asys
        return row
    return [
        r("CCO", "CCCO", "human liver microsomes", 1),
        r("CCO", "c1ccccc1", "human liver microsomes", None),  # same Z_A as row 0
        r("CCCO", "CCO", "rat liver microsomes", 0),           # different Z_A
    ]


@unittest.skipUnless(_OK, "torch/pyarrow not available")
class PrecomputeV2Test(unittest.TestCase):
    def test_dedup_index_and_pipeline(self) -> None:
        cfg = Config.from_yaml("ml/configs/default.yaml")

        def stub_molformer(smiles, cfg):
            return np.random.randn(len(smiles), 768).astype("float16")

        def stub_metadata(field_values, fields, cfg):
            n = len(next(iter(field_values.values())))
            return (np.zeros((n, len(fields), 384), dtype="float16"),
                    np.ones((n, len(fields)), dtype="uint8"))

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ds = d / "dataset.parquet"
            pq.write_table(pa.Table.from_pylist(_rows()), ds)
            emb_dir = d / "emb"
            manifest = pre.build_v2_tables(str(ds), str(emb_dir), cfg, FIELDS,
                                           molformer_fn=stub_molformer, metadata_fn=stub_metadata)
            # 3 unique smiles (CCO, CCCO, c1ccccc1); 2 unique Z_A tuples.
            self.assertEqual(manifest["n_unique_smiles"], 3)
            self.assertEqual(manifest["n_unique_metadata"], 2)

            smiles_to_row = json.loads((emb_dir / "smiles_index.json").read_text())
            meta_to_row = json.loads((emb_dir / "meta_index.json").read_text())

            # The produced maps drive the data stage without KeyError.
            datamod.build_split_memmap(str(ds), str(d / "mm"), "train", smiles_to_row, meta_to_row)
            dsobj = datamod.PairDataset(str(d / "mm"), "train")
            batch = datamod.collate_pairs([dsobj[i] for i in range(len(dsobj))])

            mol_emb = np.load(emb_dir / "molformer_emb.npy")
            meta_emb = np.load(emb_dir / "metadata_emb.npy")
            meta_present = np.load(emb_dir / "metadata_present.npy")
            model = TransferPairModel(
                ModelConfig(mol_hidden=16, mol_out=8, meta_field_proj=4, d_model=16, d_ff=32, n_blocks=1),
                LossConfig(aux_binary_weight=0.5), mol_emb.astype("float32"),
                meta_emb.astype("float32"), meta_present.astype("float32"),
            )
            out = model(**batch)
            self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
