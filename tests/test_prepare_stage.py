"""End-to-end test for the prepare stage on a synthetic source parquet."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.stages import prepare  # noqa: E402


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _q4_table() -> pa.Table:
    # q4 schema columns used by the resolver + a couple of metadata columns.
    return pa.table(
        {
            "pmid": ["1", "2", "3", "4"],
            "extraction_id": ["a", "b", "c", "d"],
            "smiles": ["CCO", "CCO", "c1ccccc1", ""],  # last row: missing smiles -> drop
            "metric_type": [
                "metabolic_half_life",  # enabled
                "extraction_ratio",  # enabled
                "intrinsic_clearance",  # disabled -> quarantine
                "metabolic_half_life",
            ],
            "reported_value": ["120", "40", "12", "30"],
            "reported_units": ["min", "%", "mL/min/kg", "min"],
            "species": ["human", "rat", "human", "dog"],
            "assay_system": ["human_liver_microsomes", "in_vivo", "x", "y"],
            "enzyme_or_pathway": ["CYP3A4", None, None, None],
        }
    )


class PrepareStageTest(unittest.TestCase):
    def test_prepare_q4_assigns_canonicalizes_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "extractions.parquet"
            pq.write_table(_q4_table(), src)
            out = Path(d) / "base"
            report = prepare.build(
                _Args(source="q4", input=src, source_dir=None, output_dir=out, compression="zstd")
            )

            # 4 rows: half_life (kept) + extraction_ratio (kept) + intrinsic_clearance
            # (disabled -> quarantine) + half_life with blank smiles (dropped) => 2 kept.
            self.assertEqual(report["records_kept"], 2)
            self.assertEqual(report["quarantine_by_reason"].get("unmapped_or_disabled_endpoint"), 1)
            self.assertEqual(report["quarantine_by_reason"].get("missing_smiles"), 1)

            table = pq.read_table(out / "base.parquet")
            recs = table.to_pylist()
            by_ep = {r["canonical_endpoint_id"]: r for r in recs}
            self.assertIn("q4.metabolic_half_life", by_ep)
            self.assertIn("q4.extraction_ratio", by_ep)

            hl = by_ep["q4.metabolic_half_life"]
            self.assertEqual(hl["property_value_native"], 120.0)  # minutes
            self.assertAlmostEqual(hl["property_value"], math.log10(120.0), places=6)
            self.assertEqual(hl["metric_type"], "half_life")
            self.assertEqual(hl["species_exact"], "human")
            self.assertEqual(hl["canonical_endpoint_key"], "q4.metabolic_half_life")
            # raw metadata retained for model input.
            self.assertEqual(hl["assay_system"], "human_liver_microsomes")

            er = by_ep["q4.extraction_ratio"]
            self.assertAlmostEqual(er["property_value_native"], 0.4)  # 40% -> fraction

    def test_report_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "extractions.parquet"
            pq.write_table(_q4_table(), src)
            out = Path(d) / "base"
            prepare.build(_Args(source="q4", input=src, source_dir=None, output_dir=out, compression="zstd"))
            report = json.loads((out / "prepare_report.json").read_text())
            self.assertEqual(report["source"], "q4")
            self.assertGreater(report["species_exact_coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
