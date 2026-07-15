from __future__ import annotations

from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (_REPO_ROOT, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from prepare_assay_base import _prepare_record  # noqa: E402
from pipeline.taskspecs import get_task  # noqa: E402


class PrepareDerivedSpeciesTest(unittest.TestCase):
    def _row(self, species: str) -> dict[str, str]:
        return {
            "global_identifier": "SMILES:1",
            "extracted.measured_result": "0.5",
            "extracted.property_type": "fraction_absorbed",
            "extracted.species_or_model": species,
            "pmid": "1",
            "extraction_id": "example",
        }

    def test_prepare_materializes_explicit_species_exact(self) -> None:
        record, rejection = _prepare_record(
            self._row("Wistar rat"),
            get_task("fa"),
            {"SMILES:1": "CC"},
        )
        self.assertIsNone(rejection)
        self.assertEqual(record["species_or_population"], "Wistar rat")
        self.assertEqual(record["species_exact"], "rat")

    def test_prepare_leaves_population_only_species_null(self) -> None:
        record, rejection = _prepare_record(
            self._row("healthy volunteers"),
            get_task("fa"),
            {"SMILES:1": "CC"},
        )
        self.assertIsNone(rejection)
        self.assertEqual(record["species_or_population"], "healthy volunteers")
        self.assertIsNone(record["species_exact"])


if __name__ == "__main__":
    unittest.main()
