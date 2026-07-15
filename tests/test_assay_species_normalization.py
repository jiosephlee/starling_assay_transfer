from __future__ import annotations

from pathlib import Path
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.normalize.assay_species_normalization import (  # noqa: E402
    resolve_assay_species,
    resolve_species_record,
)
from pipeline.normalize.species_normalization import normalized_species_or_population  # noqa: E402


class UnifiedFacadeTest(unittest.TestCase):
    def test_facade_returns_exact_species(self) -> None:
        self.assertEqual(normalized_species_or_population("Wistar rat"), "rat")
        self.assertEqual(normalized_species_or_population("mouse"), "mouse")

    def test_facade_drops_multi_species_values(self) -> None:
        self.assertIsNone(normalized_species_or_population("human and rat"))

    def test_oral_and_assay_resolvers_share_the_ontology(self) -> None:
        row = {"species_or_population": "cynomolgus monkeys"}
        oral = resolve_species_record(row, "oral_bioavailability")
        assay = resolve_assay_species(explicit="cynomolgus monkeys")
        self.assertEqual(oral, assay)


class ExplicitSpeciesResolutionTest(unittest.TestCase):
    def test_exact_rodents_remain_distinct(self) -> None:
        self.assertEqual(resolve_assay_species(explicit="male Wistar rat"), "rat")
        self.assertEqual(resolve_assay_species(explicit="mouse"), "mouse")

    def test_generic_primate_does_not_become_exact(self) -> None:
        self.assertEqual(
            resolve_assay_species(explicit="cynomolgus monkey"),
            "cynomolgus_monkey",
        )
        self.assertIsNone(resolve_assay_species(explicit="monkey"))

    def test_population_vocabulary_does_not_imply_human(self) -> None:
        values = ("healthy volunteers", "adult patients", "children", "participants")
        for value in values:
            with self.subTest(value=value):
                self.assertIsNone(resolve_assay_species(explicit=value))

    def test_explicit_species_word_is_still_recovered(self) -> None:
        self.assertEqual(resolve_assay_species(explicit="canine patients"), "dog")

    def test_cell_line_origin_does_not_imply_species(self) -> None:
        self.assertIsNone(resolve_assay_species(assay_systems="Caco-2 transport"))
        self.assertIsNone(resolve_assay_species(assay_systems="MDCK-MDR1 permeability"))

    def test_species_independent_system_is_null(self) -> None:
        self.assertIsNone(resolve_assay_species(assay_systems="PAMPA artificial membrane"))

    def test_multiple_species_are_null(self) -> None:
        self.assertIsNone(resolve_assay_species(explicit="human and rat"))

    def test_longest_alias_prevents_guinea_pig_overlap(self) -> None:
        self.assertEqual(resolve_assay_species(explicit="guinea pig"), "guinea_pig")

    def test_specific_mallard_suppresses_generic_duck(self) -> None:
        self.assertEqual(resolve_assay_species(contexts="fed mallard ducks"), "mallard")

    def test_separate_nested_species_mentions_remain_ambiguous(self) -> None:
        self.assertIsNone(resolve_assay_species(contexts="mallard duck and duck"))

    def test_only_explicit_species_text_is_used_in_mixed_system(self) -> None:
        system = "human MDR1-transfected MDCK cells"
        self.assertEqual(resolve_assay_species(assay_systems=system), "human")

    def test_q2_record_uses_approved_structured_text(self) -> None:
        row = {"biological_context": "rat jejunum", "assay_system": "in situ perfusion"}
        self.assertEqual(resolve_species_record(row, "q2"), "rat")

    def test_q1_record_uses_approved_structured_text(self) -> None:
        row = {"study_context": "Greyhound dogs, plasma, single oral dose"}
        self.assertEqual(resolve_species_record(row, "q1"), "dog")

    def test_q4_falls_back_when_dedicated_species_is_missing(self) -> None:
        row = {"species": None, "assay_system": "pooled human liver microsomes"}
        self.assertEqual(resolve_species_record(row, "q4"), "human")

    def test_q4_dedicated_species_precedes_assay_text(self) -> None:
        row = {"species": "rat", "assay_system": "recombinant human CYP3A4"}
        self.assertEqual(resolve_species_record(row, "q4"), "rat")

    def test_populated_generic_species_blocks_fallback(self) -> None:
        row = {"species": "monkey", "assay_system": "human liver microsomes"}
        self.assertIsNone(resolve_species_record(row, "q4"))

    def test_unapproved_support_text_is_not_parsed(self) -> None:
        row = {"study_context": None, "support_text": "study in human volunteers"}
        self.assertIsNone(resolve_species_record(row, "q1"))


if __name__ == "__main__":
    unittest.main()
