"""Configure the V12 pair engine with V12.1 target generation."""

from __future__ import annotations

import functools
from pathlib import Path

import pipeline.pair_core as pair_core
import scripts.build_raw_pair as pair_driver
import scripts.build_v7_intern_raw_pair as v7
from pipeline.v11_prompt_rendering import render_prompt
from pipeline.v12_1_targets import TARGET_CONTRACT, is_decisive, target_for
from pipeline.v12_contract import DEFAULT_REGISTRY
from scripts import build_v11_intern_raw_pair as v11
from scripts import build_v12_intern_raw_pair as v12


def _metadata(row: dict) -> None:
    v11._fix_metadata_v11(row, target_contract=TARGET_CONTRACT)


def configure_engine(registry_path: Path = DEFAULT_REGISTRY) -> None:
    pair_core.target_for = target_for
    pair_driver.target_for = target_for
    v7.target_for = target_for
    v7.row_for = v12.row_for
    v7._fix_metadata = _metadata
    v7._is_decisive = is_decisive
    pair_core.render_prompt = functools.partial(render_prompt, registry_path=registry_path)
    pair_core.eligible = v12.eligible
    v7.TRAIN_RETRIEVAL_DEGREE_CAP = v12.TRAIN_RETRIEVAL_DEGREE_CAP
    v7.TRAIN_QUERY_DEGREE_CAP = v12.TRAIN_QUERY_DEGREE_CAP


__all__ = ["configure_engine"]
