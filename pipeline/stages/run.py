#!/usr/bin/env python3
"""Build DAG runner (skeleton).

Reads a build config and executes the assay-transfer pipeline stages in order:

    prepare -> split -> pairs -> materialize -> render_hf

Each stage writes its output under the stage-first ``datasets/`` layout keyed by the
build name, plus a ``manifest.json`` pinning the policy/version axes. This module is a
placeholder; the stage runners are implemented in :mod:`pipeline.stages` and wired here
in the stage-runner phase of the migration.
"""

from __future__ import annotations

STAGES = ("prepare", "split", "pairs", "materialize", "render_hf")
