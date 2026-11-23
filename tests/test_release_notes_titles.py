"""scripts/release_notes.py: what a line of the release notes shows of a pull request title.

The notes are for the reader of a release, so the notes that point at issues (`refs #388`, `closes #371`, `rank 2 of #388`) are left out of a line and a pull request that only merges a release line into main is not listed. Pure functions only: no network, no gh.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_notes.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["release_notes"] = module
    spec.loader.exec_module(module)
    return module


notes = _load()


@pytest.mark.parametrize(
    "title,number,shown",
    [
        (
            "Native count(): returns the rows (rank 2 of #388) (#389)",
            389,
            "Native count(): returns the rows",
        ),
        (
            "JVM Vertex.new_edge in one crossing (refs #388) (#396)",
            396,
            "JVM Vertex.new_edge in one crossing",
        ),
        (
            "Native: query by address (refs #388, #315) (#398)",
            398,
            "Native: query by address",
        ),
        (
            "Bulk step before json.dumps (closes #371) (#377)",
            377,
            "Bulk step before json.dumps",
        ),
        ("Fix a thing (fixes #5) (#6)", 6, "Fix a thing"),
        (
            "Docs: refs, rank and closes in a sentence (#7)",
            7,
            "Docs: refs, rank and closes in a sentence",
        ),
        (
            "A change that names (refs #1) in the middle and ends plain (#8)",
            8,
            "A change that names (refs #1) in the middle and ends plain",
        ),
        (
            "fix(#423): find_nearest(allowed_rids=[]) returns nothing (#425)",
            425,
            "fix(#423): find_nearest(allowed_rids=[]) returns nothing",
        ),
        ("(refs #9) (#9)", 9, "(refs #9)"),
    ],
)
def test_a_line_leaves_out_the_notes_that_point_at_issues(title, number, shown):
    assert notes.clean_title(title, number) == shown


@pytest.mark.parametrize(
    "title,is_merge",
    [
        ("Merge the 26.11.1.post3 release line into main", True),
        ("Merge the 26.11.1.post2 release line into main (#351)", True),
        ("Merge branch main into a feature", False),
        ("Merge the vector index docs", False),
    ],
)
def test_a_pull_request_that_only_merges_a_release_line_is_not_listed(title, is_merge):
    assert bool(notes.MERGE_LINE.match(title)) is is_merge
