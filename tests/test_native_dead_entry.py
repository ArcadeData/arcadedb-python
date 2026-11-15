"""``arcadedb_command`` is gone: nothing in the wheel called it, and it carried every row of a command as JSON for nobody.

The entry point came from the first spike of the native library (``ArcadeLib.java``): it ran one statement and returned the rows as the
engine's own ``Result.toJSON()`` in a JSON array. The glue has since used ``arcadedb_n_exec``, ``arcadedb_n_cmd`` and the cursor entry
points, whose replies the Python side reads, and no Python file, test, script or document named the old one. These tests keep it that way:
the sources of the library and of the wheel never name it, and the loaded library does not export it, while the entry points the glue
does call still resolve.

conftest.py collects this file on the native backend only: the second test loads the native library.
"""

import pathlib
import re

import pytest

from arcadedb._native import lib

ROOT = pathlib.Path(__file__).resolve().parent.parent
GONE = "arcadedb_command"
NAMED = re.compile(r"\b" + GONE + r"\b")


@pytest.mark.parametrize(
    "folder, suffix",
    [(("native", "src"), ".java"), (("src", "arcadedb"), ".py")],
    ids=["library sources", "python glue"],
)
def test_no_source_of_the_wheel_names_the_removed_entry_point(folder, suffix):
    base = ROOT.joinpath(*folder)
    assert (
        base.is_dir()
    ), base  # a moved folder must not turn the scan into an empty pass
    files = sorted(base.rglob("*" + suffix))
    assert files, f"no {suffix} file under {base}"
    named = [
        str(f.relative_to(ROOT))
        for f in files
        if NAMED.search(f.read_text(encoding="utf-8"))
    ]
    assert named == []


def test_the_loaded_library_does_not_export_the_removed_entry_point():
    library = lib._lib or lib._load()
    assert not hasattr(library, GONE)
    # the entry points that the glue calls for every reply and every statement are there
    assert hasattr(library, "arcadedb_free")
    assert hasattr(library, "arcadedb_n_exec")
    assert hasattr(library, "arcadedb_n_cmd")
