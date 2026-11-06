"""Every Python block of the two Antora pages ``how-to/import-export.adoc`` and ``how-to/dataframes-and-columns.adoc`` runs, and prints what the
page says (ArcadeData/arcadedb-python#267).

Same pattern as ``test_docs_transactions_page.py``: the blocks of one page are chained on purpose, they run in the order of the page, in one
namespace, against one fresh temporary database that the namespace holds as ``db`` (and an empty folder as ``workdir``). A block that is followed
by a ``[source,text]`` block must print exactly that text. What each statement of the pages claims is pinned on its own in
``test_import_export_behavior.py`` and ``test_dataframe_columns_behavior.py``. The tests use no Java object, so they run on the JVM build and the
native build alike.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

PAGES = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
)

# a Python listing, then (optionally, directly after it) the text listing with its output
BLOCK_RE = re.compile(
    r"\[source,python[^\]\n]*\]\n----\n(?P<code>.*?)\n----\n"
    r"(?:\n\[source,text\]\n----\n(?P<out>.*?)\n----\n)?",
    re.DOTALL,
)

# page, number of Python blocks, number of them that print
EXPECTED = {
    "import-export.adoc": (11, 10),
    "dataframes-and-columns.adoc": (8, 7),
}


def _blocks(page):
    text = (PAGES / page).read_text(encoding="utf-8")
    found = [(m.group("code"), m.group("out")) for m in BLOCK_RE.finditer(text)]
    # no Python listing may escape the extraction (a different spelling of the block header would run nowhere)
    assert text.count("[source,python") == len(found)
    return found


@pytest.mark.parametrize("page", sorted(EXPECTED))
def test_the_page_has_its_blocks(page):
    blocks = _blocks(page)
    assert len(blocks) == EXPECTED[page][0]
    assert sum(1 for _, out in blocks if out is not None) == EXPECTED[page][1]


def _run(page, namespace):
    for number, (code, expected) in enumerate(_blocks(page), start=1):
        captured = io.StringIO()
        compiled = compile(code, f"{page} python block {number}", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, namespace)  # nosec B102 - the pages of this repository
        if expected is not None:
            assert (
                captured.getvalue().strip() == expected.strip()
            ), f"block {number} of {page} printed other text than the page shows:\n{code}"
        else:
            assert (
                captured.getvalue() == ""
            ), f"block {number} of {page} prints, the page shows no output"


def test_every_block_of_the_import_and_export_page_runs_in_order(temp_db, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    _run(
        "import-export.adoc",
        {"db": temp_db, "workdir": workdir, "__name__": "import_export_page"},
    )


def test_every_block_of_the_dataframe_page_runs_in_order(temp_db):
    for library in ("numpy", "pandas", "polars", "pyarrow"):
        pytest.importorskip(library)
    _run(
        "dataframes-and-columns.adoc",
        {"db": temp_db, "__name__": "dataframes_page"},
    )
