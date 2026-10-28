"""Every Python block of the Antora page ``docs/modules/ROOT/pages/how-to/vectors.adoc`` runs, and prints what the page says.

The blocks of the page are chained on purpose: they run in the order of the page, in one namespace, against one fresh
temporary database that the namespace holds as ``db`` (the page says that each block assumes an open ``db``). A block that
is followed by a ``[source,text]`` block must print exactly that text. The test uses no Java object, so it runs on the JVM
build and the native build alike.
"""

import contextlib
import io
import re
from pathlib import Path

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
    / "vectors.adoc"
)

# a Python listing, then (optionally, directly after it) the text listing with its output
BLOCK_RE = re.compile(
    r"\[source,python[^\]\n]*\]\n----\n(?P<code>.*?)\n----\n"
    r"(?:\n\[source,text\]\n----\n(?P<out>.*?)\n----\n)?",
    re.DOTALL,
)


def _blocks():
    text = PAGE.read_text(encoding="utf-8")
    found = [(m.group("code"), m.group("out")) for m in BLOCK_RE.finditer(text)]
    # no Python listing may escape the extraction (a different spelling of the block header would run nowhere)
    assert text.count("[source,python") == len(found)
    return found


def test_the_page_has_its_blocks():
    blocks = _blocks()
    assert len(blocks) >= 10
    assert sum(1 for _, out in blocks if out is not None) >= 8


def test_every_block_runs_in_order_and_prints_what_the_page_says(temp_db):
    namespace = {"db": temp_db, "__name__": "vectors_page"}
    for number, (code, expected) in enumerate(_blocks(), start=1):
        captured = io.StringIO()
        compiled = compile(code, f"vectors.adoc python block {number}", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, namespace)  # nosec B102 - the page of this repository
        if expected is not None:
            assert (
                captured.getvalue().strip() == expected.strip()
            ), f"block {number} printed other text than the page shows:\n{code}"
        else:
            assert (
                captured.getvalue() == ""
            ), f"block {number} prints, the page shows no output"
