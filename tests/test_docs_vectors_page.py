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


HIERARCHY_SINCE = (
    26,
    12,
    1,
)  # the engine builds a hierarchy by default from here (ArcadeData/arcadedb#12616)


def _engine_version(db):
    """The engine's x.y.z through SQL `SELECT version()`; a SNAPSHOT or a build suffix is dropped."""
    raw = str(db.query("sql", "SELECT version() AS v").first().get("v"))
    parts = [
        int(p) if p.isdigit() else 0
        for p in raw.split()[0].split("-")[0].split(".")[:3]
    ]
    return tuple(parts + [0] * (3 - len(parts)))


def test_the_page_has_its_blocks():
    blocks = _blocks()
    assert len(blocks) >= 10
    assert sum(1 for _, out in blocks if out is not None) >= 8


def test_every_block_runs_in_order_and_prints_what_the_page_says(temp_db):
    namespace = {"db": temp_db, "__name__": "vectors_page"}
    newer = _engine_version(temp_db) >= HIERARCHY_SINCE
    for number, (code, expected) in enumerate(_blocks(), start=1):
        captured = io.StringIO()
        compiled = compile(code, f"vectors.adoc python block {number}", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, namespace)  # nosec B102 - the page of this repository
        if expected is not None:
            # the page shows the engine of this release; an older engine (the parity job's 26.11.1) builds a flat graph from SQL
            want = expected if newer else expected.replace("NONE True", "NONE False")
            assert (
                captured.getvalue().strip() == want.strip()
            ), f"block {number} printed other text than the page shows:\n{code}"
        else:
            assert (
                captured.getvalue() == ""
            ), f"block {number} prints, the page shows no output"
