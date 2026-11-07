"""Every Python block of the Antora page ``docs/modules/ROOT/pages/how-to/facets.adoc`` runs, and prints what the page says.

The blocks of the page are chained on purpose: they run in the order of the page, in one namespace, against one fresh temporary
database that the namespace holds as ``db`` (the page says that each block assumes an open ``db``). A block that is followed by a
``[source,text]`` block must print exactly that text. The test uses no Java object, so it runs on the JVM build and the native
build alike.

Facets are new in engine 26.12.1 (see tests/test_columnar_engine_smoke.py). The engine of a lane is whatever the lane links: the
JVM lane runs the ``latest`` image, the native pull request gate links 26.11.1. A block of the page that creates a COLUMNAR index
or runs a ``FACETS`` query carries ``role=needs-engine`` and the title "Needs engine 26.12.1 or later". On a newer engine such a
block runs and prints what the page shows. On an older engine the same block must be refused with the engine's own error, which
is what the page documents, and the test asserts that refusal instead of the output. Nothing is skipped (the repository accepts no
skips, scripts/check_test_skips.py). The version is read through the test's database (``SELECT version()``), so the same check
runs on both builds.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

FACETS_SINCE = (26, 12, 1)

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
    / "facets.adoc"
)

# a Python listing (its header may name a role, and a title line may come before it), then (optionally, directly after it) the
# text listing with its output
BLOCK_RE = re.compile(
    r"(?:^(?P<title>\.[^\n]*)\n)?"
    r"\[source,python(?P<attrs>[^\]\n]*)\]\n----\n(?P<code>.*?)\n----\n"
    r"(?:\n\[source,text\]\n----\n(?P<out>.*?)\n----\n)?",
    re.DOTALL | re.MULTILINE,
)

NEEDS_TITLE = ".Needs engine 26.12.1 or later"

# what an engine before 26.12.1 answers: a FACETS clause is a syntax error, a COLUMNAR index is an unknown index type
REFUSAL = r"SQL syntax error|Index type 'COLUMNAR' is not supported|COLUMNAR"


def _blocks():
    """The Python blocks as (code, expected output or None, needs the new engine)."""
    text = PAGE.read_text(encoding="utf-8")
    found = []
    for m in BLOCK_RE.finditer(text):
        needs = "role=needs-engine" in m.group("attrs")
        # the title and the role say the same thing, in both directions
        assert needs == (m.group("title") == NEEDS_TITLE), m.group("code")[:80]
        found.append((m.group("code"), m.group("out"), needs))
    # no Python listing may escape the extraction (a different spelling of the block header would run nowhere)
    assert text.count("[source,python") == len(found)
    return found


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
    assert len(blocks) >= 18
    assert sum(1 for _, out, _ in blocks if out is not None) >= 14
    needing = [code for code, _, needs in blocks if needs]
    assert len(needing) >= 12
    # every block that needs the new engine says so: its first engine call is a COLUMNAR index or a FACETS query
    for code in needing:
        assert "COLUMNAR" in code or "FACETS" in code, code[:80]
    # and no other block does: they all run on an engine without facets
    for code, _, needs in blocks:
        if not needs:
            assert "FACETS" not in code and "COLUMNAR" not in code, code[:80]


def test_every_block_runs_in_order_and_prints_what_the_page_says(temp_db):
    namespace = {"db": temp_db, "__name__": "facets_page"}
    newer = _engine_version(temp_db) >= FACETS_SINCE
    refused = 0
    for number, (code, expected, needs) in enumerate(_blocks(), start=1):
        compiled = compile(code, f"facets.adoc python block {number}", "exec")
        captured = io.StringIO()
        if needs and not newer:
            # an older engine: the block is refused with the engine's own error, before it prints anything
            with pytest.raises(Exception, match=REFUSAL):
                with contextlib.redirect_stdout(captured):
                    exec(
                        compiled, namespace
                    )  # nosec B102 - the page of this repository
            assert captured.getvalue() == "", f"block {number} printed before it failed"
            refused += 1
            continue
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

    # the page's own version check agrees with the test's, on the engine of this lane
    assert namespace["has_facets"] is newer
    assert (refused == 0) == newer
