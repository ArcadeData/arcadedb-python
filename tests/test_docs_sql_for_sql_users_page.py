"""Every Python block of the Antora page ``docs/modules/ROOT/pages/how-to/sql-for-sql-users.adoc`` runs, and prints what the page says.

Same pattern as ``test_docs_transactions_page.py``: the blocks are chained on purpose, they run in the order of the page, in one
namespace. The page opens its own database under the current directory, so the run happens in a temporary directory. A block that is
followed by a ``[source,text]`` block must print exactly that text. The statements that the page says are refused are pinned below.
The tests use no Java object, so they run on the JVM build and the native build alike.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

from arcadedb import ArcadeDBError

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
    / "sql-for-sql-users.adoc"
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
    assert len(blocks) >= 7
    assert sum(1 for _, out in blocks if out is not None) >= 5


def test_every_block_runs_in_order_and_prints_what_the_page_says(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the page opens ./sqlpage
    namespace = {"__name__": "sql_for_sql_users_page"}
    for number, (code, expected) in enumerate(_blocks(), start=1):
        captured = io.StringIO()
        compiled = compile(
            code, f"sql-for-sql-users.adoc python block {number}", "exec"
        )
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


def _schema(db):
    db.command("sql", "CREATE VERTEX TYPE Author")
    db.command("sql", "CREATE VERTEX TYPE Book")
    db.command("sql", "CREATE EDGE TYPE Wrote")


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE EDGE Wrote FROM (SELECT FROM Author) TO (SELECT FROM Book)",
        "DELETE VERTEX Author WHERE name = 'Ann'",
        "SELECT FROM Author WHERE lower(name) = 'ann'",
        "SELECT FROM Author WHERE upper(name) = 'ANN'",
        "CREATE VERTEX TYPE Author",
    ],
)
def test_the_statements_the_page_calls_refused_are_refused(temp_db, statement):
    _schema(temp_db)
    temp_db.command("sql", "CREATE VERTEX Author SET name = 'Ann'")
    with pytest.raises(ArcadeDBError):
        if statement.startswith("SELECT"):
            temp_db.query("sql", statement).to_list()
        else:
            temp_db.command("sql", statement)


def test_a_list_alone_is_read_as_the_positional_parameters(temp_db):
    _schema(temp_db)
    with pytest.raises(TypeError):
        temp_db.query("sql", "SELECT FROM Book WHERE title IN ?", ["Emma", "Solaris"])
