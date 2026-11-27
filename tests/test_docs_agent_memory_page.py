"""Every Python block of the Antora page ``docs/modules/ROOT/pages/how-to/agent-memory.adoc`` runs, and prints what the page says.

The blocks of the page are chained on purpose: they run in the order of the page, in one namespace. A block that is followed by a
``[source,text]`` block must print exactly that text, a block without one must print nothing, and every block that differs is
reported, not only the first. The other tests call the functions that the page defines on more data than the page uses, and pin what
its text says. The tests use no Java object, so they run on the JVM build and the native build alike.

The page opens ``./agent_memory``. The tests put an absolute path of their own in that place and do not change the working
directory: after ``os.chdir`` a relative database path did not name one directory on either build (a later test opened the database
of an earlier one, or one without its types), as if the engine keeps the working directory that it started in.
"""

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pytest

import arcadedb
from arcadedb import ArcadeDBError

PAGE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "modules"
    / "ROOT"
    / "pages"
    / "how-to"
    / "agent-memory.adoc"
)
PAGE_PATH = '"./agent_memory"'
DATABASE = "__database_path__"  # the namespace key of a test's path for the page

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


@pytest.fixture
def page(tmp_path):
    """The namespace that the blocks of the page run in. Its database is closed afterwards."""
    namespace = {
        "__name__": "agent_memory_page",
        DATABASE: str(tmp_path / "agent_memory"),
    }
    yield namespace
    db = namespace.get("db")
    if db is not None and db.is_open():
        db.close()


def _run_block(namespace, number, code):
    code = code.replace(PAGE_PATH, repr(namespace[DATABASE]))
    captured = io.StringIO()
    compiled = compile(code, f"agent-memory.adoc python block {number}", "exec")
    with contextlib.redirect_stdout(captured):
        exec(compiled, namespace)  # nosec B102 - the page of this repository
    return captured.getvalue().strip()


def _run_page(namespace):
    wrong = []
    for number, (code, expected) in enumerate(_blocks(), start=1):
        printed = _run_block(namespace, number, code)
        shown = (expected or "").strip()
        if printed != shown:
            wrong.append(
                f"block {number} printed:\n{printed}\nthe page shows:\n{shown}"
            )
    assert not wrong, "\n\n".join(wrong)


def _reopened(namespace):
    """The whole page, then its database open again: the functions of the page use the name ``db``."""
    _run_page(namespace)
    namespace["db"] = arcadedb.open_database(namespace[DATABASE])
    return namespace["db"]


def _mentions(db, key):
    rows = db.query(
        "opencypher",
        "MATCH (:Memory {key: $key})-[:Mentions]->(e:Entity) "
        "RETURN e.name AS name ORDER BY name",
        {"key": key},
    )
    return [row.get("name") for row in rows]


def test_the_page_has_its_blocks():
    blocks = _blocks()
    assert len(blocks) >= 10
    assert sum(1 for _, out in blocks if out is not None) >= 9
    # the one place where the tests put their own path
    assert sum(code.count(PAGE_PATH) for code, _ in blocks) == 1


def test_every_block_runs_in_order_and_prints_what_the_page_says(page):
    _run_page(page)


def test_the_first_block_runs_again_at_the_next_start(page):
    first = _blocks()[0][0]
    _run_block(page, 1, first)
    page["db"].command("sql", "INSERT INTO Message SET turn = 1, text = 'kept'")
    page["db"].close()
    _run_block(page, 1, first)
    assert page["db"].count_type("Message") == 1


def test_remember_replaces_every_link_of_a_fact(page):
    db = _reopened(page)
    links = db.count_type("Mentions")
    many = [f"thing-{number}" for number in range(100)]
    page["remember"]("bulk", "ann", "many things", [0.0, 0.0, 1.0, 0.0], many)
    assert _mentions(db, "bulk") == sorted(many)

    few = ["thing-1", "thing-2", "new-thing"]
    page["remember"]("bulk", "ann", "few things", [0.0, 0.0, 1.0, 0.0], few)
    assert _mentions(db, "bulk") == sorted(few)
    assert db.count_type("Mentions") == links + len(few)


def test_forget_deletes_every_fact_that_mentions_the_entity(page):
    db = _reopened(page)
    memories, entities, links = page["counts"]()
    for number in range(100):
        vector = [0.0, 0.0, 1.0, number / 1000]
        page["remember"](
            f"cat-{number}", "cat", f"fact {number}", vector, ["Cat", "tea"]
        )
    assert db.count_type("Memory") == memories + 100

    page["forget"]("Cat")
    # "tea" stays: only the entity that was named is deleted, with the facts that mention it
    assert page["counts"]() == [memories, entities + 1, links]
    assert page["nearest"]([0.0, 0.0, 1.0, 0.0], "cat") == []


def test_upsert_needs_a_unique_index(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Plain")
    with pytest.raises(ArcadeDBError):
        result = temp_db.command(
            "sql", "UPDATE Plain SET name = ? UPSERT WHERE name = ?", "a", "a"
        )
        if result is not None:
            result.to_list()
    assert temp_db.count_type("Plain") == 0


def test_the_rebuild_threshold_is_a_fifth_of_a_larger_graph(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE Memory")
    temp_db.command("sql", "CREATE PROPERTY Memory.embedding ARRAY_OF_FLOATS")
    vectors = np.random.default_rng(0).random((2000, 4), dtype=np.float32)
    temp_db.insert_many("Memory", [{"embedding": vector} for vector in vectors])
    temp_db.command(
        "sql",
        "CREATE INDEX ON Memory (embedding) LSM_VECTOR "
        'METADATA {"dimensions": 4, "similarity": "COSINE"}',
    )
    index = temp_db.schema.get_vector_index("Memory", "embedding")
    index.build_graph_now()
    assert index.get_stats()["effectiveMutationsThreshold"] == 400
