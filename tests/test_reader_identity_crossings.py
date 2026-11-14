"""Crossings of ``Result.get_rid()``, ``get_vertex()``, ``get_edge()``, ``get_element()`` and ``Document.get_rid()`` on the JVM build.

``Result.get_rid()`` made four JPype calls (``getIdentity``, ``isPresent``, ``get``, ``toString``), each of the three element getters made three
(``getVertex()``, ``isPresent()``, ``get()``), and ``Document.get_rid()`` two. ``ReaderCalls`` now answers each in one call. Every test here pins two
things: the calls into the JVM (counted on a proxy of the Java row or record and of the bridge class, the technique of ``test_reader_crossings.py``;
a call on the Java row itself would show up in the log) and that the answer is the one the per-call path gives (the oracle is that path, written out
below), so a count cannot be bought with a change of behaviour.
"""

import pytest

from arcadedb import results
from arcadedb.graph import Document, Edge, Vertex


class Counting:
    """Counts the method calls made on a Java object or a Java class; a ``Counting`` argument is unwrapped, return values are the real objects."""

    def __init__(self, target, log, label):
        self._target = target
        self._log = log
        self._label = label

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        def call(*args):
            self._log.append(f"{self._label}.{name}")
            args = [a._target if isinstance(a, Counting) else a for a in args]
            return attr(*args)

        return call


def _old_rid(java_row):
    """``Result.get_rid()`` before the bridge call."""
    identity = java_row.getIdentity()
    return str(identity.get().toString()) if identity.isPresent() else None


def _old_element(java_row, getter):
    """``getVertex()``, ``getEdge()`` or ``getElement()`` of a Java row, read as the three calls did."""
    optional = getattr(java_row, getter)()
    return optional.get() if optional.isPresent() else None


@pytest.fixture
def graph_db(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE V")
    temp_db.command("sql", "CREATE EDGE TYPE E")
    temp_db.command("sql", "CREATE DOCUMENT TYPE D")
    with temp_db.transaction():
        a = temp_db.command("sql", "CREATE VERTEX V SET name = 'a'").first()
        b = temp_db.command("sql", "CREATE VERTEX V SET name = 'b'").first()
        temp_db.command(
            "sql",
            "CREATE EDGE E FROM :a TO :b SET w = 1",
            {"a": a.get_rid(), "b": b.get_rid()},
        ).close()
        temp_db.command("sql", "INSERT INTO D SET k = 1").close()
    return temp_db


@pytest.fixture
def log(graph_db, monkeypatch):
    """The list every counted call is appended to; the bridge class ``ReaderCalls`` is counted."""
    calls = []
    real = results._bridge_class("ReaderCalls")
    assert real is not None, "the bridge jar must carry ReaderCalls"
    monkeypatch.setitem(
        results._BRIDGE_CLASSES, "ReaderCalls", Counting(real, calls, "ReaderCalls")
    )
    return calls


# a vertex, an edge, a plain document, a projection that is no record, an aggregate
ROWS = {
    "vertex": "SELECT FROM V WHERE name = 'a'",
    "edge": "SELECT FROM E",
    "document": "SELECT FROM D",
    "projection": "SELECT 1 AS x",
    "aggregate": "SELECT count(*) AS c FROM V",
}


def _row(db, kind):
    return db.query("sql", ROWS[kind]).first()


@pytest.mark.parametrize("kind", ROWS)
def test_get_rid_is_one_call_and_the_rid_the_per_call_path_gives(graph_db, log, kind):
    row = _row(graph_db, kind)
    want = _old_rid(row._java_result)
    row._java_result = Counting(row._java_result, log, "row")
    del log[:]
    got = row.get_rid()
    assert log == ["ReaderCalls.ridOf"]
    assert got == want
    assert (got is None) == (kind in ("projection", "aggregate"))


@pytest.mark.parametrize("kind", ROWS)
@pytest.mark.parametrize(
    "method, getter, bridge, wrapper",
    [
        ("get_vertex", "getVertex", "vertexOrNull", Vertex),
        ("get_edge", "getEdge", "edgeOrNull", Edge),
        ("get_element", "getElement", "elementOrNull", Document),
    ],
)
def test_the_element_getters_are_one_call_and_return_what_the_per_call_path_does(
    graph_db, log, kind, method, getter, bridge, wrapper
):
    row = _row(graph_db, kind)
    want = _old_element(row._java_result, getter)
    row._java_result = Counting(row._java_result, log, "row")
    del log[:]
    got = getattr(row, method)()
    assert log == [f"ReaderCalls.{bridge}"]
    if want is None:
        assert got is None
        return
    assert isinstance(got, wrapper)
    assert str(got._java_document.getClass().getName()) == str(
        want.getClass().getName()
    )
    assert str(got.get_rid()) == str(want.getIdentity())


def test_each_kind_of_row_has_the_element_it_should(graph_db):
    vertex = _row(graph_db, "vertex")
    assert isinstance(vertex.get_vertex(), Vertex) and vertex.get_edge() is None
    edge = _row(graph_db, "edge")
    assert isinstance(edge.get_edge(), Edge) and edge.get_vertex() is None
    assert isinstance(_row(graph_db, "document").get_element(), Document)
    assert _row(graph_db, "projection").get_element() is None


def test_document_get_rid_is_one_call_and_the_text_str_of_the_identity_gives(
    graph_db, log
):
    doc = _row(graph_db, "document").get_element()
    want = str(doc._java_document.getIdentity())
    doc._java_document = Counting(doc._java_document, log, "doc")
    del log[:]
    assert doc.get_rid() == want
    assert log == ["ReaderCalls.documentRid"]


def test_a_record_with_no_identity_yet_reads_as_it_did(graph_db):
    with graph_db.transaction():
        doc = graph_db.new_document("D")
        # str() of whatever the engine gives an unsaved record: the same text as before, "None" for no identity at all
        assert doc.get_rid() == str(doc._java_document.getIdentity())


@pytest.mark.parametrize("kind", ROWS)
def test_without_the_bridge_class_every_getter_answers_as_with_it(
    graph_db, monkeypatch, kind
):
    """No ``ReaderCalls`` (a broken checkout, the case ``_bridge_class`` answers None for): the plain path, the same answers, no AttributeError."""

    def answers():
        row = _row(graph_db, kind)
        element = row.get_element()
        return {
            "rid": row.get_rid(),
            "vertex": None if row.get_vertex() is None else row.get_vertex().get_rid(),
            "edge": None if row.get_edge() is None else row.get_edge().get_rid(),
            "element": None if element is None else element.get_rid(),
            "types": [
                type(x).__name__ for x in (row.get_vertex(), row.get_edge(), element)
            ],
        }

    with_bridge = answers()
    monkeypatch.setitem(results._BRIDGE_CLASSES, "ReaderCalls", None)
    assert answers() == with_bridge
    assert (with_bridge["rid"] is None) == (kind in ("projection", "aggregate"))
