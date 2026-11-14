"""Crossings of ``Vertex.new_edge`` on the JVM build.

``new_edge`` resolved both ends by record id in Python: ``getIdentity``, ``isValid``, ``lookupByRID`` and ``asVertex`` for the source,
``getIdentity`` and ``isValid`` for the target, then ``newEdge``: seven calls into the JVM. ``GraphCalls.newEdge`` does all of it in one. The first
test pins the number of calls (counted on proxies of the source and target vertex, the database and the bridge class, the technique of
``test_reader_crossings.py``: a call on any of the Java objects would show up in the log). The rest pin that the edge is the one the seven-call
path writes, which ``_old_new_edge`` below reproduces from the former code, so a count cannot be bought with a change of behaviour.
"""

import pytest

from arcadedb import graph
from arcadedb.type_conversion import convert_python_to_java


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


def _old_new_edge(vertex, label, target, **kwargs):
    """``Vertex.new_edge`` as it was before ``GraphCalls.newEdge``: seven calls into the JVM."""
    target_java = (
        target.get_java_document() if isinstance(target, graph.Vertex) else target
    )
    source_java = vertex._java_document
    source_rid = source_java.getIdentity()
    if source_rid is not None and source_rid.isValid() and vertex._database is not None:
        source_java = vertex._database._java_db.lookupByRID(source_rid, True).asVertex()
    if hasattr(target_java, "getIdentity"):
        target_rid = target_java.getIdentity()
        if target_rid is not None and target_rid.isValid():
            target_java = target_rid
    props = []
    for k, v in kwargs.items():
        props.append(k)
        props.append(convert_python_to_java(v))
    return graph.Edge(source_java.newEdge(label, target_java, *props), vertex._database)


@pytest.fixture
def ends(temp_db):
    temp_db.command("sql", "CREATE VERTEX TYPE V")
    temp_db.command("sql", "CREATE EDGE TYPE E")
    with temp_db.transaction():
        for name in "abcdef":
            temp_db.command("sql", "CREATE VERTEX V SET name = ?", name)
    rows = temp_db.query("sql", "SELECT @rid AS r FROM V ORDER BY name").to_list()
    return temp_db, [temp_db.lookup_by_rid(str(row["r"])) for row in rows]


def _edges(db, vertex, direction):
    """The RIDs of the "E" edges of ``vertex``, read through a vertex looked up now.

    A vertex object is the state it had when it was read: the engine (``GraphEngine.getMostUpdatedVertex``) hands it the
    newest version of its record only inside the transaction that wrote it, so an object read before an edge was
    committed lists none of it. ``new_edge`` does not refresh the object it is called on (its docstring says so), which
    holds for the seven-call path as well, so the lists are read through a fresh lookup.
    """
    fresh = db.lookup_by_rid(vertex.get_rid())
    listed = fresh.get_out_edges("E") if direction == "out" else fresh.get_in_edges("E")
    return sorted(e.get_rid() for e in listed)


KW = {"s": "text", "i": 7, "big": 2**40, "f": 1.5, "flag": True, "none": None}


def test_new_edge_is_one_call_into_the_bridge(ends, monkeypatch):
    db, (a, b, *_) = ends
    calls = []
    real = graph._graph_calls()
    assert real is not None, "the bridge jar must carry GraphCalls"
    with db.transaction():
        # the proxies are on only for the call itself: the transaction's own begin and
        # commit (``DbCalls.stackMark(db)`` among them) need the real database object
        with monkeypatch.context() as patched:
            patched.setattr(graph, "_GRAPH_CALLS", Counting(real, calls, "GraphCalls"))
            patched.setattr(db, "_java_db", Counting(db._java_db, calls, "db"))
            patched.setattr(a, "_java_document", Counting(a._java_document, calls, "a"))
            patched.setattr(b, "_java_document", Counting(b._java_document, calls, "b"))
            edge = a.new_edge("E", b, **KW)
    assert calls == ["GraphCalls.newEdge"]
    assert edge.get_rid().startswith("#")


def test_the_edge_is_the_one_the_seven_call_path_writes(ends):
    db, (a, b, c, d, *_) = ends
    with db.transaction():
        new = a.new_edge("E", b, **KW)
        old = _old_new_edge(c, "E", d, **KW)
    assert new.to_dict() == old.to_dict()
    assert new.get("i") == 7 and new.get("s") == "text"
    assert type(new) is type(old)
    assert str(new._java_document.getClass().getName()) == str(
        old._java_document.getClass().getName()
    )
    assert _edges(db, a, "out") == [new.get_rid()]
    assert _edges(db, b, "in") == [new.get_rid()]
    assert _edges(db, c, "out") == [old.get_rid()]
    assert _edges(db, d, "in") == [old.get_rid()]
    # the object new_edge was called on is left as it was read, as on the seven-call path
    assert len(a.get_out_edges("E")) == len(c.get_out_edges("E"))


@pytest.mark.parametrize("form", ["vertex", "java vertex", "java rid"])
def test_every_form_of_target_reaches_the_same_vertex(ends, form):
    db, (a, b, *_) = ends
    target = {
        "vertex": b,
        "java vertex": b.get_java_document(),
        "java rid": b.get_identity(),
    }[form]
    with db.transaction():
        edge = a.new_edge("E", target, w=3)
        assert edge.get_raw("w") == 3
        assert [e.get_rid() for e in b.get_in_edges("E")] == [edge.get_rid()]


def test_an_edge_without_properties_and_several_edges_of_one_vertex(ends):
    db, (a, b, c, d, *_) = ends
    with db.transaction():
        e1 = a.new_edge("E", b)
        e2 = a.new_edge("E", c)
    with db.transaction():
        e3 = a.new_edge("E", d, w=1)  # a was read before the two commits above
    assert e1.to_dict() == e2.to_dict()
    assert e3.get("w") == 1
    assert _edges(db, a, "out") == sorted(e.get_rid() for e in (e1, e2, e3))


def _outcome(db, action):
    """None when ``action`` ran inside one transaction, else the class of what it raised."""
    try:
        with db.transaction():
            action()
    except Exception as exc:  # the parity of the failure is what is compared
        return type(exc)
    return None


def test_a_failure_is_the_one_the_seven_call_path_raises(ends):
    db, (a, b, c, d, *_) = ends
    assert _outcome(db, lambda: a.new_edge("NoSuchEdgeType", b)) == _outcome(
        db, lambda: _old_new_edge(c, "NoSuchEdgeType", d)
    )
    assert _outcome(db, lambda: a.new_edge("E", b, w=object())) == _outcome(
        db, lambda: _old_new_edge(c, "E", d, w=object())
    )
    unsaved_new = _outcome(db, lambda: db.new_vertex("V").new_edge("E", b))
    unsaved_old = _outcome(db, lambda: _old_new_edge(db.new_vertex("V"), "E", d))
    assert unsaved_new == unsaved_old
