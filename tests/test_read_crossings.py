"""Crossings of the JVM read path (one JPype call is a few microseconds of dispatch).

Each read the bindings make used to cross into the JVM several times: ``Document.get_raw`` asked
``has()`` and then ``get()``, ``ResultSet.one()`` and ``count()`` crossed per row, ``lookup_by_key``
built two Java arrays per call, and the bookkeeping around ``with db.transaction()`` crossed twice
just to take the transaction-stack mark. Every test here pins two things: the number of calls into
the JVM (counted on a proxy of the Java object or bridge class, the technique of
``test_percall_calls.py``) and that the answer is the one the plain path gives (same value, same
errors, same result-set state), so a count cannot be bought with a change of behaviour.
"""

import pytest

from arcadedb import core, results


class Counting:
    """Counts the method calls made on a Java object or a Java class.

    Arguments that are themselves a ``Counting`` are unwrapped before the call, so the proxy of the
    database can be handed to a bridge class as it is. Return values are the real objects.
    """

    def __init__(self, target, log, label):
        self._target = target
        self._log = log
        self._label = label

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self._log.append(f"{self._label}.{name}")
            args = [a._target if isinstance(a, Counting) else a for a in args]
            return attr(*args, **kwargs)

        return call


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE T")
    temp_db.command("sql", "CREATE PROPERTY T.k INTEGER")
    temp_db.command("sql", "CREATE INDEX ON T (k) UNIQUE")
    with temp_db.transaction():
        for key in range(5):
            temp_db.command("sql", "INSERT INTO T SET k = ?, v = ?", key, key * 10)
    return temp_db


@pytest.fixture
def log(db, monkeypatch):
    """The list every counted call is appended to; the database and the bridge classes are counted."""
    calls = []
    real = {name: results._bridge_class(name) for name in ("RowAccess", "DbCalls")}
    monkeypatch.setattr(db, "_java_db", Counting(db._java_db, calls, "db"))
    for name, cls in real.items():
        monkeypatch.setitem(results._BRIDGE_CLASSES, name, Counting(cls, calls, name))
    monkeypatch.setattr(core, "_DB_CALLS", Counting(real["DbCalls"], calls, "DbCalls"))
    return calls


def _no_bridge(monkeypatch, *names):
    """Send ``names`` down the plain per-row path (the bridge class reported missing)."""
    for name in names:
        monkeypatch.setitem(results._BRIDGE_CLASSES, name, None)
    monkeypatch.setattr(core, "_DB_CALLS", False)


# ---------------------------------------------------------------- Document.get_raw


def test_get_raw_is_one_crossing_and_one_buffer_scan(db):
    doc = db.query("sql", "SELECT FROM T WHERE k = 3").first().get_element()
    calls = []
    doc._java_document = Counting(doc._java_document, calls, "doc")
    assert doc.get_raw("v") == 30
    assert calls == ["doc.get"]


def test_get_raw_absent_property_is_none_in_one_crossing(db):
    doc = db.query("sql", "SELECT FROM T WHERE k = 3").first().get_element()
    calls = []
    doc._java_document = Counting(doc._java_document, calls, "doc")
    assert doc.get_raw("missing") is None
    assert doc.get("missing") is None
    assert calls == ["doc.get", "doc.get"]


def test_get_raw_null_valued_and_absent_properties_both_read_none(db):
    """A property set to null and one never set read the same through get(); has() still tells them apart."""
    with db.transaction():
        doc = db.new_document("T")
        doc.set("k", 99)
        doc.set("nothing", None)
        doc.save()
        rid = doc.get_identity()
    with db.transaction():
        mutable = db.new_document("T")
        mutable.set("k", 100)
        mutable.set("nothing", None)
        assert mutable.get_raw("nothing") is None
        assert mutable.get_raw("never") is None
        assert mutable.has_property("nothing") is True
        assert mutable.has_property("never") is False
        mutable.save()
    stored = db.lookup_by_rid(str(rid))
    assert stored.get_raw("k") == 99
    assert stored.get_raw("never") is None


def test_get_raw_on_vertex_and_edge(db):
    db.command("sql", "CREATE VERTEX TYPE V")
    db.command("sql", "CREATE EDGE TYPE E")
    with db.transaction():
        a = db.command("sql", "CREATE VERTEX V SET name = 'a'").first().get_vertex()
        b = db.command("sql", "CREATE VERTEX V SET name = 'b'").first().get_vertex()
        edge = a.new_edge("E", b, w=7)
    vertex = db.query("sql", "SELECT FROM V WHERE name = 'a'").first().get_vertex()
    assert vertex.get_raw("name") == "a"
    assert vertex.get_raw("nope") is None
    assert edge.get_raw("w") == 7
    assert edge.get_raw("nope") is None
