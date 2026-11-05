"""The Python part of the point operations of the native build: ``lookup_by_key``, ``lookup_by_rid``, the end vertices of an edge and the edge
lists of a vertex (conftest.py collects this file on the native backend only).

The native library does the work of each of these in one crossing; what remained around it in Python was repeated work: a parameter map with a
record id went through the general encoder (``_prep``, a hook closure, an encoder object) for every call, ``lookup_by_key`` built a cursor
object to hand out one row and imported a module and built the same JSON through the general encoder on each call, a step or an edge end built
the same statement text again for every record, and a step made a result object for every row only to wrap it as a record. These tests pin that
each of those is gone (spies on the codec, the cursor, the row object and the text handed to the engine) and that every result, text and error is
the one the long path gives (the oracle is the general encoder, ``to_list()`` and the JSON text of the engine reply).
"""

import datetime as dt
import decimal
import json

import numpy as np
import pytest

from arcadedb._native import codec, extras, lib, records
from arcadedb._native.database import NativeDatabase, NativeResult, NativeResultSet
from arcadedb.exceptions import ArcadeDBError
from arcadedb.graph import Edge, Vertex


class _Name(str):
    pass


NAMED = [
    {"rid": codec.RID("#12:3")},
    {"rid": codec.RID("#1:0"), "n": 5},
    {"a": 1, "b": "text", "c": None, "d": True, "e": 1.5},
    {"id": -(2**70)},
    {"ünï": "café \U0001f600", "q": 'say "hi" \n \u0001'},
    {},
    {1: "int key"},
    {"x": float("nan")},
]
# every named map the short encoder must leave to the general one
NAMED_GENERAL = [
    {"d": dt.datetime(2026, 11, 4, 12, 30)},
    {"rid": codec.RID("#1:2"), "when": dt.date(2026, 11, 4)},
    {"n": decimal.Decimal("1.5")},
    {"v": np.arange(4, dtype=np.float32)},
    {"l": [1, 2, 3]},
    {"m": {"k": 1}},
    {"s": _Name("subclass")},
    {"i": np.int64(7)},
]


@pytest.fixture
def oracle(monkeypatch):
    """encode_params as it was before the short encoder: the general path for every parameter."""

    def general(args, binary=False):
        monkeypatch.setattr(codec, "_fast", False)
        try:
            return codec.encode_params(args, binary)
        finally:
            monkeypatch.setattr(codec, "_fast", True)

    return general


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("named", NAMED + NAMED_GENERAL, ids=repr)
def test_a_named_map_encodes_as_the_general_path_does(named, binary, oracle):
    for args in ((named,), ([named],)):
        assert codec.encode_params(args, binary) == oracle(args, binary)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("named", NAMED, ids=repr)
def test_a_named_map_of_plain_values_and_rids_uses_no_hook_encoder(
    named, binary, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("the general encoder ran")

    monkeypatch.setattr(codec, "_prep", forbidden)
    monkeypatch.setattr(codec, "_dump", forbidden)
    text = codec.encode_params((named,), binary)
    text = text[0] if binary else text
    assert json.loads(text)["named"].keys() == {str(k) for k in named}


def test_a_named_map_with_another_value_still_takes_the_general_path(monkeypatch):
    seen = []
    real = codec._dump
    monkeypatch.setattr(codec, "_dump", lambda *a: seen.append(a[0]) or real(*a))
    codec.encode_params(({"when": dt.datetime(2026, 11, 4)},), True)
    codec.encode_params(({"rid": codec.RID("#1:2"), "l": [1]},), True)
    assert len(seen) == 2


def test_a_rid_subclass_takes_the_general_path(oracle):
    class Sub(codec.RID):
        pass

    args = ({"rid": Sub("#3:4")},)
    assert codec.encode_params(args, True) == oracle(args, True)


LOOKUPS = [
    (["id"], [9]),
    (["name"], ["café \U0001f600"]),
    (["a", "b"], [1, "x"]),
    ([_Name("id")], [9]),
    (["id"], [None]),
    (["id"], [True]),
    (["id"], [1.5]),
    (["id"], [dt.date(2026, 11, 4)]),
    (["id"], [dt.datetime(2026, 11, 4, 1, 2, 3)]),
    (["id"], [decimal.Decimal("2.50")]),
    (["id"], [codec.RID("#4:5")]),
    (["id"], [np.int64(3)]),
    (["id"], [_Name("sub")]),
    (["id"], [[1, 2, 3]]),
    (["id"], [{1: "x"}]),
    (("id",), (7,)),
    (["k%d" % i for i in range(12)], list(range(12))),
    (["k%d" % i for i in range(12)], [float(i) for i in range(12)]),
    (["id"], [b"\x00\x01"]),
    ([], []),
]


@pytest.mark.parametrize("keys,values", LOOKUPS, ids=repr)
def test_the_lookup_text_is_the_one_the_general_encoder_gives(keys, values):
    expected = extras._enc({"keys": [str(k) for k in keys], "values": list(values)})
    assert codec.encode_lookup(keys, values) == expected


def _documents(db, n=30):
    db.command("sql", "CREATE DOCUMENT TYPE P")
    db.command("sql", "CREATE PROPERTY P.id INTEGER")
    db.command("sql", "CREATE PROPERTY P.name STRING")
    db.command("sql", "CREATE INDEX ON P (id) UNIQUE")
    db.command("sql", "CREATE INDEX ON P (name) UNIQUE")
    with db.transaction():
        for i in range(n):
            db.command("sql", "INSERT INTO P SET id = ?, name = ?", i, f"name {i} é")


def _graph(db, count=8):
    db.command("sql", "CREATE VERTEX TYPE Person")
    db.command("sql", "CREATE PROPERTY Person.id INTEGER")
    db.command("sql", "CREATE INDEX ON Person (id) UNIQUE")
    db.command("sql", "CREATE EDGE TYPE Knows")
    db.command("sql", "CREATE EDGE TYPE Likes")
    with db.transaction():
        people = [db.new_vertex("Person").set("id", i).save() for i in range(count)]
        for i in range(count - 1):
            people[i].new_edge("Knows", people[i + 1], since=2000 + i)
            people[i].new_edge("Likes", people[(i + 2) % count])
    return people


def test_lookup_by_key_makes_no_cursor_and_no_general_encoding(temp_db, monkeypatch):
    db = temp_db
    _documents(db)
    built = []
    real_init = NativeResultSet.__init__
    monkeypatch.setattr(
        NativeResultSet,
        "__init__",
        lambda self, *a, **k: built.append(1) or real_init(self, *a, **k),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("the general encoder ran")

    monkeypatch.setattr(extras, "_enc", forbidden)
    monkeypatch.setattr(codec, "_prep", forbidden)
    record = db.lookup_by_key("P", ["id"], [9])
    assert record.get("id") == 9 and record.get("name") == "name 9 é"
    assert record.get_type_name() == "P"
    assert not built  # a reply that holds the whole result needs no cursor object


def test_lookup_by_key_results_are_unchanged(temp_db):
    db = temp_db
    _documents(db)
    for key in (0, 13, 29):
        record = db.lookup_by_key("P", ["id"], [key])
        assert record.to_dict() == {"id": key, "name": f"name {key} é"}
        assert record.get_rid().startswith("#")
        assert db.lookup_by_rid(record.get_rid()).to_dict() == record.to_dict()
    assert db.lookup_by_key("P", ["name"], ["name 4 é"]).get("id") == 4
    assert db.lookup_by_key("P", ["id"], [999]) is None
    assert db.lookup_by_key("P", ("id",), (5,)).get("id") == 5
    with pytest.raises(ArcadeDBError, match="Failed to lookup by key in 'P'"):
        db.lookup_by_key("P", ["nosuchproperty"], [1])
    with pytest.raises(ArcadeDBError, match="Failed to lookup by key in 'Nope'"):
        db.lookup_by_key("Nope", ["id"], [1])


def test_lookup_by_key_makes_one_crossing(temp_db, monkeypatch):
    db = temp_db
    _documents(db)
    calls = []
    real = lib.call
    monkeypatch.setattr(
        lib, "call", lambda name, *a: calls.append(name) or real(name, *a)
    )
    db.lookup_by_key("P", ["id"], [3])
    db.lookup_by_key("P", ["id"], [-3])
    assert calls == ["arcadedb_n_lookup"] * 2


def test_lookup_by_key_records_hold_their_own_properties(temp_db):
    db = temp_db
    _documents(db)
    first = db.lookup_by_key("P", ["id"], [2])
    second = db.lookup_by_key("P", ["id"], [2])
    assert first._props is not second._props
    first.modify().set("name", "changed")
    assert second.get("name") == "name 2 é"
    changed = first.modify()
    changed.set("name", "other")
    assert first.get("name") == "name 2 é"


def test_lookup_by_key_of_a_vertex_and_a_closed_database(temp_db_path):
    import arcadedb

    with arcadedb.create_database(temp_db_path) as db:
        people = _graph(db)
        vertex = db.lookup_by_key("Person", ["id"], [3])
        assert isinstance(vertex, Vertex) and vertex.get_rid() == people[3].get_rid()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        db.lookup_by_key("Person", ["id"], [3])
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        vertex.get_out_edges()


def test_lookup_by_rid_results_and_errors_are_unchanged(temp_db):
    db = temp_db
    people = _graph(db)
    rid = people[2].get_rid()
    for key in (rid, codec.RID(rid)):
        record = db.lookup_by_rid(key)
        assert isinstance(record, Vertex) and record.get("id") == 2
    edge = people[0].get_out_edges("Knows")[0]
    assert isinstance(db.lookup_by_rid(edge.get_rid()), Edge)
    with pytest.raises(
        ArcadeDBError, match="Failed to lookup RID 'nope': not a record id"
    ):
        db.lookup_by_rid("nope")
    with pytest.raises(
        ArcadeDBError
    ):  # a position the bucket does not hold: the engine's answer, as before
        db.lookup_by_rid(rid[: rid.index(":") + 1] + "99999")


def test_an_edge_end_and_a_step_send_one_text_object_for_every_record(
    temp_db, monkeypatch
):
    db = temp_db
    people = _graph(db)
    seen = []
    original = NativeDatabase._exec

    def recording(self, flags, prefix, language, command, args):
        seen.append(command)
        return original(self, flags, prefix, language, command, args)

    monkeypatch.setattr(NativeDatabase, "_exec", recording)
    edges = [p.get_out_edges("Knows")[0] for p in people[:-1]]
    steps = list(seen)
    del seen[:]
    for edge in edges:
        edge.get_out()
        edge.get_in()
    ends = list(seen)
    del seen[:]
    for person in people:
        person.get_in_edges()
        person.get_both_edges("Knows", "Likes")
    more = list(seen)
    # the same statement is the same object: nothing is built again per call (the engine keys its statement cache by text)
    assert len({id(t) for t in steps}) == 1 and len(set(steps)) == 1
    assert (
        len({id(t) for t in ends[0::2]}) == 1 and len({id(t) for t in ends[1::2]}) == 1
    )
    assert ends[0] != ends[1]
    assert (
        len({id(t) for t in more[0::2]}) == 1 and len({id(t) for t in more[1::2]}) == 1
    )
    assert steps[0] == "SELECT expand(outE('Knows')) FROM :rid"
    assert ends[0] == "SELECT expand(outV()) FROM :rid"
    assert ends[1] == "SELECT expand(inV()) FROM :rid"
    assert more[0] == "SELECT expand(inE()) FROM :rid"
    assert more[1] == "SELECT expand(bothE('Knows', 'Likes')) FROM :rid"


def test_a_label_with_a_quote_is_escaped_and_a_bad_label_fails_as_before(temp_db):
    db = temp_db
    people = _graph(db)
    assert (
        records._edges_sql("outE", ("it's",))
        == "SELECT expand(outE('it\\'s')) FROM :rid"
    )
    for bad in ([1], [["a"]], [None]):
        with pytest.raises(AttributeError, match="replace"):
            people[0].get_out_edges(*bad)


def test_the_statement_texts_kept_are_bounded(monkeypatch):
    monkeypatch.setattr(records, "_EDGES_SQL", {})
    monkeypatch.setattr(records, "_EDGES_SQL_MAX", 3)
    texts = [records._edges_sql("outE", (f"L{i}",)) for i in range(10)]
    assert len(records._EDGES_SQL) == 3
    assert texts[9] == "SELECT expand(outE('L9')) FROM :rid"
    assert records._edges_sql("outE", ("L0",)) is records._edges_sql("outE", ("L0",))


def test_a_step_builds_no_row_objects(temp_db, monkeypatch):
    db = temp_db
    people = _graph(db)
    oracle = [
        (row.get_rid(), row._meta[1], row.to_dict())
        for row in db.query(
            "sql",
            "SELECT expand(outE()) FROM :rid",
            {"rid": codec.RID(people[0].get_rid())},
        )
    ]
    assert len(oracle) == 2
    built = []
    real_init = NativeResult.__init__
    monkeypatch.setattr(
        NativeResult,
        "__init__",
        lambda self, *a, **k: built.append(1) or real_init(self, *a, **k),
    )
    edges = people[0].get_out_edges()
    assert all(isinstance(e, Edge) for e in edges)
    assert (
        not built
    )  # one record per row, no result object between the row and the record
    assert [(e.get_rid(), e.get_type_name(), e.to_dict()) for e in edges] == oracle
    assert people[0].get_out_edges("Likes") and people[7].get_out_edges("Knows") == []
    assert not built


def test_the_end_vertices_and_steps_are_the_records_the_long_path_gives(temp_db):
    db = temp_db
    people = _graph(db)
    by_rid = {p.get_rid(): p for p in people}
    for i, person in enumerate(people[:-1]):
        edge = person.get_out_edges("Knows")[0]
        assert edge.get("since") == 2000 + i
        assert edge.get_out().get_rid() == person.get_rid()
        assert edge.get_in().get_rid() == people[i + 1].get_rid()
        assert edge.get_in().to_dict() == by_rid[edge.get_in().get_rid()].to_dict()
        assert isinstance(edge.get_in(), Vertex)
        assert {e.get_rid() for e in edge.get_in().get_in_edges("Knows")} == {
            edge.get_rid()
        }
    both = people[3].get_both_edges()
    assert len(both) == 4
    one = people[3].get_out_edges("Knows")[0]
    other = people[3].get_out_edges("Knows")[0]
    assert one._props is not other._props  # each record has its own dict
    assert one.modify().set("since", 1).get("since") == 1 and other.get("since") == 2003
    assert people[3].get_out_edges("Knows")[0].get("since") == 2003


def test_a_step_and_an_end_make_one_crossing(temp_db, monkeypatch):
    db = temp_db
    people = _graph(db)
    edge = people[0].get_out_edges("Knows")[0]
    calls = []
    real_exec, real_call_bin, real_call = lib.exec_query, lib.call_bin, lib.call
    monkeypatch.setattr(
        lib, "exec_query", lambda *a: calls.append("exec_query") or real_exec(*a)
    )
    monkeypatch.setattr(
        lib, "call_bin", lambda n, *a: calls.append(n) or real_call_bin(n, *a)
    )
    monkeypatch.setattr(lib, "call", lambda n, *a: calls.append(n) or real_call(n, *a))
    people[0].get_out_edges("Knows")
    edge.get_in()
    edge.get_out()
    assert calls == ["exec_query"] * 3
