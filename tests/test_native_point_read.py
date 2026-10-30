"""The short call path of a point read on the native build (conftest.py collects this file on the native backend only).

``query("sql", "SELECT FROM T WHERE id = ?", v).first()`` is the call a key-value style workload makes a million times. Its Python part
is short on purpose: one int or str parameter is written as JSON text directly (``codec.encode_params``), ``arcadedb_n_exec`` is called
without the by-name lookup (``lib.exec_query``), ``first()`` takes the row that is already in the first batch, and a row is wrapped as a
record without an import statement. These tests pin that every result is the one the long path gives (the oracle is the general encoder,
the iterator and ``to_list()``), that every other parameter kind and every other shape of result still takes the long path, and that the
short path is the one taken (a spy on ``lib``), so it cannot silently stop being used.
"""

import decimal
import enum
import json
from datetime import date, datetime, timezone

import numpy as np
import pytest

from arcadedb._native import codec, lib
from arcadedb.exceptions import ArcadeDBError
from arcadedb.graph import Document, Edge, Vertex


class _Color(enum.IntEnum):
    RED = 1


class _Name(str):
    pass


ONE_PARAMETER = [
    0,
    7,
    -1,
    2**31,
    2**63,  # beyond a long: the engine decides, the text is the same digits
    -(2**100),
    "",
    "plain",
    'quote " backslash \\ slash / newline \n tab \t control \u0001 \u007f',
    "café 日本語 \U0001f600 end",  # 2-, 3- and 4-byte UTF-8 sequences, escaped as é ... and a surrogate pair
    "\ud800",  # a lone surrogate: json.dumps writes the escape, so does the short path
    "\x00 nul",
    "</script> 'single' &amp;",
]

# every parameter the short path must leave to the general encoder
GENERAL = [
    True,
    False,
    None,
    1.5,
    float("inf"),
    decimal.Decimal("1.50"),
    _Color.RED,  # an int subclass
    _Name("subclass"),  # a str subclass
    np.int64(5),
    np.float32(0.5),
    datetime(2026, 10, 30, 12, 30, tzinfo=timezone.utc),
    date(2026, 10, 30),
    b"\x00\x01",
    {"a": 1},
    [1, 2, 3],
    (1, 2),
    codec.RID("#1:2"),
]


@pytest.fixture
def oracle(monkeypatch):
    """encode_params as it was before the short path: the general encoder for every parameter."""

    def general(args, binary=False):
        monkeypatch.setattr(codec, "_fast", False)
        try:
            return codec.encode_params(args, binary)
        finally:
            monkeypatch.setattr(codec, "_fast", True)

    return general


@pytest.mark.parametrize("value", ONE_PARAMETER, ids=repr)
@pytest.mark.parametrize("binary", [False, True])
def test_one_int_or_str_parameter_is_the_text_json_dumps_gives(value, binary):
    expected = json.dumps({"pos": [value]})
    for args in (
        (value,),
        ([value],),
        ((value,),),
    ):  # a lone list or tuple is the positional list itself
        got = codec.encode_params(args, binary)
        assert got == ((expected, None, 0) if binary else expected)


@pytest.mark.parametrize("value", ONE_PARAMETER + GENERAL, ids=repr)
@pytest.mark.parametrize("binary", [False, True])
def test_every_one_parameter_encodes_as_the_general_path_does(value, binary, oracle):
    for args in ((value,), ([value],)):
        assert codec.encode_params(args, binary) == oracle(args, binary)


def test_other_parameter_counts_and_shapes_are_unchanged(oracle):
    for args in (
        (),
        (1, 2),
        ("a", 2),
        ({"x": 1},),
        ([],),
        ([1, "a"],),
        ((1.5, "a", None),),
        ([1] * 12,),
    ):
        assert codec.encode_params(args) == oracle(args)
        assert codec.encode_params(args, True) == oracle(args, True)


def test_the_short_path_needs_the_exact_type(oracle):
    # True is an int to isinstance, 1 to '%d' would be wrong ("1" instead of true)
    assert codec.encode_params((True,)) == '{"pos": [true]}'
    assert codec.encode_params((_Color.RED,)) == oracle((_Color.RED,))


def _vectors_db(db, n=300):
    db.command("sql", "CREATE DOCUMENT TYPE P")
    db.command("sql", "CREATE PROPERTY P.id INTEGER")
    db.command("sql", "CREATE PROPERTY P.name STRING")
    db.command("sql", "CREATE PROPERTY P.vec ARRAY_OF_FLOATS")
    db.command("sql", "CREATE INDEX ON P (id) UNIQUE")
    db.insert_many(
        "P",
        [
            {
                "id": i,
                "name": f"name {i} café \U0001f600",
                "vec": [np.float32((i * 16 + k) / 1000.0) for k in range(16)],
            }
            for i in range(n)
        ],
    )
    return db


def _long_path(db, sql, *args):
    """The rows through the iterator and to_list(), never first()."""
    rows = db.query("sql", sql, *args).to_list()
    via_iter = [r.to_dict() for r in db.query("sql", sql, *args)]
    assert rows == via_iter
    return rows


def test_a_point_read_gives_the_row_the_long_path_gives(temp_db):
    db = _vectors_db(temp_db)
    for key in (0, 17, 299):
        row = db.query("sql", "SELECT FROM P WHERE id = ?", key).first()
        assert row.to_dict() == _long_path(db, "SELECT FROM P WHERE id = ?", key)[0]
        assert row.get("id") == key
        assert len(row.get("vec")) == 16  # arrived in the float32 tail of the reply
    for param in (
        [17],
        (17,),
    ):  # the list and tuple shapes of one parameter are the same read
        got = db.query("sql", "SELECT FROM P WHERE id = ?", param).first().to_dict()
        assert got == _long_path(db, "SELECT FROM P WHERE id = ?", 17)[0]
    assert db.query("sql", "SELECT FROM P WHERE id = ?", 10_000).first() is None
    assert db.query("sql", "SELECT FROM P WHERE id = ?", -1).first() is None


def test_a_str_point_read_gives_the_row_the_long_path_gives(temp_db):
    db = _vectors_db(temp_db, 20)
    name = "name 7 café \U0001f600"
    row = db.query("sql", "SELECT FROM P WHERE name = ?", name).first()
    assert row.get("id") == 7
    assert row.to_dict() == _long_path(db, "SELECT FROM P WHERE name = ?", name)[0]
    # a text the JSON writer must escape is read back as it was written
    odd = 'say "hi" \\ \n \u0001 \U0001f600'
    with db.transaction():
        db.command("sql", "INSERT INTO P SET id = 500, name = ?", odd)
    assert db.query("sql", "SELECT FROM P WHERE name = ?", odd).first().get("id") == 500


def test_a_projection_and_a_scalar_parameter_use_the_same_path(temp_db):
    db = _vectors_db(temp_db, 20)
    row = db.query(
        "sql", "SELECT name, id + ? AS shifted FROM P WHERE id = 3", 10
    ).first()
    assert row.to_dict() == {"name": "name 3 café \U0001f600", "shifted": 13}
    assert row.get_element() is None  # a projection is not a record


def test_first_closes_the_result_and_releases_a_cursor_that_is_not_drained(temp_db):
    db = _vectors_db(temp_db, 600)
    rs = db.query(
        "sql", "SELECT FROM P ORDER BY id"
    )  # 256 rows in the first batch, 344 in the cursor
    assert rs._more is True
    row = rs.first()
    assert row.get("id") == 0
    assert rs._closed and rs._more is False and not rs._buf
    with pytest.raises(ArcadeDBError, match="closed before all its rows"):
        rs.first()
    one = db.query("sql", "SELECT FROM P WHERE id = ?", 5)
    assert one.first().get("id") == 5
    assert one._closed
    with pytest.raises(ArcadeDBError, match="closed before all its rows"):
        one.first()  # as the iterator path: the one row came out, the result was closed, not drained


def test_first_of_a_multi_row_result_is_the_first_row(temp_db):
    db = _vectors_db(temp_db, 50)
    rows = _long_path(db, "SELECT FROM P WHERE id >= ? ORDER BY id", 10)
    assert (
        db.query("sql", "SELECT FROM P WHERE id >= ? ORDER BY id", 10).first().to_dict()
        == rows[0]
    )


def test_a_command_point_read_still_takes_the_lazy_result(temp_db):
    db = _vectors_db(temp_db, 20)
    row = db.command("sql", "SELECT FROM P WHERE id = ?", 4).first()
    assert row.get("id") == 4
    assert db.command("sql", "SELECT FROM P WHERE id = ?", 400).first() is None


def test_first_after_the_database_is_closed_raises_as_before(temp_db_path):
    import arcadedb

    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE Q")
        with db.transaction():
            db.command("sql", "INSERT INTO Q SET id = 1")
        rs = db.query("sql", "SELECT FROM Q WHERE id = ?", 1)
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        rs.first()


def test_a_row_wraps_as_the_record_of_its_kind(temp_db):
    db = temp_db
    db.command("sql", "CREATE VERTEX TYPE V1")
    db.command("sql", "CREATE EDGE TYPE E1")
    db.command("sql", "CREATE DOCUMENT TYPE D1")
    db.command("sql", "CREATE PROPERTY V1.id INTEGER")
    db.command("sql", "CREATE PROPERTY D1.id INTEGER")
    with db.transaction():
        for i in (1, 2):
            db.command("sql", "CREATE VERTEX V1 SET id = ?", i)
        db.command(
            "sql",
            "CREATE EDGE E1 FROM (SELECT FROM V1 WHERE id = 1) TO (SELECT FROM V1 WHERE id = 2) SET w = 5",
        )
        db.command("sql", "INSERT INTO D1 SET id = 1")

    vrow = db.query("sql", "SELECT FROM V1 WHERE id = ?", 1).first()
    vertex = vrow.get_vertex()
    assert isinstance(vertex, Vertex) and vertex.get("id") == 1
    assert vertex.get_rid() == vrow.get_rid() and vertex.get_type_name() == "V1"
    assert vrow.get_element().get_rid() == vrow.get_rid()
    assert vrow.get_edge() is None

    erow = db.query("sql", "SELECT FROM E1").first()
    edge = erow.get_edge()
    assert isinstance(edge, Edge) and edge.get("w") == 5
    assert erow.get_vertex() is None

    drow = db.query("sql", "SELECT FROM D1 WHERE id = ?", 1).first()
    doc = drow.get_element()
    assert isinstance(doc, Document) and not isinstance(doc, (Vertex, Edge))
    assert doc.get("id") == 1 and drow.get_vertex() is None

    # each record has its own stand-in for the Java object, as before
    other = db.query("sql", "SELECT FROM V1 WHERE id = ?", 2).first().get_vertex()
    assert vertex._java_document is not other._java_document
    assert type(vertex._java_document) is type(other._java_document)


def test_lookup_by_key_is_unchanged(temp_db):
    db = _vectors_db(temp_db, 20)
    rec = db.lookup_by_key("P", ["id"], [9])
    assert rec.get("id") == 9 and rec.get_type_name() == "P"
    assert db.lookup_by_key("P", ["id"], [999]) is None


def test_a_point_read_takes_the_short_path_and_other_calls_do_not(temp_db, monkeypatch):
    db = _vectors_db(temp_db, 20)
    calls = []
    real_exec_query, real_call_bin = lib.exec_query, lib.call_bin
    monkeypatch.setattr(
        lib, "exec_query", lambda *a: calls.append("exec_query") or real_exec_query(*a)
    )
    monkeypatch.setattr(
        lib, "call_bin", lambda name, *a: calls.append(name) or real_call_bin(name, *a)
    )

    db.query("sql", "SELECT FROM P WHERE id = ?", 3).first()
    db.query("sql", "SELECT FROM P WHERE name = ?", "name 3").first()
    db.query("sql", "SELECT FROM P WHERE id = :id", {"id": 3}).first()
    db.query("sql", "SELECT FROM P").to_list()
    assert (
        calls == ["exec_query"] * 4
    )  # a text-only parameter list, whatever its kind: no float side buffer

    calls.clear()
    vec = np.arange(16, dtype=np.float32)
    db.query("sql", "SELECT FROM P WHERE vec = ?", vec).first()
    assert calls == [
        "arcadedb_n_exec_bin"
    ]  # a float array parameter travels in the side buffer: the long path


def test_exec_query_replies_as_call_bin_does(temp_db):
    db = _vectors_db(temp_db, 20)
    sql = lib.bc("SELECT FROM P WHERE id = ?")
    for flags in (0, 4):  # 4: float arrays come back in the float32 tail
        text, _, _ = codec.encode_params((6,), True)
        short = lib.exec_query(db._h, flags, lib.bc("sql"), sql, lib.b(text), 256)
        long = lib.call_bin(
            "arcadedb_n_exec", db._h, flags, b"sql", sql, lib.b(text), 256
        )
        a, b = codec.loads(*short), codec.loads(*long)
        for reply in (a, b):
            assert reply.pop("rs") is not None
        assert a == b and a["ok"] and len(a["rows"]) == 1
        assert (short[1] is None) == (long[1] is None)
        if flags == 4:
            assert short[1] is not None and np.array_equal(short[1], long[1])
