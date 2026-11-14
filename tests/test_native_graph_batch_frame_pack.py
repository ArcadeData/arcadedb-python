"""Native GraphBatch.create_vertices sends a chunk of rows as columns and gets the RIDs back in an int64 buffer (ArcadeData/arcadedb-python #388, rank 7).

The route is ``bulk.create_vertices_columns``: ``columnar.rows_to_columns`` and ``columnar.pack`` lay the rows out as the columns of
``arcadedb_n_insert_colp``, one crossing of ``arcadedb_n_gb_colp`` per chunk, and the library writes ``(bucket, position)`` as two int64 per
row into a buffer that Python hands over; no JSON text goes out and none comes back. A chunk the columns cannot carry (rows with other keys or
the same keys in another order, a nested list or map, a column of mixed types, a set, a NaT, a numpy scalar, an integer past 64 bits, a lone
surrogate, an empty chunk) keeps the JSON route it had, so nothing is lost or changed.

No native library is needed here: ``lib.call`` is replaced by a stand-in that records the crossing, reads the addresses it is given and
writes RIDs into the buffer, as the library does. What the library stores is compared between the two routes in
``test_native_graph_batch_frame.py`` (native backend only).
"""

import ctypes
import json
from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from arcadedb._native import bulk, codec, columnar, extras, lib

COLP = "arcadedb_n_gb_colp"
OLD = ("arcadedb_n_gb", "arcadedb_n_gb_bin")
BASE = 2**40  # a position that needs more than 32 bits


class _Db:
    def is_transaction_active(self):
        return False

    def query(self, language, text):  # the schema read of columnar.Declared
        self.schema_reads = getattr(self, "schema_reads", 0) + 1
        rows = [
            {"name": "P", "properties": [{"name": "v", "type": "ARRAY_OF_FLOATS"}]},
        ]
        return SimpleNamespace(to_list=lambda: rows)


class _Library:
    """Stands in for ``lib.call``: records every GraphBatch crossing and answers as the library does."""

    def __init__(self, monkeypatch):
        self.crossings, self.columns, self.next = [], [], 0
        monkeypatch.setattr(lib, "call", self.call)
        monkeypatch.setattr(
            lib, "_lib", SimpleNamespace(arcadedb_n_insert_colp=1, arcadedb_n_gb_colp=1)
        )
        monkeypatch.setattr(lib, "GB_COLP", True)
        monkeypatch.setattr(lib, "GB_BIN", True)
        monkeypatch.setattr(bulk, "_ENABLED", True)

    def call(self, name, *args):
        if name == COLP:
            _gb, type_name, header, spec, nspec, out, slots = args
            head = json.loads(header)
            n = head["n"]
            assert slots == 2 * n and nspec == 6 * len(head["cols"])
            self.crossings.append((name, type_name.decode(), head))
            self.columns.append(
                [
                    ctypes.string_at(spec[6 * c], spec[6 * c + 1])
                    for c in range(len(head["cols"]))
                ]
            )
            buf = (ctypes.c_int64 * slots).from_address(out)
            for i in range(
                n
            ):  # row i is #12:(BASE + next + i), as the engine hands them out
                buf[2 * i], buf[2 * i + 1] = 12, BASE + self.next + i
            self.next += n
            return json.dumps({"ok": True, "n": n})
        assert name in OLD, name
        body = json.loads(args[2])
        rows = (
            len(body["rows"]) if "rows" in body else body["count"]
        )  # the count form has no rows
        rids = [f"#1:{self.next + i}" for i in range(rows)]
        self.crossings.append((name, args[1].decode(), None))
        self.next += rows
        return json.dumps({"ok": True, "rids": rids})

    def entries(self):
        return [c[0] for c in self.crossings if c[0] in (COLP, *OLD)]


@pytest.fixture
def library(monkeypatch):
    return _Library(monkeypatch)


def _batch(chunk=100):
    return extras.NativeGraphBatch(_Db(), 7, commit_every=chunk)


def _expect_columns(library, rows, chunk=100):
    gb = _batch(chunk)
    rids = gb.create_vertices("P", rows)
    assert set(library.entries()) == {COLP}, library.entries()
    return rids


def test_plain_rows_are_one_columnar_crossing_and_the_rids_come_back_as_text(library):
    rows = [
        {"id": i, "name": f"né{i}", "score": i * 0.5, "flag": i % 2 == 0}
        for i in range(30)
    ]
    rids = _expect_columns(library, rows)
    assert library.entries() == [COLP]
    head = library.crossings[0][2]
    assert head["n"] == 30
    assert head["cols"] == [
        ["id", "q", 0],
        ["name", "s", 0],
        ["score", "d", 0],
        ["flag", "b", 0],
    ]
    assert library.columns[0][0] == np.arange(30, dtype=np.int64).tobytes()
    assert type(rids) is list and all(type(r) is str for r in rids)
    assert rids == [f"#12:{BASE + i}" for i in range(30)]  # row order, bucket:position


def test_the_chunks_of_one_call_are_one_crossing_each_and_the_rids_stay_in_order(
    library,
):
    rows = [{"id": i} for i in range(250)]
    rids = _expect_columns(library, rows, chunk=100)
    assert [c[2]["n"] for c in library.crossings] == [100, 100, 50]
    assert rids == [f"#12:{BASE + i}" for i in range(250)]
    library.crossings.clear()
    assert _batch(0)._vertex_chunk == extras.NativeGraphBatch._BULK_CHUNK
    _batch(0).create_vertices("P", rows)  # commit_every 0: all 250 rows in one chunk
    assert [c[2]["n"] for c in library.crossings] == [250]


def test_a_batch_of_one_row_and_no_per_row_python_in_the_codec(library, monkeypatch):
    seen = {"prep": 0, "default": 0}
    for key, module, name in (("prep", codec, "_prep"), ("default", codec, "_default")):
        real = getattr(module, name)

        def counting(*args, _real=real, _key=key, **kwargs):
            seen[_key] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, counting)
    assert _expect_columns(library, [{"id": 1}]) == [f"#12:{BASE}"]
    rows = [
        {"id": i, "name": f"p{i}", "born": datetime(2020, 1, 2, 3, 4, 5, i)}
        for i in range(100)
    ]
    _expect_columns(library, rows)
    assert seen == {
        "prep": 0,
        "default": 0,
    }  # the rows were never walked or written as JSON


@pytest.mark.parametrize(
    "rows, kinds",
    [
        ([{"a": 2**53 + 1}, {"a": -(2**63)}, {"a": 2**63 - 1}], "q"),
        ([{"a": float("nan")}, {"a": float("inf")}, {"a": -0.0}], "d"),
        ([{"a": None}, {"a": 1}, {"a": None}], "q"),
        ([{"a": None}, {"a": None}], "s"),
        ([{"a": date(2020, 1, 2)}, {"a": date(1969, 12, 31)}], "D"),
        ([{"a": datetime(2020, 1, 2, 3, 4, 5, 6)}, {"a": None}], "T"),
        ([{"a": Decimal("1.50")}, {"a": Decimal("-0.001")}], "n"),
        ([{"a": b"\x00\xff"}, {"a": b""}], "x"),
        ([{"a": True}, {"a": False}], "b"),
        ([{"a": "x", "b": 1}, {"a": "y", "b": 2}], "s"),
    ],
)
def test_every_value_a_column_holds_crosses_as_a_column(library, rows, kinds):
    _expect_columns(library, rows)
    assert library.crossings[0][2]["cols"][0][1] == kinds


def test_float32_vectors_cross_as_one_matrix_column(library):
    x = np.random.default_rng(1).standard_normal((12, 8)).astype(np.float32)
    rows = [{"id": i, "embedding": x[i]} for i in range(12)]
    _expect_columns(library, rows)
    cols = library.crossings[0][2]["cols"]
    assert cols == [["id", "q", 0], ["embedding", "f", 8]]
    assert library.crossings[0][2]["last"] == [
        "embedding"
    ]  # stored after the other properties, as the old route did
    assert library.columns[0][1] == x.tobytes()


def test_a_declared_vector_property_is_read_once_for_all_chunks_of_a_call(library):
    db = _Db()
    gb = extras.NativeGraphBatch(db, 7, commit_every=10)
    gb.create_vertices("P", [{"v": [i * 0.5, i + 0.25, 3.0]} for i in range(25)])
    assert library.entries() == [COLP] * 3
    assert db.schema_reads == 1  # the schema of the type, once for the three chunks
    assert [c[2]["cols"][0] for c in library.crossings] == [["v", "f", 3]] * 3
    assert (
        "last" not in library.crossings[0][2]
    )  # a list column stays in the HashMap order, as in the JSON rows
    expected = np.array([[i * 0.5, i + 0.25, 3.0] for i in range(10)], np.float32)
    assert library.columns[0][0] == expected.tobytes()


FALLBACKS = {
    "other keys": [{"a": 1}, {"a": 2, "b": 3}],
    "same keys, another order": [{"a": 1, "b": 2}, {"b": 3, "a": 4}],
    "missing key": [{"a": 1, "b": 2}, {"a": 3}],
    "a none row among dict rows": [{"a": 1}, None],
    "only none rows": [None, None],
    "nested list": [{"a": [1, 2]}, {"a": [3]}],
    "nested map": [{"a": {"x": 1}}, {"a": {"x": 2}}],
    "int and float in one column": [{"a": 1}, {"a": 2.5}],
    "bool and int in one column": [{"a": True}, {"a": 2}],
    "numpy scalars": [{"a": np.int64(1)}, {"a": np.int64(2)}],
    "a set": [{"a": {1, 2}}, {"a": {3}}],
    "integer past 64 bits": [{"a": 1}, {"a": 2**70}],
    "non-finite decimal": [{"a": Decimal("NaN")}, {"a": Decimal(1)}],
    "lone surrogate": [{"a": "ok"}, {"a": "\ud800"}],
    "float64 vectors": [{"a": np.zeros(3)}, {"a": np.ones(3)}],
    "ragged vectors": [{"a": np.zeros(3, np.float32)}, {"a": np.zeros(4, np.float32)}],
    "two-dimensional value": [{"a": np.zeros((2, 2), np.float32)}] * 2,
    "list of floats on an undeclared property": [{"a": [0.5, 1.5]}, {"a": [2.5, 3.5]}],
}


@pytest.mark.parametrize("shape", sorted(FALLBACKS))
def test_a_shape_the_columns_cannot_carry_keeps_the_json_route(library, shape):
    rows = FALLBACKS[shape]
    rids = _batch().create_vertices("P", list(rows))
    assert set(library.entries()) <= set(OLD) and library.entries(), library.entries()
    assert len(rids) == len(rows) and all(type(r) is str for r in rids)


def test_no_rows_and_a_count_keep_the_json_route(library):
    assert _batch().create_vertices("P", []) == []
    assert len(_batch().create_vertices("P", 3)) == 3
    assert set(library.entries()) <= set(OLD)


def test_a_chunk_that_cannot_be_carried_does_not_stop_the_others(library):
    rows = (
        [{"id": i} for i in range(10)]
        + [
            {"id": 10 + i, **({"x": 1} if i == 3 else {})} for i in range(10)
        ]  # a key the others lack
        + [{"id": i} for i in range(20, 30)]
    )
    rids = _batch(chunk=10).create_vertices("P", rows)
    entries = library.entries()
    assert (
        len(entries) == 3
        and entries[0] == COLP
        and entries[1] in OLD
        and entries[2] == COLP
    )
    assert len(rids) == 30


@pytest.mark.parametrize("switch", ["env", "bulk", "library"])
def test_the_switches_turn_the_route_off(library, monkeypatch, switch):
    if switch == "env":
        monkeypatch.setattr(lib, "GB_COLP", False)
    elif switch == "bulk":
        monkeypatch.setattr(bulk, "_ENABLED", False)
    else:
        monkeypatch.setattr(lib, "_lib", SimpleNamespace(arcadedb_n_insert_colp=1))
        monkeypatch.setattr(
            lib, "GB_COLP", False
        )  # what lib._load sets for a library without the entry
    _batch().create_vertices("P", [{"id": 1}, {"id": 2}])
    assert set(library.entries()) <= set(OLD) and library.entries()


def test_a_count_that_disagrees_is_an_error_not_a_short_list(library, monkeypatch):
    real = library.call

    def short(name, *args):
        if name == COLP:
            return json.dumps({"ok": True, "n": 1})
        return real(name, *args)

    monkeypatch.setattr(lib, "call", short)
    with pytest.raises(Exception, match="created 1 vertices for 2 rows"):
        _batch().create_vertices("P", [{"id": 1}, {"id": 2}])


def test_the_rid_text_is_the_engines_text_for_any_bucket_and_position():
    out = np.array([0, 0, 7, 2**62, 2**31 - 1, 2**31], dtype=np.int64)
    assert bulk.rids_of(out) == ["#0:0", f"#7:{2**62}", f"#{2**31 - 1}:{2**31}"]
    assert bulk.rids_of(np.empty(0, dtype=np.int64)) == []


def test_same_order_is_a_rule_of_the_row_check_only():
    rows = [{"a": 1, "b": 2}, {"b": 3, "a": 4}]
    assert columnar.rows_to_columns(rows) == {
        "a": [1, 4],
        "b": [2, 3],
    }  # insert_many's rule: the same keys
    assert columnar.rows_to_columns(rows, same_order=True) is None
    assert columnar.rows_to_columns(rows[:1] * 3, same_order=True) == {
        "a": [1] * 3,
        "b": [2] * 3,
    }


def test_a_pandas_timestamp_with_nanoseconds_keeps_the_json_route_of_a_vertex_load(
    library,
):
    """GraphBatch's JSON route writes a Timestamp with isoformat(), which keeps the nanoseconds, and microsecond ticks cannot; insert_many's
    JSON route keeps microseconds (test_native_columnar_temporal.py), so only this route asks the packer to refuse them.
    """
    import pandas as pd

    declared = columnar.Declared(known={})
    stamp = pd.Timestamp("2020-01-01 00:00:00.000000123")
    assert columnar.pack({"t": [stamp]}, declared) is not None  # insert_many
    assert columnar.pack({"t": [stamp]}, declared, exact_nanos=True) is None
    micros = pd.Timestamp("2020-01-01 00:00:00.000123")
    assert columnar.pack({"t": [micros]}, declared, exact_nanos=True) is not None
    _batch().create_vertices("P", [{"t": stamp}, {"t": stamp}])
    assert set(library.entries()) <= set(OLD) and library.entries()
    library.crossings.clear()
    _batch().create_vertices("P", [{"t": micros}, {"t": micros}])
    assert library.entries() == [COLP]
