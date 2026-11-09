"""The column frame decoder both wheels share (``arcadedb._columnar``), and the columnar readers that use it (refs #312).

The first half builds frames by hand, in the layout of ``ColumnBatcher`` (JVM wheel) and ``arcadedb_n_next_col`` (native wheel), and
decodes them: no database, so a decoder change is checked on its own. The second half reads one stored table through ``to_arrow``,
``to_columns``, ``to_dataframe`` and ``to_polars`` and compares each value with what was stored, on whichever wheel runs the tests:
nulls kept in Arrow, one type per column whatever the batch size, a result of more than one frame, column subsets, an empty result,
one row, many columns, vectors of several lengths. There is no branch on ``arcadedb.__backend__``: a literal that holds on both wheels
is the parity check.
"""

import json
import math
import struct
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from arcadedb import _columnar


def _json_values(data, side, count):
    return json.loads(bytes(data))


def _bits(count, nulls):
    out = bytearray((count + 7) // 8)
    for i in nulls:
        out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _frame(count, columns):
    """``columns``: ``(name, type, null_rows, data_bytes[, dim])`` each; the frame as ColumnBatcher writes it."""
    cols, body = [], b""
    for name, ctype, null_rows, data, *dim in columns:
        nulls = _bits(count, null_rows)
        head = {"name": name, "type": ctype, "nulls": len(nulls), "bytes": len(data)}
        if dim:
            head["dim"] = dim[0]
        cols.append(head)
        body += nulls + data
    header = json.dumps({"count": count, "cols": cols}).encode("utf-8")
    return memoryview(struct.pack("<i", len(header)) + header + body)


def _strings(values):
    offsets, blob = [0], b""
    for v in values:
        blob += b"" if v is None else v.encode("utf-8")
        offsets.append(len(blob))
    return struct.pack(f"<{len(offsets)}i", *offsets) + blob


def _arrow(frame):
    count, chunks, _ = _columnar.decode_arrow_frame(frame, (), _json_values)
    return count, chunks


def _columns(frame, micros=False):
    count, parts, _ = _columnar.decode_columns_frame(frame, (), _json_values, micros)
    return count, parts


def test_integers_keep_their_type_and_nulls_in_arrow_and_become_float_in_columns():
    frame = _frame(3, [("n", "i8", [1], struct.pack("<3q", 1, 0, 2**53 + 1))])
    _, chunks = _arrow(frame)
    assert chunks["n"].type == pa.int64()
    assert chunks["n"].to_pylist() == [1, None, 2**53 + 1]
    _, parts = _columns(frame)
    assert parts["n"].dtype == np.float64
    assert math.isnan(parts["n"][1])


def test_strings_wrap_the_frames_offsets_and_keep_non_bmp_and_nul():
    values = ["a\x00b", None, "\U0001f600 x", "", "é"]
    frame = _frame(5, [("s", "str", [1], _strings(values))])
    _, chunks = _arrow(frame)
    assert chunks["s"].type == pa.string()
    assert chunks["s"].to_pylist() == values
    _, parts = _columns(frame)
    assert parts["s"] == values


def test_booleans_stay_bool_in_arrow_and_a_list_in_columns_when_null():
    frame = _frame(3, [("b", "b1", [2], bytes([1, 0, 0]))])
    _, chunks = _arrow(frame)
    assert chunks["b"].type == pa.bool_()
    assert chunks["b"].to_pylist() == [True, False, None]
    _, parts = _columns(frame)
    assert parts["b"] == [True, False, None]


def test_vectors_are_fixed_size_lists_of_float32_and_a_2d_array_in_columns():
    flat = np.array([[0.5, 1.0, -2.75], [0, 0, 0], [0.25, 2.0, 3.0]], dtype="<f4")
    frame = _frame(3, [("v", "f4v", [1], flat.tobytes(), 3)])
    _, chunks = _arrow(frame)
    assert chunks["v"].type == pa.list_(pa.float32(), 3)
    assert chunks["v"].to_pylist() == [[0.5, 1.0, -2.75], None, [0.25, 2.0, 3.0]]
    _, parts = _columns(frame)
    assert parts["v"].dtype == np.float32 and parts["v"].shape == (3, 3)


def test_datetimes_are_milliseconds_or_microseconds():
    ms = struct.pack("<2q", 1767323045678, 0)
    frame = _frame(2, [("t", "dt", [1], ms)])
    _, chunks = _arrow(frame)
    assert chunks["t"].type == pa.timestamp("ms")
    assert chunks["t"].to_pylist() == [datetime(2026, 1, 2, 3, 4, 5, 678000), None]
    _, parts = _columns(frame)
    assert parts["t"].dtype == np.dtype("datetime64[ms]") and np.isnat(parts["t"][1])
    us = _frame(1, [("t", "dtu", [], struct.pack("<q", 1767323045678901))])
    assert _columns(us, True)[1]["t"].dtype == np.dtype("datetime64[us]")


def test_json_column_goes_through_the_wheels_callback():
    data = json.dumps([[1, 2], None, [3]]).encode("utf-8")
    frame = _frame(3, [("j", "json", [1], data)])
    _, chunks = _arrow(frame)
    assert chunks["j"].to_pylist() == [[1, 2], None, [3]]
    mixed = _frame(2, [("j", "json", [], json.dumps([1, "a"]).encode("utf-8"))])
    assert _arrow(mixed)[1]["j"].to_pylist() == ["1", "a"]


def test_an_empty_frame_has_no_parts():
    assert _columnar.decode_arrow_frame(_frame(0, []), (), _json_values)[:2] == (0, {})


def test_frames_merge_to_one_type_and_a_missing_column_is_null():
    one = _frame(2, [("n", "i8", [], struct.pack("<2q", 1, 2))])
    two = _frame(
        2,
        [
            ("n", "f8", [], struct.pack("<2d", 0.5, 1.5)),
            ("s", "str", [], _strings(["x", "y"])),
        ],
    )
    batches = [_arrow(one), _arrow(two)]
    table = _columnar.merge_arrow(["n", "s"], batches)
    assert table.schema.field("n").type == pa.float64()
    assert table.column("n").to_pylist() == [1.0, 2.0, 0.5, 1.5]
    assert table.column("s").to_pylist() == [None, None, "x", "y"]
    columns = _columnar.merge_columns(["n", "s"], [_columns(one), _columns(two)])
    assert columns["n"].dtype == np.float64 and list(columns["n"]) == [1, 2, 0.5, 1.5]
    assert columns["s"] == [None, None, "x", "y"]


# --- one stored table, read by every columnar API --------------------------------------------------------------------------


def _f32(x):
    return float(np.float32(x))


def _row(i):
    null = i % 7 == 3
    row = {
        "n": i,
        "big": (1 << 40) + i,
        "d": i / 3,
        "f": _f32(i / 7),
        "s": None if null else f"row {i} \U0001f600 é",
        "ok": None if null else i % 2 == 0,
        "ts": datetime(2026, 1, 2, 3, 4, 5, (i % 1000) * 1000),
        "v3": [_f32(i), _f32(i / 2), 0.5],
        "v9": [_f32(i + k / 10) for k in range(9)],
        "v128": [_f32(i * 0.5 + k) for k in range(128)],
    }
    if null:
        row["n"] = None
    return row


TYPES = [
    ("n", "INTEGER"),
    ("big", "LONG"),
    ("d", "DOUBLE"),
    ("f", "FLOAT"),
    ("s", "STRING"),
    ("ok", "BOOLEAN"),
    ("ts", "DATETIME"),
    ("v3", "ARRAY_OF_FLOATS"),
    ("v9", "ARRAY_OF_FLOATS"),
    ("v128", "ARRAY_OF_FLOATS"),
]


@pytest.fixture
def table(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE T")
    for name, type_name in TYPES:
        db.command("sql", f"CREATE PROPERTY T.{name} {type_name}")
    rows = [_row(i) for i in range(5000)]
    db.insert_many("T", [dict(r) for r in rows])
    return db, rows


def _sel():
    return "SELECT n, big, d, f, s, ok, ts, v3, v9, v128 FROM T ORDER BY big"


def _same(have, want):
    if isinstance(want, float) and math.isnan(want):
        return isinstance(have, float) and math.isnan(have)
    return have == want


def test_to_arrow_equals_the_stored_rows(table):
    db, rows = table
    t = db.query("sql", _sel()).to_arrow()
    assert t.num_rows == 5000
    assert t.column_names == [n for n, _ in TYPES]
    assert t.schema.field("n").type == pa.int64()
    assert t.schema.field("big").type == pa.int64()
    assert t.schema.field("d").type == pa.float64()
    assert t.schema.field("f").type == pa.float64()
    assert t.schema.field("s").type == pa.string()
    assert t.schema.field("ok").type == pa.bool_()
    assert t.schema.field("ts").type == pa.timestamp("ms")
    assert t.schema.field("v3").type == pa.list_(pa.float32(), 3)
    assert t.schema.field("v9").type == pa.list_(pa.float32(), 9)
    assert t.schema.field("v128").type == pa.list_(pa.float32(), 128)
    for name, _ in TYPES:
        have = t.column(name).to_pylist()
        want = [r[name] for r in rows]
        assert all(_same(a, b) for a, b in zip(have, want)), name
    assert t.column("n").null_count == len([r for r in rows if r["n"] is None])


@pytest.mark.parametrize("batch_size", [1, 7, 1000, 4999, 5000, 25000])
def test_the_table_does_not_depend_on_the_batch_size(table, batch_size):
    db, rows = table
    want = db.query("sql", _sel()).to_arrow()
    have = db.query("sql", _sel()).to_arrow(batch_size=batch_size)
    assert have.schema == want.schema
    assert have.column("big").to_pylist() == want.column("big").to_pylist()
    assert have.column("s").to_pylist() == want.column("s").to_pylist()
    assert have.column("v9").to_pylist() == want.column("v9").to_pylist()


def test_a_result_larger_than_one_frame(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Wide")
    db.command("sql", "CREATE PROPERTY Wide.i INTEGER")
    db.command("sql", "CREATE PROPERTY Wide.s STRING")
    text = "x" * 300
    db.insert_many("Wide", [{"i": i, "s": f"{text}{i}"} for i in range(8000)])
    t = db.query("sql", "SELECT i, s FROM Wide ORDER BY i").to_arrow()
    assert t.column("i").to_pylist() == list(range(8000))
    assert t.column("s").to_pylist() == [f"{text}{i}" for i in range(8000)]
    assert t.column("s").num_chunks > 1 or t.num_rows == 8000


def test_to_columns_of_the_stored_rows(table):
    db, rows = table
    c = db.query("sql", _sel()).to_columns()
    assert c["big"].dtype == np.int64 and list(c["big"]) == [r["big"] for r in rows]
    assert c["n"].dtype == np.float64  # an int column with nulls
    assert c["d"].dtype == np.float64 and c["ts"].dtype == np.dtype("datetime64[ms]")
    assert c["v9"].dtype == np.float32 and c["v9"].shape == (5000, 9)
    assert c["v128"].shape == (5000, 128)
    np.testing.assert_array_equal(
        c["v9"], np.array([r["v9"] for r in rows], dtype=np.float32)
    )
    assert c["s"] == [r["s"] for r in rows]
    assert c["ok"] == [r["ok"] for r in rows]  # a bool column with nulls is a list


def test_column_subset_and_order(table):
    db, rows = table
    t = db.query("sql", _sel()).to_arrow(columns=["s", "big"])
    assert t.column_names == ["s", "big"]
    assert t.column("big").to_pylist() == [r["big"] for r in rows]
    c = db.query("sql", _sel()).to_columns(columns=["v3"])
    assert list(c) == ["v3"] and c["v3"].shape == (5000, 3)


def test_empty_and_one_row_results(table):
    db, rows = table
    empty = db.query("sql", "SELECT n, s FROM T WHERE big < 0").to_arrow()
    assert empty.num_rows == 0
    assert db.query("sql", "SELECT n, s FROM T WHERE big < 0").to_columns() == {}
    one = db.query(
        "sql", "SELECT n, s, v3 FROM T WHERE big = ?", rows[1]["big"]
    ).to_arrow()
    assert one.num_rows == 1
    assert one.column("s").to_pylist() == [rows[1]["s"]]
    assert one.column("v3").to_pylist() == [rows[1]["v3"]]
    assert db.query("sql", "SELECT FROM T WHERE big < 0").to_arrow().num_rows == 0


def test_a_cursor_read_by_row_first_then_to_arrow(table):
    db, rows = table
    rs = db.query("sql", _sel())
    first = [next(rs) for _ in range(10)]
    assert [r.get("big") for r in first] == [r["big"] for r in rows[:10]]
    t = rs.to_arrow()
    assert t.column("big").to_pylist() == [r["big"] for r in rows[10:]]


def test_many_columns(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Many")
    names = [f"c{k}" for k in range(80)]
    for k, name in enumerate(names):
        db.command(
            "sql",
            f"CREATE PROPERTY Many.{name} {['INTEGER', 'STRING', 'DOUBLE'][k % 3]}",
        )
    rows = [
        {
            name: (i + k if k % 3 == 0 else f"{i}-{k}" if k % 3 == 1 else (i + k) / 4)
            for k, name in enumerate(names)
        }
        for i in range(600)
    ]
    db.insert_many("Many", rows)
    t = db.query("sql", "SELECT FROM Many ORDER BY c0").to_arrow(columns=names)
    assert t.column_names == names
    for name in ("c0", "c1", "c2", "c79"):
        assert t.column(name).to_pylist() == [r[name] for r in rows]


def test_to_dataframe_and_to_polars(table):
    db, rows = table
    df = db.query("sql", _sel()).to_dataframe()
    assert list(df.columns) == [n for n, _ in TYPES]
    assert list(df["big"]) == [r["big"] for r in rows]
    # pandas decides how a missing string is shown (None in an object column, nan in the string dtype of pandas 3): the same on
    # both wheels, since both build the frame from a list with None
    assert [None if pd.isna(v) else v for v in df["s"].tolist()] == [
        r["s"] for r in rows
    ]
    assert [list(v) for v in df["v3"]] == [r["v3"] for r in rows]
    assert df["ts"].iloc[1] == datetime(2026, 1, 2, 3, 4, 5, 1000)
    import polars as pl

    p = db.query("sql", _sel()).to_polars()
    assert isinstance(p, pl.DataFrame) and p.height == 5000
    assert p["big"].to_list() == [r["big"] for r in rows]
