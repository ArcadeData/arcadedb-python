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
    assert str(df["ts"].dtype).startswith(
        "datetime64[us"
    )  # microseconds on both wheels, whatever pandas infers from datetimes
    # a vector cell is a numpy float32 array, not a Python list (documented on to_dataframe)
    assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in df["v3"])
    import polars as pl

    p = db.query("sql", _sel()).to_polars()
    assert isinstance(p, pl.DataFrame) and p.height == 5000
    assert p.schema["v3"] == pl.Array(pl.Float32, 3)
    assert p["big"].to_list() == [r["big"] for r in rows]


# --- what a column's type depends on: the whole result, not the size of the result or where a batch ends ---------------------


@pytest.mark.parametrize("batch_size", [2**31 - 1, 2**31, 2**32, 10**10])
def test_a_huge_batch_size_means_as_many_as_one_batch_takes(table, batch_size):
    db, rows = table
    t = db.query("sql", _sel()).to_arrow(batch_size=batch_size)
    assert t.num_rows == 5000
    c = db.query("sql", _sel()).to_columns(batch_size=batch_size)
    assert len(c["big"]) == 5000


@pytest.mark.parametrize("batch_size", [0, -1])
def test_a_batch_size_below_one_is_refused(table, batch_size):
    db, _ = table
    with pytest.raises(ValueError):
        db.query("sql", _sel()).to_arrow(batch_size=batch_size)
    with pytest.raises(ValueError):
        db.query("sql", _sel()).to_columns(batch_size=batch_size)


def test_no_rows_with_columns_given_returns_those_columns_empty(table):
    db, _ = table
    sql = "SELECT n, s FROM T WHERE big < 0"
    t = db.query("sql", sql).to_arrow(columns=["s", "n"])
    assert t.column_names == ["s", "n"] and t.num_rows == 0
    assert db.query("sql", sql).to_columns(columns=["s", "n"]) == {"s": [], "n": []}
    assert db.query("sql", sql).to_columns() == {}
    assert db.query("sql", sql).to_arrow().num_columns == 0


@pytest.mark.parametrize("limit", [1, 10, 300])
def test_to_columns_arrays_are_writable(table, limit):
    db, _ = table
    c = db.query(
        "sql",
        f"SELECT big, d, ts, v3 FROM T ORDER BY big LIMIT {limit}",  # nosec B608 - an integer of the test
    ).to_columns()
    for name in ("big", "d", "ts", "v3"):
        assert c[name].flags.writeable, (name, limit)
        c[name][0] = c[name][0]


@pytest.mark.parametrize("rows", [255, 256, 257, 511, 512, 513])
def test_sizes_around_the_first_reply_and_two_of_them(table, rows):
    db, stored = table
    t = db.query("sql", _sel() + f" LIMIT {rows}").to_arrow()
    assert t.num_rows == rows
    assert t.column("big").to_pylist() == [r["big"] for r in stored[:rows]]
    assert t.column("v9").to_pylist() == [r["v9"] for r in stored[:rows]]
    assert t.schema.field("v9").type == pa.list_(pa.float32(), 9)
    assert t.schema.field("s").type == pa.string()


def test_two_cursors_open_at_once_and_a_cursor_closed_early(table):
    db, rows = table
    one = db.query("sql", _sel())
    two = db.query("sql", _sel())
    assert next(one).get("big") == rows[0]["big"]
    assert next(two).get("big") == rows[0]["big"]
    t_two = two.to_arrow()
    t_one = one.to_arrow()
    assert t_one.column("big").to_pylist() == [r["big"] for r in rows[1:]]
    assert t_two.column("big").to_pylist() == [r["big"] for r in rows[1:]]
    partly = db.query("sql", _sel())
    for _ in range(300):
        next(partly)
    partly.close()
    fresh = db.query("sql", _sel() + " LIMIT 400").to_arrow()
    assert fresh.num_rows == 400


def test_a_cursor_with_nothing_read_closed_many_times_over(table):
    db, _ = table
    # No ORDER BY: a sorted query holds its whole sorted buffer in the engine's query heap budget (2 GB by default) from the moment
    # it starts, so 300 of them open at once exhausted it (macOS arm64 CI and a laptop, 2026-11-10). This test is about cursors that
    # were never read, so the query streams.
    unsorted = "SELECT n, big, d, f, s, ok, ts, v3, v9, v128 FROM T"
    cursors = [db.query("sql", unsorted) for _ in range(300)]
    for cursor in cursors:
        cursor.close()
    assert db.query("sql", _sel() + " LIMIT 3").to_arrow().num_rows == 3


@pytest.fixture
def declared(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Dec")
    db.command("sql", "CREATE PROPERTY Dec.i INTEGER")
    db.command("sql", "CREATE PROPERTY Dec.arr ARRAY_OF_FLOATS")
    db.command("sql", "CREATE PROPERTY Dec.dbl ARRAY_OF_DOUBLES")
    db.command("sql", "CREATE PROPERTY Dec.lst LIST")
    rows = [
        {
            "i": i,
            "arr": [0.5 * i, 1.0, 2.0],
            "dbl": [0.5 * i, 1.5, 2.5],
            "lst": [0.5 * i, 1.0, 2.0],
        }
        for i in range(3000)
    ]
    db.insert_many("Dec", rows)
    return db, rows


@pytest.mark.parametrize("rows", [10, 300, 3000])
def test_a_vector_column_is_a_declared_array_whatever_the_result_size(declared, rows):
    db, stored = declared
    sql = f"SELECT i, arr, dbl, lst FROM Dec ORDER BY i LIMIT {rows}"  # nosec B608 - an integer of the test
    c = db.query("sql", sql).to_columns()
    assert isinstance(c["arr"], np.ndarray) and c["arr"].dtype == np.float32
    assert isinstance(c["dbl"], np.ndarray) and c["dbl"].dtype == np.float64
    assert isinstance(c["lst"], list), type(
        c["lst"]
    )  # a LIST of doubles is lists, not a float32 matrix that happens to fit
    assert c["lst"] == [r["lst"] for r in stored[:rows]]
    t = db.query("sql", sql).to_arrow()
    assert t.schema.field("arr").type == pa.list_(pa.float32(), 3)
    assert t.schema.field("dbl").type == pa.list_(pa.float64(), 3)
    assert t.schema.field("lst").type == pa.list_(pa.float64())
    assert t.column("dbl").to_pylist() == [r["dbl"] for r in stored[:rows]]


def test_a_double_array_stays_double_after_rows_read_past_the_first_reply(declared):
    db, stored = declared
    rs = db.query("sql", "SELECT i, dbl, arr FROM Dec ORDER BY i")
    for _ in range(300):
        next(rs)
    t = rs.to_arrow()
    assert t.schema.field("dbl").type == pa.list_(pa.float64(), 3)
    assert t.schema.field("arr").type == pa.list_(pa.float32(), 3)
    assert t.column("dbl").to_pylist() == [r["dbl"] for r in stored[300:]]


def test_a_property_that_is_numeric_from_row_30000_on_has_one_type(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Late")
    db.command("sql", "CREATE PROPERTY Late.i INTEGER")
    rows = [{"i": i} for i in range(30000)] + [
        {"i": i, "late": i * 2} for i in range(30000, 30100)
    ]
    db.insert_many("Late", rows)
    sql = "SELECT i, late FROM Late ORDER BY i"
    t = db.query("sql", sql).to_arrow(batch_size=25_000, columns=["i", "late"])
    assert t.schema.field("late").type == pa.int64()
    late = t.column("late").to_pylist()
    assert late[:30000] == [None] * 30000 and late[30000:] == [
        i * 2 for i in range(30000, 30100)
    ]
    c = db.query("sql", sql).to_columns(batch_size=25_000, columns=["i", "late"])
    assert isinstance(c["late"], np.ndarray) and c["late"].dtype == np.float64
    assert np.isnan(c["late"][:30000]).all()
    assert list(c["late"][30000:]) == [i * 2 for i in range(30000, 30100)]


@pytest.mark.parametrize("rows", [10, 300])
def test_an_integer_beyond_2_53_next_to_a_float_stays_exact(temp_db, rows):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Mix")
    db.command("sql", "CREATE PROPERTY Mix.i INTEGER")
    big = 2**60 + 1
    db.insert_many(
        "Mix", [{"i": i, "x": big if i % 2 == 0 else 1.5} for i in range(rows)]
    )
    c = db.query("sql", "SELECT i, x FROM Mix ORDER BY i").to_columns()
    want = [big if i % 2 == 0 else 1.5 for i in range(rows)]
    assert list(c["x"]) == want
    assert all(type(a) is type(b) for a, b in zip(c["x"], want))
    t = db.query("sql", "SELECT i, x FROM Mix ORDER BY i").to_arrow()
    assert t.column("x").to_pylist() == [
        str(v) for v in want
    ]  # no common numeric type: exact text, as across batches


# --- to_dataframe and an integer beyond 2**53 (#340) -----------------------------------------------------------------------
# to_columns() gives the exact Python list and to_arrow() the exact text; to_dataframe() handed that list to pandas, which infers
# float64 for ints and floats and rounded the integer. Each test reads the frame, so the same literal holds on both wheels.


def _mix_frame(db, values):
    """The ``to_dataframe()`` of ``values`` stored in property ``x`` of one document each, in order (None = the row lacks ``x``)."""
    db.command("sql", "CREATE DOCUMENT TYPE Mix")
    db.command("sql", "CREATE PROPERTY Mix.i INTEGER")
    db.insert_many(
        "Mix",
        [{"i": i} if v is None else {"i": i, "x": v} for i, v in enumerate(values)],
    )
    return db.query("sql", "SELECT i, x FROM Mix ORDER BY i").to_dataframe()


def _same_values_and_types(have, want):
    return list(have) == list(want) and all(
        type(a) is type(b) for a, b in zip(have, want)
    )


@pytest.mark.parametrize("layout", ["alternating", "blocks"])
@pytest.mark.parametrize("rows", [10, 300])
def test_to_dataframe_keeps_an_integer_beyond_2_53_next_to_a_float(
    temp_db, rows, layout
):
    big = 2**60 + 1
    if layout == "alternating":
        want = [big if i % 2 == 0 else 1.5 for i in range(rows)]
    else:
        want = [big if i < rows // 2 else 1.5 for i in range(rows)]
    df = _mix_frame(temp_db, want)
    assert df["x"].dtype == object  # float64 would round the integer
    assert _same_values_and_types(df["x"].tolist(), want)  # an int stays an int
    assert df["i"].dtype == np.int64 and df["i"].tolist() == list(range(rows))


@pytest.mark.parametrize(
    "big", [2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63) + 1, -(2**63)]
)
def test_to_dataframe_keeps_the_ends_of_int64_next_to_a_float(temp_db, big):
    want = [big, 1.5, big, -0.25]
    df = _mix_frame(temp_db, want)
    assert df["x"].dtype == object
    assert _same_values_and_types(df["x"].tolist(), want)


def test_to_dataframe_mixed_column_with_a_null_keeps_the_null_as_none(temp_db):
    big = 2**60 + 1
    df = _mix_frame(temp_db, [big, None, 1.5, -(2**63)])
    assert df["x"].dtype == object
    assert df["x"].tolist() == [big, None, 1.5, -(2**63)]
    assert df["x"][1] is None  # not NaN: a missing value is None as in a text column
    assert type(df["x"][0]) is int and type(df["x"][2]) is float


def test_to_dataframe_integer_beyond_2_53_next_to_text_and_a_float(temp_db):
    want = [2**60 + 1, "text", 1.5, -(2**63)]
    df = _mix_frame(temp_db, want)
    assert df["x"].dtype == object
    assert _same_values_and_types(df["x"].tolist(), want)


def test_to_dataframe_integers_alone_stay_int64_even_beyond_2_53(temp_db):
    want = [2**60 + 1, -(2**63), 2**63 - 1, 7]
    df = _mix_frame(temp_db, want)
    assert df["x"].dtype == np.int64  # unchanged: exact, no float in the column
    assert df["x"].tolist() == want


def test_to_dataframe_integers_up_to_2_53_next_to_floats_stay_float64(temp_db):
    # every integer here is an exact double, so the usual numeric column is kept (the dtype does not change for this shape)
    df = _mix_frame(temp_db, [1, 1.5, 2**53, -(2**53)])
    assert df["x"].dtype == np.float64
    assert df["x"].tolist() == [1.0, 1.5, 2.0**53, -(2.0**53)]


def test_to_dataframe_keeps_the_integer_when_the_floats_are_in_a_later_frame(temp_db):
    # docs, "What the column calls return": the integer stays exact also when it and the floats are in different frames of a
    # large result. to_dataframe() reads 25,000 rows in one frame, so rows 0 to 24,999 are integers and the last 10 are floats
    big = 2**60 + 1
    want = [big + i for i in range(25_000)] + [0.5 + i for i in range(10)]
    df = _mix_frame(temp_db, want)
    assert df["x"].dtype == object
    assert _same_values_and_types(df["x"].tolist(), want)
    # the same column in to_columns() is a list of exact values, not a float64 array
    columns = temp_db.query("sql", "SELECT i, x FROM Mix ORDER BY i").to_columns()
    assert isinstance(columns["x"], list)
    assert _same_values_and_types(columns["x"], want)


def test_an_integer_beyond_2_53_in_a_column_with_a_null_is_rounded_but_in_arrow(
    temp_db,
):
    # docs, "What the column calls return": the limit that stays. An integer column with a null is float64 with NaN in
    # to_columns() and to_dataframe() (the pandas rule), which cannot hold 2**60 + 1; to_arrow() keeps int64 with a null
    big = 2**60 + 1
    df = _mix_frame(temp_db, [big, None])
    assert df["x"].dtype == np.float64 and np.isnan(df["x"][1])
    assert int(df["x"][0]) == 2**60  # rounded to the nearest double
    query = "SELECT i, x FROM Mix ORDER BY i"
    columns = temp_db.query("sql", query).to_columns()
    assert columns["x"].dtype == np.float64 and int(columns["x"][0]) == 2**60
    arrow = temp_db.query("sql", query).to_arrow()["x"]
    assert arrow.type == pa.int64() and arrow.to_pylist() == [big, None]


def test_to_dataframe_of_an_empty_result_is_an_empty_frame(temp_db):
    _mix_frame(temp_db, [2**60 + 1, 1.5])
    df = temp_db.query("sql", "SELECT i, x FROM Mix WHERE i < 0").to_dataframe()
    assert len(df) == 0 and list(df.columns) == []


def test_frames_of_an_integer_beyond_2_53_and_frames_of_floats_merge_exactly():
    # the column's type belongs to the whole result: two frames, one of int64 (beyond 2**53) and one of float64, must not go
    # through float64 when they merge (np.concatenate would round the integers), and to_dataframe must keep the exact values
    ints = _frame(2, [("n", "i8", [], struct.pack("<2q", 2**60 + 1, -(2**63)))])
    floats = _frame(2, [("n", "f8", [], struct.pack("<2d", 0.5, 1.5))])
    columns = _columnar.merge_columns(["n"], [_columns(ints), _columns(floats)])
    want = [2**60 + 1, -(2**63), 0.5, 1.5]
    assert _same_values_and_types(columns["n"], want)
    df = _columnar.dataframe_from_columns(columns)
    assert df["n"].dtype == object
    assert _same_values_and_types(df["n"].tolist(), want)
    # integers within 2**53 next to floats keep the numeric column of the merge (float64), and so does the frame
    small = _frame(2, [("n", "i8", [], struct.pack("<2q", 1, -(2**53)))])
    columns = _columnar.merge_columns(["n"], [_columns(small), _columns(floats)])
    assert columns["n"].dtype == np.float64
    assert _columnar.dataframe_from_columns(columns)["n"].dtype == np.float64


def test_vectors_whose_dimension_changes_between_frames_are_one_list_column(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Dim")
    db.command("sql", "CREATE PROPERTY Dim.i INTEGER")
    db.command("sql", "CREATE PROPERTY Dim.v ARRAY_OF_FLOATS")
    db.insert_many(
        "Dim",
        [{"i": i, "v": [0.5, 1.0, 2.0] if i < 300 else [0.5, 1.0]} for i in range(600)],
    )
    t = db.query("sql", "SELECT i, v FROM Dim ORDER BY i").to_arrow(batch_size=100)
    assert t.schema.field("v").type == pa.list_(
        pa.float64()
    )  # a lossless widening, wherever the dimension changes
    assert t.column("v").to_pylist() == [[0.5, 1.0, 2.0]] * 300 + [[0.5, 1.0]] * 300


def test_vector_types_unify_by_value_and_dimension_never_to_text():
    def vec(dim, ctype, dtype, rows=2):
        data = np.arange(rows * dim, dtype=dtype).reshape(rows, dim).tobytes()
        return _frame(rows, [("v", ctype, [], data, dim)])

    f3, f2 = _arrow(vec(3, "f4v", "<f4")), _arrow(vec(2, "f4v", "<f4"))
    f8 = _arrow(vec(3, "f8v", "<f8"))
    table = _columnar.merge_arrow(["v"], [f3, f2])
    assert table.schema.field("v").type == pa.list_(pa.float64())
    assert table.column("v").to_pylist() == [[0, 1, 2], [3, 4, 5], [0, 1], [2, 3]]
    table = _columnar.merge_arrow(["v"], [f3, f8])
    assert table.schema.field("v").type == pa.list_(pa.float64(), 3)
    assert table.column("v").to_pylist() == [[0, 1, 2], [3, 4, 5]] * 2


def test_a_null_vector_is_none_in_the_dataframe_form_and_a_nan_row_in_the_matrix():
    data = np.array([[0.5, 1.0], [0, 0], [2.0, 3.0]], dtype="<f4").tobytes()
    frame = _frame(3, [("v", "f4v", [1], data, 2)])
    _, cells = _columns(frame, micros=True)
    cells = cells["v"]
    assert cells[1] is None
    assert all(
        isinstance(c, np.ndarray) and c.dtype == np.float32
        for c in (cells[0], cells[2])
    )
    assert cells[2].tolist() == [2.0, 3.0]
    _, matrix = _columns(frame)
    assert matrix["v"].shape == (3, 2)
    assert np.isnan(
        matrix["v"][1]
    ).all()  # the slot holds zeros in this frame: the decoder makes the null row NaN
    assert not np.isnan(matrix["v"][[0, 2]]).any() and matrix["v"].flags.writeable
    # the same after the merge with a frame that has no null and one that lacks the column
    plain = _frame(1, [("v", "f4v", [], np.array([[7, 8]], dtype="<f4").tobytes(), 2)])
    other = _frame(1, [("w", "i8", [], struct.pack("<q", 1))])
    merged = _columnar.merge_columns(
        ["v", "w"], [_columns(frame), _columns(plain), _columns(other)]
    )["v"]
    assert merged.shape == (5, 2) and merged.flags.writeable
    assert np.isnan(merged[[1, 4]]).all() and merged[3].tolist() == [7, 8]


def test_null_vectors_in_every_columnar_api(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE NV")
    db.command("sql", "CREATE PROPERTY NV.i INTEGER")
    db.command("sql", "CREATE PROPERTY NV.v ARRAY_OF_FLOATS")
    rows = [
        {"i": i, "v": None if i % 5 == 2 else [0.5 * i, 1.0, 2.0]} for i in range(700)
    ]
    db.insert_many("NV", rows)
    sql = "SELECT i, v FROM NV ORDER BY i"
    nulls = [i for i in range(700) if i % 5 == 2]
    df = db.query("sql", sql).to_dataframe()
    cells = df["v"].tolist()
    assert [i for i, c in enumerate(cells) if c is None] == nulls
    assert all(
        isinstance(c, np.ndarray) and c.dtype == np.float32
        for c in cells
        if c is not None
    )
    assert cells[3].tolist() == [1.5, 1.0, 2.0]
    t = db.query("sql", sql).to_arrow()
    assert t.column("v").null_count == len(nulls)
    assert [i for i, v in enumerate(t.column("v").to_pylist()) if v is None] == nulls
    p = db.query("sql", sql).to_polars()
    assert p["v"].null_count() == len(nulls)
    assert p["v"].to_list()[3] == [1.5, 1.0, 2.0]
    m = db.query("sql", sql).to_columns()["v"]
    assert m.shape == (700, 3) and np.isnan(m[nulls]).all()


# --- the rid, li8 and lf8 kinds (refs #388): hand-built frames, no database


def _rids(pairs):
    return b"".join(struct.pack("<2q", b, p) for b, p in pairs)


def _lists(rows, ctype):
    """The buffer of a ``li8`` or ``lf8`` column: int32 offsets (n + 1), the bitmap of null elements, the flat int64 or float64 array.
    A ``None`` row is an empty list (its null bit is the frame's)."""
    offsets, flat, null_elements = [0], [], []
    for row in rows:
        for e in row or []:
            null_elements.append(len(flat) if e is None else None)
            (
                flat.append(0 if ctype == "li8" else math.nan)
                if e is None
                else flat.append(e)
            )
        offsets.append(len(flat))
    bitmap = bytearray((len(flat) + 7) // 8)
    for j in (j for j in null_elements if j is not None):
        bitmap[j >> 3] |= 1 << (j & 7)
    fmt = "q" if ctype == "li8" else "d"
    return (
        struct.pack(f"<{len(offsets)}i", *offsets)
        + bytes(bitmap)
        + struct.pack(f"<{len(flat)}{fmt}", *flat)
    )


INT_LISTS = [[1, 2, 3], [], None, [4, None], [2**62, -(2**62), 2**53 + 1], [None]]
FLOAT_LISTS = [
    [0.5, math.nan],
    [math.inf, -math.inf, -0.0],
    [],
    None,
    [0.1, None],
    [5e-324],
]


def _null_rows(rows):
    return [i for i, row in enumerate(rows) if row is None]


def test_rid_column_is_the_text_of_each_link_in_columns_and_arrow_with_nulls():
    pairs = [(12, 0), (12, 345), (0, 0), (3, 2**40), (0, 7)]
    frame = _frame(5, [("r", "rid", [2], _rids(pairs))])
    texts = ["#12:0", "#12:345", None, "#3:1099511627776", "#0:7"]
    _, parts = _columns(frame)
    assert parts["r"] == texts
    _, chunks = _arrow(frame)
    assert chunks["r"].type == pa.string()
    assert chunks["r"].to_pylist() == texts


def test_rid_column_hands_the_pairs_to_the_wheels_hook_and_nulls_stay_none():
    seen = []

    def hook(pairs, side, col):
        seen.append((pairs.tolist(), side, col["side"]))
        return [("rid", b, p) for b, p in pairs.tolist()]

    frame = _frame(3, [("r", "rid", [1], _rids([(1, 2), (0, 0), (5, 6)]))])
    header, _ = _columnar.read_header(frame)
    header["cols"][0][
        "side"
    ] = 9  # what the bridge writes: where its RID objects start in the side array
    cols = json.dumps(header).encode("utf-8")
    frame = memoryview(
        struct.pack("<i", len(cols))
        + cols
        + bytes(frame[4 + int.from_bytes(frame[:4], "little") :])
    )
    _, parts, _ = _columnar.decode_columns_frame(
        frame, ("side",), _json_values, False, hook
    )
    assert parts["r"] == [("rid", 1, 2), None, ("rid", 5, 6)]
    assert seen == [([[1, 2], [0, 0], [5, 6]], ("side",), 9)]


@pytest.mark.parametrize("ctype,lists", [("li8", INT_LISTS), ("lf8", FLOAT_LISTS)])
def test_number_list_columns_decode_to_the_lists_and_arrow_lists_of_the_json_form(
    ctype, lists
):
    frame = _frame(len(lists), [("x", ctype, _null_rows(lists), _lists(lists, ctype))])
    # what a writer of the json kind wrote for the same column: the decoded values and the Arrow type must be the same
    as_json = json.dumps(lists, allow_nan=True).encode("utf-8")
    old = _frame(len(lists), [("x", "json", _null_rows(lists), as_json)])
    _, have = _columns(frame)
    _, want = _columns(old)
    assert [None if v is None else [norm(x) for x in v] for v in have["x"]] == [
        None if v is None else [norm(x) for x in v] for v in want["x"]
    ]
    assert [[type(x) for x in v or []] for v in have["x"]] == [
        [type(x) for x in v or []] for v in want["x"]
    ]
    _, new_arrow = _arrow(frame)
    _, old_arrow = _arrow(old)
    assert (
        new_arrow["x"].type
        == old_arrow["x"].type
        == pa.list_(pa.int64() if ctype == "li8" else pa.float64())
    )
    assert [
        None if v is None else [norm(x) for x in v] for v in new_arrow["x"].to_pylist()
    ] == [
        None if v is None else [norm(x) for x in v] for v in old_arrow["x"].to_pylist()
    ]


def norm(x):
    """Exact form of a number: NaN equal to NaN, -0.0 apart from 0.0, an int apart from a float."""
    return (type(x).__name__, repr(x))


def test_number_list_values_are_exact_python_numbers():
    _, ints = _columns(
        _frame(2, [("x", "li8", [], _lists([[2**62, -(2**63)], [2**53 + 1]], "li8"))])
    )
    assert ints["x"] == [[2**62, -(2**63)], [2**53 + 1]]
    assert all(type(e) is int for row in ints["x"] for e in row)
    _, floats = _columns(
        _frame(1, [("x", "lf8", [], _lists([[math.nan, -0.0, math.inf]], "lf8"))])
    )
    (row,) = floats["x"]
    assert math.isnan(row[0]) and math.copysign(1, row[1]) == -1 and row[2] == math.inf
    assert all(type(e) is float for e in row)


def test_list_column_of_empty_and_null_rows_and_a_list_of_one_decode():
    frame = _frame(3, [("x", "li8", [1], _lists([[], None, [9]], "li8"))])
    _, parts = _columns(frame)
    assert parts["x"] == [[], None, [9]]
    _, chunks = _arrow(frame)
    assert chunks["x"].to_pylist() == [[], None, [9]]
    assert chunks["x"].null_count == 1
    only_empty = _frame(2, [("x", "lf8", [], _lists([[], []], "lf8"))])
    assert _columns(only_empty)[1]["x"] == [[], []]
    assert _arrow(only_empty)[1]["x"].to_pylist() == [[], []]


def test_the_json_kind_still_reads_lists_maps_and_rids_of_a_writer_that_has_no_binary_kinds():
    values = [[1, 2], {"a": 1}, None, "#1:2"]
    frame = _frame(4, [("m", "json", [2], json.dumps(values).encode("utf-8"))])
    assert _columns(frame)[1]["m"] == values
    assert _arrow(frame)[1]["m"].to_pylist() == [
        str(v) if v is not None else None for v in values
    ]
