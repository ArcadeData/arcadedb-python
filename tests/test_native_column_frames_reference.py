"""The native column frames give the table the JSON row path gave (refs #312).

Before the frames, ``NativeResultSet.to_arrow``, ``to_columns`` and ``to_dataframe`` read every row as JSON text into one Python dict and
built the columns from those dicts. The module below (``REF_SOURCE``) is that assembly, copied from ``arcadedb._native.columns`` as it was
the day the frames came, and kept here as a private reference: each case reads the same query twice, once through the old route (the
rows drained as dicts, then the reference rules) and once through the frames, and compares the two column by column, value by value.

Equal output is the bar. Two types are different on purpose, because the JVM wheel's frames type them so and both wheels now read frames
with one decoder: a column of ``float[]`` is a ``fixed_size_list<float32>`` (the old rules gave ``list<double>``, the same values: a
float32 widened to float64 is exact), and a column of only nulls is a string column (the old rules gave the null type).

conftest.py collects this file on the native backend only (the old route is the native build's JSON reply).
"""

import math
import types
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

REF_SOURCE = '''"""Rows to columns for the native backend (the columnar path is Python over the rows; the JPype path decodes a Java buffer)."""

from datetime import date, datetime, timezone
from decimal import Decimal


def _names(rows, columns):
    if columns is not None:
        return list(columns)
    seen = {}
    for row in rows:
        for k in row:
            seen.setdefault(k, None)
    return list(seen)


def _kind(values):
    """'b', 'i', 'f', 'dt', 'dec', 'obj' for the non-null values of a column."""
    kinds = set()
    for v in values:
        if v is None:
            continue
        t = type(v)
        if t is bool:
            kinds.add("b")
        elif t is int:
            kinds.add("i")
        elif t is float:
            kinds.add("f")
        elif t is datetime or t is date:
            kinds.add("dt")
        elif t is Decimal:
            kinds.add("dec")
        else:
            kinds.add("obj")
    if not kinds:
        return "obj"
    if len(kinds) == 1:
        return kinds.pop()
    if kinds == {"i", "f"}:
        return "f"
    return "obj"


def _ms(v):
    if isinstance(v, datetime):
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v
    return datetime(v.year, v.month, v.day)


def _float_vectors(values):
    """A column of equal-length float lists (None allowed) that float32 holds exactly, as a (count, dim) float32 array with
    NaN rows for the nulls; None for any other column. Rows cannot say float[] from List<Double>, so only a column whose every
    value survives float32 unchanged is read as a vector column (the JPype path decides on the Java element type).
    """
    import numpy as np

    dim = 0
    for v in values:
        if v is None:
            continue
        if type(v) is not list or not v or not all(type(x) is float for x in v):
            return None
        if dim == 0:
            dim = len(v)
        elif len(v) != dim:
            return None
    if dim == 0:
        return None
    arr = np.full((len(values), dim), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = v
            if not np.array_equal(arr[i].astype(np.float64), np.array(v)):
                return None
    return arr


def to_columns(rows, columns=None):
    import numpy as np

    out = {}
    for name in _names(rows, columns):
        values = [row.get(name) for row in rows]
        if _kind(values) == "obj":
            vectors = _float_vectors(values)
            if vectors is not None:
                out[name] = vectors
                continue
        has_null = any(v is None for v in values)
        kind = _kind(values)
        if kind == "i":
            out[name] = (
                np.array([np.nan if v is None else v for v in values], dtype=np.float64)
                if has_null
                else np.array(values, dtype=np.int64)
            )
        elif kind == "f":
            out[name] = np.array(
                [np.nan if v is None else v for v in values], dtype=np.float64
            )
        elif kind == "b":
            out[name] = values if has_null else np.array(values, dtype=bool)
        elif kind == "dt":
            out[name] = np.array(
                [
                    (
                        np.datetime64("NaT", "ms")
                        if v is None
                        else np.datetime64(_ms(v), "ms")
                    )
                    for v in values
                ],
                dtype="datetime64[ms]",
            )
        elif kind == "dec":
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            out[name] = arr
        else:
            out[name] = values
    return out


def to_arrow(rows, columns=None):
    import pyarrow as pa

    arrays = {}
    for name in _names(rows, columns):
        values = [row.get(name) for row in rows]
        kind = _kind(values)
        if kind == "i":
            arrays[name] = pa.array(values, type=pa.int64())
        elif kind == "f":
            try:
                arrays[name] = pa.array(values, type=pa.float64())
            except (
                pa.ArrowInvalid
            ):  # an int beyond +-2**53 next to floats: the column degrades to exact text
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
        elif kind == "b":
            arrays[name] = pa.array(values, type=pa.bool_())
        elif kind == "dt":
            arrays[name] = pa.array(
                [None if v is None else _ms(v) for v in values], type=pa.timestamp("ms")
            )
        elif kind == "dec":
            try:
                arrays[name] = pa.array(values)
            except (
                pa.ArrowInvalid
            ):  # wider than decimal256 (76 digits): an exact string
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
        else:
            try:
                arrays[name] = pa.array([None if v is None else v for v in values])
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                arrays[name] = pa.array(
                    [None if v is None else str(v) for v in values], type=pa.string()
                )
    return pa.table(arrays)
'''

REF = types.ModuleType("columns_reference")
exec(
    compile(REF_SOURCE, "columns_reference", "exec"), REF.__dict__
)  # nosec B102 - the test's own copy of the old rules


def _f32(x):
    return float(np.float32(x))


def _row(i):
    null = i % 7 == 3
    row = {
        "n": None if null else i,
        "big": (1 << 40) + i,
        "sh": None if null else (i % 2000) - 1000,
        "bt": None if null else (i % 200) - 100,
        "d": i / 3,
        "f": None if null else _f32(i / 7),
        "s": None if null else f"row {i} \U0001f600 \x00 \u00e9",
        "ok": None if null else i % 2 == 0,
        "ts": datetime(2026, 1, 2, 3, 4, 5, (i % 1000) * 1000),
        "day": date(2000 + i % 30, 1 + i % 12, 1 + i % 28),
        "m": Decimal(f"{i}.25"),
        "v1": [_f32(i)],
        "v3": [_f32(i), _f32(i / 2), 0.5],
        "v9": None if null else [_f32(i + k / 10) for k in range(9)],
        "v128": [_f32(i * 0.5 + k) for k in range(128)],
        "pad": "p" * 400,
        "nothing": None,
    }
    return row


TYPES = [
    ("n", "INTEGER"),
    ("big", "LONG"),
    ("sh", "SHORT"),
    ("bt", "BYTE"),
    ("d", "DOUBLE"),
    ("f", "FLOAT"),
    ("s", "STRING"),
    ("ok", "BOOLEAN"),
    ("ts", "DATETIME"),
    ("day", "DATE"),
    ("m", "DECIMAL"),
    ("v1", "ARRAY_OF_FLOATS"),
    ("v3", "ARRAY_OF_FLOATS"),
    ("v9", "ARRAY_OF_FLOATS"),
    ("v128", "ARRAY_OF_FLOATS"),
    ("pad", "STRING"),
    ("nothing", "STRING"),
]
NAMES = [n for n, _ in TYPES]
ROWS = 5000


@pytest.fixture(scope="module")
def ref_db(tmp_path_factory):
    import arcadedb

    path = str(tmp_path_factory.mktemp("frames_ref") / "db")
    db = arcadedb.create_database(path)
    db.command("sql", "CREATE DOCUMENT TYPE R")
    for name, type_name in TYPES:
        db.command("sql", f"CREATE PROPERTY R.{name} {type_name}")
    db.insert_many("R", [_row(i) for i in range(ROWS)])
    yield db
    db.close()


def _eq(have, want):
    """Equal values, NaN equal to NaN, None equal only to None: a missing value is not a NaN."""
    if isinstance(have, np.ndarray) or isinstance(want, np.ndarray):
        return np.array_equal(np.asarray(have), np.asarray(want), equal_nan=True)
    if isinstance(have, (list, tuple)) and isinstance(want, (list, tuple)):
        return len(have) == len(want) and all(_eq(x, y) for x, y in zip(have, want))
    if (
        isinstance(have, float)
        and isinstance(want, float)
        and math.isnan(have)
        and math.isnan(want)
    ):
        return True
    return have == want


def _old_rows(db, sql, *args):
    return db.query("sql", sql, *args)._drain_dicts()


def _assert_arrow_equal(new, old):
    assert new.column_names == old.column_names
    assert new.num_rows == old.num_rows
    for name in old.column_names:
        have, want = new.column(name), old.column(name)
        if pa.types.is_null(want.type):
            assert have.type == pa.string() and have.null_count == len(have), name
        elif (
            pa.types.is_list(want.type)
            and want.type.value_type == pa.float64()
            and name.startswith("v")
        ):
            # a float[] column: the same values as a fixed-size list of float32 (a float32 widened to float64 is exact)
            assert have.type == pa.list_(
                pa.float32(),
                len(want[0].as_py() if want[0].is_valid else want[1].as_py()),
            ), name
        else:
            assert have.type == want.type, (name, have.type, want.type)
        assert _eq(have.to_pylist(), want.to_pylist()), name


def _assert_columns_equal(new, old):
    assert list(new) == list(old)
    for name, want in old.items():
        have = new[name]
        if isinstance(want, np.ndarray):
            assert isinstance(have, np.ndarray), (name, type(have))
            assert have.dtype == want.dtype, (name, have.dtype, want.dtype)
            assert have.shape == want.shape, name
            np.testing.assert_array_equal(have, want, err_msg=name)
        else:
            assert isinstance(have, list), (name, type(have))
            assert _eq(have, want), name


def _missing_as_none(values):
    """pandas decides how a missing value is shown (None, nan or NaT, by dtype and version): all of them are one here."""
    return [
        (
            None
            if v is None or v is pd.NaT or (isinstance(v, float) and math.isnan(v))
            else v
        )
        for v in values
    ]


def _assert_frame_equal(new, old_rows, columns=None):
    old = pd.DataFrame(old_rows)
    if columns is not None:
        old = old.reindex(columns=list(columns))
    assert list(new.columns) == list(old.columns)
    for name in old.columns:
        if name.startswith("v") and name != "vv":
            # a vector cell is a numpy float32 array (documented), the values are the stored ones
            for cell, want in zip(new[name].tolist(), old[name].tolist()):
                if want is None:
                    assert cell is None or (
                        isinstance(cell, float) and math.isnan(cell)
                    ), name
                else:
                    assert isinstance(cell, np.ndarray) and cell.dtype == np.float32, (
                        name,
                        type(cell),
                    )
                    assert _eq(cell.tolist(), want), name
            continue
        have = _missing_as_none(new[name].tolist())
        want = _missing_as_none(old[name].tolist())
        assert [
            x.to_pydatetime() if isinstance(x, pd.Timestamp) else x for x in have
        ] == [
            x.to_pydatetime() if isinstance(x, pd.Timestamp) else x for x in want
        ], name
    if "ts" in old.columns:
        assert str(new["ts"].dtype).startswith("datetime64[us"), new[
            "ts"
        ].dtype  # microseconds, whatever pandas infers from datetimes


QUERIES = [
    ("full", "SELECT FROM R ORDER BY big", None),
    (
        "projection",
        "SELECT n, big, sh, bt, d, f, s, ok, ts, day, m FROM R ORDER BY big",
        None,
    ),
    ("vectors", "SELECT big, v1, v3, v9, v128 FROM R ORDER BY big", None),
    ("limit 1", "SELECT FROM R ORDER BY big LIMIT 1", None),
    ("limit 100, one reply", "SELECT FROM R ORDER BY big LIMIT 100", None),
    ("limit 300, a reply and a cursor", "SELECT FROM R ORDER BY big LIMIT 300", None),
    ("empty", "SELECT FROM R WHERE big < 0", None),
    ("subset", "SELECT FROM R ORDER BY big", ["s", "big", "v3", "ts"]),
    ("subset with a column nobody has", "SELECT FROM R ORDER BY big", ["n", "absent"]),
    ("subset on a short result", "SELECT FROM R ORDER BY big LIMIT 20", ["m", "ok"]),
]


@pytest.mark.parametrize("label,sql,columns", QUERIES, ids=[q[0] for q in QUERIES])
def test_to_arrow_equals_the_old_route(ref_db, label, sql, columns):
    old = REF.to_arrow(_old_rows(ref_db, sql), columns)
    new = ref_db.query("sql", sql).to_arrow(columns=columns)
    _assert_arrow_equal(new, old)


@pytest.mark.parametrize("label,sql,columns", QUERIES, ids=[q[0] for q in QUERIES])
def test_to_columns_equals_the_old_route(ref_db, label, sql, columns):
    old = REF.to_columns(_old_rows(ref_db, sql), columns)
    new = ref_db.query("sql", sql).to_columns(columns=columns)
    _assert_columns_equal(new, old)


@pytest.mark.parametrize(
    "label,sql,columns", QUERIES[:7], ids=[q[0] for q in QUERIES[:7]]
)
def test_to_dataframe_equals_the_old_route(ref_db, label, sql, columns):
    rows = _old_rows(ref_db, sql)
    new = ref_db.query("sql", sql).to_dataframe()
    _assert_frame_equal(new, rows)


@pytest.mark.parametrize("batch_size", [1, 64, 255, 256, 257, 1000, 25000])
def test_the_frames_do_not_depend_on_the_batch_size(ref_db, batch_size):
    sql = "SELECT FROM R ORDER BY big"
    old = REF.to_arrow(_old_rows(ref_db, sql))
    _assert_arrow_equal(ref_db.query("sql", sql).to_arrow(batch_size=batch_size), old)


def test_a_result_larger_than_one_frame_is_read_in_several(ref_db):
    from unittest import mock

    from arcadedb._native import lib

    sizes = []
    real = lib.call_frame

    def spy(*args):
        raw = real(*args)
        sizes.append(len(raw))
        return raw

    # 5000 rows of about 2.4 KiB of values: several frames of about 1 MiB each, and the table is the same as one read of the rows
    with mock.patch.object(lib, "call_frame", spy):
        t = ref_db.query("sql", "SELECT FROM R ORDER BY big").to_arrow()
    assert t.num_rows == ROWS
    assert len(sizes) >= 4, sizes
    assert max(sizes) < 4 * (1 << 20), sizes  # a frame, not the table
    assert t.column("big").to_pylist() == [(1 << 40) + i for i in range(ROWS)]


def test_rows_read_by_iteration_first_then_the_rest_as_columns(ref_db):
    sql = "SELECT FROM R ORDER BY big"
    old = REF.to_arrow(_old_rows(ref_db, sql)).slice(40)
    rs = ref_db.query("sql", sql)
    taken = [next(rs).get("big") for _ in range(40)]
    assert taken == [(1 << 40) + i for i in range(40)]
    _assert_arrow_equal(rs.to_arrow(), old)


def test_rows_read_past_the_first_reply_then_the_rest_as_columns(ref_db):
    sql = "SELECT FROM R ORDER BY big"
    old = REF.to_arrow(_old_rows(ref_db, sql)).slice(300)
    rs = ref_db.query("sql", sql)
    for _ in range(300):
        next(rs)
    _assert_arrow_equal(rs.to_arrow(), old)


def test_a_result_is_read_once(ref_db):
    rs = ref_db.query("sql", "SELECT FROM R ORDER BY big LIMIT 500")
    assert rs.to_arrow().num_rows == 500
    assert rs.to_arrow().num_rows == 0


def test_to_list_still_reads_the_json_rows(ref_db):
    rows = ref_db.query("sql", "SELECT FROM R ORDER BY big LIMIT 300").to_list()
    assert len(rows) == 300 and rows[299]["big"] == (1 << 40) + 299


def test_a_frame_stays_near_its_budget_when_every_row_has_other_properties(tmp_path):
    """Rows that each carry a property no other row has: the frame holds rows times the union of the names, not the values read."""
    from unittest import mock

    import arcadedb
    from arcadedb._native import lib

    db = arcadedb.create_database(str(tmp_path / "sparse"))
    try:
        db.command("sql", "CREATE DOCUMENT TYPE Sparse")
        db.insert_many("Sparse", [{"i": i, f"p{i}": i} for i in range(2000)])
        sizes = []
        real = lib.call_frame

        def spy(*args):
            raw = real(*args)
            sizes.append(len(raw))
            return raw

        with mock.patch.object(lib, "call_frame", spy):
            t = db.query("sql", "SELECT FROM Sparse ORDER BY i").to_arrow()
        assert t.num_rows == 2000 and t.num_columns == 2001
        assert t.column("p1999").to_pylist() == [None] * 1999 + [1999]
        # about 1 MiB of fixed-width buffers (rows x columns x 9 bytes) is what a frame is allowed; 2000 x 2000 x 9 = 36 MiB in one frame is not
        assert len(sizes) > 5 and max(sizes) < 3 * (1 << 20), sizes
    finally:
        db.close()


def test_an_empty_frame_for_a_cursor_with_rows_left_is_an_error_not_a_loop(ref_db):
    from unittest import mock

    from arcadedb._native import lib
    from arcadedb.exceptions import ArcadeDBError

    header = b'{"count":0,"done":false,"cols":[]}'
    empty = len(header).to_bytes(4, "little") + header

    rs = ref_db.query("sql", "SELECT FROM R ORDER BY big")
    with mock.patch.object(lib, "call_frame", lambda *args: empty):
        with pytest.raises(ArcadeDBError, match="empty column frame"):
            rs.to_arrow()


def test_a_cursor_reads_a_huge_batch_size_without_wrapping(ref_db):
    sql = "SELECT big FROM R ORDER BY big"
    for batch_size in (2**31, 2**32, 2**33 + 5):
        assert ref_db.query("sql", sql).to_arrow(batch_size=batch_size).num_rows == ROWS


def test_a_list_property_is_not_a_vector_column_at_any_result_size(tmp_path):
    import arcadedb

    db = arcadedb.create_database(str(tmp_path / "lists"))
    try:
        db.command("sql", "CREATE DOCUMENT TYPE L")
        db.insert_many("L", [{"i": i, "lst": [0.5, 1.0, 2.0]} for i in range(600)])
        for rows in (5, 300, 600):
            c = db.query(
                "sql",
                f"SELECT i, lst FROM L ORDER BY i LIMIT {rows}",  # nosec B608 - an integer of the test
            ).to_columns()
            assert isinstance(c["lst"], list) and c["lst"] == [[0.5, 1.0, 2.0]] * rows
    finally:
        db.close()
