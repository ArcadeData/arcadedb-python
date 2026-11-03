"""``append_samples`` on the native build sends numpy columns by address, not as ``.tolist()`` and one JSON text per chunk.

A one-dimensional numpy int, float or bool column, and an int or ``datetime64[ns]`` timestamp array, go through ``arcadedb_n_async_bin``:
the JSON of the call holds a kind marker per such column ("q" int64, "d" float64) and the sample count, and the data crosses in an address
table, as ``arcadedb_n_insert_colp`` does for a bulk insert. Everything else (the host tag column of strings, lists, a float array as
timestamps, a column of another length) keeps the text of ``arcadedb_n_async``. The stored samples do not change: the Java side builds the
same ``long[]`` and boxed ``Object[]`` columns the text gave and calls the same ``appendSamples``. A library without the entry
(``lib.TS_BIN`` false) keeps the text path; ``primitive=True`` is the same length check as before.

The calls are observed through ``lib.call`` (the ctypes door of the native library, as in test_native_tx_crossings.py). Needs the native
library, so conftest.py collects this file on the native backend only.
"""

import json
import struct

import numpy as np
import pytest

from arcadedb import ArcadeDBError
from arcadedb._native import codec
from arcadedb._native import extras as native_extras
from arcadedb._native import lib

BASE = 1_700_000_000_000  # epoch ms


class _Calls:
    """The ``(entry point, operation, JSON text, rest of the arguments)`` of every async call made while it is installed.

    ``stub``: the calls are recorded and answered with ``{"ok":true}`` without reaching the library, for the tests that only look at
    which entry a call takes (a call that the engine would refuse, or that is not worth a database).
    """

    def __init__(self, monkeypatch, stub=False):
        self.calls = []
        real = lib.call

        def spy(name, *args):
            if name in ("arcadedb_n_async", "arcadedb_n_async_bin"):
                self.calls.append((name, args[1].decode(), args[2].decode(), args[3:]))
                if stub and args[1] == b"append_samples":
                    return '{"ok":true}'
            return real(name, *args)

        monkeypatch.setattr(lib, "call", spy)

    def entries(self, op="append_samples"):
        return [c[0] for c in self.calls if c[1] == op]

    def texts(self, op="append_samples"):
        return [json.loads(c[2]) for c in self.calls if c[1] == op]


def _type(db, name, fields="d1 DOUBLE, d2 DOUBLE, d3 DOUBLE"):
    db.command(
        "sql",
        f"CREATE TIMESERIES TYPE {name} TIMESTAMP ts TAGS (host STRING) FIELDS ({fields}) SHARDS 2",
    )


def _canon(value):
    # a float by its bits (-0.0, NaN and the last digit all count), anything else by its repr
    if isinstance(value, float):
        return struct.pack("<d", value).hex()
    return repr(value)


def _rows(db, name, columns):
    names = ["ts", "host"] + columns.split(", ")
    query = f"SELECT {', '.join(names)} FROM {name} ORDER BY ts"  # nosec B608 - test-owned names
    return [
        tuple(_canon(row[c]) for c in names) for row in db.query("sql", query).to_list()
    ]


def _count_sums(db, name, columns):
    sums = ", ".join(f"sum({c}) AS s_{c}" for c in columns)
    query = f"SELECT count(*) AS n, {sums} FROM {name}"  # nosec B608 - test-owned names
    row = db.query("sql", query).to_list()[0]
    return [int(row["n"])] + [float(row[f"s_{c}"]) for c in columns]


# ----------------------------------------------------------------------------------------------------------- the lane's shape


def test_the_lane_shape_takes_the_binary_entry_and_stores_the_source(
    temp_db, monkeypatch
):
    # l4_tsbs ArcadeNativeTS.ingest: epoch-ms int64 timestamps, a list of host strings, three float64 columns, primitive=True
    calls = _Calls(monkeypatch)
    _type(temp_db, "Lane")
    rng = np.random.default_rng(5)
    chunk, chunks, hosts = 10_000, 3, 100
    n = chunk * chunks
    ts = BASE + np.arange(n, dtype=np.int64) * 1000
    host = [f"host_{i % hosts}" for i in range(n)]
    cols = [rng.integers(0, 101, size=n).astype(np.float64) for _ in range(3)]
    ex = temp_db.async_executor()
    for lo in range(0, n, chunk):
        s = slice(lo, lo + chunk)
        ex.append_samples("Lane", ts[s], host[s], *[c[s] for c in cols], primitive=True)
    ex.wait_completion()
    assert lib.TS_BIN is True  # the loaded library has the entry and the default is on
    assert calls.entries() == ["arcadedb_n_async_bin"] * chunks
    for k, text in enumerate(calls.texts()):
        assert text["n"] == chunk and text["ts"] == "q"
        assert text["cols"][1:] == [
            "d",
            "d",
            "d",
        ]  # no number of a numeric column is in the text
        assert (
            text["cols"][0] == host[k * chunk : (k + 1) * chunk]
        )  # the host tag stays text
    assert all(
        len(c[2]) < 14 * chunk for c in calls.calls
    )  # the text path wrote about 44 bytes a point
    assert _count_sums(temp_db, "Lane", ["d1", "d2", "d3"]) == [n] + [
        float(c.sum()) for c in cols
    ]
    # windows: ten instants of one host, as the probe reads them
    lo_ts, hi_ts = BASE + 1000 * 1000, BASE + 1000 * 1000 + 9 * hosts * 1000
    got = temp_db.query(
        "sql",
        f"SELECT ts, d1 FROM Lane WHERE host = 'host_7' AND ts BETWEEN {lo_ts} AND {hi_ts} ORDER BY ts",  # nosec B608
    ).to_list()
    want = [
        float(cols[0][i])
        for i in range(n)
        if host[i] == "host_7" and lo_ts <= ts[i] <= hi_ts
    ]
    assert [float(r["d1"]) for r in got] == want and len(want) == 10


def test_a_chunk_is_not_sent_as_number_text(temp_db, monkeypatch):
    calls = _Calls(monkeypatch)
    _type(temp_db, "NoText")
    n = 5_000
    ts = BASE + np.arange(n, dtype=np.int64)
    d1 = np.random.default_rng(1).standard_normal(n)
    ex = temp_db.async_executor()
    ex.append_samples("NoText", ts, ["h"] * n, d1, d1, d1)
    ex.wait_completion()
    (text,) = [c[2] for c in calls.calls if c[1] == "append_samples"]
    assert repr(float(d1[3])) not in text and str(int(ts[3])) not in text


# ----------------------------------------------------------------------------------------------------------- same stored samples


def _matrix(n=400):
    rng = np.random.default_rng(11)
    d1 = rng.standard_normal(n)
    d1[:8] = [-0.0, np.nan, np.inf, -np.inf, 1e300, 5e-324, 0.1, 1e-5]
    d2 = rng.standard_normal(n).astype(np.float32)
    d2[:3] = [np.finfo(np.float32).max, np.finfo(np.float32).tiny, np.float32(-0.0)]
    d3 = rng.standard_normal(n).astype(np.float16)
    d4 = rng.integers(-(2**31), 2**31, size=n).astype(
        np.int32
    )  # ints into a DOUBLE column
    l1 = rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)
    l1[:5] = [-(2**63), 2**63 - 1, 0, 1, -1]
    l2 = rng.integers(0, 256, size=n).astype(np.uint8)
    l3 = rng.integers(0, 2, size=n) == 1  # a numpy bool array is a 0/1 numeric column
    l4 = np.round(
        rng.standard_normal(n) * 1000.0, 3
    )  # floats into a LONG column: the engine truncates them, the same on both paths
    return n, [d1, d2, d3, d4, l1, l2, l3, l4]


_FIELDS = (
    "d1 DOUBLE, d2 DOUBLE, d3 DOUBLE, d4 DOUBLE, l1 LONG, l2 LONG, l3 LONG, l4 LONG"
)
_NAMES = "d1, d2, d3, d4, l1, l2, l3, l4"


def test_every_numeric_dtype_is_stored_as_the_text_path_stored_it(temp_db, monkeypatch):
    calls = _Calls(monkeypatch)
    n, cols = _matrix()
    ts = BASE + np.arange(n, dtype=np.int64) * 7
    host = [f"h{i % 5}" for i in range(n)]
    ex = temp_db.async_executor()
    _type(temp_db, "Bin", _FIELDS)
    ex.append_samples("Bin", ts, host, *cols)
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async_bin"]
    assert calls.texts()[0]["cols"][1:] == ["d", "d", "d", "d", "q", "q", "q", "d"]
    monkeypatch.setattr(
        lib, "TS_BIN", False
    )  # the library as a build without the entry leaves the flag
    del calls.calls[:]
    _type(temp_db, "Txt", _FIELDS)
    ex.append_samples("Txt", ts, host, *cols)
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async"]
    binary, text = _rows(temp_db, "Bin", _NAMES), _rows(temp_db, "Txt", _NAMES)
    assert len(binary) == n == len(text)
    assert binary == text


@pytest.mark.parametrize(
    "make",
    [
        lambda: (1_000_000 + np.arange(50) * 1000).astype(np.int32),
        lambda: (np.arange(50) * 10).astype(np.uint16),
        lambda: np.uint64(BASE) + np.arange(50, dtype=np.uint64),
        lambda: (np.int64(BASE) + np.arange(50, dtype=np.int64) * 1000).view(
            "datetime64[ns]"
        ),
        lambda: (BASE + np.arange(100, dtype=np.int64) * 2)[::2],  # a strided view
    ],
    ids=["int32", "uint16", "uint64", "datetime64-ns", "int64-strided"],
)
def test_timestamp_arrays_are_stored_as_the_text_path_stored_them(
    temp_db, monkeypatch, make
):
    calls = _Calls(monkeypatch)
    ts = make()
    vals = np.arange(50, dtype=np.float64)
    ex = temp_db.async_executor()
    _type(temp_db, "TsBin", "d1 DOUBLE")
    ex.append_samples("TsBin", ts, ["a"] * 50, vals)
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async_bin"] and calls.texts()[0]["ts"] == "q"
    monkeypatch.setattr(lib, "TS_BIN", False)
    _type(temp_db, "TsTxt", "d1 DOUBLE")
    ex.append_samples("TsTxt", ts, ["a"] * 50, vals)
    ex.wait_completion()
    assert _rows(temp_db, "TsBin", "d1") == _rows(temp_db, "TsTxt", "d1")
    assert len(_rows(temp_db, "TsBin", "d1")) == 50


def test_a_non_contiguous_and_a_big_endian_column_are_stored_as_values(
    temp_db, monkeypatch
):
    calls = _Calls(monkeypatch)
    n = 60
    ts = np.repeat(BASE + np.arange(n, dtype=np.int64) * 3, 2)[::2]  # a strided view
    strided = np.arange(2 * n, dtype=np.float64)[::2]
    big = (np.arange(n) * 5).astype(">i8")
    ex = temp_db.async_executor()
    _type(temp_db, "Odd", "d1 DOUBLE, l1 LONG")
    ex.append_samples("Odd", ts, ["a"] * n, strided, big)
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async_bin"]
    got = temp_db.query("sql", "SELECT ts, d1, l1 FROM Odd ORDER BY ts").to_list()
    assert [float(r["d1"]) for r in got] == [float(x) for x in strided]
    assert [int(r["l1"]) for r in got] == [5 * i for i in range(n)]


# ----------------------------------------------------------------------------------------------------------- what keeps the text path


def test_lists_keep_the_text_entry(temp_db, monkeypatch):
    calls = _Calls(monkeypatch)
    _type(temp_db, "Lists")
    ex = temp_db.async_executor()
    ex.append_samples(
        "Lists", [BASE, BASE + 1], ["a", "b"], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
    )
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async"]
    assert _count_sums(temp_db, "Lists", ["d1", "d2", "d3"]) == [2, 3.0, 7.0, 11.0]


def test_a_timestamp_list_with_numpy_columns_sends_the_timestamps_as_text(
    temp_db, monkeypatch
):
    calls = _Calls(monkeypatch)
    _type(temp_db, "MixedTs")
    ts = [BASE + i for i in range(20)]
    d = np.arange(20, dtype=np.float64)
    ex = temp_db.async_executor()
    ex.append_samples("MixedTs", ts, ["a"] * 20, d, d, d)
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async_bin"]
    text = calls.texts()[0]
    assert text["ts"] == ts and text["cols"][1:] == ["d", "d", "d"]
    assert _count_sums(temp_db, "MixedTs", ["d1"]) == [20, float(d.sum())]


def test_float_timestamps_keep_the_text_path_and_the_columns_still_go_by_address(
    temp_db, monkeypatch
):
    calls = _Calls(monkeypatch, stub=True)
    ex = temp_db.async_executor()
    ts = (
        np.arange(5, dtype=np.float64) + BASE
    )  # int() truncates float timestamps on the text path
    ex.append_samples("T", ts, ["a"] * 5, [1.0] * 5)
    assert calls.entries() == ["arcadedb_n_async"]
    assert calls.texts()[0]["ts"] == [BASE + i for i in range(5)]
    ex.append_samples("T", ts + 0.7, ["a"] * 5, np.ones(5))
    assert calls.entries() == ["arcadedb_n_async", "arcadedb_n_async_bin"]
    text = calls.texts()[1]
    assert text["ts"] == [BASE + i for i in range(5)] and text["cols"][1] == "d"


def test_datetime64_milliseconds_are_refused_as_before(temp_db, monkeypatch):
    calls = _Calls(monkeypatch, stub=True)
    ts = np.arange(5).astype("datetime64[ms]") + np.datetime64(BASE, "ms")
    with pytest.raises(TypeError):  # int() of a datetime, on the text path
        temp_db.async_executor().append_samples("T", ts, ["a"] * 5, np.zeros(5))
    assert calls.entries() == []


def test_a_column_of_another_length_keeps_the_text_path(temp_db, monkeypatch):
    calls = _Calls(monkeypatch, stub=True)
    ex = temp_db.async_executor()
    ex.append_samples(
        "T", np.arange(10, dtype=np.int64), ["a"] * 10, np.zeros(9), np.zeros(10)
    )
    (text,) = calls.texts()
    assert calls.entries() == [
        "arcadedb_n_async_bin"
    ]  # the timestamps and the column of 10 go by address, the column of 9 is text
    assert (
        text["cols"][0] == ["a"] * 10
        and text["cols"][1] == [0.0] * 9
        and text["cols"][2] == "d"
    )
    calls.calls.clear()
    ex.append_samples(
        "T", [1, 2, 3], ["a"] * 3, np.zeros(2)
    )  # no column of the call's length and the timestamps a list
    assert calls.entries() == ["arcadedb_n_async"]


def test_primitive_true_keeps_its_length_check_and_takes_the_same_entry(
    temp_db, monkeypatch
):
    calls = _Calls(monkeypatch, stub=True)
    ex = temp_db.async_executor()
    with pytest.raises(ValueError, match="column 1 has 9 values for 10 timestamps"):
        ex.append_samples(
            "T", np.arange(10, dtype=np.int64), ["a"] * 10, np.zeros(9), primitive=True
        )
    with pytest.raises(ValueError, match="column 0 has 3 values for 10 timestamps"):
        ex.append_samples("T", np.arange(10, dtype=np.int64), ["a"] * 3, primitive=True)
    assert calls.entries() == []  # the check comes before the crossing
    ex.append_samples(
        "T", np.arange(10, dtype=np.int64), ["a"] * 10, np.zeros(10), primitive=True
    )
    ex.append_samples("T", np.arange(10, dtype=np.int64), ["a"] * 10, np.zeros(10))
    assert calls.entries() == ["arcadedb_n_async_bin"] * 2
    assert calls.texts()[0] == calls.texts()[1]


def test_a_library_without_the_entry_keeps_the_text_path(temp_db, monkeypatch):
    # the probe flag as the loader leaves it for a library built before arcadedb_n_async_bin
    monkeypatch.setattr(lib, "TS_BIN", False)
    calls = _Calls(monkeypatch)
    _type(temp_db, "Old")
    n = 1_000
    d = np.arange(n, dtype=np.float64)
    ex = temp_db.async_executor()
    ex.append_samples(
        "Old", BASE + np.arange(n, dtype=np.int64), ["a"] * n, d, d, d, primitive=True
    )
    ex.wait_completion()
    assert calls.entries() == ["arcadedb_n_async"]
    assert _count_sums(temp_db, "Old", ["d1", "d2", "d3"]) == [n] + [float(d.sum())] * 3


# ----------------------------------------------------------------------------------------------------------- sample_array


_INT64_MAX = 2**63 - 1


@pytest.mark.parametrize(
    "values, timestamps, want",
    [
        (np.arange(3, dtype=np.int8), False, np.int64),
        (np.arange(3, dtype=np.int64), True, np.int64),
        (np.arange(3, dtype=np.uint32), False, np.int64),
        (np.array([1, 2], dtype=np.uint64), False, np.int64),
        (
            np.array([1, _INT64_MAX + 1], dtype=np.uint64),
            False,
            None,
        ),  # past the int64 range: the text keeps its Python ints
        (np.array([True, False]), False, np.int64),
        (np.array([True, False]), True, np.int64),
        (np.ones(3, dtype=np.float16), False, np.float64),
        (np.ones(3, dtype=np.float32), False, np.float64),
        (np.ones(3, dtype=np.float64), False, np.float64),
        (
            np.ones(3, dtype=np.float64),
            True,
            None,
        ),  # int() truncates float timestamps on the text path
        (np.ones(3, dtype=np.longdouble), False, None),
        (np.array([1, 2], dtype="datetime64[ns]"), True, np.int64),
        (np.array([1, 2], dtype="datetime64[ns]"), False, None),
        (
            np.array(["1970-01-01T00:00:00.000000001", "NaT"], dtype="datetime64[ns]"),
            True,
            None,
        ),
        (np.array([1, 2], dtype="datetime64[ms]"), True, None),
        (np.array([1, 2], dtype=">M8[ns]"), True, None),
        (np.zeros(0, dtype=np.float64), False, None),
        (np.zeros((2, 2)), False, None),
        (
            np.ma.masked_array([1.0, 2.0], mask=[False, True]),
            False,
            None,
        ),  # tolist() gives None for the masked value
        (np.array(["a", "b"]), False, None),
        (np.array([1, 2], dtype=object), False, None),
        ([1.0, 2.0], False, None),
        (None, False, None),
    ],
)
def test_sample_array_picks_the_columns_whose_tolist_it_reproduces(
    values, timestamps, want
):
    got = codec.sample_array(values, timestamps)
    if want is None:
        assert got is None
        return
    assert got.dtype == want and got.flags.c_contiguous and got.ndim == 1
    if isinstance(values, np.ndarray) and values.dtype.kind in "iub":
        assert got.tolist() == [int(v) for v in values.tolist()]
    if isinstance(values, np.ndarray) and values.dtype.kind == "f":
        assert (
            got.tolist() == values.tolist()
        )  # the same doubles .tolist() writes into the text (a float32 widens exactly)
    if isinstance(values, np.ndarray) and values.dtype.kind == "M":
        assert (
            got.tolist() == values.tolist()
        )  # the ints .tolist() gives for datetime64[ns]


# ----------------------------------------------------------------------------------------------------------- the Java side checks the table


def _bin(db, text, table, count):
    return native_extras._call(
        "append_samples",
        "arcadedb_n_async_bin",
        db._h,
        lib.b("append_samples"),
        lib.b(json.dumps(text)),
        table.ctypes.data if table is not None else None,
        count,
    )


def test_a_wrong_address_table_is_refused_before_anything_is_written(temp_db):
    _type(temp_db, "Bad")
    ts = np.arange(4, dtype=np.int64) + BASE
    table = np.zeros(4, dtype=np.int64)
    table[0], table[1] = (
        ts.ctypes.data,
        3,
    )  # the table says 3 timestamps, the call says 4
    args = {"type": "Bad", "n": 4, "ts": "q", "cols": [["a"] * 4]}
    with pytest.raises(ArcadeDBError, match="address table"):
        _bin(temp_db, args, table, len(table))
    table[1] = 4
    with pytest.raises(ArcadeDBError, match="address table"):
        _bin(
            temp_db, args, table, 1
        )  # the table is shorter than the slots the call uses
    with pytest.raises(ArcadeDBError, match="address table"):
        _bin(temp_db, args, None, 0)
    with pytest.raises(ArcadeDBError, match="unknown kind"):
        _bin(temp_db, {**args, "ts": "x"}, table, len(table))
    with pytest.raises(ArcadeDBError, match="unknown kind"):
        _bin(temp_db, {**args, "ts": "d"}, table, len(table))  # a timestamp is int64
    with pytest.raises(ArcadeDBError, match="sample count out of range"):
        _bin(temp_db, {**args, "n": -1}, table, len(table))
    temp_db.async_executor().wait_completion()
    assert temp_db.query("sql", "SELECT count(*) AS n FROM Bad").to_list()[0]["n"] == 0
