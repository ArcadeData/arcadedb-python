"""Packing of columns for the native backend's columnar insert (``arcadedb/_native/columnar.py``).

No native library is needed: the layouts (kinds, offsets, null bitmaps, which columns are refused) are read back from the addresses the
packer would hand to ``arcadedb_n_insert_colp``. The run-time half, which compares the stored records of the JSON and the columnar paths,
is in ``test_native_columnar_insert.py``, collected on the native backend only.
"""

import ctypes

import numpy as np
import pytest

from arcadedb._native import columnar  # noqa: E402


def _bytes(addr_len):
    addr, length = addr_len
    return ctypes.string_at(addr, length) if length else b""


def _strings(col):
    offsets = np.frombuffer(_bytes(col.data), dtype=np.int32)
    blob = _bytes(col.aux)
    return [blob[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(col.n)]


def _present(col):
    bits = np.frombuffer(_bytes(col.nulls), dtype=np.uint8)
    return np.unpackbits(bits, bitorder="little")[: col.n].astype(bool).tolist()


NO_TYPES = columnar.Declared(known={})


def _pack(columns, declared=None):
    got = columnar.pack(columns, columnar.Declared(known=declared or {}))
    assert got is not None
    rows, cols = got
    return rows, {c.name: c for c in cols}


def test_numpy_columns_cross_as_their_64_bit_kinds_without_a_copy():
    ids = np.arange(5, dtype=np.int64)
    vals = np.linspace(0, 1, 5)
    rows, cols = _pack({"id": ids, "val": vals})
    assert rows == 5
    assert (cols["id"].kind, cols["val"].kind) == ("q", "d")
    assert cols["id"].data == (ids.ctypes.data, 40)  # the array itself, not a copy
    assert cols["val"].data == (vals.ctypes.data, 40)
    assert cols["id"].nulls == (0, 0) and cols["id"].aux == (0, 0)


def test_narrow_numbers_widen_unless_the_property_is_declared_that_narrow():
    small = np.array([1, -2, 3], dtype=np.int32)
    single = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    _, cols = _pack(
        {"n": small, "x": single}
    )  # undeclared: a Long and a Double, as the JSON path stores them
    assert (cols["n"].kind, cols["x"].kind) == ("q", "d")
    assert np.frombuffer(_bytes(cols["n"].data), dtype=np.int64).tolist() == [1, -2, 3]
    _, cols = _pack({"n": small, "x": single}, {"n": "INTEGER", "x": "FLOAT"})
    assert (cols["n"].kind, cols["x"].kind) == ("i", "e")
    assert cols["n"].data == (small.ctypes.data, 12) and cols["x"].data == (
        single.ctypes.data,
        12,
    )
    _, cols = _pack({"n": small.astype(np.int64)}, {"n": "INTEGER"})
    assert (
        cols["n"].kind == "q"
    )  # a 64-bit source is range-checked by the engine, as before


def test_unsigned_past_the_signed_range_is_refused_and_below_it_is_not():
    assert columnar.pack({"u": np.array([2**63], dtype=np.uint64)}, NO_TYPES) is None
    _, cols = _pack({"u": np.array([2**63 - 1, 1], dtype=np.uint64)})
    assert np.frombuffer(_bytes(cols["u"].data), dtype=np.int64).tolist() == [
        2**63 - 1,
        1,
    ]


def test_bool_and_nan_and_infinity_keep_their_values():
    _, cols = _pack(
        {"f": np.array([True, False, True]), "x": np.array([np.nan, np.inf, -np.inf])}
    )
    assert cols["f"].kind == "b" and _bytes(cols["f"].data) == b"\x01\x00\x01"
    x = np.frombuffer(_bytes(cols["x"].data), dtype=np.float64)
    assert np.isnan(x[0]) and x[1] == np.inf and x[2] == -np.inf


def test_a_float32_matrix_and_a_list_of_float32_arrays_are_vector_columns():
    m = np.arange(6, dtype=np.float32).reshape(3, 2)
    _, cols = _pack({"v": m, "w": [row for row in m]})
    for name in ("v", "w"):
        assert (cols[name].kind, cols[name].dim, cols[name].n) == ("f", 2, 3)
        assert np.frombuffer(
            _bytes(cols[name].data), dtype=np.float32
        ).tolist() == list(range(6))


def test_float64_arrays_are_not_vector_columns():
    # only float32 is a column of vectors, as on the JPype path (tests/test_float_vector_bulk.py): float64 keeps the row path
    m = np.arange(6, dtype=np.float64).reshape(3, 2)
    assert columnar.pack({"v": m}, NO_TYPES) is None
    assert columnar.pack({"v": [row for row in m]}, NO_TYPES) is None


def test_a_list_of_float_lists_is_a_vector_only_for_an_array_of_floats_property():
    lists = [[0.5, 1.5], [2.5, 3.5]]
    assert (
        columnar.pack({"v": lists}, NO_TYPES) is None
    )  # undeclared: the JSON path keeps the doubles
    _, cols = _pack({"v": lists}, {"v": "ARRAY_OF_FLOATS"})
    assert (cols["v"].kind, cols["v"].dim) == ("f", 2)
    assert (
        columnar.pack(
            {"v": [[1.0], [1.0, 2.0]]},
            columnar.Declared(known={"v": "ARRAY_OF_FLOATS"}),
        )
        is None
    )  # ragged
    assert (
        columnar.pack(
            {"v": [[], []]}, columnar.Declared(known={"v": "ARRAY_OF_FLOATS"})
        )
        is None
    )  # no dimension


def test_strings_are_arrow_offsets_plus_utf8_bytes():
    texts = ["", "abc", "héllo", "日本語", "😀", "x\x00y"]
    rows, cols = _pack({"s": texts})
    assert rows == 6 and cols["s"].kind == "s"
    assert _strings(cols["s"]) == texts
    offsets = np.frombuffer(_bytes(cols["s"].data), dtype=np.int32).tolist()
    assert (
        offsets[0] == 0
        and offsets == sorted(offsets)
        and offsets[-1] == len(_bytes(cols["s"].aux))
    )


def test_ascii_and_all_empty_strings():
    _, cols = _pack({"a": ["a", "bb", "ccc"], "e": ["", "", ""]})
    assert _strings(cols["a"]) == ["a", "bb", "ccc"]
    assert _strings(cols["e"]) == ["", "", ""] and cols["e"].aux == (0, 0)


def test_a_lone_surrogate_is_refused_so_the_json_path_decides():
    assert columnar.pack({"s": ["ok", "\ud800"]}, NO_TYPES) is None


def test_nulls_become_a_bitmap_and_a_filler_value():
    rows, cols = _pack(
        {
            "n": [1, None, 3, None],
            "x": [None, 2.5, None, 4.5],
            "s": ["a", None, "c", None],
            "b": [True, None, False, True],
            "z": [None, None, None, None],
        }
    )
    assert rows == 4
    assert _present(cols["n"]) == [True, False, True, False]
    assert _present(cols["s"]) == [True, False, True, False]
    assert _present(cols["x"]) == [False, True, False, True]
    assert _present(cols["b"]) == [True, False, True, True]
    assert _present(cols["z"]) == [False] * 4 and cols["z"].kind == "s"
    assert _strings(cols["s"])[0] == "a" and _strings(cols["s"])[2] == "c"
    assert np.frombuffer(_bytes(cols["n"].data), dtype=np.int64)[[0, 2]].tolist() == [
        1,
        3,
    ]
    assert len(_bytes(cols["n"].nulls)) == 1


def test_a_bitmap_longer_than_a_byte():
    values = [None if i % 3 == 0 else i for i in range(20)]
    _, cols = _pack({"n": values})
    assert _present(cols["n"]) == [v is not None for v in values]
    assert len(_bytes(cols["n"].nulls)) == 3


@pytest.mark.parametrize(
    "column",
    [
        [1, 2.5],  # mixed int and float
        [1, "a"],
        [True, 1],  # bool is not an int here
        [1, 2**63],  # past 64 bits
        [{"a": 1}, {"a": 2}],  # nested
        [[1], [2]],  # a list without a declared ARRAY_OF_FLOATS
        [
            b"ab",
            "cd",
        ],  # bytes and text (a column of bytes alone crosses as kind x since #316)
        [b"ab", bytearray(b"cd")],  # two bytes-like types
        [np.int64(1), np.int64(2)],  # numpy scalars in a list are not Python numbers
    ],
)
def test_columns_the_buffer_cannot_carry_unchanged_are_refused(column):
    assert columnar.pack({"c": column}, NO_TYPES) is None


def test_numpy_datetime64_arrays_and_mixed_temporal_columns_are_refused():
    import datetime
    import decimal

    assert (
        columnar.pack({"d": np.array(["2020-01-01"], dtype="datetime64[D]")}, NO_TYPES)
        is None
    )
    assert (
        columnar.pack(
            {"d": [datetime.datetime(2020, 1, 1), datetime.date(2020, 1, 1)]}, NO_TYPES
        )
        is None
    )
    assert (
        columnar.pack({"d": [decimal.Decimal("1"), decimal.Decimal("NaN")]}, NO_TYPES)
        is None
    )
    assert columnar.pack({"d": [b"a", "b"]}, NO_TYPES) is None


def _ticks(col, dtype):
    return np.frombuffer(_bytes(col.data), dtype=dtype).tolist()


def test_python_datetimes_are_utc_wall_clock_microseconds():
    import datetime

    utc9 = datetime.timezone(datetime.timedelta(hours=9))
    values = [
        datetime.datetime(1970, 1, 1),
        datetime.datetime(1969, 12, 31, 23, 59, 59, 999999),
        datetime.datetime(2020, 1, 2, 12, 0, 0, 5, tzinfo=utc9),  # 03:00:00.000005 UTC
        None,
    ]
    rows, cols = columnar.pack({"t": values}, NO_TYPES)
    col = cols[0]
    assert rows == 4 and col.kind == "T" and col.dim == 1_000_000
    assert _ticks(col, np.int64) == [0, -1, 1_577_934_000_000_005, 0]
    assert _present(col) == [True, True, True, False]


def test_python_dates_are_epoch_days():
    import datetime

    rows, cols = columnar.pack(
        {"d": [datetime.date(1970, 1, 2), datetime.date(1969, 12, 31), None]}, NO_TYPES
    )
    assert cols[0].kind == "D" and _ticks(cols[0], np.int32) == [1, -1, 0]
    assert _present(cols[0]) == [True, True, False]


def test_decimals_and_bytes_use_the_string_layout():
    import decimal

    rows, cols = columnar.pack(
        {
            "a": [decimal.Decimal("1E+3"), None, decimal.Decimal("-0.50")],
            "b": [b"\x00\xff", b"", None],
            "c": [bytearray(b"xy"), bytearray(b"z"), bytearray(b"")],
        },
        NO_TYPES,
    )
    a, b, c = cols
    assert [x.kind for x in cols] == ["n", "x", "x"]
    offsets = np.frombuffer(_bytes(a.data), dtype=np.int32).tolist()
    assert offsets == [0, 4, 4, 9] and _bytes(a.aux) == b"1E+3-0.50"
    offsets = np.frombuffer(_bytes(b.data), dtype=np.int32).tolist()
    assert offsets == [0, 2, 2, 2] and _bytes(b.aux) == b"\x00\xff"
    assert _bytes(c.aux) == b"xyz" and _present(a) == [True, False, True]


def test_arrow_timestamps_dates_decimals_and_binary():
    import decimal

    import pyarrow as pa

    stamps = pa.array([1, None, -2], pa.int64()).cast(
        pa.timestamp("ms", tz="Asia/Seoul")
    )
    days = pa.array([3, -4, None], pa.int32()).cast(pa.date32())
    day_ms = pa.array([86_400_000 * 2, -1, None], pa.int64()).cast(pa.date64())
    amounts = pa.array([decimal.Decimal("1.50"), None], pa.decimal128(10, 2))
    blobs = pa.array([b"ab", None, b"c"])
    for name, col, expected in (
        ("t", stamps, ("T", 1_000)),
        ("d", days, ("D", 0)),
        ("m", day_ms, ("D", 0)),
    ):
        packed = columnar.pack({name: col}, NO_TYPES)[1][0]
        assert (packed.kind, packed.dim) == expected
    assert _ticks(columnar.pack({"t": stamps}, NO_TYPES)[1][0], np.int64) == [1, 0, -2]
    assert _ticks(columnar.pack({"d": days}, NO_TYPES)[1][0], np.int32) == [3, -4, 0]
    # a date64 is floor-divided to days: 2 days, then -1 ms is the day before the epoch
    assert _ticks(columnar.pack({"m": day_ms}, NO_TYPES)[1][0], np.int32) == [2, -1, 0]
    n = columnar.pack({"n": amounts}, NO_TYPES)[1][0]
    assert n.kind == "n" and _bytes(n.aux) == b"1.50"
    x = columnar.pack({"x": blobs}, NO_TYPES)[1][0]
    assert (
        x.kind == "x" and _bytes(x.aux) == b"abc" and _present(x) == [True, False, True]
    )


def test_ragged_columns_and_empty_input_are_left_to_the_row_path():
    assert columnar.pack({"a": [1, 2, 3], "b": [1, 2]}, NO_TYPES) is None
    assert columnar.pack({"a": []}, NO_TYPES) is None
    assert columnar.pack({}, NO_TYPES) is None
    assert columnar.pack({"a": (x for x in range(3))}, NO_TYPES) is None


def test_the_address_table_and_header_describe_the_columns():
    rows, packed = columnar.pack({"id": np.arange(3), "s": ["a", "bc", None]}, NO_TYPES)
    spec = columnar.spec_of(packed)
    assert len(spec) == 12  # six int64 per column
    assert spec[0] == packed[0].data[0] and spec[1] == 24
    assert spec[6] == packed[1].data[0] and spec[7] == 16  # n + 1 offsets of 4 bytes
    assert spec[8] == packed[1].aux[0] and spec[9] == 3
    assert spec[10] == packed[1].nulls[0] and spec[11] == 1
    import json

    assert json.loads(columnar.header_of(rows, 10_000, packed)) == {
        "n": 3,
        "commit": 10_000,
        "cols": [["id", "q", 0], ["s", "s", 0]],
    }
    assert json.loads(columnar.header_of(rows, 2**40, packed))["commit"] == 2**31 - 1
    # "parallel" and the failure token are in the header only when a token is given (the image hands the records to the async writers then)
    parallel = json.loads(columnar.header_of(rows, 10_000, packed, fid=41))
    assert parallel["parallel"] is True and parallel["fid"] == 41
    plain = json.loads(columnar.header_of(rows, 10_000, packed))
    assert "parallel" not in plain and "fid" not in plain


def test_rows_become_columns_only_when_every_row_has_the_same_keys():
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert columnar.rows_to_columns(rows) == {"a": [1, 2], "b": ["x", "y"]}
    assert columnar.rows_to_columns([{"a": 1}, {"a": 2, "b": 3}]) is None
    assert (
        columnar.rows_to_columns([{"a": 1, "b": 2}, {"a": 1, "c": 2}]) is None
    )  # same size, other key
    assert columnar.rows_to_columns([{1: "x"}]) is None
    assert columnar.rows_to_columns([]) is None
    assert columnar.rows_to_columns([[1, 2]]) is None


def test_arrow_strings_cross_in_place_with_their_offset_and_validity():
    import pyarrow as pa

    arr = pa.array(["a", None, "héllo", "", "xyz", "tail"])
    rows, cols = _pack({"s": arr})
    assert rows == 6 and cols["s"].kind == "s"
    assert _present(cols["s"]) == [True, False, True, True, True, True]
    assert [t for t, ok in zip(_strings(cols["s"]), _present(cols["s"])) if ok] == [
        "a",
        "héllo",
        "",
        "xyz",
        "tail",
    ]
    # a slice keeps the buffers and moves the offsets: the offsets no longer start at 0
    part = arr.slice(2, 3)
    _, cols = _pack({"s": part})
    assert _strings(cols["s"]) == ["héllo", "", "xyz"]
    assert int(np.frombuffer(_bytes(cols["s"].data), dtype=np.int32)[0]) > 0
    big = pa.array(["a", "bc"], type=pa.large_string())
    _, cols = _pack({"s": big})
    assert _strings(cols["s"]) == ["a", "bc"]


def test_arrow_numbers_with_nulls_and_arrow_vectors():
    import pyarrow as pa

    _, cols = _pack(
        {
            "n": pa.array([1, None, 3], type=pa.int64()),
            "b": pa.array([True, None, False]),
        }
    )
    assert _present(cols["n"]) == [True, False, True] and cols["n"].kind == "q"
    assert cols["b"].kind == "b" and _present(cols["b"]) == [True, False, True]
    vec = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32()))
    assert columnar.pack({"v": vec}, NO_TYPES) is None
    _, cols = _pack({"v": vec}, {"v": "ARRAY_OF_FLOATS"})
    assert (cols["v"].kind, cols["v"].dim) == ("f", 2)
    fixed = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2))
    _, cols = _pack({"v": fixed}, {"v": "ARRAY_OF_FLOATS"})
    assert np.frombuffer(_bytes(cols["v"].data), dtype=np.float32).tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    ragged = pa.array([[1.0], [1.0, 2.0]], type=pa.list_(pa.float32()))
    assert (
        columnar.pack({"v": ragged}, columnar.Declared(known={"v": "ARRAY_OF_FLOATS"}))
        is None
    )


# ---- Declared: the schema is read once per object, so once per ingest call (#315)


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def to_list(self):
        return self._rows


class _Handle:
    """Just enough of a database handle for ``Declared``: a ``query`` that counts the schema reads, and ``_command_count``, which a
    test moves the way a statement run on the handle does."""

    def __init__(self, properties):
        self.properties = properties
        self.reads = 0
        self.unreadable = False
        self._command_count = 0

    def query(self, language, command):
        self.reads += 1
        if self.unreadable:
            raise RuntimeError("the schema cannot be read")
        entries = [{"name": n, "type": t} for n, t in self.properties.items()]
        return _Rows([{"name": "T", "properties": entries, "parentTypes": []}])


def test_declared_reads_the_schema_once_and_again_only_after_a_statement():
    handle = _Handle({"v": "ARRAY_OF_FLOATS"})
    declared = columnar.Declared(handle, "T")
    assert handle.reads == 0  # nothing is read until a column asks
    assert [declared.get("v") for _ in range(50)] == ["ARRAY_OF_FLOATS"] * 50
    assert declared.get("w") is None
    assert handle.reads == 1
    handle.properties["w"] = "FLOAT"
    assert declared.get("w") is None and handle.reads == 1  # no statement ran: kept
    handle._command_count += 1  # a statement ran on the handle (a DDL, say)
    assert declared.get("w") == "FLOAT" and handle.reads == 2
    assert declared.get("v") == "ARRAY_OF_FLOATS" and handle.reads == 2


def test_declared_with_known_types_never_reads_the_schema():
    handle = _Handle({"v": "FLOAT"})
    declared = columnar.Declared(handle, "T", {"v": "ARRAY_OF_FLOATS"})
    handle._command_count += 1
    assert declared.get("v") == "ARRAY_OF_FLOATS" and handle.reads == 0


def test_declared_asks_again_while_the_schema_cannot_be_read():
    handle = _Handle({"v": "FLOAT"})
    handle.unreadable = True
    declared = columnar.Declared(handle, "T")
    assert declared.get("v") is None and handle.reads == 1
    handle.unreadable = False
    assert declared.get("v") == "FLOAT" and handle.reads == 2


def test_declared_without_a_handle_or_types_answers_none():
    assert columnar.Declared().get("v") is None
