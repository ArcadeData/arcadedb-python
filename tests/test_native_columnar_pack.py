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


def test_a_list_of_equal_float_lists_is_a_vector_matrix_for_an_array_of_floats_property():
    lists = [[0.5, 1.5], [2.5, 3.5]]
    _, cols = _pack({"v": lists}, {"v": "ARRAY_OF_FLOATS"})
    assert (cols["v"].kind, cols["v"].dim) == ("f", 2)  # a matrix, not a ragged list
    _, cols = _pack({"v": lists})  # undeclared: the JSON path keeps the doubles
    assert (cols["v"].kind, cols["v"].dim) == ("D", 0)


def _lists(col):
    """(offsets, elements, present) of a ragged list column as read back from the addresses."""
    offsets = np.frombuffer(_bytes(col.data), dtype=np.int32)
    dtype = {"I": np.int32, "Q": np.int64, "E": np.float32, "D": np.float64}[col.kind]
    elements = np.frombuffer(_bytes(col.aux), dtype=dtype)
    return (
        offsets.tolist(),
        elements.tolist(),
        _present(col) if col.nulls[1] else None,
    )


def test_ragged_int_and_float_lists_are_offsets_plus_elements():
    tokens = [[5, 7, 9], [1], [], [2, 4]]
    weights = [[0.5, 1.25, 2.0], [3.5], [], [0.1, 0.2]]
    rows, cols = _pack(
        {"t": tokens, "w": weights, "t2": tokens, "w2": weights},
        {"t": "ARRAY_OF_INTEGERS", "w": "ARRAY_OF_FLOATS", "t2": "LIST", "w2": "LIST"},
    )
    assert rows == 4
    assert [cols[k].kind for k in ("t", "w", "t2", "w2")] == ["I", "E", "Q", "D"]
    assert all(cols[k].n == 4 and cols[k].dim == 0 for k in cols)
    assert _lists(cols["t"]) == ([0, 3, 4, 4, 6], [5, 7, 9, 1, 2, 4], None)
    assert _lists(cols["t2"]) == ([0, 3, 4, 4, 6], [5, 7, 9, 1, 2, 4], None)
    assert _lists(cols["w"])[0] == [0, 3, 4, 4, 6]
    assert _lists(cols["w"])[1] == np.array(sum(weights, []), dtype=np.float32).tolist()
    assert _lists(cols["w2"])[1] == sum(
        weights, []
    )  # float64: exactly the Python floats


def test_undeclared_lists_cross_as_the_64_bit_kinds_the_json_text_parses_to():
    _, cols = _pack({"t": [[1, 2], [3]], "w": [[0.5], [1.5, 2.5]]})
    assert (cols["t"].kind, cols["w"].kind) == ("Q", "D")
    big = [[-(2**63), 2**63 - 1], [2**31, -(2**31) - 1]]
    _, cols = _pack({"t": big})
    assert _lists(cols["t"])[1] == [-(2**63), 2**63 - 1, 2**31, -(2**31) - 1]


def test_a_declared_float_array_narrows_to_float32_and_a_huge_double_becomes_infinity():
    _, cols = _pack({"w": [[1e300, 0.1], [-1e300]]}, {"w": "ARRAY_OF_FLOATS"})
    got = _lists(cols["w"])[1]
    assert got[0] == float("inf") and got[2] == float("-inf")
    assert got[1] == float(np.float32(0.1))


def test_null_and_empty_rows_of_a_list_column():
    _, cols = _pack({"t": [[1, 2], None, [], [3]], "w": [None, None, [0.5], None]})
    assert _lists(cols["t"]) == ([0, 2, 2, 2, 3], [1, 2, 3], [True, False, True, True])
    assert _lists(cols["w"]) == ([0, 0, 0, 1, 1], [0.5], [False, False, True, False])
    _, cols = _pack({"t": [[], []]}, {"t": "ARRAY_OF_INTEGERS"})
    assert (cols["t"].kind, cols["t"].n, cols["t"].aux) == ("I", 2, (0, 0))
    assert _lists(cols["t"])[0] == [0, 0, 0]
    for declared, kind in (
        ({}, "Q"),
        ({"t": "LIST"}, "Q"),
        ({"t": "ARRAY_OF_FLOATS"}, "E"),
    ):
        _, cols = _pack({"t": [[], []]}, declared)
        assert cols["t"].kind == kind
    _, cols = _pack({"t": [None, [1.5]]}, {"t": "ARRAY_OF_FLOATS"})
    assert cols["t"].kind == "E"  # a null row keeps a float list off the fixed matrix
    assert _lists(cols["t"]) == ([0, 0, 1], [1.5], [False, True])


def test_the_offsets_of_a_long_ragged_list_column_index_its_elements():
    lists = [list(range(k % 7)) for k in range(25_000)]
    rows, cols = _pack({"t": lists})
    offsets, elements, _ = _lists(cols["t"])
    assert rows == 25_000 and offsets[-1] == len(elements) == sum(map(len, lists))
    assert elements[offsets[1234] : offsets[1235]] == lists[1234]


@pytest.mark.parametrize(
    "column, declared",
    [
        ([[1, 2.5]], {}),  # mixed int and float elements
        ([[True, False]], {}),  # bool is not an int here
        ([[1], [2.5]], {}),  # an int row and a float row
        ([["a"]], {}),
        ([[[1]], [[2]]], {}),  # nested
        ([[1], (2, 3)], {}),  # a tuple row
        ([(1, 2), (3,)], {}),
        ([[np.int64(1)]], {}),
        ([[1, 2**63]], {}),  # past 64 bits
        ([[float("nan")]], {}),
        ([[1.0], [2.0, float("inf")]], {"t": "ARRAY_OF_FLOATS"}),  # ragged: no matrix
        ([[2**31]], {"t": "ARRAY_OF_INTEGERS"}),  # the engine refuses it: JSON path
        ([[-(2**31) - 1]], {"t": "ARRAY_OF_INTEGERS"}),
        ([[1.5]], {"t": "ARRAY_OF_INTEGERS"}),
        ([[1]], {"t": "ARRAY_OF_LONGS"}),  # a declared type the kinds do not cover
        ([[1.0]], {"t": "ARRAY_OF_DOUBLES"}),
        ([[1]], {"t": "STRING"}),
    ],
)
def test_list_columns_the_buffers_cannot_carry_unchanged_are_refused(column, declared):
    named = {"t": column}
    assert columnar.pack(named, columnar.Declared(known=declared)) is None


def test_a_library_without_the_list_decoder_keeps_lists_on_the_json_path():
    off = columnar.Declared(
        known={"t": "ARRAY_OF_INTEGERS", "w": "ARRAY_OF_FLOATS"}, lists=False
    )
    assert columnar.pack({"t": [[1], [2, 3]]}, off) is None
    assert columnar.pack({"w": [[1.0], [2.0, 3.0]]}, off) is None
    assert (
        columnar.pack({"id": [1, 2], "w": [[1.0, 2.0], [3.0, 4.0]]}, off) is not None
    )  # a float matrix needs no decoder
    assert columnar.pack({"s": ["a", "b"]}, off) is not None


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
        [b"ab", b"cd"],
        [np.int64(1), np.int64(2)],  # numpy scalars in a list are not Python numbers
    ],
)
def test_columns_the_buffer_cannot_carry_unchanged_are_refused(column):
    assert columnar.pack({"c": column}, NO_TYPES) is None


def test_datetimes_decimals_and_object_arrays_of_them_are_refused():
    import datetime
    import decimal

    assert columnar.pack({"d": [datetime.datetime(2020, 1, 1)]}, NO_TYPES) is None
    assert columnar.pack({"d": [decimal.Decimal("1.5")]}, NO_TYPES) is None
    assert (
        columnar.pack({"d": np.array(["2020-01-01"], dtype="datetime64[D]")}, NO_TYPES)
        is None
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
