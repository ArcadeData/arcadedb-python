"""One fixed table, committed golden literals, run on both wheels: the two builds return the same values (#314).

``tests/`` runs against each installed wheel, but until now a value was checked against one literal in a test
that only one wheel could run, so a difference between the wheels stayed invisible. This file has no branch on
``arcadedb.__backend__``: the golden literals below are asserted on ``arcadedb`` (native) and on ``arcadedb-jvm``
alike, so a literal that holds on both IS the parity check.

The table holds a FLOAT, a DOUBLE, two float32 ``float[]`` (3 values, which the native build writes element by
element, and 9, which it sends as a float32 tail), a DECIMAL, a LONG beyond 2**53, a DATE, a DATETIME, a NULL and an
embedded list. Every float32 input is written as the exact float64 widening of that float32, so the stored value is
the literal. ``to_json_list()`` of a FLOAT or a float32 vector used to differ: the JVM build printed the shortest
float32 text (``0.1``), which Python reads as a different double than the exact widening
(``0.10000000149011612``) that the native build and ``to_list()`` return.
"""

from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pytest

SELECT = "SELECT n, f, d, v3, v9, dec, big, day, ts, maybe, lst FROM Golden ORDER BY n"

# float32 values written as the float64 they widen to, so each is exact in a FLOAT and in a float32 vector
F32_TENTH = 0.10000000149011612  # float32(0.1)
F32_THIRD = 0.3333333432674408  # float32(1/3)
F32_MILLI = 0.0010000000474974513  # float32(0.001)
F32_SEVEN_TENTHS = 0.699999988079071  # float32(0.7)
F32_NINE_TENTHS = 0.8999999761581421  # float32(0.9)

V9_ROW0 = [
    0.10000000149011612,
    0.20000000298023224,
    0.30000001192092896,
    0.4000000059604645,
    0.5,
    0.6000000238418579,
    0.699999988079071,
    0.800000011920929,
    0.8999999761581421,
]
V9_ROW1 = [
    F32_MILLI,
    F32_SEVEN_TENTHS,
    F32_NINE_TENTHS,
    0.5,
    F32_THIRD,
    F32_TENTH,
    1.0,
    2.0,
    -2.75,
]

ROWS = [
    {
        "n": 0,
        "f": F32_TENTH,
        "d": 0.1,
        "v3": [F32_TENTH, F32_THIRD, 0.5],
        "v9": V9_ROW0,
        "dec": Decimal("1.50"),
        "big": 2**53 + 1,
        "day": date(2026, 1, 2),
        "ts": datetime(2026, 1, 2, 3, 4, 5, 678000),
        "maybe": 0.25,
        "lst": [1, "a", 2.5],
    },
    {
        "n": 1,
        "f": 0.5,
        "d": 1 / 3,
        "v3": [0.5, 1.0, -2.75],
        "v9": V9_ROW1,
        "dec": Decimal("-0.25"),
        "big": -(2**53) - 1,
        "day": date(1969, 12, 31),
        "ts": datetime(1969, 12, 31, 23, 59, 59, 999000),
        "maybe": None,
        "lst": ["x"],
    },
]

# to_list: the Python types of the values as stored
GOLDEN_LIST = ROWS

# to_json_list: JSON-native values; DATE and DATETIME as epoch milliseconds, DECIMAL as a float
GOLDEN_JSON = [
    {
        "n": 0,
        "f": F32_TENTH,
        "d": 0.1,
        "v3": [F32_TENTH, F32_THIRD, 0.5],
        "v9": V9_ROW0,
        "dec": 1.5,
        "big": 9007199254740993,
        "day": 1767312000000,
        "ts": 1767323045678,
        "maybe": 0.25,
        "lst": [1, "a", 2.5],
    },
    {
        "n": 1,
        "f": 0.5,
        "d": 1 / 3,
        "v3": [0.5, 1.0, -2.75],
        "v9": V9_ROW1,
        "dec": -0.25,
        "big": -9007199254740993,
        "day": -86400000,
        "ts": -1,
        "maybe": None,
        "lst": ["x"],
    },
]

# to_columns: one array per column, the numpy dtype is part of the golden
GOLDEN_COLUMNS = {
    "n": np.array([0, 1], dtype=np.int64),
    "f": np.array([F32_TENTH, 0.5], dtype=np.float64),
    "d": np.array([0.1, 1 / 3], dtype=np.float64),
    "v3": np.array([[F32_TENTH, F32_THIRD, 0.5], [0.5, 1.0, -2.75]], dtype=np.float32),
    "v9": np.array([V9_ROW0, V9_ROW1], dtype=np.float32),
    "big": np.array([9007199254740993, -9007199254740993], dtype=np.int64),
    "day": np.array(["2026-01-02", "1969-12-31"], dtype="datetime64[ms]"),
    "ts": np.array(
        ["2026-01-02T03:04:05.678", "1969-12-31T23:59:59.999"], dtype="datetime64[ms]"
    ),
    "maybe": np.array([0.25, np.nan], dtype=np.float64),
}


@pytest.fixture
def golden(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE Golden")
    for name, type_name in [
        ("n", "INTEGER"),
        ("f", "FLOAT"),
        ("d", "DOUBLE"),
        ("v3", "ARRAY_OF_FLOATS"),
        ("v9", "ARRAY_OF_FLOATS"),
        ("dec", "DECIMAL"),
        ("big", "LONG"),
        ("day", "DATE"),
        ("ts", "DATETIME"),
        ("maybe", "DOUBLE"),
        ("lst", "LIST"),
    ]:
        db.command("sql", f"CREATE PROPERTY Golden.{name} {type_name}")
    db.insert_many("Golden", [dict(row) for row in ROWS])
    return db


def _json_rows(db):
    return db.query("sql", SELECT).to_json_list()


def test_to_json_list_floats_are_the_exact_widening_of_the_float32(golden):
    """The float cases of #314: FLOAT, float32 float[] of 3 and of 9 values, next to a DOUBLE and the shortest-text trap."""
    rows = _json_rows(golden)
    for have, want in zip(rows, GOLDEN_JSON):
        for name in ("f", "d", "v3", "v9", "maybe"):
            assert have[name] == want[name], (have["n"], name)
    # the shortest text of float32(0.1) is "0.1", which Python reads as a different double
    assert rows[0]["f"] != 0.1
    assert rows[0]["f"] == 0.10000000149011612


def test_to_json_list_floats_equal_to_list_floats(golden):
    """The two readers of the same rows agree: row["v9"] == to_list()[i]["v9"] (false on the JVM build before #314)."""
    as_json = _json_rows(golden)
    as_list = golden.query("sql", SELECT).to_list()
    for from_json, from_list in zip(as_json, as_list):
        for name in ("f", "d", "v3", "v9", "maybe"):
            assert from_json[name] == from_list[name], (from_json["n"], name)


def test_to_json_list_golden(golden):
    assert _json_rows(golden) == GOLDEN_JSON
    rows = _json_rows(golden)
    assert type(rows[0]["dec"]) is float
    assert type(rows[0]["big"]) is int and type(rows[0]["day"]) is int
    assert type(rows[0]["ts"]) is int
    assert all(type(x) is float for row in rows for x in row["v3"] + row["v9"])


def test_to_list_golden(golden):
    rows = golden.query("sql", SELECT).to_list()
    assert rows == GOLDEN_LIST
    assert type(rows[0]["dec"]) is Decimal
    assert type(rows[0]["big"]) is int
    assert type(rows[0]["day"]) is date
    assert type(rows[0]["ts"]) is datetime
    assert all(type(x) is float for row in rows for x in row["v3"] + row["v9"])


def _as_list(column):
    return [list(x) if isinstance(x, (list, np.ndarray)) else x for x in column]


@pytest.mark.parametrize("name", sorted(GOLDEN_COLUMNS))
def test_to_columns_golden(golden, name):
    column = golden.query("sql", SELECT).to_columns()[name]
    want = GOLDEN_COLUMNS[name]
    assert isinstance(column, np.ndarray), type(column)
    assert column.dtype == want.dtype, (column.dtype, want.dtype)
    assert column.shape == want.shape
    np.testing.assert_array_equal(column, want)


def test_to_columns_decimal_and_list_columns(golden):
    columns = golden.query("sql", SELECT).to_columns()
    assert list(columns["dec"]) == [Decimal("1.50"), Decimal("-0.25")]
    assert all(type(v) is Decimal for v in columns["dec"])
    assert _as_list(columns["lst"]) == [[1, "a", 2.5], ["x"]]
