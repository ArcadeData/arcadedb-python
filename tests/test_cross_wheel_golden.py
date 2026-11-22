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

Four more cases of #314 follow the first six tests: the text of ``Result.to_json()``, the float-list columns of ``to_arrow()``, the
distances of a vector search and an integral DECIMAL in ``to_json_list()``. Each asserts what the two builds return alike. Where they
do not return the same today, the test pins the part that is equal and its docstring gives the value of each build, measured on the
wheels CI built from ``62195709b7`` (``arcadedb-jvm`` on engine 26.12.1-SNAPSHOT, ``arcadedb`` on engine 26.11.1) and on
``arcadedb-jvm`` 26.11.1.post4. These differences are open in #314, a literal for them would fail on one build:

- ``to_json()`` writes a FLOAT and a float32 ``float[]`` as the shortest float32 text on the JVM build (``0.1``) and as the exact
  widening on the native build (``0.10000000149011612``); a DECIMAL keeps its scale on the JVM build (``1.50``) and is a float text on
  the native build (``1.5``); in a projection row a DATE is epoch milliseconds and a DATETIME is text on the JVM build, days and
  milliseconds on the native build; the keys of a record row come in a different order.
- The distance of a vector search can differ in the last bits of the float32 between the builds, and between two calls on the JVM
  build.
"""

import json
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pyarrow as pa
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


# ---- Result.to_json(): the text of a row ----

# INTEGER, DOUBLE, LONG, NULL and LIST: the text is the same on both builds, with no space and in the order of the projection
JSON_TEXT_SELECT = "SELECT n, d, big, maybe, lst FROM Golden ORDER BY n"
GOLDEN_JSON_TEXT = [
    '{"n":0,"d":0.1,"big":9007199254740993,"maybe":0.25,"lst":[1,"a",2.5]}',
    '{"n":1,"d":0.3333333333333333,"big":-9007199254740993,"maybe":null,"lst":["x"]}',
]

# a record row (SELECT FROM Golden), parsed: a DATE is epoch days there, a DATETIME epoch milliseconds, a DECIMAL a JSON number
GOLDEN_RECORD_JSON = [
    {
        "n": 0,
        "d": 0.1,
        "dec": 1.5,
        "big": 9007199254740993,
        "day": 20455,
        "ts": 1767323045678,
        "maybe": 0.25,
        "lst": [1, "a", 2.5],
        "@cat": "d",
        "@type": "Golden",
    },
    {
        "n": 1,
        "d": 1 / 3,
        "dec": -0.25,
        "big": -9007199254740993,
        "day": -1,
        "ts": -1,
        "maybe": None,
        "lst": ["x"],
        "@cat": "d",
        "@type": "Golden",
    },
]
RECORD_JSON_KEYS = {
    "n",
    "f",
    "d",
    "v3",
    "v9",
    "dec",
    "big",
    "day",
    "ts",
    "maybe",
    "lst",
    "@cat",
    "@type",
    "@rid",
}


def test_to_json_text_of_a_projection_row_golden(golden):
    """The text of ``Result.to_json()`` for the columns both builds write alike: INTEGER, DOUBLE, LONG beyond 2**53, NULL and LIST.

    The other columns are not in this projection because their text differs today (#314, open). For row 0 the JVM build wrote
    ``"f":0.1``, ``"v3":[0.1,0.33333334,0.5]``, ``"dec":1.50``, ``"day":1767312000000`` and ``"ts":"2026-01-02 03:04:05"``; the native
    build wrote ``"f":0.10000000149011612``, ``"v3":[0.10000000149011612,0.3333333432674408,0.5]``, ``"dec":1.5``, ``"day":20455`` and
    ``"ts":1767323045678``.
    """
    texts = [row.to_json() for row in golden.query("sql", JSON_TEXT_SELECT)]
    assert texts == GOLDEN_JSON_TEXT


def test_to_json_of_a_record_row_golden(golden):
    """``Result.to_json()`` of a record row reads back to the same values on both builds, the float32 values as float32.

    The text itself is not compared: the keys come in a different order on the two builds, the JVM build writes a FLOAT and a
    float32 ``float[]`` as the shortest float32 text (``0.1``) where the native build writes the exact widening
    (``0.10000000149011612``), and a DECIMAL as ``1.50`` where the native build writes ``1.5`` (#314, open). Both texts of a float32
    value name the same float32, which is what the last three assertions pin.
    """
    rows = list(golden.query("sql", "SELECT FROM Golden ORDER BY n"))
    assert len(rows) == len(GOLDEN_RECORD_JSON)
    for row, want, stored in zip(rows, GOLDEN_RECORD_JSON, ROWS):
        have = json.loads(row.to_json())
        assert set(have) == RECORD_JSON_KEYS
        assert {name: have[name] for name in want} == want
        assert have["@rid"] == str(row.get_rid())
        assert np.float32(have["f"]) == np.float32(stored["f"])
        for name in ("v3", "v9"):
            np.testing.assert_array_equal(
                np.array(have[name], dtype=np.float32),
                np.array(stored[name], dtype=np.float32),
            )


# ---- to_arrow(): float-list columns, and to_json_list(): an integral DECIMAL ----

# fl holds doubles that every float32 can hold, fd holds doubles that no float32 can hold: both are a LIST of doubles
MORE_ROWS = [
    {
        "n": 0,
        "whole": Decimal("3"),
        "scaled": Decimal("2.00"),
        "huge": Decimal("12345678901234567890"),
        "fl": [0.5, 0.25, -2.75],
        "fd": [0.1, 1 / 3],
    },
    {
        "n": 1,
        "whole": Decimal("-7"),
        "scaled": Decimal("-4.0"),
        "huge": Decimal("-12345678901234567890"),
        "fl": [1.0, 2.0, 4.0],
        "fd": [0.2, 0.7],
    },
]
MORE_SELECT = "SELECT n, whole, scaled, fl, fd FROM GoldenMore ORDER BY n"
DECIMAL_SELECT = "SELECT n, whole, scaled, huge FROM GoldenMore ORDER BY n"


@pytest.fixture
def golden_more(temp_db):
    db = temp_db
    db.command("sql", "CREATE DOCUMENT TYPE GoldenMore")
    for name, type_name in [
        ("n", "INTEGER"),
        ("whole", "DECIMAL"),
        ("scaled", "DECIMAL"),
        ("huge", "DECIMAL"),
        ("fl", "LIST"),
        ("fd", "LIST"),
    ]:
        db.command("sql", f"CREATE PROPERTY GoldenMore.{name} {type_name}")
    db.insert_many("GoldenMore", [dict(row) for row in MORE_ROWS])
    return db


def test_to_arrow_float32_vector_columns_golden(golden):
    """A float32 ``float[]`` column is a ``FixedSizeList<float32>`` Arrow column on both builds, its values the stored float32."""
    table = golden.query("sql", SELECT).to_arrow()
    for name, size in (("v3", 3), ("v9", 9)):
        column_type = table.schema.field(name).type
        assert column_type == pa.list_(pa.float32(), size), (name, column_type)
        assert column_type.value_type == pa.float32()
        assert column_type.list_size == size
        assert table.column(name).to_pylist() == [row[name] for row in ROWS], name


@pytest.mark.parametrize("name", ["fl", "fd"])
def test_to_arrow_list_of_doubles_column_golden(golden_more, name):
    """A LIST of floats is a ``list<double>`` Arrow column on both builds, also when every value would fit a float32."""
    table = golden_more.query("sql", MORE_SELECT).to_arrow()
    column_type = table.schema.field(name).type
    assert column_type == pa.list_(pa.float64()), column_type
    assert column_type.value_type == pa.float64()
    assert table.column(name).to_pylist() == [row[name] for row in MORE_ROWS]


def test_to_json_list_integral_decimal(golden_more):
    """Every DECIMAL in ``to_json_list()`` is a float on both builds, whatever its scale (#452).

    ``to_json_list()`` documents a DECIMAL as a float. The JVM build used to return the int ``3`` for ``Decimal("3")`` (a DECIMAL
    of scale 0 has no fraction digit in its JSON) and the int ``12345678901234567890`` for the 20-digit value, where the native
    build returned the floats ``3.0`` and ``1.2345678901234567e+19``, so a column held two Python types and the numbers differed.
    Both builds return the float of the value now: ``3.0``, ``-7.0``, the scaled ``2.0`` (a float on both before) and the
    20-digit values, rounded to the nearest double. ``to_list()`` returns the ``Decimal`` with every digit on both.
    """
    rows = golden_more.query("sql", DECIMAL_SELECT).to_json_list()
    assert [row["whole"] for row in rows] == [3.0, -7.0]
    assert [row["scaled"] for row in rows] == [2.0, -4.0]
    assert [row["huge"] for row in rows] == [
        1.2345678901234567e19,
        -1.2345678901234567e19,
    ]
    for name in ("whole", "scaled", "huge"):
        assert all(type(row[name]) is float for row in rows), name
    batches = golden_more.query("sql", DECIMAL_SELECT).iter_json_batches(batch_size=1)
    assert [row for batch in batches for row in batch] == rows
    as_list = golden_more.query("sql", DECIMAL_SELECT).to_list()
    assert [row["whole"] for row in as_list] == [Decimal("3"), Decimal("-7")]
    assert [row["huge"] for row in as_list] == [
        Decimal("12345678901234567890"),
        Decimal("-12345678901234567890"),
    ]
    for name in ("whole", "scaled", "huge"):
        assert all(type(row[name]) is Decimal for row in as_list), name


# ---- vector distances ----

VECTORS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
    [3.0, 4.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    # float32(0.1), float32(0.2), float32(0.3), float32(0.4)
    [0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645],
    # float32(0.7), float32(0.9), float32(1/3), float32(0.001)
    [0.699999988079071, 0.8999999761581421, 0.3333333432674408, 0.0010000000474974513],
]
UNIT_QUERY = [1.0, 0.0, 0.0, 0.0]
# float32(0.3), float32(0.6), float32(0.9), float32(0.1)
F32_QUERY = [
    0.30000001192092896,
    0.6000000238418579,
    0.8999999761581421,
    0.10000000149011612,
]

# UNIT_QUERY against vectors 0 to 5: every product and every partial sum is exact in a float32, so the order of the additions
# cannot change a bit. EUCLIDEAN is the squared distance, COSINE is 1 - cosine: 0.3999999761581421 is 1 - float32(0.6), not 0.4.
GOLDEN_EXACT_DISTANCES = {
    "euclidean": {0: 0.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 20.0, 5: 4.0},
    "cosine": {0: 0.0, 1: 1.0, 3: 0.5, 4: 0.3999999761581421, 5: 2.0},
}
# F32_QUERY against all eight: the values of the native build, which returns the same bits on every call
GOLDEN_F32_DISTANCES = {
    "euclidean": {
        0: 1.6700000762939453,
        1: 1.0699999332427979,
        2: 1.4699997901916504,
        3: 0.3700000047683716,
        4: 19.670001983642578,
        5: 2.869999885559082,
        6: 0.6499999761581421,
        7: 0.5809118747711182,
    },
    "cosine": {
        0: 0.7337930202484131,
        1: 0.46758604049682617,
        2: 0.43528974056243896,
        3: 0.1570112705230713,
        4: 0.4143446683883667,
        5: 1.266206979751587,
        6: 0.25476133823394775,
        7: 0.21558105945587158,
    },
}
# a few float32 steps: the builds differed by up to 4 steps of the last float32 bit (relative 6e-7), this leaves room for the
# SIMD widths of the other platforms and still tells a distance from a wrong one
F32_DISTANCE_TOLERANCE = 1e-5


def _vector_index(db, metric):
    type_name = f"GoldenVec{metric.capitalize()}"
    db.command("sql", f"CREATE VERTEX TYPE {type_name}")
    db.command("sql", f"CREATE PROPERTY {type_name}.id INTEGER")
    db.command("sql", f"CREATE PROPERTY {type_name}.emb ARRAY_OF_FLOATS")
    with db.transaction():
        for i, vector in enumerate(VECTORS):
            db.command("sql", f"INSERT INTO {type_name} SET id = ?, emb = ?", i, vector)
    # quantization NONE: the distance is that of the stored float32 values, not of an INT8 approximation
    return db.create_vector_index(
        type_name,
        "emb",
        dimensions=4,
        id_property="id",
        distance_function=metric,
        quantization="NONE",
    )


def _distances(index, query):
    hits = index.find_nearest(query, k=len(VECTORS))
    assert all(type(score) is float for _, score in hits)
    return {int(record.get("id")): score for record, score in hits}


@pytest.mark.parametrize("metric", sorted(GOLDEN_EXACT_DISTANCES))
def test_vector_distances_exact_golden(temp_db, metric):
    """Distances that are exact in float32 arithmetic are the same bits on both builds, on every call."""
    index = _vector_index(temp_db, metric)
    want = GOLDEN_EXACT_DISTANCES[metric]
    for _ in range(20):
        have = _distances(index, UNIT_QUERY)
        assert set(have) == set(range(len(VECTORS)))
        assert {i: have[i] for i in want} == want


@pytest.mark.parametrize("metric", sorted(GOLDEN_F32_DISTANCES))
def test_vector_distances_float32_golden(temp_db, metric):
    """Distances that round in float32 are equal to the golden within a few float32 steps, not to the bit (#314, open).

    Measured with this table: the native build returned the golden bits on each of 400 calls. The JVM build on engine
    26.12.1-SNAPSHOT returned ``19.670000076293945`` for vector 4 and ``2.869999408721924`` for vector 5 (EUCLIDEAN) on 396 of 400
    calls and the golden bits on the other 4, and ``0.2547614574432373`` and ``0.21558117866516113`` for vectors 6 and 7 (COSINE) on
    every call. Why the JVM build changes between calls was not looked into here.
    """
    index = _vector_index(temp_db, metric)
    want = GOLDEN_F32_DISTANCES[metric]
    for _ in range(20):
        have = _distances(index, F32_QUERY)
        assert set(have) == set(want)
        for i, distance in want.items():
            assert have[i] == pytest.approx(
                distance, rel=F32_DISTANCE_TOLERANCE, abs=F32_DISTANCE_TOLERANCE
            ), (i, have[i], distance)
