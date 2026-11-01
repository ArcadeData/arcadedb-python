"""NaN, Infinity and -Infinity survive every reader, at the top level and inside a list or a map.

JSON has no literal for them and the engine's JSON writers turn them into null, so the JVM build
returned None from ``to_json_list()`` and, for a value inside a list or a map, from ``to_columns()``,
``to_dataframe()`` and ``to_arrow()`` (the native build kept them). The bridge now writes them as a
tag that Python turns back into a float, so both builds return a Python float where the object reader
``to_list()`` always did. ``to_list()`` is the control in every test below.

A top-level number is a float64 column in the columnar readers, which kept NaN already.
Shared by both builds: ``tests/conftest.py`` runs this file against the native wheel as well.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

import arcadedb

VALUES = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="ninf"),
]
SHAPES = ["top", "list", "map"]


def _wrap(shape, value):
    if shape == "list":
        return [value]
    if shape == "map":
        return {"x": value}
    return value


def _same(actual, expected):
    """One float: NaN is NaN, an infinity is that infinity (with its sign), and None fails."""
    assert actual is not None, "a non-finite float came back as None"
    actual = float(actual)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert math.isinf(actual) and actual == expected


def _check(actual, shape, value):
    if shape == "list":
        assert len(actual) == 1
        _same(actual[0], value)
    elif shape == "map":
        assert list(actual) == ["x"]
        _same(actual["x"], value)
    else:
        _same(actual, value)


@pytest.fixture
def db(temp_db_path):
    with arcadedb.create_database(temp_db_path) as database:
        database.command("sql", "CREATE DOCUMENT TYPE Num")
        yield database


def _insert(db, shape, value, k=0):
    with db.transaction():
        db.command(
            "sql",
            "INSERT INTO Num SET k = :k, v = :v",
            {"k": k, "v": _wrap(shape, value)},
        )


QUERY = "SELECT v FROM Num ORDER BY k"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_to_list_keeps_it(db, shape, value):
    """The control: the object reader always kept it."""
    _insert(db, shape, value)
    _check(db.query("sql", QUERY).to_list()[0]["v"], shape, value)


@pytest.mark.parametrize("batch_size", [10_000, 1])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_to_json_list_returns_a_float(db, shape, value, batch_size):
    """to_json_list() gives Python floats, as the native build does (JSON-native types otherwise)."""
    _insert(db, shape, value)
    row = db.query("sql", QUERY).to_json_list(batch_size=batch_size)[0]
    _check(row["v"], shape, value)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_iter_json_batches_returns_a_float(db, shape, value):
    _insert(db, shape, value)
    (batch,) = list(db.query("sql", QUERY).iter_json_batches(batch_size=5))
    _check(batch[0]["v"], shape, value)


@pytest.mark.parametrize("batch_size", [25_000, 2, 1])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_to_columns_keeps_it(db, shape, value, batch_size):
    # a list of floats is a numpy matrix on the native build and a list of lists on the JVM build:
    # tolist() reads both
    _insert(db, shape, value)
    column = db.query("sql", QUERY).to_columns(batch_size=batch_size)["v"]
    cell = column[0]
    _check(cell.tolist() if isinstance(cell, np.ndarray) else cell, shape, value)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_to_dataframe_keeps_it(db, shape, value):
    _insert(db, shape, value)
    cell = db.query("sql", QUERY).to_dataframe()["v"].iloc[0]
    _check(cell.tolist() if isinstance(cell, np.ndarray) else cell, shape, value)


@pytest.mark.parametrize("batch_size", [25_000, 2, 1])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value", VALUES)
def test_to_arrow_keeps_it(db, shape, value, batch_size):
    _insert(db, shape, value)
    table = db.query("sql", QUERY).to_arrow(batch_size=batch_size)
    assert isinstance(table, pa.Table)
    _check(table.column("v").to_pylist()[0], shape, value)


@pytest.mark.parametrize("shape", ["list", "map"])
def test_rows_around_the_non_finite_one_are_unchanged(db, shape):
    """A row with no NaN, a row with one, a row whose text holds the tag character, in batches of one and two."""
    plain = [1.5, 2.5] if shape == "list" else {"x": 1.5, "y": "a\x01b"}
    holder = (
        [math.nan, math.inf, -math.inf]
        if shape == "list"
        else {"x": math.nan, "y": "a\x01b"}
    )
    with db.transaction():
        db.command("sql", "INSERT INTO Num SET k = 0, v = :v", {"v": plain})
        db.command("sql", "INSERT INTO Num SET k = 1, v = :v", {"v": holder})
        db.command("sql", "INSERT INTO Num SET k = 2, v = :v", {"v": plain})
    for batch_size in (2, 1):
        rows = db.query("sql", QUERY).to_json_list(batch_size=batch_size)
        assert rows[0]["v"] == plain and rows[2]["v"] == plain
        if shape == "list":
            assert math.isnan(rows[1]["v"][0])
            assert rows[1]["v"][1:] == [math.inf, -math.inf]
        else:
            assert math.isnan(rows[1]["v"]["x"]) and rows[1]["v"]["y"] == "a\x01b"
    for batch_size in (2, 1):
        arrow = (
            db.query("sql", QUERY)
            .to_arrow(batch_size=batch_size)
            .column("v")
            .to_pylist()
        )
        assert arrow[0] == plain
        if shape == "list":
            assert math.isnan(arrow[1][0]) and arrow[1][1:] == [math.inf, -math.inf]
        else:
            assert math.isnan(arrow[1]["x"])


def test_a_string_holding_the_tag_character_is_not_a_float(db):
    with db.transaction():
        db.command("sql", "INSERT INTO Num SET k = 0, v = :v", {"v": "\x01"})
        db.command(
            "sql", "INSERT INTO Num SET k = 1, v = :v", {"v": ["\x01", "f", "NaN"]}
        )
    rows = db.query("sql", QUERY).to_json_list()
    assert [r["v"] for r in rows] == ["\x01", ["\x01", "f", "NaN"]]
