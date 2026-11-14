"""``count()`` on the native build builds no result object for the rows it counts (#388, #315).

``NativeResultSet.count()`` used to walk the whole result with ``for _ in self``: one ``NativeResult`` was built per row (a tuple popped
from the buffer, an object with its own dict and meta) just to be counted and thrown away. It now takes the rows of a reply as ``len``.

The counts here are objects built, not seconds: deterministic. The behaviour tests pin what ``count()`` is for (the number of rows still
to be read, the state the result is left in, the errors) at sizes around the batch limits (256 rows in the first reply, 2048 per later
reply), so a cheaper count cannot buy a different answer. The oracle for the number is the rows the same query gives through
``to_list()``.

Needs the native library: conftest.py does not collect this file on the JVM build.
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from arcadedb import ArcadeDBError
from arcadedb._native import database, lib

ROWS = 5000  # 256 in the first reply, then 2048 + 2048 + 648
EDGE_SIZES = [0, 1, 255, 256, 257, 2303, 2304, 2305, ROWS]


@pytest.fixture
def db(temp_db):
    temp_db.command(
        "sql", "CREATE DOCUMENT TYPE R BUCKETS 1"
    )  # one bucket: a scan is in insertion order
    temp_db.command("sql", "CREATE PROPERTY R.id INTEGER")
    temp_db.command("sql", "CREATE PROPERTY R.txt STRING")
    temp_db.command("sql", "CREATE PROPERTY R.g INTEGER")
    temp_db.insert_many(
        "R", [{"id": i, "txt": f"t{i}", "g": i % 7} for i in range(ROWS)]
    )
    lib.flush_closes()  # a cursor closed earlier is not part of a call measured below
    return temp_db


@pytest.fixture
def built(monkeypatch):
    """The number of ``NativeResult`` objects built while it is installed."""
    seen = []
    real = database.NativeResult

    def counting(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(database, "NativeResult", counting)
    return seen


def _state(rs):
    return rs._closed, rs._exhausted, rs._more


# -- the number: the same as the rows the same query gives


@pytest.mark.parametrize("size", EDGE_SIZES)
def test_count_is_the_number_of_rows_to_list_gives(db, size):
    query = f"SELECT id, txt FROM R LIMIT {size}"
    rows = db.query("sql", query).to_list()
    assert len(rows) == size
    assert db.query("sql", query).count() == size


def test_count_of_typed_wide_rows_is_the_number_of_rows(db):
    db.command("sql", "CREATE DOCUMENT TYPE W")
    n = 2500
    rows = [
        {
            "i": i,
            "d": date(2020, 1, 1 + i % 28),
            "dt": datetime(2021, 2, 3, 4, 5, i % 60),
            "dec": Decimal(f"{i}.25"),
            "b": bytes([i % 256, 7]),
            "big": 2**60 + i,
            "v": [float(i), 1.5, 2.5],
            "m": {"a": i, "n": [i, {"z": None}]},
            "n": None,
        }
        for i in range(n)
    ]
    db.insert_many("W", rows)
    want = len(db.query("sql", "SELECT FROM W").to_list())
    assert want == n
    assert db.query("sql", "SELECT FROM W").count() == n


def test_count_of_aggregates_and_projections(db):
    assert db.query("sql", "SELECT count(*) AS c FROM R").count() == 1
    assert db.query("sql", "SELECT g, count(*) AS c FROM R GROUP BY g").count() == 7
    assert (
        db.query("sql", "SELECT @rid AS rid, @type AS t FROM R WHERE id < 3000").count()
        == 3000
    )


def test_count_of_a_cypher_result(db):
    db.command("sql", "CREATE VERTEX TYPE V")
    db.insert_many("V", [{"k": i} for i in range(2600)])
    assert db.query("opencypher", "MATCH (n:V) RETURN n.k AS k").count() == 2600


def test_count_of_a_command_drains_it_and_the_statement_ran_to_the_end(db):
    """A command's rows are pulled by the library when the statement runs and read late: counting them runs the statement through."""
    db.command("sql", "CREATE DOCUMENT TYPE B")
    with db.transaction():
        rs = db.command("sql", "INSERT INTO B FROM SELECT FROM R")
        assert rs.count() == ROWS
    assert db.query("sql", "SELECT count(*) AS c FROM B").first().get("c") == ROWS


# -- the state a result is left in is the one iteration leaves it in


def test_count_counts_only_the_rows_not_yet_read(db):
    rs = db.query("sql", "SELECT id FROM R")
    assert [next(rs).get("id") for _ in range(3)] == [0, 1, 2]
    assert rs.count() == ROWS - 3
    assert _state(rs) == (True, True, False)
    assert rs.count() == 0  # exhausted: nothing remains
    assert list(rs) == [] and rs.to_list() == [] and rs.first() is None


def test_count_reads_across_the_first_reply_after_rows_were_taken_from_it(db):
    rs = db.query("sql", "SELECT id FROM R")
    for _ in range(
        300
    ):  # past the 256 rows of the first reply: the cursor was read once
        next(rs)
    assert rs.count() == ROWS - 300


def test_count_of_an_empty_result_leaves_it_exhausted(db):
    rs = db.query("sql", "SELECT FROM R WHERE id < 0")
    assert rs.count() == 0
    assert _state(rs) == (True, True, False)
    assert rs.count() == 0


def test_count_after_the_result_was_closed_early_raises(db):
    rs = db.query("sql", "SELECT FROM R")
    rs.close()
    with pytest.raises(ArcadeDBError, match="closed before all its rows were read"):
        rs.count()


def test_count_on_a_closed_database_raises(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE T")
    temp_db.insert_many("T", [{"k": i} for i in range(600)])
    rs = temp_db.query("sql", "SELECT FROM T")
    temp_db.close()
    with pytest.raises(ArcadeDBError, match="Database is closed"):
        rs.count()


def test_count_raises_an_engine_error_met_after_the_first_reply_and_closes_the_result(
    db,
):
    # the first 256 rows are fine; the division by zero is at id 2999, in a later reply
    rs = db.query("sql", "SELECT 1 / (id - 2999) AS f FROM R")
    with pytest.raises(ArcadeDBError) as info:
        rs.count()
    assert str(info.value).startswith("Reading the result set failed: ")
    assert "Exception: " in str(info.value)
    assert _state(rs) == (True, True, False)
    assert list(rs) == []
    # the cursor is closed on the Java side, not left open: the database still reads and drops
    assert db.query("sql", "SELECT count(*) AS c FROM R").first().get("c") == ROWS


# -- objects: a row that is only counted is not turned into a result object


def test_count_builds_no_result_object(db, built):
    rs = db.query("sql", "SELECT id, txt FROM R")
    assert rs.count() == ROWS
    assert built == []
    rs = db.query("sql", "SELECT id, txt FROM R")
    next(rs)
    assert len(built) == 1  # the spy is in place: a row read by next() is one object
    assert rs.count() == ROWS - 1
    assert len(built) == 1
