"""``iter_json_batches`` on the native build reads the cursor one batch at a time (#239).

It used to drain the whole result into dicts, convert every row, and only then cut the list into batches, so ``batch_size`` bounded nothing:
the peak held every row of the result. Now at most one reply of the cursor (``_NEXT_BATCH`` rows) and one output batch are held, and the
cursor is read when the next batch is asked for. The rows, their order, the batch boundaries and the types are those of ``to_json_list``
(compared here on a result of several replies: ``arcadedb_n_exec`` returns the first 256 rows, ``arcadedb_n_next`` the following 2048 at a
time). conftest.py collects this file on the native backend only: it observes the result set's buffer, which the JVM build does not have.
"""

from datetime import date

import pytest

import arcadedb
from arcadedb._native import database

ROWS = 5000  # 256 in the first reply, then 2048 + 2048 + 648


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE R")
    temp_db.command("sql", "CREATE PROPERTY R.id INTEGER")
    temp_db.command("sql", "CREATE PROPERTY R.txt STRING")
    temp_db.command("sql", "CREATE PROPERTY R.val DOUBLE")
    temp_db.insert_many(
        "R", [{"id": i, "txt": f"t{i}", "val": i / 8.0} for i in range(ROWS)]
    )
    return temp_db


def _select(db):
    return db.query("sql", "SELECT id, txt, val FROM R ORDER BY id")


@pytest.fixture
def fills(monkeypatch):
    """The number of rows each read of the cursor (``_fill``) added to the result set's buffer, one entry per read."""
    seen = []
    real = database.NativeResultSet._fill

    def counting(self):
        before = len(self._buf)
        got = real(self)
        seen.append(len(self._buf) - before)
        return got

    monkeypatch.setattr(database.NativeResultSet, "_fill", counting)
    return seen


@pytest.mark.parametrize("size", [1, 7, 255, 256, 257, 2048, 2049, 4999, 5000, 5001])
def test_batches_equal_to_json_list_cut_at_batch_size(db, size):
    want = _select(db).to_json_list()
    assert len(want) == ROWS
    got = list(_select(db).iter_json_batches(batch_size=size))
    assert [len(b) for b in got] == [size] * (ROWS // size) + (
        [ROWS % size] if ROWS % size else []
    )
    assert [row for batch in got for row in batch] == want
    assert want[0] == {"id": 0, "txt": "t0", "val": 0.0}
    assert want[-1] == {"id": ROWS - 1, "txt": f"t{ROWS - 1}", "val": (ROWS - 1) / 8.0}


def test_default_batch_size_and_an_empty_result(db):
    rows = [r for b in _select(db).iter_json_batches() for r in b]
    assert [r["id"] for r in rows] == list(range(ROWS))
    assert list(db.query("sql", "SELECT FROM R WHERE id < 0").iter_json_batches()) == []


def test_the_last_batch_finishes_the_cursor(db):
    rs = db.query("sql", "SELECT id FROM R WHERE id < 10 ORDER BY id")
    (batch,) = list(rs.iter_json_batches(batch_size=10))  # exactly full, and the last
    assert len(batch) == 10
    assert (
        list(rs.iter_json_batches()) == [] and rs.to_list() == []
    )  # read to the end, not closed early


def test_rows_read_ahead_stay_within_one_reply_not_the_result(db, fills):
    """Rows read from the cursor but not yet handed to the caller stay within one reply plus one batch.

    The old code drained the whole result into dicts before it cut the first batch, so after the first batch it had read all
    ROWS rows and the caller held 100: 4900 rows ahead. The buffer alone cannot show that (the drain empties it as it goes), so
    this counts the rows read from the cursor against the rows delivered, at every batch.
    """
    rs = _select(db)
    size = 100
    read = len(rs._buf)  # the first reply, read when the query ran
    delivered = 0
    ahead = []
    for batch in rs.iter_json_batches(batch_size=size):
        delivered += len(batch)
        ahead.append(read + sum(fills) - delivered)
    assert delivered == ROWS and read + sum(fills) == ROWS
    assert len(ahead) == ROWS // size
    assert fills, "the cursor was never read past its first reply"
    assert max(fills) <= database._NEXT_BATCH
    assert ahead[0] <= database._NEXT_BATCH  # not ROWS - size
    assert max(ahead) <= database._NEXT_BATCH + size
    assert max(ahead) < ROWS // 2


def test_the_cursor_is_read_when_the_next_batch_is_asked_for(db, fills):
    rs = _select(db)
    batches = rs.iter_json_batches(batch_size=1000)
    first = next(batches)
    assert len(first) == 1000
    # the first batch needs one read past the first reply of 256 rows, not all of the 5000 rows
    assert len(fills) <= 1
    assert rs._more and not rs._closed
    rest = [r for b in batches for r in b]
    assert [r["id"] for r in first + rest] == list(range(ROWS))
    assert rs._exhausted and rs._closed


def test_close_between_two_batches_raises_and_closes_the_cursor(db):
    rs = _select(db)
    batches = rs.iter_json_batches(batch_size=100)
    assert len(next(batches)) == 100
    rs.close()
    assert rs._closed and not rs._more and not rs._buf
    with pytest.raises(arcadedb.ArcadeDBError, match="closed before"):
        next(batches)


def test_leaving_the_with_block_mid_iteration_closes_the_cursor(db):
    with _select(db) as rs:
        batches = rs.iter_json_batches(batch_size=100)
        assert len(next(batches)) == 100
    assert rs._closed and not rs._exhausted
    with pytest.raises(arcadedb.ArcadeDBError, match="closed before"):
        rs.to_list()


def test_an_abandoned_generator_leaves_the_result_to_its_owner(db):
    rs = _select(db)
    batches = rs.iter_json_batches(batch_size=100)
    next(batches)
    batches.close()  # the generator is gone, the cursor is the result set's
    assert not rs._closed
    rs.close()
    assert rs._closed


def test_a_batch_size_below_one_is_refused_when_the_first_batch_is_asked_for(db):
    batches = _select(db).iter_json_batches(batch_size=0)
    with pytest.raises(ValueError, match="at least 1"):
        next(batches)


@pytest.fixture
def walked(monkeypatch):
    """The number of rows ``json_native`` (the walk that converts DATE, DATETIME, DECIMAL, sets and record ids) was called for."""
    seen = [0]
    real = database.json_native

    def counting(value):
        seen[0] += 1
        return real(value)

    monkeypatch.setattr(database, "json_native", counting)
    return seen


def test_batches_of_an_untagged_result_are_not_walked(db, walked):
    got = [r for b in _select(db).iter_json_batches(batch_size=100) for r in b]
    assert len(got) == ROWS
    assert (
        walked[0] == 0
    ), "no reply was tagged, so iter_json_batches must skip json_native like to_json_list does"
    assert got == _select(db).to_json_list()


@pytest.mark.parametrize("size", [100, 1000])
def test_a_tag_in_a_later_reply_is_converted_in_every_batch_it_reaches(
    temp_db, walked, size
):
    """The first reply (256 rows) has no tagged value and a later one does: its rows must still arrive as JSON-native values."""
    temp_db.command("sql", "CREATE DOCUMENT TYPE D")
    temp_db.command("sql", "CREATE PROPERTY D.id INTEGER")
    temp_db.command("sql", "CREATE PROPERTY D.d DATE")
    rows = [{"id": i} for i in range(600)]
    for row in rows[300:]:
        row["d"] = date(2026, 1, 2)
    temp_db.insert_many("D", rows)
    query = "SELECT id, d FROM D ORDER BY id"
    want = temp_db.query("sql", query).to_json_list()
    walked[0] = 0
    got = [r for b in temp_db.query("sql", query).iter_json_batches(size) for r in b]
    assert got == want
    assert want[0].get("d") is None
    assert (
        type(want[599]["d"]) is int
    ), "the DATE is epoch milliseconds, not a date object"
    assert walked[0] > 0
