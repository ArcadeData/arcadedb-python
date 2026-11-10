"""Review of the native parallel loads (#328): nothing is lost silently, an interrupt leaves no writer running and no token behind.

``insert_columns(parallel=True)`` hands the records to the async writers in one crossing and waits for them before it returns. These
tests pin what a caller relies on: a record that a writer rejects is counted (``stored + reported failed >= rows``) and no key is stored twice
(a writer that meets a duplicate abandons its whole open batch, and reports it); a load at the writers' batch boundary stores every row and nothing lands after the call returns;
a ^C during the crossing, or between the chunks of the row path, waits for the writers and reads (so removes) the failure token; a wait
that fails still reads the tokens; and the message of a failed parallel batch inside a caller's transaction does not claim that a
rollback removes rows the writers committed. Needs the native library: conftest.py collects this file on the native backend only.
"""

import re
import time

import numpy as np
import pytest

from arcadedb._native import database as native_database
from arcadedb._native import extras
from arcadedb.exceptions import ArcadeDBError

COLP = "arcadedb_n_insert_colp"


def _count(db, name):
    sql = f"SELECT count(*) AS n FROM {name}"  # nosec B608 - test-owned type name
    return db.query("sql", sql).to_list()[0]["n"]


def _type(db, name="Col", unique=False):
    db.command("sql", f"CREATE DOCUMENT TYPE {name} BUCKETS 3")
    db.command("sql", f"CREATE PROPERTY {name}.id INTEGER")
    db.command("sql", f"CREATE PROPERTY {name}.txt STRING")
    if unique:
        db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")


class TokenSpy:
    """The failure tokens that were made and the ones that were read (a read removes the token's entries in the image)."""

    def __init__(self, monkeypatch):
        self.made, self.read = [], []
        real_make, real_read = extras.failure_token, extras.failures
        real_rows = extras.insert_parallel

        def make(database):
            fid = real_make(database)
            self.made.append(fid)
            return fid

        def failures(database, token):
            self.read.append(token)
            return real_read(database, token)

        def rows(database, type_name, chunk):
            fid = real_rows(database, type_name, chunk)
            self.made.append(fid["fid"])
            return fid

        monkeypatch.setattr(extras, "failure_token", make)
        monkeypatch.setattr(extras, "failures", failures)
        monkeypatch.setattr(extras, "insert_parallel", rows)


@pytest.fixture
def tokens(monkeypatch):
    return TokenSpy(monkeypatch)


def _failed_count(error):
    return int(re.search(r"reported (\d+) failed record", str(error)).group(1))


def test_a_duplicate_key_is_reported_and_no_row_is_stored_twice(temp_db):
    _type(temp_db, unique=True)
    n, dupes = 3_000, 50
    ids = np.arange(n, dtype=np.int64)
    ids[n - dupes :] = np.arange(dupes)  # the last 50 repeat the first 50 keys
    with pytest.raises(ArcadeDBError, match="failed record") as err:
        temp_db.insert_columns("Col", {"id": ids, "txt": ["t"] * n}, parallel=True)
    stored_ids = [r["id"] for r in temp_db.query("sql", "SELECT id FROM Col").to_list()]
    stored = len(stored_ids)
    assert (
        stored + _failed_count(err.value) >= n
    )  # every row is a stored row or a reported one
    # A writer that meets a duplicate key rolls back its WHOLE open batch (up to commit_every rows), so the valid rows of that batch are
    # abandoned and reported too: `stored` can be below the number of distinct keys, never above it.
    assert stored <= len(set(ids.tolist()))
    assert set(stored_ids) <= set(ids.tolist())  # only ids that were sent
    assert len(stored_ids) == len(set(stored_ids))  # none stored twice


def test_a_load_at_the_writers_batch_boundary_stores_every_row_and_nothing_later(
    temp_db,
):
    _type(temp_db)
    threads = temp_db.async_executor().get_parallel_level()
    n = threads * 10_240 + 1  # one row past a full batch for every writer
    ids = np.arange(n, dtype=np.int64)
    assert (
        temp_db.insert_columns("Col", {"id": ids, "txt": ["t"] * n}, parallel=True) == n
    )
    stored = _count(temp_db, "Col")
    assert stored == n
    time.sleep(0.3)
    temp_db.async_executor().wait_completion()
    assert _count(temp_db, "Col") == n  # nothing landed after the call returned
    distinct = temp_db.query("sql", "SELECT count(DISTINCT id) AS c FROM Col")
    assert distinct.to_list()[0]["c"] == n


def test_an_interrupt_in_the_crossing_waits_for_the_writers_and_reads_the_token(
    temp_db, tokens, monkeypatch
):
    _type(temp_db)
    real_call = native_database._call

    def interrupted(prefix, name, *args):
        reply = real_call(prefix, name, *args)
        if name == COLP:
            raise KeyboardInterrupt  # the records are handed over; the reply never reaches the caller
        return reply

    monkeypatch.setattr(native_database, "_call", interrupted)
    n = 30_000
    with pytest.raises(KeyboardInterrupt):
        temp_db.insert_columns(
            "Col", {"id": np.arange(n, dtype=np.int64), "txt": ["t"] * n}, parallel=True
        )
    assert (
        temp_db.async_executor().is_processing() is False
    )  # waited: no writer runs on
    assert len(tokens.made) == 1 and tokens.read == tokens.made  # and the token is gone
    assert (
        _count(temp_db, "Col") == n
    )  # the rows handed over were all written, none half


def test_an_interrupt_between_the_chunks_of_the_row_path_waits_and_reads_the_tokens(
    temp_db, tokens, monkeypatch
):
    _type(temp_db)
    real_rows = extras.insert_parallel
    seen = []

    def second_chunk_interrupted(database, type_name, chunk):
        if seen:
            raise KeyboardInterrupt
        seen.append(1)
        return real_rows(database, type_name, chunk)

    monkeypatch.setattr(extras, "insert_parallel", second_chunk_interrupted)
    rows = [{"id": i, "txt": "t"} for i in range(25_000)]
    with pytest.raises(KeyboardInterrupt):
        temp_db.insert_many("Col", rows, parallel=True)
    assert temp_db.async_executor().is_processing() is False
    assert _count(temp_db, "Col") == 10_000  # the first chunk, whole
    assert len(tokens.made) == 1 and tokens.read == tokens.made


def test_a_wait_that_fails_still_reads_the_tokens(temp_db, tokens, monkeypatch):
    _type(temp_db)
    real_wait = extras.NativeAsyncExecutor.wait_completion
    state = {"raised": False}

    def failing_wait(self, timeout_ms=None):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("the wait failed")
        return real_wait(self, timeout_ms)

    monkeypatch.setattr(extras.NativeAsyncExecutor, "wait_completion", failing_wait)
    n = 2_000
    with pytest.raises(ArcadeDBError, match="the wait failed"):
        temp_db.insert_columns(
            "Col", {"id": np.arange(n, dtype=np.int64), "txt": ["t"] * n}, parallel=True
        )
    assert len(tokens.made) == 1 and tokens.read == tokens.made
    temp_db.async_executor().wait_completion()
    assert _count(temp_db, "Col") == n


def test_a_failed_parallel_batch_in_a_caller_transaction_does_not_promise_a_rollback(
    temp_db,
):
    import pyarrow as pa

    _type(temp_db, unique=True)
    ids = list(range(300))
    ids[250] = 7  # a duplicate key in the third batch of 100
    table = pa.table({"id": pa.array(ids, pa.int64()), "txt": ["t"] * 300})
    temp_db.begin()
    try:
        with pytest.raises(ArcadeDBError) as err:
            temp_db.insert_arrow("Col", table, batch_size=100, parallel=True)
        message = str(err.value)
        assert "not committed" not in message and "your transaction" in message
        assert "async writers" in message and "does not remove them" in message
    finally:
        temp_db.rollback()
    temp_db.async_executor().wait_completion()
    assert (
        _count(temp_db, "Col") >= 200
    )  # the writers committed the first two batches: the rollback did not remove them
