"""Review of the vertex loads through the columnar insert (#335): a commit conflict is retried, a failed later batch is exact.

The old route, ``GraphBatch.create_vertices``, rolled a vertex chunk back and wrote it again (with a back-off, up to ``commitRetries``)
when its commit raised a ``NeedRetryException``: an MVCC conflict with another writer, a lock timeout. The columnar insert is one plain
transaction per chunk, so the same conflict would have surfaced as an error with earlier chunks already committed. The chunk loop of
``BulkOps.insertColumns`` (native wheel) and ``DocumentBatcher.insertColumns`` (JVM wheel) now does the retry, only in a transaction it
opened itself. Runs on both wheels.

* several threads load one vertex type with a NOTUNIQUE index on a handful of key values, in many small batches: every row is stored;
* a duplicate key in a later batch: the earlier batches stay committed, the failing batch leaves no row and no index entry, the message
  states the committed count, and the fixed tail loads afterwards.
"""

import threading

import numpy as np
import pyarrow as pa
import pytest

from arcadedb.exceptions import ArcadeDBError


def _count(db, name):
    sql = f"SELECT count(*) AS n FROM {name}"  # nosec B608 - test-owned type name
    return db.query("sql", sql).first().get("n")


def _vertex_type(db, name, unique=False):
    db.command("sql", f"CREATE VERTEX TYPE {name} BUCKETS 3")
    db.command("sql", f"CREATE PROPERTY {name}.id LONG")
    db.command("sql", f"CREATE PROPERTY {name}.grp INTEGER")
    db.command("sql", f"CREATE INDEX ON {name} (grp) NOTUNIQUE")
    if unique:
        db.command("sql", f"CREATE INDEX ON {name} (id) UNIQUE")


def _table(first, n, grps=5):
    ids = np.arange(first, first + n, dtype=np.int64)
    return pa.table({"id": ids, "grp": pa.array(ids % grps, pa.int32())})


@pytest.mark.parametrize("threads", [2, 4])
def test_threads_loading_one_vertex_type_in_small_batches_lose_no_row(temp_db, threads):
    _vertex_type(temp_db, "V")
    batches, batch_size = 20, 25
    errors = []

    def load(worker):
        try:
            table = _table(worker * 1_000_000, batches * batch_size)
            n = temp_db.insert_arrow("V", table, batch_size=batch_size)
            assert n == batches * batch_size
        except BaseException as e:  # noqa: BLE001 - reported below, in the test thread
            errors.append(repr(e))

    pool = [threading.Thread(target=load, args=(w,)) for w in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join(timeout=300)
    assert not errors, errors
    total = threads * batches * batch_size
    assert _count(temp_db, "V") == total
    distinct = temp_db.query("sql", "SELECT count(DISTINCT id) AS n FROM V")
    assert distinct.first().get("n") == total  # no row written twice by a retry
    by_group = temp_db.query("sql", "SELECT grp, count(*) AS n FROM V GROUP BY grp")
    assert sum(r.get("n") for r in by_group.to_list()) == total  # the index agrees


def test_a_duplicate_in_a_later_batch_keeps_the_earlier_batches_and_nothing_of_its_own(
    temp_db,
):
    _vertex_type(temp_db, "Dup", unique=True)
    ids = np.arange(500, dtype=np.int64)
    ids[250] = 7  # the third batch of five repeats a key that the first batch committed
    table = pa.table({"id": ids, "grp": pa.array(ids % 5, pa.int32())})
    with pytest.raises(ArcadeDBError, match="Duplicat") as err:
        temp_db.insert_arrow("Dup", table, batch_size=100)
    assert "the 200 row(s) of the batches before this one were committed" in str(
        err.value
    )
    assert temp_db.is_transaction_active() is False
    assert _count(temp_db, "Dup") == 200  # batches 1 and 2, whole
    for key in (200, 249, 251, 299):  # the failing batch left no index entry behind
        assert temp_db.lookup_by_key("Dup", ["id"], [key]) is None
    assert temp_db.lookup_by_key("Dup", ["id"], [7]).get("id") == 7  # the committed one

    ids[250] = 250  # the fixed tail loads
    tail = pa.table({"id": ids[200:], "grp": pa.array(ids[200:] % 5, pa.int32())})
    assert temp_db.insert_arrow("Dup", tail, batch_size=100) == 300
    assert _count(temp_db, "Dup") == 500
    assert temp_db.lookup_by_key("Dup", ["id"], [250]).get("id") == 250
