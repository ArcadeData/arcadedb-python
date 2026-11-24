"""A write outside a transaction commits on its own: the default of both packages, and the strict mode that turns it off.

The edge cases of flipping the default are pinned one by one: an explicit ``begin()`` must not autocommit, nested blocks, a failed
statement, a script, a query, ``save()`` and ``delete()``, the batch calls, another thread, the embedded server, a read-only
database, and the asyncio facade. The tests use SQL and the record API only, so they run on the JVM build and the native build
alike (ArcadeData/arcadedb-python, the transactions page ``docs/modules/ROOT/pages/how-to/transactions.adoc``).
"""

import asyncio
import threading

import pytest

import arcadedb
from arcadedb import ArcadeDBError
from arcadedb.aio import AsyncDatabase
from tests.conftest import TEST_PASSWORD, free_port


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE T")
    temp_db.command("sql", "CREATE PROPERTY T.k INTEGER")
    temp_db.command("sql", "CREATE INDEX ON T (k) UNIQUE")
    return temp_db


def _keys(db):
    return sorted(r["k"] for r in db.query("sql", "SELECT k FROM T").to_list())


def _insert(db, k):
    db.command("sql", "INSERT INTO T SET k = ?", k)


def test_a_database_that_was_just_created_has_auto_transaction_on(temp_db):
    assert temp_db.is_auto_transaction() is True


def test_a_database_that_was_opened_again_has_it_on_whatever_the_last_run_set(
    temp_db_path,
):
    with arcadedb.create_database(temp_db_path) as db:
        db.set_auto_transaction(False)
    with arcadedb.open_database(temp_db_path) as db:
        assert db.is_auto_transaction() is True
        db.command("sql", "CREATE DOCUMENT TYPE T")
        db.command("sql", "INSERT INTO T SET k = 1")
        assert db.is_transaction_active() is False
        assert db.query("sql", "SELECT count(*) AS n FROM T").first().get("n") == 1


def test_a_read_only_database_still_refuses_a_write_and_says_why(temp_db_path):
    with arcadedb.create_database(temp_db_path) as db:
        db.command("sql", "CREATE DOCUMENT TYPE T")
    with arcadedb.open_database(temp_db_path, read_only=True) as db:
        with pytest.raises(ArcadeDBError) as caught:
            db.command("sql", "INSERT INTO T SET k = 1")
        assert "Transaction not begun" not in str(caught.value)
        assert db.query("sql", "SELECT count(*) AS n FROM T").first().get("n") == 0


def test_a_bare_write_commits_at_once_and_leaves_no_transaction(db):
    _insert(db, 1)
    assert db.is_transaction_active() is False
    assert _keys(db) == [1]
    seen = []
    thread = threading.Thread(target=lambda: seen.append(_keys(db)))
    thread.start()
    thread.join()
    assert seen == [[1]]


def test_an_explicit_begin_is_not_committed_for_you(db):
    db.begin()
    _insert(db, 1)
    _insert(db, 2)
    assert db.is_transaction_active() is True
    db.rollback()
    assert _keys(db) == []
    with pytest.raises(RuntimeError, match="stop"):
        with db.transaction():
            _insert(db, 3)
            raise RuntimeError("stop")
    assert _keys(db) == []


def test_nested_blocks_keep_their_rules_and_an_inner_failure_rolls_back_only_the_inner(
    db,
):
    with db.transaction():
        _insert(db, 10)
        try:
            with db.transaction():
                _insert(db, 11)
                raise ValueError("inner")
        except ValueError:
            pass
        _insert(db, 12)
    assert _keys(db) == [10, 12]


def test_a_statement_that_fails_in_the_middle_leaves_nothing_of_its_own_rows(db):
    _insert(db, 1)
    with pytest.raises(ArcadeDBError, match="DuplicatedKeyException"):
        db.command("sql", "INSERT INTO T (k) VALUES (20), (21), (20)")
    assert db.is_transaction_active() is False
    assert _keys(db) == [1]
    _insert(db, 22)  # the handle is usable at once
    assert _keys(db) == [1, 22]


def test_a_script_commits_each_statement_alone_unless_it_has_begin_and_commit(db):
    with pytest.raises(ArcadeDBError, match="DuplicatedKeyException"):
        db.command("sqlscript", "INSERT INTO T SET k = 30; INSERT INTO T SET k = 30;")
    assert _keys(db) == [30]  # the first statement committed alone

    db.command(
        "sqlscript",
        "BEGIN; INSERT INTO T SET k = 31; INSERT INTO T SET k = 32; COMMIT;",
    )
    assert _keys(db) == [30, 31, 32]
    assert db.is_transaction_active() is False

    with pytest.raises(ArcadeDBError, match="DuplicatedKeyException"):
        db.command(
            "sqlscript",
            "BEGIN; INSERT INTO T SET k = 40; INSERT INTO T SET k = 30; COMMIT;",
        )
    assert 40 not in _keys(db)  # the whole script was undone
    assert db.is_transaction_active() is False


def test_a_query_opens_no_transaction_and_a_write_through_it_is_refused(db):
    db.query("sql", "SELECT FROM T").to_list()
    assert db.is_transaction_active() is False
    with pytest.raises(ArcadeDBError):
        db.query("sql", "INSERT INTO T SET k = 99").to_list()
    assert _keys(db) == []


def test_a_commit_with_nothing_open_still_raises_and_a_rollback_does_not(db):
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        db.commit()
    db.rollback()


def test_save_and_delete_of_a_record_commit_on_their_own(db):
    db.command("sql", "CREATE VERTEX TYPE V")
    db.command("sql", "CREATE EDGE TYPE E")
    first = db.new_vertex("V").set("n", 1).save()
    second = db.new_vertex("V").set("n", 2).save()
    first.new_edge("E", second).save()
    assert db.is_transaction_active() is False
    assert db.query("sql", "SELECT count(*) AS n FROM V").first().get("n") == 2
    assert db.query("sql", "SELECT count(*) AS n FROM E").first().get("n") == 1

    changed = db.lookup_by_rid(first.get_rid()).modify().set("n", 100)
    changed.save()
    assert db.is_transaction_active() is False
    assert db.lookup_by_rid(first.get_rid()).get("n") == 100

    gone = db.lookup_by_rid(second.get_rid()).modify()
    gone.delete()
    assert db.query("sql", "SELECT count(*) AS n FROM V").first().get("n") == 1


def test_in_strict_mode_a_save_outside_a_transaction_raises_and_writes_nothing(db):
    db.set_auto_transaction(False)
    with pytest.raises(ArcadeDBError, match="Transaction not begun"):
        db.new_document("T").set("k", 5).save()
    assert _keys(db) == []
    with db.transaction():
        db.new_document("T").set("k", 5).save()
    assert _keys(db) == [5]


def test_insert_many_keeps_its_own_transactions_with_the_default_on_or_off(db):
    for auto in (True, False):
        db.set_auto_transaction(auto)
        base = 1000 * (1 + int(auto))
        count = db.insert_many(
            "T", [{"k": base + i} for i in range(1000)], commit_every=250
        )
        assert count == 1000
        assert db.is_transaction_active() is False
        assert len([k for k in _keys(db) if base <= k < base + 1000]) == 1000


def test_insert_many_inside_a_transaction_of_yours_joins_it(db):
    db.begin()
    db.insert_many("T", [{"k": 2000 + i} for i in range(10)])
    assert db.is_transaction_active() is True
    db.rollback()
    assert _keys(db) == []


def test_graph_batch_keeps_its_own_transactions_and_still_refuses_yours(db):
    db.command("sql", "CREATE VERTEX TYPE City")
    with db.graph_batch(parallel_flush=False) as batch:
        batch.create_vertices("City", 500)
    assert db.is_transaction_active() is False
    assert db.query("sql", "SELECT count(*) AS n FROM City").first().get("n") == 500
    with db.graph_batch(parallel_flush=False) as batch:
        db.begin()
        try:
            with pytest.raises(ArcadeDBError):
                batch.create_vertices("City", 2)
            assert db.is_transaction_active() is True
        finally:
            db.rollback()


def test_a_thread_without_a_transaction_commits_alone_while_another_has_one_open(db):
    db.begin()
    _insert(db, 3000)
    outcome = []

    def worker():
        try:
            _insert(db, 3001)
            outcome.append("ok")
        except Exception as error:  # noqa: BLE001
            outcome.append(repr(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    db.rollback()
    assert outcome == ["ok"]
    assert _keys(db) == [
        3001
    ]  # the worker's row is its own commit; mine was rolled back


def test_run_in_transaction_still_runs_the_function_in_one_transaction(db):
    def work():
        _insert(db, 1)
        _insert(db, 2)
        assert db.is_transaction_active() is True
        return "done"

    assert db.run_in_transaction(work) == "done"
    assert _keys(db) == [1, 2]


def test_the_asyncio_facade_commits_a_bare_command(db):
    async def main():
        async with AsyncDatabase(db) as adb:
            await adb.command("sql", "INSERT INTO T SET k = 7")
            rows = await adb.query("sql", "SELECT k FROM T")
            return [r["k"] for r in rows]

    assert asyncio.run(main()) == [7]


def test_the_database_of_an_embedded_server_has_it_on_too(temp_server_root):
    with arcadedb.create_server(
        root_path=temp_server_root,
        root_password=TEST_PASSWORD,
        config={"http_port": free_port()},
    ) as server:
        db = server.create_database("autotx")
        assert db.is_auto_transaction() is True
        db.command("sql", "CREATE DOCUMENT TYPE T")
        db.command("sql", "INSERT INTO T SET k = 1")
        assert db.query("sql", "SELECT count(*) AS n FROM T").first().get("n") == 1
        again = server.get_database("autotx")
        assert again.is_auto_transaction() is True
