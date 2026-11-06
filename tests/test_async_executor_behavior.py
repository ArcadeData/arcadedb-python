"""The async executor, as the how-to page ``docs/modules/ROOT/pages/how-to/async-executor.adoc`` says it (ArcadeData/arcadedb-python#267, row 11).

``test_async_executor.py`` and ``test_async_transaction.py`` already pin most of the calls: exactness at parallel level 1, the
callbacks, ``wait_completion`` and its timeouts, ``close``, the ``transaction`` block and slot order. This file pins what the page
adds: one engine executor behind every ``db.async_executor()`` call, what ``close()`` does to the other callers, that callbacks run
off the calling thread, that a statement failure without an error callback is silent, and submitting from several threads. The
tests use SQL and no Java object, so they run on the JVM build and the native build alike.
"""

import threading

import pytest

from arcadedb import ArcadeDBError

TIMEOUT_MS = 30_000


@pytest.fixture
def db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    return temp_db


def _count(db):
    return int(db.query("sql", "SELECT count(*) AS c FROM Item").first().get("c"))


def test_every_call_wraps_the_one_executor_of_the_database(db):
    first = db.async_executor()
    second = db.async_executor()
    assert first is not second

    first.set_parallel_level(3).set_commit_every(77).set_transaction_sync(
        "yes_nometadata"
    )
    assert second.get_parallel_level() == 3
    assert second.get_commit_every() == 77
    assert second.get_transaction_sync() == "yes_nometadata"

    second.set_commit_every(5)
    assert first.get_commit_every() == 5


def test_close_shuts_the_executor_down_for_every_wrapper_of_it(db):
    first = db.async_executor()
    second = db.async_executor()
    first.close()

    assert first.is_closed() is True
    assert second.is_closed() is False
    with pytest.raises(Exception) as info:
        second.transaction(lambda: None)
    assert "shut down" in str(info.value)


def test_the_with_statement_closes_the_executor_at_the_end(db):
    with db.async_executor() as executor:
        executor.command("sql", "INSERT INTO Item SET n = 1")
        executor.wait_completion(TIMEOUT_MS)
        assert executor.is_closed() is False
    assert executor.is_closed() is True
    assert _count(db) == 1


def test_commands_are_stored_once_wait_completion_returns(db):
    executor = db.async_executor().set_parallel_level(1).set_commit_every(1000)
    for n in range(50):
        executor.command("sql", "INSERT INTO Item SET n = ?", args=[n])
    executor.wait_completion(TIMEOUT_MS)
    assert _count(db) == 50


def test_callbacks_run_on_a_thread_of_the_engine_not_the_one_that_submitted(db):
    caller = threading.get_ident()
    threads = {}

    def on_row(_row):
        threads["result"] = threading.get_ident()

    def on_error(_error):
        threads["error"] = threading.get_ident()

    executor = db.async_executor().set_parallel_level(1)
    executor.command("sql", "INSERT INTO Item SET n = 1", callback=on_row)
    executor.command("sql", "INSERT INTO Missing SET n = 1", error_callback=on_error)
    executor.wait_completion(TIMEOUT_MS)

    assert threads["result"] != caller
    assert threads["error"] != caller


def test_a_failed_statement_without_an_error_callback_raises_nowhere(db):
    executor = db.async_executor().set_parallel_level(1)
    executor.command("sql", "INSERT INTO Missing SET n = 1")
    executor.command("sql", "INSERT INTO Item SET n = 2")
    executor.wait_completion(TIMEOUT_MS)
    assert _count(db) == 1


def test_a_failed_statement_does_not_stop_the_ones_after_it(db):
    errors = []
    executor = db.async_executor().set_parallel_level(1)
    executor.command(
        "sql", "INSERT INTO Missing SET n = 1", error_callback=errors.append
    )
    executor.command("sql", "INSERT INTO Item SET n = 2")
    executor.wait_completion(TIMEOUT_MS)
    assert len(errors) == 1
    assert _count(db) == 1


def test_positional_and_named_parameters_together_raise_before_anything_is_queued(db):
    executor = db.async_executor()
    with pytest.raises(ValueError, match="positional args or named params"):
        executor.command("sql", "INSERT INTO Item SET n = ?", args=[1], n=2)
    with pytest.raises(ValueError, match="positional args or named params"):
        executor.query("sql", "SELECT FROM Item", lambda row: None, args=[1], n=2)
    executor.wait_completion(TIMEOUT_MS)
    assert _count(db) == 0


def test_a_query_callback_gets_every_row_of_the_result(db):
    for n in range(5):
        with db.transaction():
            db.command("sql", "INSERT INTO Item SET n = ?", n)
    seen = []
    executor = db.async_executor().set_parallel_level(1)
    executor.query(
        "sql", "SELECT n FROM Item ORDER BY n", lambda row: seen.append(row.get("n"))
    )
    executor.wait_completion(TIMEOUT_MS)
    assert seen == [0, 1, 2, 3, 4]


def test_several_threads_can_submit_to_one_executor(db):
    executor = db.async_executor().set_parallel_level(1).set_commit_every(1000)
    errors = []

    def submit(worker):
        try:
            for n in range(25):
                executor.command(
                    "sql", "INSERT INTO Item SET worker = ?, n = ?", args=[worker, n]
                )
        except Exception as error:  # noqa: BLE001 - reported by the assert below
            errors.append(error)

    threads = [threading.Thread(target=submit, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    executor.wait_completion(TIMEOUT_MS)

    assert errors == []
    assert _count(db) == 100


def test_closing_the_database_closes_every_executor_it_handed_out(temp_db_path):
    import arcadedb

    db = arcadedb.create_database(temp_db_path)
    executors = [db.async_executor(), db.async_executor()]
    db.close()
    assert [e.is_closed() for e in executors] == [True, True]


def test_an_executor_of_a_closed_database_cannot_be_asked_for(temp_db_path):
    import arcadedb

    db = arcadedb.create_database(temp_db_path)
    db.close()
    with pytest.raises(ArcadeDBError, match="closed"):
        db.async_executor()
