"""Native build: the async executor releases the callback ids of its operations.

Every callback of an async call is a Python function that the library calls back by number (``extras._callbacks``). Only
``transaction`` used to give its numbers back, so a long-lived executor grew the table with every ``create_record``,
``query``, ``command``, ``new_edge``, ``new_edge_by_keys`` and ``scan_type`` call that carried a callback. The numbers of
the calls that were queued are released when ``wait_completion`` returns, or when the executor is closed. The two
executor-wide handlers, ``on_ok`` and ``on_error``, last until the executor is closed (a new handler replaces the old one).
"""

import pytest

import arcadedb
from arcadedb._native import extras

ROUNDS = 25


def _size():
    with extras._callbacks_lock:
        return len(extras._callbacks)


@pytest.fixture
def graph_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    temp_db.command("sql", "CREATE VERTEX TYPE P")
    temp_db.command("sql", "CREATE PROPERTY P.n INTEGER")
    temp_db.command("sql", "CREATE INDEX ON P (n) UNIQUE")
    temp_db.command("sql", "CREATE EDGE TYPE K")
    with temp_db.transaction():
        temp_db.new_vertex("P").set("n", 1).save()
        temp_db.new_vertex("P").set("n", 2).save()
    return temp_db


def _submit_every_kind(db, executor, seen):
    """One call of each kind that registers a callback, each with its own handlers."""
    a = db.query("sql", "SELECT FROM P WHERE n = 1").first().get_rid()
    b = db.query("sql", "SELECT FROM P WHERE n = 2").first().get_rid()
    doc = db.new_document("Item")
    doc.set("n", 1)
    executor.create_record(
        doc, callback=lambda r: seen.append("create_record"), error_callback=seen.append
    )
    executor.query(
        "sql",
        "SELECT FROM Item",
        lambda row: seen.append("query"),
        error_callback=seen.append,
    )
    executor.command(
        "sql",
        "INSERT INTO Item SET n = ?",
        callback=lambda row: seen.append("command"),
        args=(2,),
        error_callback=seen.append,
    )
    executor.new_edge(a, "K", b, callback=lambda *edge: seen.append("new_edge"))
    executor.new_edge_by_keys(
        "P",
        "n",
        1,
        "P",
        "n",
        2,
        False,
        "K",
        False,
        False,
        callback=lambda *edge: seen.append("new_edge_by_keys"),
    )
    executor.scan_type(
        "Item",
        lambda record: seen.append("scan_type"),
        error_callback=lambda rid, exc: seen.append(exc),
    )
    executor.transaction(
        lambda: db.command("sql", "INSERT INTO Item SET n = 3"),
        ok_callback=lambda: seen.append("transaction"),
        error_callback=seen.append,
    )


def test_wait_completion_gives_back_the_ids_of_every_kind_of_call(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    seen = []
    for _ in range(ROUNDS):
        _submit_every_kind(graph_db, executor, seen)
        assert _size() > baseline  # the calls hold their ids while they are queued
        executor.wait_completion(30_000)
        assert _size() == baseline
    executor.close()

    assert _size() == baseline
    # every kind ran its callback, so the ids were released after use and not instead of it
    assert {
        "create_record",
        "query",
        "command",
        "new_edge",
        "new_edge_by_keys",
        "scan_type",
        "transaction",
    } <= set(seen)
    assert not [s for s in seen if isinstance(s, Exception)]


def test_a_poll_that_finds_the_executor_busy_keeps_the_ids(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    seen = []
    _submit_every_kind(graph_db, executor, seen)
    try:
        executor.wait_completion(
            0
        )  # raises while the writers are busy, passes once they are not
    except TimeoutError:
        assert _size() > baseline
    executor.wait_completion(30_000)
    assert _size() == baseline
    executor.close()


def test_a_call_the_engine_refuses_gives_back_its_ids(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    with pytest.raises(arcadedb.ArcadeDBError):
        executor.new_edge(
            "#99:99",  # no such record: the engine refuses the call itself, no callback will ever run
            "K",
            "#99:98",
            callback=lambda *edge: None,
        )
    assert _size() == baseline
    executor.close()


def test_close_gives_back_the_ids_of_calls_that_were_never_waited_for(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    seen = []
    for _ in range(5):
        _submit_every_kind(graph_db, executor, seen)
    assert _size() > baseline
    executor.close()  # no wait_completion: closing is the other release point

    assert _size() == baseline


def test_closing_the_database_gives_back_the_ids_of_its_executors(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    _submit_every_kind(graph_db, executor, [])
    assert _size() > baseline
    graph_db.close()

    assert executor.is_closed()
    assert _size() == baseline


def test_on_ok_and_on_error_last_until_close_and_a_new_one_replaces_the_old(graph_db):
    baseline = _size()
    executor = graph_db.async_executor()
    executor.on_ok(lambda: None).on_error(lambda exc: None)
    assert _size() == baseline + 2
    for _ in range(ROUNDS):  # the same two slots every time, not two more each time
        executor.on_ok(lambda: None).on_error(lambda exc: None)
    assert _size() == baseline + 2

    executor.command("sql", "INSERT INTO Item SET n = 9")
    executor.wait_completion(30_000)
    assert _size() == baseline + 2  # a wait does not unhook the executor-wide handlers

    executor.close()
    assert _size() == baseline


def test_the_executor_wide_handlers_still_fire_after_many_waits(graph_db):
    ok = []
    executor = graph_db.async_executor().on_ok(lambda: ok.append(1))
    for n in range(5):
        executor.command("sql", "INSERT INTO Item SET n = ?", n)
        executor.wait_completion(30_000)
    executor.close()

    assert ok


def test_a_new_on_ok_replaces_the_old_one_in_the_engine(graph_db):
    # the id of a replaced handler goes back, so the engine must not call the replaced handler any more
    first, second = [], []
    executor = graph_db.async_executor().on_ok(lambda: first.append(1))
    executor.on_ok(lambda: second.append(1))
    executor.command("sql", "INSERT INTO Item SET n = 1")
    executor.wait_completion(30_000)
    executor.close()

    assert second
    assert not first


def test_calls_queued_while_another_thread_waits_keep_their_callbacks(graph_db):
    import threading

    baseline = _size()
    executor = graph_db.async_executor()
    ran, errors = [], []
    producers = 4
    per_producer = 50

    def produce(tag):
        try:
            for n in range(per_producer):
                executor.command(
                    "sql",
                    "INSERT INTO Item SET n = ?",
                    callback=lambda row: ran.append(1),
                    args=(tag * 1000 + n,),
                )
        except Exception as exc:  # pragma: no cover - reported by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=produce, args=(t,)) for t in range(producers)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        executor.wait_completion(
            30_000
        )  # releases the calls queued so far, not the later ones
    for thread in threads:
        thread.join()
    executor.wait_completion(30_000)
    executor.close()

    assert not errors
    assert len(ran) == producers * per_producer
    assert _size() == baseline
