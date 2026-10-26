"""``AsyncExecutor.transaction``: a block that runs on an async writer inside a transaction, on the JVM wheel and the native wheel."""

import threading

import pytest

import arcadedb

TIMEOUT = 30


@pytest.fixture
def items_db(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Item")
    temp_db.command("sql", "CREATE PROPERTY Item.n INTEGER")
    return temp_db


def _count(db):
    return int(db.query("sql", "SELECT count(*) AS c FROM Item").first().get("c"))


def _insert(db, n):
    db.command("sql", "INSERT INTO Item SET n = ?", n)


def test_the_block_runs_in_a_transaction_and_commits(items_db):
    async_exec = items_db.async_executor()
    ran_on = []

    def block():
        ran_on.append(threading.current_thread())
        _insert(items_db, 1)
        _insert(items_db, 2)

    async_exec.transaction(block)
    async_exec.wait_completion(TIMEOUT * 1000)
    async_exec.close()

    assert _count(items_db) == 2
    # an async writer ran it, not the calling thread
    assert len(ran_on) == 1 and ran_on[0] is not threading.current_thread()


def test_the_ok_callback_fires_after_the_commit(items_db):
    async_exec = items_db.async_executor()
    done = threading.Event()
    seen_count = []

    async_exec.transaction(
        lambda: _insert(items_db, 7),
        retries=3,
        ok_callback=lambda: (seen_count.append(_count(items_db)), done.set()),
    )
    assert done.wait(TIMEOUT)
    async_exec.wait_completion(TIMEOUT * 1000)
    async_exec.close()

    assert seen_count == [1]
    assert _count(items_db) == 1


def test_a_failing_block_rolls_back_and_reaches_the_error_callback(items_db):
    async_exec = items_db.async_executor()
    failed = threading.Event()
    errors = []

    def block():
        _insert(items_db, 1)
        raise ValueError("boom")

    def on_error(exc):
        errors.append(exc)
        failed.set()

    async_exec.transaction(block, retries=1, error_callback=on_error)
    assert failed.wait(TIMEOUT)
    async_exec.wait_completion(TIMEOUT * 1000)
    async_exec.close()

    assert len(errors) == 1 and isinstance(errors[0], Exception)
    assert _count(items_db) == 0


def test_blocks_on_one_slot_run_in_submission_order(items_db):
    async_exec = items_db.async_executor()
    order = []

    for i in range(5):
        async_exec.transaction(
            lambda i=i: (order.append(i), _insert(items_db, i)), retries=1, slot=0
        )
    async_exec.wait_completion(TIMEOUT * 1000)
    async_exec.close()

    assert order == [0, 1, 2, 3, 4]
    assert _count(items_db) == 5
