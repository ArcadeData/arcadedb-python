"""``arcadedb.aio.AsyncDatabase``: the asyncio facade over the blocking calls of a ``Database`` (ArcadeData/arcadedb-python#306).

The facade runs each call on a worker thread of its own bounded pool, so these tests pin what the docs page ``how-to/asyncio.adoc`` says:
the rows and errors come back to the awaiting task, ``run_in_transaction`` keeps begin, the retries and the commit on one worker and rolls
back on an exception, many tasks can write at once, the pool never runs more than ``max_workers`` calls and does run that many, a cancelled
await abandons the result without stopping a call that started, and ``close()`` waits for the calls that were submitted. The file uses no
Java object, so it runs on the JVM build and on the native build alike. The tests wait on events and barriers, never on the wall clock.
"""

import asyncio
import contextvars
import threading
import time

import pytest

import arcadedb
from arcadedb.aio import AsyncDatabase

WAIT_S = (
    30  # a ceiling for an event or a barrier, so a broken test fails instead of hanging
)


def _count(db, type_name="Note"):
    return db.query("sql", f"SELECT count(*) AS n FROM {type_name}").first().get("n")


@pytest.fixture
def notes(temp_db):
    temp_db.command("sql", "CREATE DOCUMENT TYPE Note")
    return temp_db


def test_query_and_command_return_plain_rows_and_take_the_parameters_of_the_sync_calls(
    notes,
):
    async def main():
        async with AsyncDatabase(notes) as adb:
            created = await adb.command("sql", "CREATE DOCUMENT TYPE Other")
            assert isinstance(created, list)
            await adb.run_in_transaction(
                lambda: notes.command("sql", "INSERT INTO Note SET n = ?", 7)
            )
            positional = await adb.query("sql", "SELECT n FROM Note WHERE n = ?", 7)
            named = await adb.query("sql", "SELECT n FROM Note WHERE n = :n", {"n": 7})
            nothing = await adb.query("sql", "SELECT n FROM Note WHERE n = ?", 8)
        return positional, named, nothing

    positional, named, nothing = asyncio.run(main())
    for rows in (positional, named):
        assert all(isinstance(row, dict) for row in rows)
        assert [row["n"] for row in rows] == [7]
    assert nothing == []
    assert notes.is_open(), "closing the facade must not close the database"


def test_run_in_transaction_commits_on_one_worker_and_returns_what_the_function_returns(
    notes,
):
    threads = []

    def write():
        threads.append(threading.current_thread().name)
        notes.command("sql", "INSERT INTO Note SET n = 1")
        notes.command("sql", "INSERT INTO Note SET n = 2")
        threads.append(threading.current_thread().name)
        return "written"

    async def main():
        async with AsyncDatabase(notes, max_workers=4) as adb:
            return await adb.run_in_transaction(write)

    assert asyncio.run(main()) == "written"
    assert _count(notes) == 2
    assert len(set(threads)) == 1 and threads[0].startswith("arcadedb-aio")


def test_an_exception_in_run_in_transaction_rolls_back_and_reaches_the_task(notes):
    def write_then_fail():
        notes.command("sql", "INSERT INTO Note SET n = 1")
        raise ValueError("boom")

    async def main():
        # one worker: the check after the failure runs on the thread that ran the transaction, where a leftover transaction would be
        async with AsyncDatabase(notes, max_workers=1) as adb:
            with pytest.raises(ValueError, match="boom"):
                await adb.run_in_transaction(write_then_fail)
            assert await adb.run(notes.is_transaction_active) is False
            assert await adb.run(_count, notes) == 0
            # the worker is usable afterwards
            await adb.run_in_transaction(
                lambda: notes.command("sql", "INSERT INTO Note SET n = 2")
            )
            assert await adb.run(_count, notes) == 1

    asyncio.run(main())


def test_an_engine_error_is_raised_in_the_awaiting_task(notes):
    async def main():
        async with AsyncDatabase(notes) as adb:
            with pytest.raises(arcadedb.ArcadeDBError):
                await adb.query("sql", "SELEC nothing FROM Note")
            # the pool carries on after an error
            assert await adb.query("sql", "SELECT count(*) AS n FROM Note")

    asyncio.run(main())


def test_many_tasks_write_and_read_at_once_and_every_write_lands(notes):
    tasks = 200
    notes.command("sql", "CREATE DOCUMENT TYPE Counter")
    with notes.transaction():
        notes.command("sql", "INSERT INTO Counter SET id = 1, n = 0")

    async def one(adb, i):
        def work():
            notes.command("sql", "UPDATE Counter SET n = n + 1 WHERE id = 1")
            notes.command("sql", "INSERT INTO Note SET n = ?", i)

        await adb.run_in_transaction(work, retries=60)
        rows = await adb.query("sql", "SELECT count(*) AS n FROM Note")
        return rows[0]["n"]

    async def main():
        async with AsyncDatabase(notes, max_workers=8) as adb:
            return await asyncio.gather(*(one(adb, i) for i in range(tasks)))

    seen = asyncio.run(main())
    assert all(1 <= n <= tasks for n in seen)
    assert _count(notes) == tasks
    assert _count(notes, "Counter") == 1
    counter = notes.query("sql", "SELECT n FROM Counter WHERE id = 1").first().get("n")
    assert counter == tasks, f"{tasks - counter} of {tasks} increments were lost"
    values = [row.get("n") for row in notes.query("sql", "SELECT n FROM Note")]
    assert sorted(values) == list(range(tasks))


def test_the_pool_runs_max_workers_calls_at_once_and_never_more():
    class Plain:  # run() needs no engine call
        pass

    workers = 4
    barrier = threading.Barrier(workers)
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def call():
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.01)
        with lock:
            state["now"] -= 1

    async def main():
        async with AsyncDatabase(Plain(), max_workers=workers) as adb:
            assert adb.max_workers == workers
            # all four calls must be inside the barrier together: with fewer running at once the wait times out and raises
            await asyncio.gather(
                *(adb.run(barrier.wait, WAIT_S) for _ in range(workers))
            )
            await asyncio.gather(*(adb.run(call) for _ in range(5 * workers)))

    asyncio.run(main())
    assert 1 <= state["peak"] <= workers


def test_close_waits_for_a_running_call_and_a_closed_facade_refuses_new_calls():
    class Plain:
        pass

    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        assert release.wait(WAIT_S)
        return "done"

    async def main():
        adb = AsyncDatabase(Plain(), max_workers=2)
        call = asyncio.ensure_future(adb.run(slow))
        assert await asyncio.to_thread(started.wait, WAIT_S)
        closing = asyncio.ensure_future(adb.close())
        await asyncio.sleep(0.2)  # a close that did not wait would be done long before
        assert not closing.done(), "close() returned while a call was still running"
        assert adb.closed
        with pytest.raises(RuntimeError, match="closed"):
            await adb.run(lambda: None)
        release.set()
        assert await call == "done"
        await closing
        await adb.close()  # a second close has nothing left to wait for

    asyncio.run(main())


def test_close_also_runs_the_calls_that_are_still_queued():
    class Plain:
        pass

    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(WAIT_S)

    async def main():
        adb = AsyncDatabase(Plain(), max_workers=1)
        first = asyncio.ensure_future(adb.run(blocker))
        assert await asyncio.to_thread(started.wait, WAIT_S)
        second = asyncio.ensure_future(adb.run(ran.append, "second"))
        await asyncio.sleep(
            0.05
        )  # let the second call reach the queue behind the blocked worker
        closing = asyncio.ensure_future(adb.close())
        release.set()
        await first
        await second
        await closing

    asyncio.run(main())
    assert ran == ["second"]


def test_cancelling_the_await_abandons_the_result_but_a_started_call_finishes():
    class Plain:
        pass

    started, release, finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def slow():
        started.set()
        assert release.wait(WAIT_S)
        finished.set()

    async def main():
        adb = AsyncDatabase(Plain(), max_workers=1)
        task = asyncio.ensure_future(adb.run(slow))
        assert await asyncio.to_thread(started.wait, WAIT_S)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
        release.set()
        await adb.close()  # waits for the abandoned call
        assert finished.is_set()

    asyncio.run(main())


def test_a_call_that_still_waits_for_a_worker_does_not_run_when_its_task_is_cancelled():
    class Plain:
        pass

    started, release = threading.Event(), threading.Event()
    ran = []

    def blocker():
        started.set()
        assert release.wait(WAIT_S)

    async def main():
        adb = AsyncDatabase(Plain(), max_workers=1)
        first = asyncio.ensure_future(adb.run(blocker))
        assert await asyncio.to_thread(started.wait, WAIT_S)
        second = asyncio.ensure_future(adb.run(ran.append, "second"))
        await asyncio.sleep(
            0.05
        )  # the second call is queued behind the one blocked worker
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await first
        await adb.close()

    asyncio.run(main())
    assert ran == [], "a cancelled call that had not started still ran"


def test_the_context_variables_of_the_task_reach_the_call():
    class Plain:
        pass

    agent = contextvars.ContextVar("agent")

    async def main():
        async with AsyncDatabase(Plain()) as adb:
            agent.set("agent-7")
            return await adb.run(agent.get)

    assert asyncio.run(main()) == "agent-7"


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "8", None])
def test_max_workers_must_be_an_int_of_at_least_one(bad):
    with pytest.raises(ValueError, match="max_workers"):
        AsyncDatabase(object(), max_workers=bad)
