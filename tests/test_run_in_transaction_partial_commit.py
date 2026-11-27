"""
``Database.run_in_transaction`` never replays a block that already committed part of its work (#7916), on both wheels.

A statement with a ``BATCH n`` boundary, or a ``commit()`` inside the block, makes part of the block's work durable. When the block
then hits a retryable conflict, the rollback cannot take that part back, and a replay would apply it twice and report clean success.
The conflict has to be raised instead. The conflict here is a real one (a second thread commits to a record the block also wrote), so
the same test runs against the JPype wheel and the native wheel, and it needs neither a Java class nor a fabricated error.

The native wheel's guard once sat in a ``try`` whose own ``except ArcadeDBError: pass`` swallowed the ``raise`` it was written to
make, so a native block that had committed part of its work was replayed.
"""

import threading

import pytest

import arcadedb as arcadedb
from arcadedb import ArcadeDBError, ConflictError


@pytest.fixture
def db(temp_db_path):
    database = arcadedb.create_database(temp_db_path)
    database.command("sql", "CREATE DOCUMENT TYPE R")
    database.command("sql", "CREATE PROPERTY R.k INTEGER")
    database.command("sql", "CREATE PROPERTY R.v INTEGER")
    database.command("sql", "CREATE DOCUMENT TYPE C")
    database.command("sql", "CREATE PROPERTY C.n INTEGER")
    with database.transaction():
        database.command("sql", "INSERT INTO C SET n = 0")
    yield database
    database.drop()


def _count(db, where="true"):
    sql = f"SELECT count(*) AS c FROM R WHERE {where}"  # nosec B608 - test-owned predicate text, no user input
    return db.query("sql", sql).first().get("c")


def _commit_from_another_thread(db):
    """Commit a write to the record the block also wrote, so the block's own commit meets a ConcurrentModificationException."""
    errors = []

    def other():
        try:
            with db.transaction():
                db.command("sql", "UPDATE C SET n = n + 100")
        except BaseException as e:  # reported on the calling thread
            errors.append(e)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert errors == []


def _commit_inside_the_block(db):
    """Make the rows written so far durable with an explicit commit, then carry on in a new transaction."""
    db.command("sql", "UPDATE R SET v = v + 1")
    db.commit()
    db.begin()


def _batch_boundary(db):
    """The rows the BATCH boundaries committed (20 of 25) are durable; the remaining 5 sit in the open transaction."""
    db.command("sql", "UPDATE R SET v = v + 1 BATCH 10")


@pytest.mark.parametrize(
    "publish, durable_v1",
    [
        pytest.param(_batch_boundary, 20, id="batch-boundary"),
        pytest.param(_commit_inside_the_block, 25, id="commit-inside-the-block"),
    ],
)
def test_a_block_that_committed_part_of_its_work_is_not_replayed(
    db, publish, durable_v1
):
    with db.transaction():
        for _ in range(25):
            db.command("sql", "INSERT INTO R SET k = 0, v = 0")
    attempts = []

    def fn():
        attempts.append(1)
        publish(db)
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)
        return "done"

    with pytest.raises(ArcadeDBError, match="ConcurrentModification"):
        db.run_in_transaction(fn, retries=5, backoff_s=0)

    assert (
        len(attempts) == 1
    ), "the block was replayed after it had committed part of its work"
    # a replay would have raised every durable row to v = 2 and reported success
    assert _count(db, "v = 2") == 0
    assert _count(db, "v = 1") == durable_v1
    assert not db.is_transaction_active()


def test_a_block_that_committed_nothing_is_still_retried(db):
    """The guard must not turn every conflict into a refusal: a block with no durable part is replayed, once, and succeeds."""
    attempts = []

    def fn():
        attempts.append(1)
        db.command("sql", "INSERT INTO R SET k = 1, v = 0")
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)
        return "done"

    assert db.run_in_transaction(fn, retries=5, backoff_s=0) == "done"
    assert len(attempts) == 2
    assert _count(db, "k = 1") == 1
    # the first attempt was rolled back; the other thread added 100 and the second attempt 1
    assert db.query("sql", "SELECT n FROM C").first().get("n") == 101


@pytest.mark.parametrize(
    "publish",
    [
        pytest.param(_batch_boundary, id="batch-boundary"),
        pytest.param(_commit_inside_the_block, id="commit-inside-the-block"),
    ],
)
def test_a_caller_that_follows_the_hint_applies_nothing_twice(db, publish):
    """The error's hint says to run the whole transaction again, and a program retries on ``ConflictError`` (#504)."""
    _insert_rows(db)
    attempts = []

    def fn():
        attempts.append(1)
        publish(db)
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)

    try:
        db.run_in_transaction(fn, retries=5, backoff_s=0)
    except arcadedb.ConflictError as e:
        print(type(e).__name__, e.code, e.hint)
        db.run_in_transaction(fn, retries=5, backoff_s=0)
    except ArcadeDBError:
        pass

    assert _count(db, "v = 2") == 0, "the committed part was applied a second time"


def _insert_rows(db):
    with db.transaction():
        for _ in range(25):
            db.command("sql", "INSERT INTO R SET k = 0, v = 0")


def _engine_raises_its_own_refusal(db):
    """Whether the engine's retry loops answer such a block with a ``TransactionException`` of their own (engine PR 13406).

    None when it cannot be told from here: the native build on a 26.12.1 snapshot, which may be older or newer than the change.
    """
    if arcadedb.__backend__ != "native":
        import jpype

        context = jpype.JClass("com.arcadedb.database.TransactionContext")
        return hasattr(context, "notReplayable")
    version = str(db.query("sql", "SELECT version() AS v").first().get("v")).split()[0]
    if version.startswith("26.11."):
        return False
    return None if "SNAPSHOT" in version else True


def test_the_engines_own_refusal_gets_the_same_class_and_code(db):
    """``COMMIT RETRY`` is a retry loop of the engine. From engine 26.12.1 it refuses a block that committed part of its work with
    a ``TransactionException`` of its own; an older engine raises the conflict.
    """
    _insert_rows(db)
    calls = []

    def conflict_once():
        calls.append(1)
        if len(calls) == 1:
            _commit_from_another_thread(db)
        return len(calls)

    db.register_function("conflict_once", conflict_once)
    script = (
        "BEGIN;"
        "UPDATE R SET v = v + 1 BATCH 10;"
        "UPDATE C SET n = n + 1;"
        "LET called = SELECT conflict_once() AS c;"
        "COMMIT RETRY 5;"
    )
    try:
        with pytest.raises(ArcadeDBError) as raised:
            db.command("sqlscript", script).to_list()
    finally:
        db.unregister_function("conflict_once")

    error = raised.value
    assert calls == [1], "the engine replayed the block"
    assert _count(db, "v = 1") == 20
    refused = "committed part of its work" in str(error)
    expected = _engine_raises_its_own_refusal(db)
    if expected is not None:
        assert refused is expected, str(error)
    if refused:
        assert type(error).__name__ == "PartialCommitError", str(error)
        assert error.code == "partial_commit"
        assert error.cause == "com.arcadedb.exception.TransactionException"
    else:
        assert type(error) is ConflictError, str(error)


def test_a_nested_block_that_committed_nothing_is_still_retried(db):
    """Inside a caller's transaction the block has a transaction of its own, whose commits are what counts."""
    _insert_rows(db)
    attempts = []

    def fn():
        attempts.append(1)
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)

    with db.transaction():
        db.run_in_transaction(fn, retries=5, backoff_s=0)

    assert len(attempts) == 2


def test_a_nested_block_with_a_commit_is_not_replayed_on_a_new_thread(db):
    _insert_rows(db)
    attempts = []

    def fn():
        attempts.append(1)
        _commit_inside_the_block(db)
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)

    def caller():
        try:
            with db.transaction():
                db.run_in_transaction(fn, retries=5, backoff_s=0)
        except ArcadeDBError:
            pass

    thread = threading.Thread(target=caller)
    thread.start()
    thread.join()

    assert len(attempts) == 1
    assert _count(db, "v = 2") == 0, "the committed part was applied a second time"
