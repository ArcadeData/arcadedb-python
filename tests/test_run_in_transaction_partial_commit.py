"""
A block that already committed part of its work is never replayed, and its error says so, on both wheels
(#7916, ArcadeData/arcadedb-python#504).

A statement with a ``BATCH n`` boundary, or a ``commit()`` inside the block, makes part of the block's work durable. When the block
then loses a conflict, the rollback cannot take that part back, and a replay would apply it twice and report clean success. So
``run_in_transaction`` and ``with db.transaction():`` raise ``PartialCommitError``: not a ``ConflictError``, whose hint says to run
the transaction again, and with the conflict as its ``__context__``, not as its ``__cause__``, so that no retry loop finds it. The
conflict here is a real one (a second thread commits to a record the block also wrote), so the same test runs against the JPype
wheel and the native wheel, and it needs neither a Java class nor a fabricated error.

The native wheel's guard once sat in a ``try`` whose own ``except ArcadeDBError: pass`` swallowed the ``raise`` it was written to
make, so a native block that had committed part of its work was replayed.
"""

import threading

import pytest

import arcadedb as arcadedb
from arcadedb import ArcadeDBError, ConflictError, DuplicateKeyError, PartialCommitError


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


def _insert_rows(db):
    with db.transaction():
        for _ in range(25):
            db.command("sql", "INSERT INTO R SET k = 0, v = 0")


def _count(db, where="true"):
    sql = f"SELECT count(*) AS c FROM R WHERE {where}"  # nosec B608 - test-owned predicate text, no user input
    return db.query("sql", sql).first().get("c")


def _counter(db):
    return db.query("sql", "SELECT n FROM C").first().get("n")


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


PUBLISH = [
    pytest.param(_batch_boundary, 20, id="batch-boundary"),
    pytest.param(_commit_inside_the_block, 25, id="commit-inside-the-block"),
]


def _losing_block(db, publish, attempts):
    """A block that commits part of its work (``publish``), then loses a conflict at its commit on the first attempt."""

    def fn():
        attempts.append(1)
        if publish is not None:
            publish(db)
        db.command("sql", "UPDATE C SET n = n + 1")
        if len(attempts) == 1:
            _commit_from_another_thread(db)
        return "done"

    return fn


def _assert_refusal(error, block):
    assert type(error) is PartialCommitError
    assert not isinstance(error, ConflictError)
    assert error.code == "partial_commit"
    assert "Part of the work is already committed" in error.hint
    assert "check what was applied" in error.hint
    text = str(error)
    assert "run the whole transaction again" not in text
    assert (
        f"{block} committed part of its work before failing with ConcurrentModificationException"
        in text
    )
    # the conflict stays visible, but nothing that follows the cause chain to decide on a retry finds it
    assert error.__cause__ is None
    assert isinstance(error.__context__, ArcadeDBError)
    assert error.__context__.code == "conflict"


@pytest.mark.parametrize("retries", [5, 0], ids=["retries-left", "last-attempt"])
@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_a_block_that_committed_part_of_its_work_is_not_replayed(
    db, publish, durable_v1, retries
):
    _insert_rows(db)
    attempts = []

    with pytest.raises(PartialCommitError) as raised:
        db.run_in_transaction(
            _losing_block(db, publish, attempts), retries=retries, backoff_s=0
        )

    _assert_refusal(raised.value, "run_in_transaction block")
    assert (
        len(attempts) == 1
    ), "the block was replayed after it had committed part of its work"
    # a replay would have raised every durable row to v = 2 and reported success
    assert _count(db, "v = 2") == 0
    assert _count(db, "v = 1") == durable_v1
    assert not db.is_transaction_active()


@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_a_caller_that_follows_the_hint_applies_nothing_twice(db, publish, durable_v1):
    """A conflict's hint says to run the whole transaction again, and a program retries on ``ConflictError`` (#504)."""
    _insert_rows(db)
    attempts = []
    fn = _losing_block(db, publish, attempts)

    try:
        db.run_in_transaction(fn, retries=5, backoff_s=0)
    except ConflictError:
        db.run_in_transaction(fn, retries=5, backoff_s=0)
    except PartialCommitError:
        pass

    assert _count(db, "v = 2") == 0, "the committed part was applied a second time"
    assert _count(db, "v = 1") == durable_v1


@pytest.mark.parametrize("raised_by", ["end-of-the-block", "commit-inside-the-block"])
@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_a_with_block_that_committed_part_of_its_work_raises_the_same_error(
    db, publish, durable_v1, raised_by
):
    _insert_rows(db)

    with pytest.raises(PartialCommitError) as raised:
        with db.transaction():
            publish(db)
            db.command("sql", "UPDATE C SET n = n + 1")
            _commit_from_another_thread(db)
            if raised_by == "commit-inside-the-block":
                db.commit()

    _assert_refusal(raised.value, "Transaction block")
    assert _count(db, "v = 1") == durable_v1
    assert _counter(db) == 100
    assert not db.is_transaction_active()


def test_a_with_block_that_committed_nothing_still_raises_a_conflict(db):
    _insert_rows(db)  # the thread's transaction context has commits behind it

    with pytest.raises(ConflictError) as raised:
        with db.transaction():
            db.command("sql", "INSERT INTO R SET k = 1, v = 0")
            db.command("sql", "UPDATE C SET n = n + 1")
            _commit_from_another_thread(db)

    assert raised.value.code == "conflict"
    assert _count(db, "k = 1") == 0
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
    assert _counter(db) == 101


def test_a_nested_block_that_committed_nothing_is_still_retried(db):
    """Inside a caller's transaction the block has a transaction of its own, whose commits are what counts."""
    _insert_rows(db)  # the caller's context has commits behind it, the block's has none
    attempts = []

    with db.transaction():
        assert (
            db.run_in_transaction(
                _losing_block(db, None, attempts), retries=5, backoff_s=0
            )
            == "done"
        )

    assert len(attempts) == 2
    assert _counter(db) == 101


def test_a_nested_with_block_that_committed_nothing_still_raises_a_conflict(db):
    _insert_rows(db)

    with db.transaction():
        with pytest.raises(ConflictError):
            with db.transaction():
                db.command("sql", "UPDATE C SET n = n + 1")
                _commit_from_another_thread(db)

    assert _counter(db) == 100


@pytest.mark.parametrize("first_use", [False, True], ids=["used-thread", "new-thread"])
@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_a_nested_block_that_committed_part_of_its_work_is_not_replayed(
    db, publish, durable_v1, first_use
):
    """On a new thread the caller's context has committed nothing, on a used one it has: neither says what the block did."""
    _insert_rows(db)
    attempts = []
    outcome = []

    def caller():
        try:
            with db.transaction():
                db.run_in_transaction(
                    _losing_block(db, publish, attempts), retries=5, backoff_s=0
                )
        except BaseException as e:  # checked on the test's thread
            outcome.append(e)

    if first_use:
        thread = threading.Thread(target=caller)
        thread.start()
        thread.join()
    else:
        caller()

    assert len(attempts) == 1
    assert len(outcome) == 1
    _assert_refusal(outcome[0], "run_in_transaction block")
    assert _count(db, "v = 2") == 0
    assert _count(db, "v = 1") == durable_v1


@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_a_nested_with_block_that_committed_part_of_its_work_raises_the_same_error(
    db, publish, durable_v1
):
    _insert_rows(db)

    with db.transaction():
        with pytest.raises(PartialCommitError) as raised:
            with db.transaction():
                publish(db)
                db.command("sql", "UPDATE C SET n = n + 1")
                _commit_from_another_thread(db)

    _assert_refusal(raised.value, "Transaction block")
    assert _count(db, "v = 1") == durable_v1


@pytest.mark.parametrize("publish, durable_v1", PUBLISH)
def test_an_outer_run_in_transaction_does_not_replay_the_refusal(
    db, publish, durable_v1
):
    _insert_rows(db)
    outer_attempts = []
    inner_attempts = []

    def outer():
        outer_attempts.append(1)
        db.run_in_transaction(
            _losing_block(db, publish, inner_attempts), retries=5, backoff_s=0
        )

    with pytest.raises(PartialCommitError):
        db.run_in_transaction(outer, retries=5, backoff_s=0)

    assert (len(outer_attempts), len(inner_attempts)) == (1, 1)
    assert _count(db, "v = 1") == durable_v1
    assert not db.is_transaction_active()


def test_a_conflict_that_the_block_wraps_is_refused_too(db):
    """An exception of the program's own, raised from the conflict, is what ``run_in_transaction`` would retry on the JVM build."""
    _insert_rows(db)
    attempts = []

    def fn():
        attempts.append(1)
        _commit_inside_the_block(db)
        db.command("sql", "UPDATE C SET n = n + 1")
        _commit_from_another_thread(db)
        try:
            db.commit()
        except ConflictError as e:
            raise RuntimeError("could not store the step") from e

    with pytest.raises(PartialCommitError) as raised:
        db.run_in_transaction(fn, retries=5, backoff_s=0)

    assert len(attempts) == 1
    assert isinstance(raised.value.__context__, RuntimeError)
    assert raised.value.__cause__ is None


def test_a_duplicated_key_after_a_commit_is_raised_as_it_is(db):
    """Nobody is told to retry a duplicated key, so it keeps its class, as in the engine; the block is not run again either."""
    db.command("sql", "CREATE DOCUMENT TYPE U")
    db.command("sql", "CREATE PROPERTY U.k INTEGER")
    db.command("sql", "CREATE INDEX ON U (k) UNIQUE")
    attempts = []

    def fn():
        attempts.append(1)
        db.command("sql", "INSERT INTO U SET k = 1")
        db.commit()
        db.begin()
        db.command("sql", "INSERT INTO U SET k = 1")

    with pytest.raises(DuplicateKeyError):
        db.run_in_transaction(fn, retries=5, backoff_s=0)

    assert len(attempts) == 1


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
        assert type(error) is PartialCommitError, str(error)
        assert error.code == "partial_commit"
        assert error.cause == "com.arcadedb.exception.TransactionException"
    else:
        assert type(error) is ConflictError, str(error)
